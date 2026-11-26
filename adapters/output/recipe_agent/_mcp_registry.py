import asyncio
import json
import time
from typing import Any

import fastmcp
from fastmcp.client.transports import StreamableHttpTransport
import httpx
from pydantic import BaseModel

from adapters.output.recipe_agent._logger import log as logger


def _to_plain_data(value: Any) -> Any:
    """Convert FastMCP generated Pydantic result models into plain data."""

    if isinstance(value, BaseModel):
        return _to_plain_data(value.model_dump(mode="json"))
    if isinstance(value, dict):
        return {key: _to_plain_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_data(item) for item in value]
    return value


class _ClientCredentialsTokenProvider:
    def __init__(self, *, token_url: str, client_id: str, client_secret: str, scopes: list[str]) -> None:
        self._token_url = token_url
        self._client_id = client_id
        self._client_secret = client_secret
        self._scopes = scopes
        self._access_token: str | None = None
        self._expires_at = 0.0
        self._lock = asyncio.Lock()

    async def get_token(self) -> str:
        if self._access_token is not None and time.monotonic() < self._expires_at:
            return self._access_token
        async with self._lock:
            if self._access_token is not None and time.monotonic() < self._expires_at:
                return self._access_token
            await self._refresh()
            if self._access_token is None:
                raise RuntimeError("Keycloak did not return an OAuth access token")
            return self._access_token

    def invalidate(self) -> None:
        self._access_token = None
        self._expires_at = 0.0

    async def _refresh(self) -> None:
        data = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scopes:
            data["scope"] = " ".join(self._scopes)
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(self._token_url, data=data)
            response.raise_for_status()
            payload = response.json()
        token = payload.get("access_token")
        if not isinstance(token, str) or not token:
            raise RuntimeError("Keycloak did not return an OAuth access token")
        expires_in = max(float(payload.get("expires_in", 60)), 1.0)
        refresh_margin = min(30.0, expires_in / 10)
        self._access_token = token
        self._expires_at = time.monotonic() + max(expires_in - refresh_margin, 1.0)


class _MCPRecipeRegistry:
    def __init__(
        self,
        url: str,
        api_url: str,
        tenant_uri: str,
        *,
        oauth_token_url: str | None = None,
        oauth_client_id: str | None = None,
        oauth_client_secret: str | None = None,
        oauth_scopes: list[str] | None = None,
    ) -> None:
        self._mcp_url = url
        self._api_url = api_url.rstrip("/")
        self._tenant_uri = tenant_uri
        self._token_provider = None
        if oauth_token_url and oauth_client_id and oauth_client_secret:
            self._token_provider = _ClientCredentialsTokenProvider(
                token_url=oauth_token_url,
                client_id=oauth_client_id,
                client_secret=oauth_client_secret,
                scopes=oauth_scopes or [],
            )

    async def _call(self, tool_name: str, args: dict) -> Any:
        logger.debug(f"  → MCP {tool_name}({json.dumps(args, ensure_ascii = False)})")
        t0 = time.perf_counter()
        result = await self._call_http(tool_name, args)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if result is None:
            logger.debug(f"  ← MCP {tool_name} None ({elapsed_ms:.0f}ms)")
            return None
        if not result:
            logger.debug(f"  ← MCP {tool_name} [] ({elapsed_ms:.0f}ms)")
            return []
        if isinstance(result, list) and isinstance(result[0], dict) and result[0].get("type") == "text":
            parsed = json.loads(result[0]["text"])
            logger.debug(f"  ← MCP {tool_name} {json.dumps(parsed, ensure_ascii = False)[:200]} ({elapsed_ms:.0f}ms)")
            return parsed

        logger.debug(f"  ← MCP {tool_name} {repr(result)[:200]} ({elapsed_ms:.0f}ms)")
        return result

    async def _call_http(self, tool_name: str, args: dict) -> Any:
        try:
            return await self._call_http_once(tool_name, args)
        except httpx.HTTPStatusError as exc:
            if self._token_provider is None or exc.response.status_code != 401:
                raise
            self._token_provider.invalidate()
            return await self._call_http_once(tool_name, args)

    async def _call_http_once(self, tool_name: str, args: dict) -> Any:
        headers = {"X-Tenant-URI": self._tenant_uri}
        if self._token_provider is not None:
            headers["Authorization"] = f"Bearer {await self._token_provider.get_token()}"
        transport = StreamableHttpTransport(
            self._mcp_url,
            headers=headers,
        )
        async with fastmcp.Client(transport, timeout=120.0) as client:
            result = await client.call_tool(tool_name, args, timeout=120.0)
        if result.structured_content:
            if set(result.structured_content) == {"result"}:
                return result.structured_content["result"]
            return result.structured_content
        for block in result.content:
            text = getattr(block, "text", None)
            if text:
                return json.loads(text)
        if result.data is not None:
            return _to_plain_data(result.data)
        return None

    # ── Ingredients ───────────────────────────────────────────────────────────

    async def list_ingredients(self, name: str) -> list[dict]:
        return await self._call("list_ingredients", {"name": name})

    async def create_ingredient(self, name: str, unit: str | None) -> dict:
        return await self._call("create_ingredient", {"name": name, "unit": unit})

    async def get_ingredient(self, uuid: str) -> dict | None:
        return await self._call("get_ingredient", {"uuid": uuid})

    async def update_ingredient(self, uuid: str, name: str, unit: str | None) -> dict:
        return await self._call("update_ingredient", {"uuid": uuid, "name": name, "unit": unit})

    async def delete_ingredient(self, uuid: str) -> None:
        await self._call("delete_ingredient", {"uuid": uuid})

    async def duplicate_ingredient(self, uuid: str) -> dict:
        return await self._call("duplicate_ingredient", {"uuid": uuid})

    async def purge_ingredients(self) -> dict:
        return await self._call("purge_ingredients", {})

    # ── Equipment ─────────────────────────────────────────────────────────────

    async def list_equipment(self, name: str) -> list[dict]:
        return await self._call("list_equipment", {"name": name})

    async def create_equipment(self, name: str) -> dict:
        return await self._call("create_equipment", {"name": name})

    async def list_ustensils(self, name: str) -> list[dict]:
        return await self.list_equipment(name)

    async def create_ustensil(self, name: str) -> dict:
        return await self.create_equipment(name)

    # ── Tags ──────────────────────────────────────────────────────────────────

    async def list_tags(self) -> list[dict]:
        return await self._call("list_tags", {})

    # ── Recipes ───────────────────────────────────────────────────────────────

    async def list_recipes(self, name: str) -> list[dict]:
        return await self._call("list_recipes", {"name": name})

    async def create_recipe(self, payload: dict) -> dict:
        return await self._call("create_recipe", {"payload": payload})

    async def get_recipe(self, uuid: str) -> dict | None:
        return await self._call("get_recipe", {"uuid": uuid})

    async def normalize_origin_country(self, origin_country: str | None) -> dict:
        return await self._call("normalize_origin_country", {"origin_country": origin_country})

    async def update_recipe(self, uuid: str, payload: dict) -> dict:
        return await self._call("update_recipe", {"uuid": uuid, "payload": payload})

    async def generate_recipe_image(self, uuid: str) -> dict:
        return await self._call(
            "generate_recipe_image",
            {"uuid": uuid, "replace_existing": False},
        )

    async def delete_recipe(self, uuid: str) -> None:
        await self._call("delete_recipe", {"uuid": uuid})

    async def duplicate_recipe(self, uuid: str) -> dict:
        return await self._call("duplicate_recipe", {"uuid": uuid})

    async def purge_recipes(self) -> dict:
        return await self._call("purge_recipes", {})
