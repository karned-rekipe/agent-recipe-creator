import json
import time
from typing import Any

import fastmcp
from fastmcp.client.transports import StreamableHttpTransport
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


class _MCPRecipeRegistry:
    def __init__(self, url: str, api_url: str, tenant_uri: str) -> None:
        self._mcp_url = url
        self._api_url = api_url.rstrip("/")
        self._tenant_uri = tenant_uri

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
        transport = StreamableHttpTransport(
            self._mcp_url,
            headers={"X-Tenant-URI": self._tenant_uri},
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

    # ── Recipes ───────────────────────────────────────────────────────────────

    async def list_recipes(self, name: str) -> list[dict]:
        return await self._call("list_recipes", {"name": name})

    async def create_recipe(self, payload: dict) -> dict:
        return await self._call("create_recipe", {"payload": payload})

    async def get_recipe(self, uuid: str) -> dict | None:
        return await self._call("get_recipe", {"uuid": uuid})

    async def update_recipe(self, uuid: str, payload: dict) -> dict:
        return await self._call("update_recipe", {"uuid": uuid, "payload": payload})

    async def generate_recipe_image(self, uuid: str) -> dict:
        return await self._call(
            "generate_recipe_image",
            {"uuid": uuid, "prompt": None, "replace_existing": False},
        )

    async def delete_recipe(self, uuid: str) -> None:
        await self._call("delete_recipe", {"uuid": uuid})

    async def duplicate_recipe(self, uuid: str) -> dict:
        return await self._call("duplicate_recipe", {"uuid": uuid})

    async def purge_recipes(self) -> dict:
        return await self._call("purge_recipes", {})
