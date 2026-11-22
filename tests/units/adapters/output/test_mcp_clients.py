import json
from types import SimpleNamespace

import httpx
from pydantic import RootModel

from adapters.output.ocr.mcp_ocr_client import MCPOCRClient, _result_data
from adapters.output.recipe_agent._mcp_registry import (
    _ClientCredentialsTokenProvider,
    _MCPRecipeRegistry,
)


def test_ocr_result_data_supports_all_fastmcp_shapes():
    assert _result_data(SimpleNamespace(data={"text": "x"}, structured_content=None, content=[])) == {"text": "x"}
    assert _result_data(SimpleNamespace(data=None, structured_content={"result": {"text": "y"}}, content=[])) == {"text": "y"}
    assert _result_data(SimpleNamespace(data=None, structured_content={"text": "z"}, content=[])) == {"text": "z"}
    block = SimpleNamespace(text=json.dumps({"text": "t"}))
    assert _result_data(SimpleNamespace(data=None, structured_content=None, content=[block])) == {"text": "t"}
    assert _result_data(SimpleNamespace(data=None, structured_content=None, content=[])) is None


async def test_ocr_client_calls_tool(monkeypatch):
    captured = {}

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def call_tool(self, name, arguments, **kwargs):
            captured.update(name=name, arguments=arguments)
            return SimpleNamespace(data={"text": "Texte OCR suffisamment long"}, structured_content=None, content=[])

    monkeypatch.setattr("adapters.output.ocr.mcp_ocr_client.fastmcp.Client", _Client)
    result = await MCPOCRClient("http://ocr/mcp").extract(image_url="https://example.test/image.jpg")
    assert result.startswith("Texte OCR")
    assert captured["name"] == "extract_recipe_text"


async def test_recipe_registry_call_and_public_wrappers(monkeypatch):
    registry = _MCPRecipeRegistry("http://recipe/mcp", "http://unused", "tenant")
    calls = []

    async def fake_call(name, args):
        calls.append((name, args))
        return {"uuid": "id"}

    monkeypatch.setattr(registry, "_call", fake_call)
    await registry.create_ingredient("farine", "g")
    await registry.create_equipment("fouet")
    await registry.create_recipe({"name": "Cake"})
    await registry.get_recipe("id")
    await registry.update_recipe("id", {"name": "Cake"})
    await registry.generate_recipe_image("id")
    assert [name for name, _ in calls] == [
        "create_ingredient",
        "create_equipment",
        "create_recipe",
        "get_recipe",
        "update_recipe",
        "generate_recipe_image",
    ]


async def test_recipe_registry_mcp_transport_result(monkeypatch):
    captured = {}

    class _Client:
        def __init__(self, *args, **kwargs):
            captured["headers"] = args[0].headers

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def call_tool(self, name, args, **kwargs):
            return SimpleNamespace(data={"uuid": "recipe-1"}, structured_content=None, content=[])

    monkeypatch.setattr("adapters.output.recipe_agent._mcp_registry.fastmcp.Client", _Client)
    registry = _MCPRecipeRegistry(
        "http://recipe/mcp",
        "http://unused",
        "tenant",
        oauth_token_url="http://keycloak/token",
        oauth_client_id="agent",
        oauth_client_secret="secret",
        oauth_scopes=["recipes:read"],
    )
    assert registry._token_provider is not None
    monkeypatch.setattr(registry._token_provider, "get_token", lambda: _async_value("access-token"))
    assert await registry.get_recipe("recipe-1") == {"uuid": "recipe-1"}
    assert captured["headers"] == {
        "X-Tenant-URI": "tenant",
        "Authorization": "Bearer access-token",
    }


async def _async_value(value):
    return value


async def test_client_credentials_token_provider_fetches_and_caches(monkeypatch):
    captured = []

    class _Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {"access_token": "access-token", "expires_in": 300}

    class _AsyncClient:
        def __init__(self, *args, **kwargs):
            captured.append(("client", kwargs))

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, *, data):
            captured.append((url, data))
            return _Response()

    monkeypatch.setattr("adapters.output.recipe_agent._mcp_registry.httpx.AsyncClient", _AsyncClient)
    provider = _ClientCredentialsTokenProvider(
        token_url="http://keycloak/token",
        client_id="agent",
        client_secret="secret",
        scopes=["recipes:read", "recipes:write"],
    )

    assert await provider.get_token() == "access-token"
    assert await provider.get_token() == "access-token"
    assert captured == [
        ("client", {"timeout": 15.0}),
        (
            "http://keycloak/token",
            {
                "grant_type": "client_credentials",
                "client_id": "agent",
                "client_secret": "secret",
                "scope": "recipes:read recipes:write",
            },
        ),
    ]


async def test_recipe_registry_retries_once_after_unauthorized(monkeypatch):
    registry = _MCPRecipeRegistry(
        "http://recipe/mcp",
        "http://unused",
        "tenant",
        oauth_token_url="http://keycloak/token",
        oauth_client_id="agent",
        oauth_client_secret="secret",
    )
    response = httpx.Response(401, request=httpx.Request("POST", "http://recipe/mcp"))
    calls = []

    async def fake_call(name, args):
        calls.append((name, args))
        if len(calls) == 1:
            raise httpx.HTTPStatusError("unauthorized", request=response.request, response=response)
        return {"uuid": "recipe-1"}

    assert registry._token_provider is not None
    invalidations = []
    monkeypatch.setattr(registry, "_call_http_once", fake_call)
    monkeypatch.setattr(registry._token_provider, "invalidate", lambda: invalidations.append(True))

    assert await registry._call_http("get_recipe", {"uuid": "recipe-1"}) == {"uuid": "recipe-1"}
    assert invalidations == [True]
    assert len(calls) == 2


async def test_recipe_registry_converts_generated_root_models(monkeypatch):
    class IngredientResult(RootModel[dict[str, str]]):
        pass

    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def call_tool(self, name, args, **kwargs):
            return SimpleNamespace(
                data=[IngredientResult({"uuid": "ingredient-1", "name": "oeuf"})],
                structured_content={"result": [{"uuid": "ingredient-1", "name": "oeuf"}]},
                content=[],
            )

    monkeypatch.setattr("adapters.output.recipe_agent._mcp_registry.fastmcp.Client", _Client)
    registry = _MCPRecipeRegistry("http://recipe/mcp", "http://unused", "tenant")
    assert await registry.list_ingredients("oeuf") == [{"uuid": "ingredient-1", "name": "oeuf"}]
