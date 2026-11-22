import json
from types import SimpleNamespace

from pydantic import RootModel

from adapters.output.ocr.mcp_ocr_client import MCPOCRClient, _result_data
from adapters.output.recipe_agent._mcp_registry import _MCPRecipeRegistry


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
    class _Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def call_tool(self, name, args, **kwargs):
            return SimpleNamespace(data={"uuid": "recipe-1"}, structured_content=None, content=[])

    monkeypatch.setattr("adapters.output.recipe_agent._mcp_registry.fastmcp.Client", _Client)
    registry = _MCPRecipeRegistry("http://recipe/mcp", "http://unused", "tenant")
    assert await registry.get_recipe("recipe-1") == {"uuid": "recipe-1"}


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
