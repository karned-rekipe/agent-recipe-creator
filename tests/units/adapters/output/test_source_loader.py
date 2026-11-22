import json

import httpx
import pytest

from adapters.output.source_loader.web_recipe_source_loader import (
    UnsafeRecipeURL,
    WebRecipeSourceLoader,
    _find_recipe_json_ld,
    _format_recipe,
    _instruction_text,
)


def _transport(response_factory):
    async def handler(request: httpx.Request) -> httpx.Response:
        return response_factory(request)

    return httpx.MockTransport(handler)


def _public_host(_: str) -> list[str]:
    return ["8.8.8.8"]


async def test_load_extracts_schema_org_recipe_and_source():
    recipe = {
        "@context": "https://schema.org",
        "@type": "Recipe",
        "name": "Tarte Tatin",
        "description": "Tarte aux pommes caramélisées",
        "recipeYield": "6 personnes",
        "recipeCuisine": "Française",
        "prepTime": "PT20M",
        "cookTime": "PT40M",
        "totalTime": "PT60M",
        "recipeIngredient": ["6 pommes", "100 g de sucre"],
        "recipeInstructions": [
            {"@type": "HowToStep", "text": "Peler les pommes."},
            {"@type": "HowToSection", "name": "Cuisson", "itemListElement": [
                {"@type": "HowToStep", "text": "Cuire 40 minutes."}
            ]},
        ],
    }
    html = f'<html><script type="application/ld+json">{json.dumps(recipe)}</script></html>'
    loader = WebRecipeSourceLoader(
        transport=_transport(lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"})),
        host_resolver=_public_host,
    )
    result = await loader.load("https://recipes.example/tatin")
    assert "Nom: Tarte Tatin" in result.raw_text
    assert "6 pommes" in result.raw_text
    assert "Cuire 40 minutes" in result.raw_text
    assert result.sources[0].uri == "https://recipes.example/tatin"


async def test_load_falls_back_to_visible_html_text():
    html = "<html><style>hidden</style><script>alert(1)</script><body><h1>Soupe</h1><p>Deux carottes et une pomme de terre. Mixer puis servir.</p></body></html>"
    loader = WebRecipeSourceLoader(
        transport=_transport(lambda request: httpx.Response(200, text=html, headers={"content-type": "text/html"})),
        host_resolver=_public_host,
    )
    result = await loader.load("https://recipes.example/soupe")
    assert "Deux carottes" in result.raw_text
    assert "hidden" not in result.raw_text
    assert "alert" not in result.raw_text


async def test_load_follows_a_public_redirect():
    def response(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/old":
            return httpx.Response(302, headers={"location": "/new"})
        return httpx.Response(
            200,
            text="<html><body>Recette de soupe avec légumes, cuire vingt minutes.</body></html>",
            headers={"content-type": "text/html"},
        )

    loader = WebRecipeSourceLoader(transport=_transport(response), host_resolver=_public_host)
    result = await loader.load("https://recipes.example/old")
    assert result.sources[0].uri == "https://recipes.example/new"


@pytest.mark.parametrize(
    "url",
    ["file:///tmp/recipe.html", "https://user:pass@example.test/recipe"],
)
async def test_load_rejects_invalid_url_shape(url: str):
    loader = WebRecipeSourceLoader(transport=_transport(lambda request: httpx.Response(200)), host_resolver=_public_host)
    with pytest.raises(UnsafeRecipeURL):
        await loader.load(url)


async def test_load_rejects_private_address():
    loader = WebRecipeSourceLoader(
        transport=_transport(lambda request: httpx.Response(200)),
        host_resolver=lambda host: ["127.0.0.1"],
    )
    with pytest.raises(UnsafeRecipeURL, match="public"):
        await loader.load("http://localhost/recipe")


async def test_load_rejects_non_html_and_empty_pages():
    non_html = WebRecipeSourceLoader(
        transport=_transport(lambda request: httpx.Response(200, content=b"pdf", headers={"content-type": "application/pdf"})),
        host_resolver=_public_host,
    )
    with pytest.raises(ValueError, match="HTML"):
        await non_html.load("https://recipes.example/file")

    empty = WebRecipeSourceLoader(
        transport=_transport(lambda request: httpx.Response(200, text="<html></html>", headers={"content-type": "text/html"})),
        host_resolver=_public_host,
    )
    with pytest.raises(ValueError, match="No usable"):
        await empty.load("https://recipes.example/empty")


def test_json_ld_helpers_handle_graph_invalid_blocks_and_nested_instructions():
    payload = {"@graph": [{"@type": ["Thing", "Recipe"], "name": "Cake"}]}
    assert _find_recipe_json_ld(["invalid", json.dumps(payload)])["name"] == "Cake"
    assert _find_recipe_json_ld([json.dumps({"@type": "Thing"})]) is None
    assert _instruction_text("Mélanger") == ["Mélanger"]
    assert _instruction_text({"name": "Section", "itemListElement": ["Cuire"]}) == ["Section", "Cuire"]
    assert _instruction_text(42) == []
    assert "Nom: Cake" in _format_recipe({"name": "Cake", "recipeCategory": "Dessert"})
