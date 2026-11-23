from adapters.output.recipe_agent._nodes import (
    make_generate_image_node,
    make_normalize_origin_country_node,
    make_update_recipe_node,
)
from domain.models.recipe import IngredientLine, RecipePlan, RecipeSource, RecipeStep, UstensilLine
from domain.ports.recipe_agent import RecipeEnrichmentStage


class _Registry:
    def __init__(
        self,
        *,
        image_error: Exception | None = None,
        country_normalization: dict | Exception | None = None,
    ) -> None:
        self.updated: tuple[str, dict] | None = None
        self.generated: list[str] = []
        self.image_error = image_error
        self.country_normalization = country_normalization or {"origin_country": "FR", "warning": None}

    async def normalize_origin_country(self, origin_country: str) -> dict:
        if isinstance(self.country_normalization, Exception):
            raise self.country_normalization
        return self.country_normalization

    async def update_recipe(self, uuid: str, payload: dict) -> dict:
        self.updated = (uuid, payload)
        return {"uuid": uuid, **payload}

    async def generate_recipe_image(self, uuid: str) -> dict:
        if self.image_error:
            raise self.image_error
        self.generated.append(uuid)
        return {"recipe": {"uuid": uuid, "main_image": "/image.png"}}


def _state(**updates):
    base = {
        "messages": [],
        "plan": RecipePlan(
            name="Soupe",
            description="Une soupe courte.",
            servings="4 personnes",
            origin_country="FR",
            difficulty=2,
            price=1,
            sources=[RecipeSource(name="Livre")],
            ingredients=[IngredientLine(name="carotte", quantity="3", unit="pièce")],
            ustensils=[UstensilLine(name="casserole")],
            steps=[RecipeStep(
                title="Cuire",
                instruction="Cuire les légumes.",
                preparation_minutes=5,
                cooking_minutes=20,
                rest_minutes=2,
            )],
        ),
        "resolved_ingredients": {"carotte": "ingredient-1"},
        "resolved_ustensils": {"casserole": "equipment-1"},
        "available_tags": [],
        "resolved_tags": {"soupe": "tag-2"},
        "resolved_components": [],
        "recipe_uuid": "recipe-1",
        "recipe_exists": False,
        "allow_duplicate": False,
        "existing_recipe_uuid": "recipe-1",
        "existing_recipe_name": "Soupe",
        "existing_recipe": {
            "uuid": "recipe-1",
            "name": "Soupe",
            "origin_country": None,
            "difficulty": None,
            "price": None,
            "main_image": None,
            "secondary_images": ["/old.png"],
            "favorite": True,
            "tag_uuids": ["tag-1"],
            "sources": [{"name": "Cahier", "uri": None}],
            "components": [],
            "status": "draft",
            "quality_status": "text_only",
        },
        "target_recipe_uuid": "recipe-1",
        "input_sources": [],
        "generate_image": True,
        "image_generated": False,
        "warnings": [],
        "error": None,
    }
    return base | updates


async def test_update_recipe_reclassifies_and_preserves_unrelated_fields():
    registry = _Registry()
    result = await make_update_recipe_node(registry)(_state())
    assert registry.updated is not None
    uuid, payload = registry.updated
    assert uuid == "recipe-1"
    assert payload["origin_country"] == "FR"
    assert payload["favorite"] is True
    assert payload["secondary_images"] == ["/old.png"]
    assert payload["status"] == "active"
    assert payload["quality_status"] == "structured"
    assert payload["tag_uuids"] == ["tag-1", "tag-2"]
    assert payload["ingredients"][0]["ingredient_uuid"] == "ingredient-1"
    assert payload["steps"][0] == {
        "name": "Cuire",
        "description": "Cuire les légumes.",
        "preparation_time": 5,
        "cooking_time": 20,
        "rest_time": 2,
        "rank": 1,
    }
    assert {item["name"] for item in payload["sources"]} == {"Livre", "Cahier"}
    assert "mise à niveau" in result["messages"][0]["content"]


async def test_generate_image_success_skip_and_warning():
    registry = _Registry()
    node = make_generate_image_node(registry)
    assert (await node(_state()))["image_generated"] is True
    assert registry.generated == ["recipe-1"]

    assert (await node(_state(generate_image=False)))["image_generated"] is False
    assert (await node(_state(existing_recipe={"main_image": "/existing.png"})))["image_generated"] is False

    failed = await make_generate_image_node(_Registry(image_error=RuntimeError("media down")))(_state())
    assert failed["image_generated"] is False
    assert "media down" in failed["warnings"][0]


async def test_normalize_origin_country_ignores_unknown_code_without_stopping_enrichment():
    state = _state(plan=RecipePlan(name="Soupe", origin_country="XK"))
    result = await make_normalize_origin_country_node(_Registry(
        country_normalization={
            "origin_country": None,
            "warning": "Le code pays IA 'XK' est inconnu ou incompatible ; le pays d'origine a été ignoré.",
        },
    ))(state)

    assert result["plan"].origin_country is None
    assert "a été ignoré" in result["warnings"][0]


async def test_normalize_origin_country_continues_when_validation_service_fails():
    state = _state(plan=RecipePlan(name="Soupe", origin_country="FR"))
    result = await make_normalize_origin_country_node(_Registry(
        country_normalization=RuntimeError("backend down"),
    ))(state)

    assert result["plan"].origin_country is None
    assert "n'a pas pu être validé" in result["warnings"][0]


async def test_enrichment_nodes_report_real_progress_stages():
    stages: list[RecipeEnrichmentStage] = []

    async def report(stage: RecipeEnrichmentStage) -> None:
        stages.append(stage)

    state = _state(progress_callback=report)
    await make_update_recipe_node(_Registry())(state)
    await make_generate_image_node(_Registry())(state)

    assert stages == ["saving", "image"]
