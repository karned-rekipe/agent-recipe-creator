from langchain_core.messages import AIMessage

from adapters.output.recipe_agent.agent_adapter import RecipeAgentAdapter
from domain.models.recipe import RecipePlan


class _Graph:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.state: dict | None = None

    async def ainvoke(self, state: dict) -> dict:
        self.state = state
        return self.result


class _Registry:
    async def get_recipe(self, uuid: str) -> dict:
        return {
            "uuid": uuid,
            "name": "Soupe",
            "description": "2 carottes. Cuire vingt minutes.",
            "ingredients": [],
            "steps": [],
        }


def _graph_result(recipe_uuid="recipe-1"):
    return {
        "messages": [AIMessage(content="ok")],
        "plan": RecipePlan(name="Soupe"),
        "resolved_ingredients": {"carotte": "ingredient-1"},
        "resolved_ustensils": {},
        "resolved_components": [],
        "recipe_uuid": recipe_uuid,
        "recipe_exists": False,
        "existing_recipe_uuid": None,
        "existing_recipe_name": None,
        "image_generated": True,
        "warnings": [],
    }


async def test_process_builds_extended_state_and_result():
    adapter = object.__new__(RecipeAgentAdapter)
    graph = _Graph(_graph_result())
    adapter._graph = graph
    result = await adapter.process("Recette de soupe", "run-1", generate_image=True)
    assert result.created is True
    assert result.image_generated is True
    assert graph.state is not None
    assert graph.state["generate_image"] is True


async def test_enrich_loads_existing_recipe_and_updates_in_place():
    adapter = object.__new__(RecipeAgentAdapter)
    graph = _Graph(_graph_result())
    adapter._enrichment_graph = graph
    adapter._registry = _Registry()
    result = await adapter.enrich("recipe-1", "run-1")
    assert result.created is False
    assert result.updated is True
    assert graph.state is not None
    assert graph.state["existing_recipe"]["name"] == "Soupe"
    assert "FICHE EXISTANTE JSON" in graph.state["messages"][0].content
