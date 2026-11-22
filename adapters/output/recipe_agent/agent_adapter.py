from langchain_core.messages import HumanMessage
from langgraph.graph import END, StateGraph

from adapters.output.recipe_agent._fuzzy import _make_fuzzy_matcher
from adapters.output.recipe_agent._mcp_registry import _MCPRecipeRegistry
from adapters.output.recipe_agent._nodes import (
    make_check_recipe_node,
    make_create_components_node,
    make_create_recipe_node,
    make_generate_image_node,
    make_plan_node,
    make_review_plan_node,
    make_resolve_ingredients_node,
    make_resolve_ustensils_node,
    make_update_recipe_node,
)
from adapters.output.recipe_agent._planner import _PydanticAIPlanner
from adapters.output.recipe_agent._state import RecipeAgentState
from domain.models.recipe import RecipeResult, RecipeSource
from domain.ports.recipe_agent import RecipeAgentPort, RecipeProgressCallback
from infrastructure.config import AgentConfig


def _route_check_recipe(state: RecipeAgentState) -> str:
    return "exists" if state["recipe_exists"] else "new"


def _build_graph(planner: _PydanticAIPlanner, registry: _MCPRecipeRegistry, matcher):
    g = StateGraph(RecipeAgentState)

    g.add_node("plan", make_plan_node(planner))
    g.add_node("review_plan", make_review_plan_node(planner))
    g.add_node("check_recipe", make_check_recipe_node(registry, matcher))
    g.add_node("create_components", make_create_components_node(registry, matcher))
    g.add_node("resolve_ingredients", make_resolve_ingredients_node(registry, matcher))
    g.add_node("resolve_ustensils", make_resolve_ustensils_node(registry, matcher))
    g.add_node("create_recipe", make_create_recipe_node(registry))
    g.add_node("generate_image", make_generate_image_node(registry))

    g.set_entry_point("plan")
    g.add_edge("plan", "review_plan")
    g.add_edge("review_plan", "check_recipe")
    g.add_conditional_edges("check_recipe", _route_check_recipe, {"exists": END, "new": "create_components"})
    g.add_edge("create_components", "resolve_ingredients")
    g.add_edge("resolve_ingredients", "resolve_ustensils")
    g.add_edge("resolve_ustensils", "create_recipe")
    g.add_edge("create_recipe", "generate_image")
    g.add_edge("generate_image", END)

    return g.compile()


def _build_enrichment_graph(planner: _PydanticAIPlanner, registry: _MCPRecipeRegistry, matcher):
    g = StateGraph(RecipeAgentState)
    g.add_node("plan", make_plan_node(planner))
    g.add_node("review_plan", make_review_plan_node(planner))
    g.add_node("create_components", make_create_components_node(registry, matcher))
    g.add_node("resolve_ingredients", make_resolve_ingredients_node(registry, matcher))
    g.add_node("resolve_ustensils", make_resolve_ustensils_node(registry, matcher))
    g.add_node("update_recipe", make_update_recipe_node(registry))
    g.add_node("generate_image", make_generate_image_node(registry))
    g.set_entry_point("plan")
    g.add_edge("plan", "review_plan")
    g.add_edge("review_plan", "create_components")
    g.add_edge("create_components", "resolve_ingredients")
    g.add_edge("resolve_ingredients", "resolve_ustensils")
    g.add_edge("resolve_ustensils", "update_recipe")
    g.add_edge("update_recipe", "generate_image")
    g.add_edge("generate_image", END)
    return g.compile()


class RecipeAgentAdapter(RecipeAgentPort):
    def __init__(self, config: AgentConfig) -> None:
        planner = _PydanticAIPlanner(config.lm.planner)
        registry = _MCPRecipeRegistry(config.mcp_registry.url, config.recipe_api.url, config.recipe_api.tenant_uri)
        matcher = _make_fuzzy_matcher(config.fuzzy.threshold)
        self._graph = _build_graph(planner, registry, matcher)
        self._enrichment_graph = _build_enrichment_graph(planner, registry, matcher)
        self._registry = registry

    async def process(
            self,
            raw_text: str,
            run_uuid: str,
            *,
            allow_duplicate: bool = False,
            sources: list[RecipeSource] | None = None,
            generate_image: bool = True,
            progress_callback: RecipeProgressCallback | None = None,
    ) -> RecipeResult:
        initial_state: RecipeAgentState = {
            "messages": [HumanMessage(content = raw_text)],
            "plan": None,
            "resolved_ingredients": {},
            "resolved_ustensils": {},
            "resolved_components": [],
            "recipe_uuid": None,
            "recipe_exists": None,
            "allow_duplicate": allow_duplicate,
            "existing_recipe_uuid": None,
            "existing_recipe_name": None,
            "existing_recipe": None,
            "target_recipe_uuid": None,
            "input_sources": sources or [],
            "generate_image": generate_image,
            "image_generated": False,
            "warnings": [],
            "error": None,
            "progress_callback": progress_callback,
        }
        result = await self._graph.ainvoke(initial_state)
        last_message = result["messages"][-1]
        duplicate_confirmation_required = bool(result["recipe_exists"])
        return RecipeResult(
            recipe_uuid = result["recipe_uuid"],
            recipe_name = result["plan"].name,
            resolved_ingredients = result["resolved_ingredients"],
            resolved_ustensils = result["resolved_ustensils"],
            formatted_response = (
                last_message.content if hasattr(last_message, "content") else str(last_message)
            ),
            created = not duplicate_confirmation_required,
            duplicate_confirmation_required = duplicate_confirmation_required,
            existing_recipe_uuid = result["existing_recipe_uuid"],
            existing_recipe_name = result["existing_recipe_name"],
            image_generated=result["image_generated"],
            warnings=result["warnings"],
        )

    async def enrich(
        self,
        recipe_uuid: str,
        run_uuid: str,
        *,
        generate_image: bool = True,
        progress_callback: RecipeProgressCallback | None = None,
    ) -> RecipeResult:
        import json

        if progress_callback is not None:
            await progress_callback("loading_recipe")
        existing = await self._registry.get_recipe(recipe_uuid)
        if existing is None:
            raise KeyError(f"Recipe with uuid {recipe_uuid} not found")
        raw_text = (
            "Mets à niveau cette fiche existante. Préserve les données structurées correctes, complète les métadonnées "
            "absentes et reclasse les ingrédients/étapes présents dans la description.\n\n"
            f"FICHE EXISTANTE JSON:\n{json.dumps(existing, ensure_ascii=False)}"
        )
        initial_state: RecipeAgentState = {
            "messages": [HumanMessage(content=raw_text)],
            "plan": None,
            "resolved_ingredients": {},
            "resolved_ustensils": {},
            "resolved_components": [],
            "recipe_uuid": recipe_uuid,
            "recipe_exists": False,
            "allow_duplicate": False,
            "existing_recipe_uuid": recipe_uuid,
            "existing_recipe_name": existing.get("name"),
            "existing_recipe": existing,
            "target_recipe_uuid": recipe_uuid,
            "input_sources": [],
            "generate_image": generate_image,
            "image_generated": False,
            "warnings": [],
            "error": None,
            "progress_callback": progress_callback,
        }
        result = await self._enrichment_graph.ainvoke(initial_state)
        last_message = result["messages"][-1]
        return RecipeResult(
            recipe_uuid=recipe_uuid,
            recipe_name=result["plan"].name,
            resolved_ingredients=result["resolved_ingredients"],
            resolved_ustensils=result["resolved_ustensils"],
            formatted_response=last_message.content if hasattr(last_message, "content") else str(last_message),
            created=False,
            updated=True,
            image_generated=result["image_generated"],
            warnings=result["warnings"],
        )
