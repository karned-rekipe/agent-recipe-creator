from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from domain.models.recipe import RecipePlan, RecipeSource
from domain.ports.recipe_agent import RecipeProgressCallback


class RecipeAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: RecipePlan | None
    resolved_ingredients: dict[str, str]  # name → uuid
    resolved_ustensils: dict[str, str]  # name → uuid
    available_tags: list[dict]
    resolved_tags: dict[str, str]  # name → uuid
    resolved_components: list[dict]
    recipe_uuid: str | None
    recipe_exists: bool | None
    allow_duplicate: bool
    existing_recipe_uuid: str | None
    existing_recipe_name: str | None
    existing_recipe: dict | None
    target_recipe_uuid: str | None
    input_sources: list[RecipeSource]
    generate_image: bool
    image_generated: bool
    warnings: Annotated[list[str], add]
    error: str | None
    progress_callback: RecipeProgressCallback | None
