from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from domain.models.recipe import RecipePlan


class RecipeAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    plan: RecipePlan | None
    resolved_ingredients: dict[str, str]  # name → uuid
    resolved_ustensils: dict[str, str]  # name → uuid
    recipe_uuid: str | None
    recipe_exists: bool | None
    error: str | None
