from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Literal

from domain.models.recipe import RecipeResult, RecipeSource

RecipeEnrichmentStage = Literal[
    "starting",
    "loading_recipe",
    "analyzing",
    "reviewing",
    "organizing",
    "ingredients",
    "equipment",
    "saving",
    "image",
    "complete",
    "failed",
]
RecipeProgressCallback = Callable[[RecipeEnrichmentStage], Awaitable[None]]


class RecipeAgentPort(ABC):
    @abstractmethod
    async def process(
        self,
        raw_text: str,
        run_uuid: str,
        *,
        allow_duplicate: bool = False,
        sources: list[RecipeSource] | None = None,
        generate_image: bool = True,
    ) -> RecipeResult: ...

    async def enrich(
        self,
        recipe_uuid: str,
        run_uuid: str,
        *,
        generate_image: bool = True,
        progress_callback: RecipeProgressCallback | None = None,
    ) -> RecipeResult:
        raise NotImplementedError
