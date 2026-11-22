from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from typing import Literal

from domain.models.recipe import RecipeResult, RecipeSource

RecipeProcessingStage = Literal[
    "starting",
    "reading_source",
    "loading_recipe",
    "analyzing",
    "reviewing",
    "checking_duplicate",
    "organizing",
    "ingredients",
    "equipment",
    "tags",
    "saving",
    "image",
    "complete",
    "failed",
]
RecipeEnrichmentStage = RecipeProcessingStage
RecipeProgressCallback = Callable[[RecipeProcessingStage], Awaitable[None]]


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
        progress_callback: RecipeProgressCallback | None = None,
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
