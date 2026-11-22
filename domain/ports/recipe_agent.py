from abc import ABC, abstractmethod

from domain.models.recipe import RecipeResult, RecipeSource


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

    async def enrich(self, recipe_uuid: str, run_uuid: str, *, generate_image: bool = True) -> RecipeResult:
        raise NotImplementedError
