from abc import ABC, abstractmethod

from domain.models.recipe_intake import ResolvedRecipeInput


class RecipeSourceLoaderPort(ABC):
    @abstractmethod
    async def load(self, url: str) -> ResolvedRecipeInput: ...
