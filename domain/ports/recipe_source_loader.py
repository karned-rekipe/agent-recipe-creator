from abc import ABC, abstractmethod

from domain.models.recipe_intake import ResolvedRecipeInput


class RemoteRecipeDocument(ValueError):
    def __init__(self, url: str, media_type: str) -> None:
        super().__init__(f"Recipe URL points to a document ({media_type})")
        self.url = url
        self.media_type = media_type


class RecipeSourceLoaderPort(ABC):
    @abstractmethod
    async def load(self, url: str) -> ResolvedRecipeInput: ...
