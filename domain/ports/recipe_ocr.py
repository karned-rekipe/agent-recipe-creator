from abc import ABC, abstractmethod


class RecipeOCRPort(ABC):
    @abstractmethod
    async def extract(
        self,
        *,
        image_url: str | None = None,
        image_base64: str | None = None,
        image_media_type: str | None = None,
    ) -> str: ...
