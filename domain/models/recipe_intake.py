from enum import Enum

from pydantic import BaseModel, Field, model_validator

from domain.models.recipe import RecipeSource


class RecipeInputKind(str, Enum):
    text = "text"
    url = "url"
    image_url = "image_url"
    image_upload = "image_upload"


class RecipeIntake(BaseModel):
    raw_text: str | None = Field(default=None, min_length=10, max_length=80_000)
    source_url: str | None = Field(default=None, max_length=2_048)
    image_url: str | None = Field(default=None, max_length=2_048)
    image_base64: str | None = Field(default=None, max_length=14_000_000)
    image_media_type: str | None = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def require_exactly_one_source(self) -> "RecipeIntake":
        provided = [
            self.raw_text is not None,
            self.source_url is not None,
            self.image_url is not None,
            self.image_base64 is not None,
        ]
        if sum(provided) != 1:
            raise ValueError("Provide exactly one recipe source: raw_text, source_url, image_url or image_base64")
        if self.image_base64 is not None and not self.image_media_type:
            raise ValueError("image_media_type is required with image_base64")
        return self

    @property
    def kind(self) -> RecipeInputKind:
        if self.raw_text is not None:
            return RecipeInputKind.text
        if self.source_url is not None:
            return RecipeInputKind.url
        if self.image_url is not None:
            return RecipeInputKind.image_url
        return RecipeInputKind.image_upload

    @property
    def audit_label(self) -> str:
        match self.kind:
            case RecipeInputKind.text:
                return self.raw_text or ""
            case RecipeInputKind.url:
                return f"recipe-url:{self.source_url}"
            case RecipeInputKind.image_url:
                return f"recipe-image-url:{self.image_url}"
            case RecipeInputKind.image_upload:
                return f"recipe-image-upload:{self.image_media_type}"


class ResolvedRecipeInput(BaseModel):
    raw_text: str = Field(min_length=10)
    audit_label: str
    sources: list[RecipeSource] = Field(default_factory=list)
