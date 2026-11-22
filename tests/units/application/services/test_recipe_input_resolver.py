import pytest
from pydantic import ValidationError

from application.services.recipe_input_resolver import RecipeInputResolver
from domain.models.recipe import RecipeSource
from domain.models.recipe_intake import RecipeInputKind, RecipeIntake, ResolvedRecipeInput


class _Loader:
    async def load(self, url: str) -> ResolvedRecipeInput:
        return ResolvedRecipeInput(
            raw_text="Recette Web complète avec ingrédients et plusieurs étapes.",
            audit_label=f"recipe-url:{url}",
            sources=[RecipeSource(name="Web", uri=url)],
        )


class _OCR:
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def extract(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "Recette OCR complète avec ingrédients et plusieurs étapes."


def test_intake_requires_one_source_and_upload_media_type():
    with pytest.raises(ValidationError):
        RecipeIntake()
    with pytest.raises(ValidationError):
        RecipeIntake(raw_text="une recette valide", source_url="https://example.test")
    with pytest.raises(ValidationError):
        RecipeIntake(image_base64="YQ==")


@pytest.mark.parametrize(
    ("intake", "kind", "audit_prefix"),
    [
        (RecipeIntake(raw_text="Une recette texte valide"), RecipeInputKind.text, "Une recette"),
        (RecipeIntake(source_url="https://example.test/r"), RecipeInputKind.url, "recipe-url:"),
        (RecipeIntake(image_url="https://example.test/r.jpg"), RecipeInputKind.image_url, "recipe-image-url:"),
        (RecipeIntake(image_base64="YQ==", image_media_type="image/jpeg"), RecipeInputKind.image_upload, "recipe-image-upload:"),
    ],
)
def test_intake_kind_and_audit_label(intake: RecipeIntake, kind: RecipeInputKind, audit_prefix: str):
    assert intake.kind == kind
    assert intake.audit_label.startswith(audit_prefix)


async def test_resolver_supports_text_url_and_both_image_sources():
    ocr = _OCR()
    resolver = RecipeInputResolver(_Loader(), ocr)
    text = await resolver.resolve(RecipeIntake(raw_text="Une recette texte valide"))
    assert text.raw_text == "Une recette texte valide"

    web = await resolver.resolve(RecipeIntake(source_url="https://example.test/r"))
    assert web.sources[0].name == "Web"

    image_url = await resolver.resolve(RecipeIntake(image_url="https://example.test/r.jpg"))
    assert image_url.sources[0].uri == "https://example.test/r.jpg"

    upload = await resolver.resolve(RecipeIntake(image_base64="YQ==", image_media_type="image/jpeg"))
    assert "OCR" in upload.raw_text
    assert ocr.calls == [
        {"image_url": "https://example.test/r.jpg"},
        {"image_base64": "YQ==", "image_media_type": "image/jpeg"},
    ]
