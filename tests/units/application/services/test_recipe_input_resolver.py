import pytest
from pydantic import ValidationError

from application.services.recipe_input_resolver import RecipeInputResolver
from domain.models.recipe import RecipeSource
from domain.models.recipe_intake import RecipeInputKind, RecipeIntake, ResolvedRecipeInput
from domain.ports.recipe_ocr import RecipeOCRPort
from domain.ports.recipe_source_loader import RecipeSourceLoaderPort, RemoteRecipeDocument


class _Loader(RecipeSourceLoaderPort):
    async def load(self, url: str) -> ResolvedRecipeInput:
        return ResolvedRecipeInput(
            raw_text="Recette Web complète avec ingrédients et plusieurs étapes.",
            audit_label=f"recipe-url:{url}",
            sources=[RecipeSource(name="Web", uri=url)],
        )


class _OCR(RecipeOCRPort):
    def __init__(self) -> None:
        self.calls: list[dict[str, str]] = []

    async def extract(self, **kwargs) -> str:
        self.calls.append(kwargs)
        return "Recette OCR complète avec ingrédients et plusieurs étapes."


class _DocumentLoader(RecipeSourceLoaderPort):
    def __init__(self, media_type: str) -> None:
        self._media_type = media_type

    async def load(self, url: str) -> ResolvedRecipeInput:
        raise RemoteRecipeDocument(url, self._media_type)


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


@pytest.mark.parametrize(
    ("media_type", "source_name", "audit_prefix"),
    [
        ("image/png", "Image de recette", "recipe-image-url:"),
        ("application/pdf", "Document de recette", "recipe-document-url:"),
    ],
)
async def test_resolver_routes_linked_images_and_pdfs_to_ocr(
    media_type: str,
    source_name: str,
    audit_prefix: str,
):
    ocr = _OCR()
    resolver = RecipeInputResolver(_DocumentLoader(media_type), ocr)

    result = await resolver.resolve(RecipeIntake(source_url="https://example.test/source"))

    assert result.audit_label.startswith(audit_prefix)
    assert result.sources[0].name == source_name
    assert ocr.calls == [{"image_url": "https://example.test/source"}]
