from domain.models.recipe import RecipeSource
from domain.models.recipe_intake import RecipeInputKind, RecipeIntake, ResolvedRecipeInput
from domain.ports.recipe_ocr import RecipeOCRPort
from domain.ports.recipe_source_loader import RecipeSourceLoaderPort


class RecipeInputResolver:
    def __init__(self, source_loader: RecipeSourceLoaderPort, ocr: RecipeOCRPort) -> None:
        self._source_loader = source_loader
        self._ocr = ocr

    async def resolve(self, intake: RecipeIntake) -> ResolvedRecipeInput:
        match intake.kind:
            case RecipeInputKind.text:
                return ResolvedRecipeInput(
                    raw_text=intake.raw_text or "",
                    audit_label=intake.audit_label,
                )
            case RecipeInputKind.url:
                return await self._source_loader.load(intake.source_url or "")
            case RecipeInputKind.image_url:
                text = await self._ocr.extract(image_url=intake.image_url)
                return ResolvedRecipeInput(
                    raw_text=text,
                    audit_label=intake.audit_label,
                    sources=[RecipeSource(
                        name="Image de recette",
                        description="Texte extrait par OCR",
                        uri=intake.image_url,
                    )],
                )
            case RecipeInputKind.image_upload:
                text = await self._ocr.extract(
                    image_base64=intake.image_base64,
                    image_media_type=intake.image_media_type,
                )
                return ResolvedRecipeInput(raw_text=text, audit_label=intake.audit_label)
