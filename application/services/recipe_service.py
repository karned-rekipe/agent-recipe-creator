from application.services.recipe_input_resolver import RecipeInputResolver
from application.use_cases.enrich_existing_recipe import EnrichExistingRecipeUseCase
from application.use_cases.process_raw_recipe import ProcessRawRecipeUseCase
from domain.models.recipe import RecipeResult
from domain.models.recipe_intake import RecipeIntake


class RecipeService:
    def __init__(
        self,
        use_case: ProcessRawRecipeUseCase,
        input_resolver: RecipeInputResolver | None = None,
        enrich_use_case: EnrichExistingRecipeUseCase | None = None,
    ) -> None:
        self._use_case = use_case
        self._input_resolver = input_resolver
        self._enrich_use_case = enrich_use_case

    async def ai_create(
        self,
        recipe_input: str | RecipeIntake,
        *,
        allow_duplicate: bool = False,
        generate_image: bool = True,
    ) -> RecipeResult:
        if isinstance(recipe_input, str):
            return await self._use_case.execute(
                recipe_input,
                allow_duplicate=allow_duplicate,
                generate_image=generate_image,
            )
        if self._input_resolver is None:
            if recipe_input.raw_text is not None:
                return await self._use_case.execute(
                    recipe_input.raw_text,
                    allow_duplicate=allow_duplicate,
                    generate_image=generate_image,
                )
            raise RuntimeError("Recipe input resolver is not configured")
        resolved = await self._input_resolver.resolve(recipe_input)
        return await self._use_case.execute(
            resolved.raw_text,
            allow_duplicate=allow_duplicate,
            sources=resolved.sources,
            audit_input=resolved.audit_label,
            generate_image=generate_image,
        )

    async def ai_enrich(self, recipe_uuid: str, *, generate_image: bool = True) -> RecipeResult:
        if self._enrich_use_case is None:
            raise RuntimeError("Existing recipe enrichment is not configured")
        return await self._enrich_use_case.execute(recipe_uuid, generate_image=generate_image)
