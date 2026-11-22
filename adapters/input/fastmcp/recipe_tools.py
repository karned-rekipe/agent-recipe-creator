from typing import Annotated

import fastmcp
from adapters.input.schemas.recipe_schema import AiCreateResponseSchema
from application.services.recipe_service import RecipeService
from arclith.domain.ports.logger import Logger
from pydantic import Field


class RecipeMCP:
    def __init__(self, service: RecipeService, logger: Logger, mcp: fastmcp.FastMCP) -> None:
        self._service = service
        self._logger = logger
        self._mcp = mcp
        self._register_tools()

    def _register_tools(self) -> None:
        service = self._service
        logger = self._logger

        @self._mcp.tool
        async def ai_create_recipe(
                raw_text: Annotated[
                    str,
                    Field(
                        description = "Texte brut de la recette : copier/coller internet, OCR, saisie libre…",
                        examples = ["Tarte tatin : 6 pommes, 150g beurre, 200g sucre…"],
                    ),
                ],
                allow_duplicate: Annotated[
                    bool,
                    Field(description = "Créer un doublon si une recette équivalente existe déjà."),
                ] = False,
        ) -> dict:
            """Structure and register a raw recipe via AI agent."""
            try:
                result = await service.ai_create(raw_text, allow_duplicate = allow_duplicate)
            except Exception as exc:
                logger.error("💥 ai_create_recipe failed via MCP", error = str(exc))
                raise
            logger.info(
                "✅ Recipe processed via MCP",
                recipe_uuid = result.recipe_uuid,
                name = result.recipe_name,
                created = result.created,
            )
            return AiCreateResponseSchema(
                recipe_uuid = result.recipe_uuid,
                recipe_name = result.recipe_name,
                formatted_response = result.formatted_response,
                created = result.created,
                duplicate_confirmation_required = result.duplicate_confirmation_required,
                existing_recipe_uuid = result.existing_recipe_uuid,
                existing_recipe_name = result.existing_recipe_name,
                updated=result.updated,
                image_generated=result.image_generated,
                warnings=result.warnings,
            ).model_dump()

        @self._mcp.tool
        async def ai_enrich_recipe(
                recipe_uuid: Annotated[
                    str,
                    Field(description="UUID de la fiche recette existante à mettre à niveau."),
                ],
                generate_image: Annotated[
                    bool,
                    Field(description="Générer une image principale lorsque la fiche n'en possède pas."),
                ] = True,
        ) -> dict:
            """Complete and reorganize an existing recipe in place via the AI agent."""
            try:
                result = await service.ai_enrich(recipe_uuid, generate_image=generate_image)
            except Exception as exc:
                logger.error(
                    "💥 ai_enrich_recipe failed via MCP",
                    error=str(exc),
                    recipe_uuid=recipe_uuid,
                )
                raise
            logger.info(
                "✅ Existing recipe enriched via MCP",
                recipe_uuid=result.recipe_uuid,
                name=result.recipe_name,
                image_generated=result.image_generated,
            )
            return AiCreateResponseSchema(
                recipe_uuid=result.recipe_uuid,
                recipe_name=result.recipe_name,
                formatted_response=result.formatted_response,
                created=False,
                duplicate_confirmation_required=False,
                updated=result.updated,
                image_generated=result.image_generated,
                warnings=result.warnings,
            ).model_dump()
