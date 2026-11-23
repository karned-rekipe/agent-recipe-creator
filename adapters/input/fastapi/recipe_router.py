from adapters.input.schemas.recipe_schema import AiCreateRequestSchema, AiCreateResponseSchema, AiEnrichRequestSchema
from application.services.recipe_service import RecipeService
from arclith.domain.ports.logger import Logger
from domain.errors import AIProviderCreditExhaustedError
from fastapi import APIRouter, HTTPException, Response, status


class RecipeRouter:
    def __init__(self, service: RecipeService, logger: Logger) -> None:
        self._service = service
        self._logger = logger
        self.router = APIRouter(prefix = "/v1/recipes", tags = ["recipes"])
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.add_api_route(
            "/ai-create",
            self.ai_create,
            methods = ["POST"],
            response_model = AiCreateResponseSchema,
            status_code = 201,
            responses={503: {"description": "Crédit du fournisseur IA épuisé"}},
        )
        self.router.add_api_route(
            "/{recipe_uuid}/ai-enrich",
            self.ai_enrich,
            methods=["POST"],
            response_model=AiCreateResponseSchema,
            status_code=200,
            responses={503: {"description": "Crédit du fournisseur IA épuisé"}},
        )

    async def ai_create(
            self,
            payload: AiCreateRequestSchema,
            response: Response,
    ) -> AiCreateResponseSchema:
        """Structure and register a raw recipe via AI agent."""
        try:
            result = await self._service.ai_create(
                payload.to_domain(),
                allow_duplicate=payload.allow_duplicate,
                generate_image=payload.generate_image,
                tracking_id=payload.tracking_id,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except AIProviderCreditExhaustedError as exc:
            self._logger.error("💳 OpenAI credit exhausted via HTTP", error_code=exc.code)
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except Exception as exc:
            self._logger.error("💥 ai_create failed via HTTP", error = str(exc))
            raise HTTPException(status_code = 500, detail = str(exc)) from exc
        if not result.created:
            response.status_code = status.HTTP_200_OK
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
        )

    async def ai_enrich(self, recipe_uuid: str, payload: AiEnrichRequestSchema) -> AiCreateResponseSchema:
        """Fill missing metadata and reclassify recipe content stored in its description."""
        try:
            result = await self._service.ai_enrich(recipe_uuid, generate_image=payload.generate_image)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AIProviderCreditExhaustedError as exc:
            self._logger.error(
                "💳 OpenAI credit exhausted via HTTP",
                error_code=exc.code,
                recipe_uuid=recipe_uuid,
            )
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc
        except Exception as exc:
            self._logger.error("💥 ai_enrich failed via HTTP", error=str(exc), recipe_uuid=recipe_uuid)
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        return AiCreateResponseSchema(
            recipe_uuid=result.recipe_uuid,
            recipe_name=result.recipe_name,
            formatted_response=result.formatted_response,
            created=False,
            duplicate_confirmation_required=False,
            updated=result.updated,
            image_generated=result.image_generated,
            warnings=result.warnings,
        )
