import pytest
from fastapi import HTTPException, Response

from adapters.input.fastapi.recipe_router import RecipeRouter
from adapters.input.schemas.recipe_schema import AiCreateRequestSchema
from application.services.recipe_service import RecipeService
from application.use_cases.process_raw_recipe import ProcessRawRecipeUseCase
from arclith import InMemoryRepository
from domain.models.agent_run import AgentRun
from domain.models.recipe import RecipeResult, RecipeSource
from domain.ports.recipe_agent import RecipeAgentPort
from tests.units.conftest import NullLogger


class _SuccessAgent(RecipeAgentPort):
    async def process(
            self,
            raw_text: str,
            run_uuid: str,
            *,
            allow_duplicate: bool = False,
            sources: list[RecipeSource] | None = None,
            generate_image: bool = True,
    ) -> RecipeResult:
        return RecipeResult(
            recipe_uuid = "r-uuid",
            recipe_name = "Tarte au citron",
            resolved_ingredients = {"citron": "i-1"},
            resolved_ustensils = {"moule": "u-1"},
            formatted_response = "✅ Tarte au citron créée",
        )


class _FailingAgent(RecipeAgentPort):
    async def process(
            self,
            raw_text: str,
            run_uuid: str,
            *,
            allow_duplicate: bool = False,
            sources: list[RecipeSource] | None = None,
            generate_image: bool = True,
    ) -> RecipeResult:
        raise RuntimeError("agent failed")


class _InvalidInputAgent(RecipeAgentPort):
    async def process(
            self,
            raw_text: str,
            run_uuid: str,
            *,
            allow_duplicate: bool = False,
            sources: list[RecipeSource] | None = None,
            generate_image: bool = True,
    ) -> RecipeResult:
        raise ValueError("Recipe URL must resolve only to public addresses")


def _make_router(agent: RecipeAgentPort) -> RecipeRouter:
    use_case = ProcessRawRecipeUseCase(agent, InMemoryRepository[AgentRun](), NullLogger())
    service = RecipeService(use_case)
    return RecipeRouter(service, NullLogger())


async def test_ai_create_success():
    router = _make_router(_SuccessAgent())
    payload = AiCreateRequestSchema(raw_text = "Une recette de tarte classique au citron meringuée")
    response = Response()
    result = await router.ai_create(payload, response)
    assert result.recipe_uuid == "r-uuid"
    assert result.recipe_name == "Tarte au citron"
    assert "Tarte" in result.formatted_response
    assert result.created is True
    assert result.duplicate_confirmation_required is False


async def test_ai_create_returns_duplicate_confirmation():
    class _DuplicateAgent(RecipeAgentPort):
        async def process(
                self,
                raw_text: str,
                run_uuid: str,
                *,
                allow_duplicate: bool = False,
                sources: list[RecipeSource] | None = None,
                generate_image: bool = True,
        ) -> RecipeResult:
            return RecipeResult(
                recipe_uuid = "existing-r",
                recipe_name = "Tarte au citron",
                resolved_ingredients = {},
                resolved_ustensils = {},
                formatted_response = "Aucune modification n'a été appliquée.",
                created = False,
                duplicate_confirmation_required = True,
                existing_recipe_uuid = "existing-r",
                existing_recipe_name = "Tarte au citron",
            )

    router = _make_router(_DuplicateAgent())
    payload = AiCreateRequestSchema(raw_text = "Une recette de tarte classique au citron meringuée")
    response = Response()
    result = await router.ai_create(payload, response)
    assert result.created is False
    assert result.duplicate_confirmation_required is True
    assert result.existing_recipe_uuid == "existing-r"
    assert response.status_code == 200


async def test_ai_create_passes_allow_duplicate():
    received: list[bool] = []

    class _CapturingAgent(RecipeAgentPort):
        async def process(
                self,
                raw_text: str,
                run_uuid: str,
                *,
                allow_duplicate: bool = False,
                sources: list[RecipeSource] | None = None,
                generate_image: bool = True,
        ) -> RecipeResult:
            received.append(allow_duplicate)
            return RecipeResult(
                recipe_uuid = "r-uuid",
                recipe_name = "Tarte au citron",
                resolved_ingredients = {},
                resolved_ustensils = {},
                formatted_response = "ok",
            )

    router = _make_router(_CapturingAgent())
    payload = AiCreateRequestSchema(
        raw_text = "Une recette de tarte classique au citron meringuée",
        allow_duplicate = True,
    )
    await router.ai_create(payload, Response())
    assert received == [True]


async def test_ai_create_raises_500_on_agent_failure():
    router = _make_router(_FailingAgent())
    payload = AiCreateRequestSchema(raw_text = "Recette qui va échouer lors du traitement agent")
    with pytest.raises(HTTPException) as exc_info:
        await router.ai_create(payload, Response())
    assert exc_info.value.status_code == 500
    assert "agent failed" in exc_info.value.detail


async def test_ai_create_raises_422_on_invalid_source():
    router = _make_router(_InvalidInputAgent())
    payload = AiCreateRequestSchema(raw_text="Recette valide mais source refusée pendant la résolution")
    with pytest.raises(HTTPException) as exc_info:
        await router.ai_create(payload, Response())
    assert exc_info.value.status_code == 422
    assert "public addresses" in exc_info.value.detail
