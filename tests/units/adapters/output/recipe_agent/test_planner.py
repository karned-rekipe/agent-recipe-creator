from unittest.mock import patch

import pytest
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from adapters.output.recipe_agent._planner import (
    _PydanticAIPlanner,
    _build_model,
    _is_openai_credit_exhausted,
)
from domain.errors import AIProviderCreditExhaustedError
from infrastructure.config import LMSettings


def test_build_model_uses_responses_api_for_openai_cloud():
    model = _build_model(LMSettings(
        provider = "openai",
        model_name = "gpt-5.4-mini",
        api_key = "test-key",
    ))

    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == "gpt-5.4-mini"


def test_build_model_keeps_prompted_chat_mode_for_local_openai():
    model = _build_model(LMSettings(
        provider = "openai",
        model_name = "qwen3-9b",
        base_url = "http://localhost:1234/v1",
        api_key = "local",
    ))

    assert isinstance(model, OpenAIChatModel)
    assert model.profile.default_structured_output_mode == "prompted"


def test_build_model_keeps_anthropic_support():
    model = _build_model(LMSettings(
        provider = "anthropic",
        model_name = "claude-haiku-4-5-20251001",
        api_key = "test-key",
    ))

    assert isinstance(model, AnthropicModel)


async def test_planner_translates_openai_credit_exhaustion():
    planner = _PydanticAIPlanner(LMSettings(
        provider="openai",
        model_name="gpt-5.4-mini",
        api_key="test-key",
    ))
    provider_error = ModelHTTPError(
        status_code=429,
        model_name="gpt-5.4-mini",
        body={
            "code": "credit_balance_exhausted",
            "message": "You have no credits remaining.",
            "type": "insufficient_quota",
        },
    )

    with patch.object(planner._agent, "run", side_effect=provider_error):
        with pytest.raises(AIProviderCreditExhaustedError, match="crédit OpenAI est épuisé") as exc_info:
            await planner.plan("Recette de soupe")

    assert exc_info.value.__cause__ is provider_error


def test_credit_exhaustion_detection_accepts_nested_openai_error():
    provider_error = ModelHTTPError(
        status_code=429,
        model_name="gpt-5.4-mini",
        body={"error": {"code": "insufficient_quota", "type": "invalid_request_error"}},
    )

    assert _is_openai_credit_exhausted(provider_error) is True


def test_rate_limit_is_not_classified_as_credit_exhaustion():
    provider_error = ModelHTTPError(
        status_code=429,
        model_name="gpt-5.4-mini",
        body={"code": "rate_limit_exceeded", "type": "rate_limit_error"},
    )

    assert _is_openai_credit_exhausted(provider_error) is False
