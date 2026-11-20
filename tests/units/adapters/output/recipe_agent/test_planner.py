from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIResponsesModel

from adapters.output.recipe_agent._planner import _build_model
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
