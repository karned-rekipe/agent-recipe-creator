import pytest
from pydantic import ValidationError

from adapters.input.schemas.recipe_schema import AiCreateRequestSchema, AiCreateResponseSchema


def test_ai_create_request_valid():
    schema = AiCreateRequestSchema(raw_text = "Voici une longue recette de gâteau au chocolat")
    assert schema.raw_text == "Voici une longue recette de gâteau au chocolat"


def test_ai_create_request_too_short():
    with pytest.raises(ValidationError):
        AiCreateRequestSchema(raw_text = "court")


def test_ai_create_request_exactly_min_length():
    schema = AiCreateRequestSchema(raw_text = "1234567890")
    assert len(schema.raw_text) == 10


def test_ai_create_request_below_min_length():
    with pytest.raises(ValidationError):
        AiCreateRequestSchema(raw_text = "123456789")  # 9 chars


def test_ai_create_response():
    schema = AiCreateResponseSchema(
        recipe_uuid = "r-uuid",
        recipe_name = "Gâteau au chocolat",
        formatted_response = "✅ Gâteau créé avec succès",
    )
    assert schema.recipe_uuid == "r-uuid"
    assert schema.recipe_name == "Gâteau au chocolat"
    assert "Gâteau" in schema.formatted_response


def test_ai_create_response_required_fields():
    with pytest.raises(ValidationError):
        AiCreateResponseSchema(recipe_uuid = "r-uuid")  # type: ignore[call-arg]
