import pytest

from application.use_cases.enrich_existing_recipe import EnrichExistingRecipeUseCase
from arclith import InMemoryRepository
from domain.models.agent_run import AgentRun
from domain.models.recipe import RecipeResult, RecipeSource
from domain.ports.recipe_agent import RecipeAgentPort
from tests.units.conftest import NullLogger


def _result() -> RecipeResult:
    return RecipeResult(
        recipe_uuid="recipe-1",
        recipe_name="Soupe",
        resolved_ingredients={"carotte": "ingredient-1"},
        resolved_ustensils={},
        formatted_response="mise à niveau",
        created=False,
        updated=True,
        image_generated=True,
    )


class _Agent(RecipeAgentPort):
    async def process(
        self,
        raw_text: str,
        run_uuid: str,
        *,
        allow_duplicate: bool = False,
        sources: list[RecipeSource] | None = None,
        generate_image: bool = True,
    ) -> RecipeResult:
        return _result()

    async def enrich(self, recipe_uuid: str, run_uuid: str, *, generate_image: bool = True) -> RecipeResult:
        return _result()


async def test_enrich_existing_recipe_records_successful_run():
    repository = InMemoryRepository[AgentRun]()
    use_case = EnrichExistingRecipeUseCase(_Agent(), repository, NullLogger())
    result = await use_case.execute("recipe-1")
    assert result.updated is True
    run = (await repository.find_all())[0]
    assert run.status == "success"
    assert run.metadata["image_generated"] is True


async def test_enrich_existing_recipe_records_failure():
    class _FailingAgent(_Agent):
        async def enrich(self, recipe_uuid: str, run_uuid: str, *, generate_image: bool = True) -> RecipeResult:
            raise RuntimeError("planner down")

    repository = InMemoryRepository[AgentRun]()
    use_case = EnrichExistingRecipeUseCase(_FailingAgent(), repository, NullLogger())
    with pytest.raises(RuntimeError, match="planner down"):
        await use_case.execute("recipe-1")
    run = (await repository.find_all())[0]
    assert run.status == "failed"
    assert run.error == "planner down"
