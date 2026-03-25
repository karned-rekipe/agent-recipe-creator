import pytest

from application.use_cases.process_raw_recipe import ProcessRawRecipeUseCase
from arclith import InMemoryRepository
from domain.models.agent_run import AgentRun
from domain.models.recipe import RecipeResult
from domain.ports.output.recipe_agent import RecipeAgentPort
from tests.units.conftest import NullLogger


class _SuccessAgent(RecipeAgentPort):
    def __init__(self, result: RecipeResult) -> None:
        self._result = result

    async def process(self, raw_text: str, run_uuid: str) -> RecipeResult:
        return self._result


class _FailingAgent(RecipeAgentPort):
    async def process(self, raw_text: str, run_uuid: str) -> RecipeResult:
        raise RuntimeError("LLM unavailable")


def _make_result(**kwargs) -> RecipeResult:
    defaults = dict(
        recipe_uuid = "r-uuid",
        recipe_name = "Beignets",
        resolved_ingredients = {"farine": "i-1"},
        resolved_ustensils = {"fouet": "u-1"},
        formatted_response = "✅ Beignets",
    )
    return RecipeResult(**(defaults | kwargs))


async def test_execute_success_returns_result():
    result = _make_result()
    use_case = ProcessRawRecipeUseCase(
        _SuccessAgent(result), InMemoryRepository[AgentRun](), NullLogger()
    )
    returned = await use_case.execute("recette de beignets")
    assert returned.recipe_uuid == "r-uuid"
    assert returned.recipe_name == "Beignets"


async def test_execute_success_creates_run_with_success_status():
    repo = InMemoryRepository[AgentRun]()
    use_case = ProcessRawRecipeUseCase(_SuccessAgent(_make_result()), repo, NullLogger())
    await use_case.execute("recette de beignets")

    runs = await repo.find_all()
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "success"
    assert run.recipe_uuid == "r-uuid"
    assert run.recipe_name == "Beignets"


async def test_execute_success_stores_metadata():
    repo = InMemoryRepository[AgentRun]()
    use_case = ProcessRawRecipeUseCase(_SuccessAgent(_make_result()), repo, NullLogger())
    await use_case.execute("test")

    run = (await repo.find_all())[0]
    assert "elapsed_ms" in run.metadata
    assert run.metadata["resolved_ingredients"] == 1
    assert run.metadata["resolved_ustensils"] == 1


async def test_execute_failure_updates_run_to_failed():
    repo = InMemoryRepository[AgentRun]()
    use_case = ProcessRawRecipeUseCase(_FailingAgent(), repo, NullLogger())

    with pytest.raises(RuntimeError, match = "LLM unavailable"):
        await use_case.execute("recette échouée")

    runs = await repo.find_all()
    assert len(runs) == 1
    run = runs[0]
    assert run.status == "failed"
    assert run.error == "LLM unavailable"
    assert "elapsed_ms" in run.metadata


async def test_execute_failure_reraises_original_exception():
    repo = InMemoryRepository[AgentRun]()
    use_case = ProcessRawRecipeUseCase(_FailingAgent(), repo, NullLogger())

    with pytest.raises(RuntimeError):
        await use_case.execute("test")


async def test_execute_unwraps_single_exception_group():
    class _GroupAgent(RecipeAgentPort):
        async def process(self, raw_text: str, run_uuid: str) -> RecipeResult:
            raise BaseExceptionGroup("group", [ValueError("inner error")])

    repo = InMemoryRepository[AgentRun]()
    use_case = ProcessRawRecipeUseCase(_GroupAgent(), repo, NullLogger())

    with pytest.raises(ValueError, match = "inner error"):
        await use_case.execute("test")

    run = (await repo.find_all())[0]
    assert run.status == "failed"
    assert "inner error" in run.error


async def test_execute_reraises_multi_exception_group():
    class _MultiGroupAgent(RecipeAgentPort):
        async def process(self, raw_text: str, run_uuid: str) -> RecipeResult:
            raise BaseExceptionGroup("group", [ValueError("e1"), RuntimeError("e2")])

    repo = InMemoryRepository[AgentRun]()
    use_case = ProcessRawRecipeUseCase(_MultiGroupAgent(), repo, NullLogger())

    with pytest.raises(BaseExceptionGroup):
        await use_case.execute("test")

    run = (await repo.find_all())[0]
    assert run.status == "failed"


async def test_execute_logs_start_and_end(logger: NullLogger):
    from arclith import LogLevel

    use_case = ProcessRawRecipeUseCase(
        _SuccessAgent(_make_result()), InMemoryRepository[AgentRun](), logger
    )
    await use_case.execute("test")
    levels = [r["level"] for r in logger.records]
    assert LogLevel.INFO in levels
