from application.services.agent_run_service import AgentRunService
from arclith import InMemoryRepository
from domain.models.agent_run import AgentRun
from tests.units.conftest import NullLogger


async def test_create_and_read():
    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    run = await service.create(AgentRun(raw_input = "test recette"))
    found = await service.read(run.uuid)
    assert found is not None
    assert found.raw_input == "test recette"


async def test_read_unknown_uuid_returns_none():
    from uuid6 import uuid7

    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    result = await service.read(uuid7())
    assert result is None


async def test_find_all_empty():
    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    result = await service.find_all()
    assert result == []


async def test_find_all_returns_all_active():
    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    await service.create(AgentRun(raw_input = "recette 1"))
    await service.create(AgentRun(raw_input = "recette 2"))
    result = await service.find_all()
    assert len(result) == 2


async def test_update_changes_field():
    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    run = await service.create(AgentRun(raw_input = "test"))
    updated = await service.update(run.model_copy(update = {"status": "success", "recipe_name": "Soupe"}))
    assert updated.status == "success"
    assert updated.recipe_name == "Soupe"


async def test_delete_soft_deletes():
    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    run = await service.create(AgentRun(raw_input = "test"))
    await service.delete(run.uuid)
    active = await service.find_all()
    assert len(active) == 0


async def test_duplicate_creates_new_uuid():
    repo = InMemoryRepository[AgentRun]()
    service = AgentRunService(repo, NullLogger())
    run = await service.create(AgentRun(raw_input = "original"))
    clone = await service.duplicate(run.uuid)
    assert clone.uuid != run.uuid
    assert clone.raw_input == "original"
