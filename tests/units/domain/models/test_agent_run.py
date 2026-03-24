from domain.models.agent_run import AgentRun


def test_agent_run_defaults():
    run = AgentRun(raw_input = "recette de beignets")
    assert run.status == "pending"
    assert run.recipe_uuid is None
    assert run.recipe_name is None
    assert run.error is None
    assert run.metadata == {}
    assert run.is_deleted is False


def test_agent_run_custom_status():
    run = AgentRun(raw_input = "test", status = "running")
    assert run.status == "running"


def test_agent_run_success_fields():
    run = AgentRun(
        raw_input = "carbonara",
        status = "success",
        recipe_uuid = "r-uuid-123",
        recipe_name = "Carbonara",
        metadata = {"elapsed_ms": 1200, "resolved_ingredients": 3},
    )
    assert run.status == "success"
    assert run.recipe_uuid == "r-uuid-123"
    assert run.recipe_name == "Carbonara"
    assert run.metadata["elapsed_ms"] == 1200


def test_agent_run_failed_fields():
    run = AgentRun(raw_input = "test", status = "failed", error = "LLM unavailable")
    assert run.status == "failed"
    assert run.error == "LLM unavailable"


def test_agent_run_unique_uuids():
    run1 = AgentRun(raw_input = "a")
    run2 = AgentRun(raw_input = "b")
    assert run1.uuid != run2.uuid


def test_agent_run_has_audit_fields():
    run = AgentRun(raw_input = "test")
    assert run.created_at is not None
    assert run.updated_at is not None


def test_agent_run_model_copy_preserves_uuid():
    run = AgentRun(raw_input = "test")
    updated = run.model_copy(update = {"status": "running"})
    assert updated.uuid == run.uuid
    assert updated.status == "running"
    assert updated.raw_input == "test"


def test_agent_run_all_statuses_are_valid():
    for status in ("pending", "running", "success", "failed", "cancelled"):
        run = AgentRun(raw_input = "test", status = status)
        assert run.status == status
