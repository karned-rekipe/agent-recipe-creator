from arclith import BaseService, Logger
from arclith.domain.ports.repository import Repository

from domain.models.agent_run import AgentRun


class AgentRunService(BaseService[AgentRun]):
    def __init__(self, repository: Repository[AgentRun], logger: Logger) -> None:
        super().__init__(repository, logger)

    async def find_running_recipe_enrichment(self, recipe_uuid: str) -> AgentRun | None:
        raw_input = f"existing-recipe:{recipe_uuid}"
        runs = [
            run
            for run in await self.find_all()
            if run.raw_input == raw_input and run.status == "running"
        ]
        return max(runs, key=lambda run: run.created_at) if runs else None

    async def find_running_recipe_creation(self, tracking_id: str) -> AgentRun | None:
        runs = [
            run
            for run in await self.find_all()
            if run.metadata.get("tracking_id") == tracking_id and run.status == "running"
        ]
        return max(runs, key=lambda run: run.created_at) if runs else None
