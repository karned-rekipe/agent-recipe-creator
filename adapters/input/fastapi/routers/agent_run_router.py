from fastapi import APIRouter, HTTPException
from uuid import UUID as StdUUID
from uuid6 import UUID

from adapters.input.schemas.agent_run_schema import AgentRunSchema
from application.services.agent_run_service import AgentRunService
from arclith.domain.ports.logger import Logger


class AgentRunRouter:
    def __init__(self, service: AgentRunService, logger: Logger) -> None:
        self._service = service
        self._logger = logger
        self.router = APIRouter(prefix = "/v1/agent-runs", tags = ["agent-runs"])
        self._register_routes()

    def _register_routes(self) -> None:
        self.router.add_api_route(
            methods = ["GET"],
            path = "/",
            endpoint = self.list_runs,
            summary = "List agent runs",
            response_model = list[AgentRunSchema],
            response_description = "List of all agent runs",
        )
        self.router.add_api_route(
            methods = ["GET"],
            path = "/{uuid}",
            endpoint = self.get_run,
            summary = "Get agent run",
            response_model = AgentRunSchema,
            response_description = "The agent run",
            responses = {404: {"description": "Agent run not found"}},
        )

    async def list_runs(self) -> list[AgentRunSchema]:
        """List all agent runs."""
        runs = await self._service.find_all()
        return [AgentRunSchema.model_validate(r, from_attributes = True) for r in runs]

    async def get_run(self, uuid: StdUUID) -> AgentRunSchema:
        """Get a specific agent run by UUID."""
        run = await self._service.read(UUID(str(uuid)))
        if run is None:
            raise HTTPException(status_code = 404, detail = "Agent run not found")
        return AgentRunSchema.model_validate(run, from_attributes = True)
