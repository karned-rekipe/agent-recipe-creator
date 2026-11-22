import time

from arclith.domain.ports.logger import Logger
from arclith.domain.ports.repository import Repository
from domain.models.agent_run import AgentRun
from domain.models.recipe import RecipeResult
from domain.ports.recipe_agent import RecipeAgentPort


class EnrichExistingRecipeUseCase:
    def __init__(self, agent: RecipeAgentPort, run_repository: Repository[AgentRun], logger: Logger) -> None:
        self._agent = agent
        self._run_repository = run_repository
        self._logger = logger

    async def execute(self, recipe_uuid: str, *, generate_image: bool = True) -> RecipeResult:
        run = await self._run_repository.create(AgentRun(
            raw_input=f"existing-recipe:{recipe_uuid}",
            status="running",
        ))
        run_uuid = str(run.uuid)
        started = time.perf_counter()
        try:
            result = await self._agent.enrich(recipe_uuid, run_uuid, generate_image=generate_image)
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            await self._run_repository.update(run.model_copy(update={
                "status": "success",
                "recipe_uuid": result.recipe_uuid,
                "recipe_name": result.recipe_name,
                "metadata": {
                    "elapsed_ms": elapsed_ms,
                    "updated": result.updated,
                    "image_generated": result.image_generated,
                    "warnings": result.warnings,
                    "resolved_ingredients": len(result.resolved_ingredients),
                    "resolved_ustensils": len(result.resolved_ustensils),
                },
            }))
            return result
        except BaseException as exc:
            if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
            elapsed_ms = round((time.perf_counter() - started) * 1000)
            await self._run_repository.update(run.model_copy(update={
                "status": "failed",
                "error": str(exc),
                "metadata": {"elapsed_ms": elapsed_ms},
            }))
            self._logger.error("💥 Existing recipe enrichment failed", run_uuid=run_uuid, error=str(exc))
            raise exc
