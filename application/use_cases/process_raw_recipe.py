import time

from arclith.domain.ports.logger import Logger
from arclith.domain.ports.repository import Repository
from domain.models.agent_run import AgentRun
from domain.models.recipe import RecipeResult, RecipeSource
from domain.ports.recipe_agent import RecipeAgentPort, RecipeProcessingStage


class ProcessRawRecipeUseCase:
    def __init__(
            self,
            agent: RecipeAgentPort,
            run_repository: Repository[AgentRun],
            logger: Logger,
    ) -> None:
        self._agent = agent
        self._run_repository = run_repository
        self._logger = logger

    async def execute(
            self,
            raw_text: str,
            *,
            allow_duplicate: bool = False,
            sources: list[RecipeSource] | None = None,
            audit_input: str | None = None,
            generate_image: bool = True,
            tracking_id: str | None = None,
    ) -> RecipeResult:
        run = await self._run_repository.create(AgentRun(
            raw_input=audit_input or raw_text,
            status="running",
            metadata={
                "current_stage": "starting",
                "stage_history": ["starting"],
                **({"tracking_id": tracking_id} if tracking_id else {}),
            },
        ))
        run_uuid = str(run.uuid)
        self._logger.info("▶ Agent run started", run_uuid = run_uuid, input_length = len(raw_text))
        t0 = time.perf_counter()

        async def report_progress(stage: RecipeProcessingStage) -> None:
            nonlocal run
            history = list(run.metadata.get("stage_history", []))
            if not history or history[-1] != stage:
                history.append(stage)
            run = await self._run_repository.update(run.model_copy(update={
                "metadata": {
                    **run.metadata,
                    "current_stage": stage,
                    "stage_history": history,
                },
            }))

        try:
            result = await self._agent.process(
                raw_text,
                run_uuid,
                allow_duplicate=allow_duplicate,
                sources=sources,
                generate_image=generate_image,
                progress_callback=report_progress,
            )
            await report_progress("complete")
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            await self._run_repository.update(
                run.model_copy(update = {
                    "status": "success",
                    "recipe_uuid": result.recipe_uuid,
                    "recipe_name": result.recipe_name,
                    "metadata": {
                        **run.metadata,
                        "elapsed_ms": elapsed_ms,
                        "resolved_ingredients": len(result.resolved_ingredients),
                        "resolved_ustensils": len(result.resolved_ustensils),
                        "created": result.created,
                        "duplicate_confirmation_required": result.duplicate_confirmation_required,
                        "existing_recipe_uuid": result.existing_recipe_uuid,
                        "image_generated": result.image_generated,
                        "warnings": result.warnings,
                    },
                })
            )
            self._logger.info(
                "◀ Agent run succeeded",
                run_uuid = run_uuid,
                recipe_uuid = result.recipe_uuid,
                elapsed_ms = elapsed_ms,
            )
        except BaseException as exc:
            # Unwrap ExceptionGroup (asyncio.TaskGroup / anyio) pour exposer la vraie erreur
            if isinstance(exc, BaseExceptionGroup) and len(exc.exceptions) == 1:
                exc = exc.exceptions[0]
            elapsed_ms = round((time.perf_counter() - t0) * 1000)
            await report_progress("failed")
            await self._run_repository.update(
                run.model_copy(update = {
                    "status": "failed",
                    "error": str(exc),
                    "metadata": {**run.metadata, "elapsed_ms": elapsed_ms},
                })
            )
            self._logger.error("💥 Agent run failed", run_uuid = run_uuid, error = str(exc))
            raise exc

        return result
