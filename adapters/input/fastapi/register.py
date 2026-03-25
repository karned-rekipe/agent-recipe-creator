from pathlib import Path

from fastapi import FastAPI

from adapters.input.fastapi.routers.agent_run_router import AgentRunRouter
from adapters.input.fastapi.routers.recipe_router import RecipeRouter
from arclith import Arclith
from infrastructure.config import load_agent_config
from infrastructure.containers.container import build_container


def register_routers(app: FastAPI, arclith: Arclith) -> None:
    agent_config = load_agent_config(Path(__file__).parent.parent.parent.parent / "config.yaml")
    recipe_service, run_service, logger = build_container(arclith, agent_config)
    app.include_router(RecipeRouter(recipe_service, logger).router)
    app.include_router(AgentRunRouter(run_service, logger).router)
