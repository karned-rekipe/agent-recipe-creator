from copy import deepcopy
import os
from pathlib import Path
from typing import Any

from arclith.infrastructure.secret_factory import build_secret_resolver
import yaml
from pydantic import BaseModel, ConfigDict


class LMSettings(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    model_name: str
    provider: str = "openai"  # "openai" | "anthropic"
    base_url: str | None = None
    api_key: str = "local"


class LMConfig(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    planner: LMSettings


class MCPRegistrySettings(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    url: str


class FuzzySettings(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    threshold: int = 80


class AgentConfig(BaseModel):
    model_config = ConfigDict(extra = "ignore")

    mcp_registry: MCPRegistrySettings
    lm: LMConfig
    fuzzy: FuzzySettings = FuzzySettings()


def load_agent_config(config_path: Path) -> AgentConfig:
    with config_path.open() as f:
        data = yaml.safe_load(f)
    data = _expand_env(data or {})
    data = _resolve_config_secrets(data)
    return AgentConfig.model_validate(data)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, list):
        return [_expand_env(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}
    return value


def _set_nested(data: dict[str, Any], path: str, value: str) -> None:
    current: dict[str, Any] = data
    keys = path.split(".")
    for key in keys[:-1]:
        next_value = current.get(key)
        if not isinstance(next_value, dict):
            next_value = {}
            current[key] = next_value
        current = next_value
    current[keys[-1]] = value


def _resolve_config_secrets(data: dict[str, Any]) -> dict[str, Any]:
    resolver = build_secret_resolver(data)
    if resolver is None:
        return data

    result = deepcopy(data)
    mappings: dict[str, str] = (result.get("secrets") or {}).get("mappings") or {}
    for field_path, secret_key in mappings.items():
        value = resolver.get(field_path, secret_key)
        if value is not None:
            _set_nested(result, field_path, value)
    return result
