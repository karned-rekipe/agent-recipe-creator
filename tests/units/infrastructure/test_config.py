from pathlib import Path

import yaml

from infrastructure.config import (
    AgentConfig,
    FuzzySettings,
    LMConfig,
    LMSettings,
    MCPRegistrySettings,
    load_agent_config,
)


def test_lm_settings_defaults():
    settings = LMSettings(model_name = "claude-haiku-4-5")
    assert settings.provider == "openai"
    assert settings.base_url is None
    assert settings.api_key == "local"


def test_lm_settings_custom():
    settings = LMSettings(
        model_name = "gpt-4",
        provider = "openai",
        base_url = "http://localhost:1234/v1",
        api_key = "my-key",
    )
    assert settings.model_name == "gpt-4"
    assert settings.base_url == "http://localhost:1234/v1"
    assert settings.api_key == "my-key"


def test_lm_settings_anthropic():
    settings = LMSettings(model_name = "claude-haiku-4-5", provider = "anthropic", api_key = "sk-ant-xxx")
    assert settings.provider == "anthropic"
    assert settings.api_key == "sk-ant-xxx"


def test_mcp_registry_settings():
    settings = MCPRegistrySettings(url = "http://my-server:8302/mcp")
    assert settings.url == "http://my-server:8302/mcp"


def test_fuzzy_settings_default():
    s = FuzzySettings()
    assert s.threshold == 80


def test_fuzzy_settings_custom():
    s = FuzzySettings(threshold = 60)
    assert s.threshold == 60


def test_agent_config_defaults_fuzzy():
    config = AgentConfig(
        mcp_registry = MCPRegistrySettings(url = "http://localhost:8302/mcp"),
        lm = LMConfig(
            planner = LMSettings(model_name = "gpt-4", provider = "openai", base_url = "http://localhost/v1")),
    )
    assert config.fuzzy.threshold == 80


def test_agent_config_custom_fuzzy():
    config = AgentConfig(
        mcp_registry = MCPRegistrySettings(url = "http://localhost:8302/mcp"),
        lm = LMConfig(planner = LMSettings(model_name = "gpt-4")),
        fuzzy = FuzzySettings(threshold = 65),
    )
    assert config.fuzzy.threshold == 65


def test_load_agent_config_valid(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_data = {
        "mcp_registry": {"url": "http://localhost:8302/mcp"},
        "lm": {
            "planner": {
                "provider": "openai",
                "model_name": "gpt-4",
                "base_url": "http://localhost:1234/v1",
                "api_key": "test-key",
            }
        },
        "fuzzy": {"threshold": 75},
    }
    config_file.write_text(yaml.dump(config_data))
    config = load_agent_config(config_file)
    assert config.mcp_registry.url == "http://localhost:8302/mcp"
    assert config.lm.planner.model_name == "gpt-4"
    assert config.lm.planner.provider == "openai"
    assert config.fuzzy.threshold == 75


def test_load_agent_config_default_fuzzy(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_data = {
        "mcp_registry": {"url": "http://localhost:8302/mcp"},
        "lm": {"planner": {"model_name": "claude-haiku-4-5", "provider": "anthropic", "api_key": "sk-x"}},
    }
    config_file.write_text(yaml.dump(config_data))
    config = load_agent_config(config_file)
    assert config.fuzzy.threshold == 80


def test_load_agent_config_ignores_extra_fields(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_data = {
        "mcp_registry": {"url": "http://localhost:8302/mcp"},
        "lm": {"planner": {"model_name": "gpt-4", "unknown_field": "ignored"}},
        "adapters": {"repository": "mongodb"},
        "api": {"port": 8006},
    }
    config_file.write_text(yaml.dump(config_data))
    config = load_agent_config(config_file)
    assert config.mcp_registry.url == "http://localhost:8302/mcp"


def test_load_agent_config_anthropic_provider(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_data = {
        "mcp_registry": {"url": "http://localhost:8302/mcp"},
        "lm": {
            "planner": {
                "provider": "anthropic",
                "model_name": "claude-haiku-4-5",
                "api_key": "sk-ant-test",
            }
        },
    }
    config_file.write_text(yaml.dump(config_data))
    config = load_agent_config(config_file)
    assert config.lm.planner.provider == "anthropic"
    assert config.lm.planner.api_key == "sk-ant-test"
