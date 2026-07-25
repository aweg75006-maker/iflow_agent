from __future__ import annotations

import asyncio

import pytest

from iflow_agent2.base.engine.async_llm import LLMsConfig
from iflow_agent2.tools.audio_analysis import build_audio_content
from iflow_agent2.tools.tavily_search import TavilySearchAction


def test_yaml_alias_and_environment_expansion(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-from-env")
    path = tmp_path / "models.yaml"
    path.write_text(
        """
models:
  worker:
    model: provider-model-id
    base_url: ${TEST_PROVIDER_URL:-https://provider.example/v1}
    api_key: ${TEST_PROVIDER_KEY}
""".strip(),
        encoding="utf-8",
    )

    config = LLMsConfig.load(path).get("worker")

    assert config.model == "provider-model-id"
    assert config.base_url == "https://provider.example/v1"
    assert config.key == "secret-from-env"


def test_empty_api_key_is_rejected(tmp_path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n  worker:\n    api_key: ${MISSING_TEST_KEY}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="API key"):
        LLMsConfig.load(path).get("worker")


def test_iflow_environment_fallback(monkeypatch) -> None:
    monkeypatch.setenv("IFLOW_OPENAI_API_KEY", "iflow-secret")
    monkeypatch.setenv("IFLOW_OPENAI_BASE_URL", "https://gateway.example/v1")
    monkeypatch.setenv("IFLOW_OPENAI_MODELS", "planner,worker")
    monkeypatch.setenv("IFLOW_WORKER_BASE_URL", "https://worker.example/v1")

    config = LLMsConfig._load_config_from_env()

    assert config is not None
    assert config["planner"]["api_key"] == "iflow-secret"
    assert config["planner"]["base_url"] == "https://gateway.example/v1"
    assert config["worker"]["base_url"] == "https://worker.example/v1"


def test_mimo_asr_uses_audio_only_content() -> None:
    content = build_audio_content("mimo-v2.5-asr", "encoded", "wav", "prompt")
    assert content == [
        {
            "type": "input_audio",
            "input_audio": {"data": "encoded", "format": "wav"},
        }
    ]

    compatible_content = build_audio_content(
        "gpt-4o-audio-preview", "encoded", "wav", "prompt"
    )
    assert [item["type"] for item in compatible_content] == ["text", "input_audio"]


def test_tavily_tool_reports_missing_key(monkeypatch) -> None:
    monkeypatch.setattr(TavilySearchAction, "_load_api_key", staticmethod(lambda: ""))
    result = asyncio.run(TavilySearchAction()(query="test"))
    assert result["success"] is False
    assert "Tavily API key" in result["error"]
