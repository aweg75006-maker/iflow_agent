from __future__ import annotations

import pytest

from iflow_agent2.extensions.reinforcement_learning.grpo import reward


def test_parse_and_validate_main_agent_decision() -> None:
    decision = reward.parse_decision(
        """```json
        {
          "action": "delegate_task",
          "params": {
            "tasks": [
              {
                "task_instruction": "Collect current evidence",
                "model": "worker",
                "tools": ["Search"]
              }
            ]
          }
        }
        ```"""
    )

    assert decision is not None
    assert reward.validate_decision(decision) == (1.0, 1.0)


def test_invalid_action_keeps_only_format_reward(monkeypatch) -> None:
    monkeypatch.setattr(
        reward,
        "_call_llm_judge",
        lambda prompt: pytest.fail("invalid decisions must not call the judge"),
    )

    result = reward.compute_score(
        "unit",
        '{"action": "unknown", "params": {}}',
        "",
    )

    assert result == {
        "score": 0.1,
        "acc": 0.0,
        "format_correct": 1.0,
        "action_valid": 0.0,
        "tool_reasonable": 0.0,
        "decision_quality": 0.0,
    }


def test_compute_score_combines_local_and_judge_dimensions(monkeypatch) -> None:
    monkeypatch.setattr(
        reward,
        "_call_llm_judge",
        lambda prompt: "TOOL_REASONABLE: 2\nDECISION_QUALITY: 3",
    )

    result = reward.compute_score(
        "unit",
        '{"action": "complete", "params": {"answer": "42"}}',
        "42",
        {"question": "What is the result?"},
    )

    assert result["format_correct"] == 1.0
    assert result["action_valid"] == 1.0
    assert result["tool_reasonable"] == pytest.approx(2 / 3)
    assert result["decision_quality"] == 1.0
    assert result["score"] == pytest.approx(0.1 + 0.1 + 0.2 * (2 / 3) + 0.6)
    assert result["acc"] == 1.0


def test_judge_failure_preserves_deterministic_reward(monkeypatch) -> None:
    def unavailable(prompt: str) -> str:
        raise RuntimeError("judge is not configured")

    monkeypatch.setattr(reward, "_call_llm_judge", unavailable)

    result = reward.compute_score(
        "unit",
        '{"action": "complete", "params": {"answer": "done"}}',
        "done",
    )

    assert result["score"] == pytest.approx(0.2)
    assert result["format_correct"] == 1.0
    assert result["action_valid"] == 1.0
    assert result["decision_quality"] == 0.0
