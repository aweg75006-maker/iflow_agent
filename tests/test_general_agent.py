from __future__ import annotations

import asyncio
from collections import deque
from typing import Any, Dict

from iflow_agent2.base.agent.base_action import BaseAction
from iflow_agent2.evaluation.scorers import ScorerRegistry
from iflow_agent2.general_agent import GeneralAgent


class EchoAction(BaseAction):
    name: str = "EchoAction"
    description: str = "Return the supplied text."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    async def __call__(self, text: str = "", **kwargs: Any) -> Dict[str, Any]:
        return {"success": True, "output": text}


class ScriptedLLM:
    def __init__(self, responses) -> None:
        self.responses = deque(responses)
        self.prompts = []

    async def __call__(self, prompt: str, max_tokens=None) -> str:
        self.prompts.append(prompt)
        return self.responses.popleft()

    def get_usage_summary(self) -> Dict[str, Any]:
        return {
            "model": "fake",
            "total_cost": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
        }


def test_general_agent_runs_parallel_delegation_flow() -> None:
    main_llm = ScriptedLLM(
        [
            '{"action":"delegate_task","params":{"tasks":[{"task_instruction":"Echo hello","context":"","model":"worker","tools":["EchoAction"]}]}}',
            '{"action":"complete","params":{"answer":"hello"}}',
        ]
    )
    workers = []

    def create_worker(model_name: str) -> ScriptedLLM:
        assert model_name == "worker"
        worker = ScriptedLLM(
            [
                '{"action":"EchoAction","params":{"text":"hello"},"memory":"echoed"}',
                '{"action":"finish","params":{"result":"hello","status":"done","summary":"EchoAction returned hello"},"memory":"done"}',
            ]
        )
        workers.append(worker)
        return worker

    agent = GeneralAgent(
        main_llm=main_llm,
        sub_models=["worker"],
        tools=[EchoAction()],
        llm_factory=create_worker,
        max_attempts=3,
        subagent_max_steps=4,
    )

    result = asyncio.run(
        agent.run(
            "Return hello using the tool",
            attachments=[{"type": "image", "path": "photo.png"}],
        )
    )

    assert result.status == "done"
    assert result.answer == "hello"
    assert result.attempts == 2
    assert result.subtask_entries[0]["subtask_results"][0]["result"] == "hello"
    assert "[ATTACHMENTS]" in main_llm.prompts[0]
    assert "Step 2/4" in workers[0].prompts[1]
    assert ScorerRegistry.extract_tool_calls(result) == {"EchoAction"}


def test_rejects_reserved_tool_names() -> None:
    class FinishTool(EchoAction):
        name: str = "finish"

    try:
        GeneralAgent(
            main_llm=ScriptedLLM([]),
            sub_models=["worker"],
            tools=[FinishTool()],
        )
    except ValueError as exc:
        assert "reserved" in str(exc)
    else:
        raise AssertionError("reserved tool name was accepted")
