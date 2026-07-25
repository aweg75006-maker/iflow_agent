from __future__ import annotations

import asyncio
from typing import Any, Dict

from iflow_agent2.base.agent.base_action import BaseAction
from iflow_agent2.runtime.tool_environment import ToolEnvironment


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


def test_environment_keeps_original_contract_and_clones_state() -> None:
    env = ToolEnvironment(
        "Inspect the attachment",
        [EchoAction()],
        attachments=[{"type": "image", "path": "photo.png"}],
        max_steps=3,
    )
    clone = env.clone()
    clone.restrict_tools([])

    assert env.get_basic_info().max_steps == 3
    assert "photo.png" in env.get_basic_info().instruction
    assert "EchoAction" in env.get_basic_info().action_space
    assert "EchoAction" not in clone.get_basic_info().action_space
    assert "### finish" in clone.get_basic_info().action_space


def test_environment_executes_tool_and_finish() -> None:
    async def scenario() -> None:
        env = ToolEnvironment("Echo", [EchoAction()], max_steps=3)
        await env.reset()
        observation, _, done, info = await env.step(
            {"action": "EchoAction", "params": {"text": "hello"}}
        )
        assert observation["output"] == "hello"
        assert not done
        assert info["last_action_result"]["success"]

        _, _, done, info = await env.step(
            {
                "action": "finish",
                "params": {"result": "hello", "status": "done", "summary": "echoed"},
            }
        )
        assert done
        assert info["finish_result"]["result"] == "hello"

    asyncio.run(scenario())

