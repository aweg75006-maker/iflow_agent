"""Tool execution environment for delegated agent tasks."""
from __future__ import annotations

import inspect
import json
from copy import copy
from typing import Any, Dict, Iterable, Optional, Tuple

from iflow_agent2.base.agent.base_action import BaseAction
from iflow_agent2.base.engine.logs import logger
from iflow_agent2.runtime.env import Action, BasicInfo, Environment, Observation

FINISH_ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "result": {"type": "string", "description": "Result returned to the MainAgent"},
        "status": {"type": "string", "enum": ["done", "partial", "failed"]},
        "summary": {"type": "string", "description": "Brief evidence and limitations"},
    },
    "required": ["result", "status"],
}


class ToolEnvironment(Environment):
    """Runs registered BaseAction-compatible tools behind the Environment protocol."""

    def __init__(
        self,
        instruction: str,
        tools: Iterable[BaseAction],
        *,
        attachments: Optional[Iterable[Any]] = None,
        max_steps: int = 30,
        env_id: str = "general-task",
        meta_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.instruction = instruction
        self._all_tools = {tool.name: tool for tool in tools}
        self.tools = dict(self._all_tools)
        self.attachments = list(attachments or [])
        self.max_steps = max_steps
        self.env_id = env_id
        self.meta_data = dict(meta_data or {})
        self.meta_data.setdefault("attachments", self.attachments)
        self._steps = 0
        self._done = False

    def clone(self) -> "ToolEnvironment":
        return ToolEnvironment(
            instruction=self.instruction,
            tools=list(self._all_tools.values()),
            attachments=copy(self.attachments),
            max_steps=self.max_steps,
            env_id=self.env_id,
            meta_data=copy(self.meta_data),
        )

    def restrict_tools(self, names: Optional[Iterable[str]]) -> None:
        if names is None:
            self.tools = dict(self._all_tools)
            return
        missing = [name for name in names if name not in self._all_tools]
        if missing:
            raise ValueError(f"Unknown delegated tools: {missing}")
        self.tools = {name: self._all_tools[name] for name in names}

    def get_basic_info(self) -> BasicInfo:
        return BasicInfo(
            env_id=self.env_id,
            instruction=self._build_instruction(),
            action_space=self._build_action_space(),
            max_steps=self.max_steps,
            meta_data=self.meta_data,
        )

    async def reset(self, seed: int | None = None) -> Observation:
        self._steps = 0
        self._done = False
        return {
            "message": "Environment ready. Use the registered tools or finish.",
            "current_step": 0,
            "max_steps": self.max_steps,
        }

    async def step(self, action: Action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Environment already finished. Call reset() first.")
        self._steps += 1
        action_name = str(action.get("action", ""))
        params = action.get("params", {})

        if action_name == "finish":
            return self._finish(params)

        tool = self.tools.get(action_name)
        if tool is None:
            observation = {
                "success": False,
                "error": f"Unknown or disallowed tool: {action_name}. Available: {list(self.tools)}",
                "current_step": self._steps,
            }
            return self._continue_or_timeout(observation, {"error": "unknown_action"})

        try:
            value = tool(**params)
            result = await value if inspect.isawaitable(value) else value
            if isinstance(result, dict):
                success = bool(result.get("success", not result.get("error")))
                output = result.get("output", result)
                error = result.get("error")
            else:
                success, output, error = True, result, None
            observation = {
                "action": action_name,
                "success": success,
                "output": output if success else None,
                "error": error if not success else None,
                "current_step": self._steps,
                "max_steps": self.max_steps,
            }
        except Exception as exc:
            logger.error(f"[ToolEnvironment] {action_name} failed: {exc}")
            observation = {
                "action": action_name,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "current_step": self._steps,
                "max_steps": self.max_steps,
            }
        return self._continue_or_timeout(observation, {"last_action_result": observation})

    async def close(self) -> None:
        # Clones share registered tool instances; their owner manages tool lifetime.
        return None

    def _build_instruction(self) -> str:
        if not self.attachments:
            return self.instruction
        lines = [self.instruction, "", "[ATTACHMENTS]"]
        for index, item in enumerate(self.attachments, 1):
            if isinstance(item, dict):
                item_type = item.get("type", item.get("kind", "file"))
                uri = item.get("path", item.get("url", item.get("uri", "")))
            else:
                item_type, uri = "file", str(item)
            lines.append(f"{index}. [{str(item_type).upper()}] {uri}")
        return "\n".join(lines)

    def _build_action_space(self) -> str:
        blocks = []
        for tool in self.tools.values():
            block = f"### {tool.name}\nDescription: {tool.description}"
            if tool.parameters:
                block += f"\nParameters: {json.dumps(tool.parameters, ensure_ascii=False)}"
            blocks.append(block)
        blocks.append(
            "### finish\nDescription: Return the delegated result to MainAgent.\n"
            f"Parameters: {json.dumps(FINISH_ACTION_SCHEMA, ensure_ascii=False)}"
        )
        return "Available actions:\n\n" + "\n\n".join(blocks)

    def _finish(self, params: Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        self._done = True
        finish_result = {
            "result": params.get("result", ""),
            "status": params.get("status", "done"),
            "summary": params.get("summary", ""),
        }
        observation = {
            "message": "Result reported to MainAgent.",
            "finish_result": finish_result,
            "current_step": self._steps,
        }
        return observation, 0.0, True, {
            "finished": True,
            "finish_result": finish_result,
            "last_action_result": observation,
        }

    def _continue_or_timeout(
        self,
        observation: Observation,
        info: Dict[str, Any],
    ) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self._steps < self.max_steps:
            return observation, 0.0, False, info
        self._done = True
        finish_result = {
            "result": "",
            "status": "partial",
            "summary": f"Reached the {self.max_steps}-step limit.",
        }
        observation["finish_result"] = finish_result
        return observation, 0.0, True, {
            **info,
            "finished": True,
            "max_steps_reached": True,
            "finish_result": finish_result,
        }
