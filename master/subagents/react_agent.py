"""ReAct SubAgent for focused tool-driven task execution."""
from __future__ import annotations

import re
from typing import Any, Dict, List

from pydantic import Field

from iflow_agent2.base.agent.base_agent import BaseAgent
from iflow_agent2.base.agent.memory import Memory
from iflow_agent2.base.engine.logs import LogLevel, logger
from iflow_agent2.base.engine.utils import parse_llm_action_response, parse_llm_output
from iflow_agent2.runtime.env import Action, BasicInfo, Observation

GENERAL_REACT_PROMPT = """You are a specialized SubAgent. Complete the assigned task efficiently.

==== Progress ====
[Step {current_step}/{max_steps}] Remaining {remaining_steps} steps
{budget_warning}

==== Assigned Task ====
{task_instruction}

==== Delegated Context ====
{context}

==== Original User Request ====
{original_question}

==== Available Tools ====
{action_space}

==== Working Rules ====
1. Focus only on the assigned task.
2. Use exact tool names and exact parameter names from Available Tools.
3. Record short durable findings in the memory field.
4. Do not invent tool results or claim to inspect an attachment without using a suitable tool/model.
5. Avoid repeating a failed behavior. Return a partial result when further attempts are unlikely to help.
6. Use finish immediately when the task is complete or only partial findings remain.

==== Output Format ====
Return exactly one JSON object without Markdown or extra text:
{{
  "action": "<exact tool name or finish>",
  "params": {{"tool-specific": "parameters"}},
  "memory": "short useful observation"
}}

To finish:
{{
  "action": "finish",
  "params": {{
    "result": "result returned to MainAgent",
    "status": "done or partial",
    "summary": "brief evidence and limitations"
  }},
  "memory": "task complete"
}}

==== Memory ====
{memory}

==== Current Observation ====
{obs}
"""


class ReActAgent(BaseAgent):
    """ReAct SubAgent running in an isolated ToolEnvironment."""

    name: str = Field(default="ReActAgent")
    description: str = Field(default="ReAct-style SubAgent for IFlow Agent")

    task_mode: str = Field(default="general")
    task_instruction: str = Field(default="")
    context: str = Field(default="")
    original_question: str = Field(default="")
    allowed_tools: List[str] | None = Field(default=None)

    current_env_instruction: str = Field(default="")
    current_action_space: str = Field(default="")
    memory: Memory = Field(default=None)

    class Config:
        arbitrary_types_allowed = True

    def reset(self, env_info: BasicInfo) -> None:
        if self.memory is None:
            self.memory = Memory(llm=self.llm, max_memory=10)
        else:
            self.memory.clear()

        if not self.original_question:
            self.original_question = env_info.instruction
        self.current_env_instruction = env_info.instruction

        if self.allowed_tools:
            self.current_action_space = self._filter_action_space(
                env_info.action_space,
                self.allowed_tools,
            )
            logger.info(f"[ReActAgent] Filtered to tools: {self.allowed_tools}")
        else:
            self.current_action_space = env_info.action_space

    def _normalize_tool_name(self, name: str) -> str:
        normalized = name.lower().replace("_", "")
        if normalized.endswith("action"):
            normalized = normalized[:-6]
        return normalized

    def _tool_matches(self, tool_name: str, allowed_tools: List[str]) -> bool:
        if tool_name in allowed_tools:
            return True
        normalized_tool = self._normalize_tool_name(tool_name)
        return any(
            self._normalize_tool_name(allowed) == normalized_tool
            for allowed in allowed_tools
        )

    def _filter_action_space(self, action_space: str, allowed_tools: List[str]) -> str:
        blocks = re.split(r"\n(?=### )", action_space)
        filtered_blocks = []
        for block in blocks:
            if block.startswith("Available actions"):
                filtered_blocks.append(block.rstrip())
                continue
            match = re.match(r"### (\w+)", block)
            if match and (
                match.group(1) == "finish"
                or self._tool_matches(match.group(1), allowed_tools)
            ):
                filtered_blocks.append(block.rstrip())
        return "\n\n".join(filtered_blocks)

    def parse_action(self, resp: str) -> Dict[str, Any]:
        return parse_llm_action_response(resp)

    def _get_memory(self) -> str:
        return self.memory.as_text()

    def _get_budget_warning(self, remaining_steps: int) -> str:
        if remaining_steps <= 3:
            return f"CRITICAL: Only {remaining_steps} steps left. Finish with the best available result now."
        if remaining_steps <= 5:
            return f"Warning: {remaining_steps} steps remain. Prepare to finish soon."
        return ""

    def _build_prompt(
        self,
        observation: Any,
        current_step: int,
        max_steps: int,
        remaining_steps: int,
        budget_warning: str,
    ) -> str:
        return GENERAL_REACT_PROMPT.format(
            task_instruction=self.task_instruction,
            context=self.context or "None",
            original_question=self.original_question,
            action_space=self.current_action_space,
            memory=self._get_memory(),
            obs=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=budget_warning,
        )

    async def step(
        self,
        observation: Observation,
        history: Any,
        current_step: int = 1,
        max_steps: int = 30,
    ) -> tuple[Action, str, str]:
        remaining_steps = max_steps - current_step
        prompt = self._build_prompt(
            observation=observation,
            current_step=current_step,
            max_steps=max_steps,
            remaining_steps=remaining_steps,
            budget_warning=self._get_budget_warning(remaining_steps),
        )
        logger.log_to_file(LogLevel.INFO, f"ReActAgent Input:\n{prompt}\n")
        try:
            resp = await self.llm(prompt)
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            resp = ""

        memory_content = parse_llm_output(resp, "memory")
        thinking = memory_content.get("memory") if isinstance(memory_content, dict) else None
        action = self.parse_action(resp)
        logger.agent_action(f"ReActAgent Action: {action}")

        agent_obs = history[-1].info.get("last_action_result") if history else observation
        await self.memory.add_memory(
            obs=agent_obs,
            action=action,
            thinking=thinking,
            raw_response=resp,
        )
        return action, resp, prompt

    async def run(self, request: str = None) -> str:
        return "Execution is managed by Runner."
