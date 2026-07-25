"""Public entrypoint for the IFlow multi-agent runtime."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Callable, Dict, Iterable, List, Optional

from iflow_agent2.base.agent.base_action import BaseAction
from iflow_agent2.base.engine.async_llm import LLMsConfig, create_llm_instance
from iflow_agent2.master.main_agent import MainAgent
from iflow_agent2.master.prompts import GeneralMainAgentPrompt
from iflow_agent2.master.tools.complete import CompleteTool
from iflow_agent2.master.tools.delegate import DelegateTaskTool
from iflow_agent2.runtime.runner import Runner
from iflow_agent2.runtime.tool_environment import ToolEnvironment


@dataclass
class GeneralAgentResult:
    answer: str
    status: str
    attempts: int
    main_cost: float = 0.0
    sub_cost: float = 0.0
    duration_seconds: float = 0.0
    main_input_tokens: int = 0
    main_output_tokens: int = 0
    sub_input_tokens: int = 0
    sub_output_tokens: int = 0
    trace: List[Dict[str, Any]] = field(default_factory=list)
    subtask_entries: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def total_cost(self) -> float:
        return self.main_cost + self.sub_cost

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["total_cost"] = self.total_cost
        return data


class GeneralAgent:
    """General-purpose facade for planning, delegation, and tool execution."""

    def __init__(
        self,
        *,
        main_llm: Any,
        sub_models: Iterable[str],
        tools: Iterable[BaseAction],
        llm_factory: Optional[Callable[[str], Any]] = None,
        max_attempts: int = 8,
        subagent_max_steps: int = 20,
        runner: Optional[Runner] = None,
        trace_summarizer: Any = None,
        mask_model_names: bool = False,
    ) -> None:
        self.main_llm = main_llm
        self.sub_models = list(sub_models)
        self.tools = list(tools)
        self.llm_factory = llm_factory or self._default_llm_factory
        self.max_attempts = max_attempts
        self.subagent_max_steps = subagent_max_steps
        self.runner = runner or Runner()
        self.trace_summarizer = trace_summarizer
        self.mask_model_names = mask_model_names
        self._validate_configuration()

    @classmethod
    def from_model_names(
        cls,
        *,
        main_model: str,
        sub_models: Iterable[str],
        tools: Iterable[BaseAction],
        **kwargs: Any,
    ) -> "GeneralAgent":
        main_llm = create_llm_instance(LLMsConfig.default().get(main_model))
        return cls(
            main_llm=main_llm,
            sub_models=sub_models,
            tools=tools,
            **kwargs,
        )

    async def run(
        self,
        instruction: str,
        *,
        attachments: Optional[Iterable[Any]] = None,
        meta_data: Optional[Dict[str, Any]] = None,
        task_id: str = "general-task",
    ) -> GeneralAgentResult:
        started_at = perf_counter()
        env = ToolEnvironment(
            instruction=instruction,
            tools=self.tools,
            attachments=attachments,
            max_steps=self.subagent_max_steps,
            env_id=task_id,
            meta_data=meta_data,
        )
        alias_to_model = (
            {f"model_{index + 1}": model for index, model in enumerate(self.sub_models)}
            if self.mask_model_names
            else {}
        )
        delegate_tool = DelegateTaskTool(
            env=env,
            runner=self.runner,
            models=self.sub_models,
            task_mode="general",
            alias_to_model=alias_to_model,
            llm_factory=self.llm_factory,
            trace_summarizer=self.trace_summarizer,
        )
        main_agent = MainAgent(
            llm=self.main_llm,
            sub_models=self.sub_models,
            tools=[delegate_tool, CompleteTool()],
            subagent_tools=self.tools,
            prompt_builder=GeneralMainAgentPrompt,
            max_attempts=self.max_attempts,
            task_mode="general",
            mask_model_names=self.mask_model_names,
        )
        main_agent.reset(env.get_basic_info())
        main_cost_before = self._usage_cost(self.main_llm)
        main_usage_before = self._usage_tokens(self.main_llm)
        trace: List[Dict[str, Any]] = []
        answer = ""
        status = "partial"

        for attempt in range(1, self.max_attempts + 1):
            action, raw_response = await main_agent.step(None, [])
            trace.append(
                {
                    "attempt": attempt,
                    "action": action,
                    "raw_response": raw_response,
                }
            )
            if action.get("action") == "complete":
                answer = str(action.get("params", {}).get("answer", ""))
                status = "done"
                break

        if not answer:
            answer = self._best_partial_answer(main_agent.task_entries)
        main_cost = max(0.0, self._usage_cost(self.main_llm) - main_cost_before)
        main_usage_after = self._usage_tokens(self.main_llm)
        main_input_tokens = max(0, main_usage_after[0] - main_usage_before[0])
        main_output_tokens = max(0, main_usage_after[1] - main_usage_before[1])
        sub_cost = self._subtask_cost(main_agent.task_entries)
        sub_input_tokens, sub_output_tokens = self._subtask_tokens(
            main_agent.task_entries
        )
        return GeneralAgentResult(
            answer=answer,
            status=status,
            attempts=len(trace),
            main_cost=main_cost,
            sub_cost=sub_cost,
            duration_seconds=perf_counter() - started_at,
            main_input_tokens=main_input_tokens,
            main_output_tokens=main_output_tokens,
            sub_input_tokens=sub_input_tokens,
            sub_output_tokens=sub_output_tokens,
            trace=trace,
            subtask_entries=main_agent.task_entries,
        )

    def _default_llm_factory(self, model_name: str) -> Any:
        return create_llm_instance(LLMsConfig.default().get(model_name))

    def _validate_configuration(self) -> None:
        if not self.sub_models:
            raise ValueError("sub_models must contain at least one model name")
        if self.max_attempts < 1 or self.subagent_max_steps < 1:
            raise ValueError("max_attempts and subagent_max_steps must be positive")
        names = [tool.name for tool in self.tools]
        if len(names) != len(set(names)):
            raise ValueError(f"tool names must be unique: {names}")
        reserved = {"finish", "delegate_task", "complete"}
        conflicts = sorted(reserved.intersection(names))
        if conflicts:
            raise ValueError(f"tool names are reserved: {conflicts}")

    @staticmethod
    def _usage_cost(llm: Any) -> float:
        getter = getattr(llm, "get_usage_summary", None)
        if not callable(getter):
            return 0.0
        return float(getter().get("total_cost", 0.0) or 0.0)

    @staticmethod
    def _usage_tokens(llm: Any) -> tuple[int, int]:
        getter = getattr(llm, "get_usage_summary", None)
        if not callable(getter):
            return 0, 0
        summary = getter()
        return (
            int(summary.get("total_input_tokens", 0) or 0),
            int(summary.get("total_output_tokens", 0) or 0),
        )

    @staticmethod
    def _subtask_cost(entries: List[Dict[str, Any]]) -> float:
        total = 0.0
        for entry in entries:
            if entry.get("is_parallel_batch"):
                total += float(entry.get("total_cost", 0.0) or 0.0)
            else:
                total += float(entry.get("cost", 0.0) or 0.0)
        return total

    @staticmethod
    def _subtask_tokens(entries: List[Dict[str, Any]]) -> tuple[int, int]:
        input_tokens = 0
        output_tokens = 0
        for entry in entries:
            candidates = (
                entry.get("subtask_results", [])
                if entry.get("is_parallel_batch")
                else [entry]
            )
            for candidate in candidates:
                input_tokens += int(candidate.get("input_tokens", 0) or 0)
                output_tokens += int(candidate.get("output_tokens", 0) or 0)
        return input_tokens, output_tokens

    @staticmethod
    def _best_partial_answer(entries: List[Dict[str, Any]]) -> str:
        answers = []
        for entry in entries:
            candidates = entry.get("subtask_results", []) if entry.get("is_parallel_batch") else [entry]
            for candidate in candidates:
                result = candidate.get("result")
                if result and result != "-":
                    answers.append(str(result))
        return "\n\n".join(answers) if answers else "No conclusive result was produced."
