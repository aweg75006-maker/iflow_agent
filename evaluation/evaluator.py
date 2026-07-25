from __future__ import annotations

import asyncio
import inspect
from time import perf_counter
from typing import Any, Callable, Iterable, Optional

from iflow_agent2.evaluation.models import (
    EvalCase,
    EvaluationReport,
    EvaluationRunResult,
    ScoreResult,
)
from iflow_agent2.evaluation.scorers import ScorerRegistry
from iflow_agent2.general_agent import GeneralAgentResult

AgentFactory = Callable[[EvalCase], Any]


class EvaluationRunner:
    def __init__(
        self,
        agent_factory: AgentFactory,
        *,
        scorers: Optional[ScorerRegistry] = None,
        max_concurrency: int = 1,
        default_timeout_seconds: Optional[float] = 900.0,
        suite_name: str = "iflow-evaluation",
    ) -> None:
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        self.agent_factory = agent_factory
        self.scorers = scorers or ScorerRegistry()
        self.max_concurrency = max_concurrency
        self.default_timeout_seconds = default_timeout_seconds
        self.suite_name = suite_name

    async def run(
        self,
        cases: Iterable[EvalCase],
        *,
        repeats: int = 1,
    ) -> EvaluationReport:
        if repeats < 1:
            raise ValueError("repeats must be positive")
        indexed = [
            (index, case, repeat)
            for index, case in enumerate(cases)
            for repeat in range(1, repeats + 1)
        ]
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def execute(index: int, case: EvalCase, repeat: int):
            async with semaphore:
                result = await self._run_once(case, repeat)
                return index, repeat, result

        completed = await asyncio.gather(
            *(execute(index, case, repeat) for index, case, repeat in indexed)
        )
        completed.sort(key=lambda item: (item[0], item[1]))
        return EvaluationReport(
            results=[item[2] for item in completed],
            suite_name=self.suite_name,
        )

    async def _run_once(self, case: EvalCase, repeat: int) -> EvaluationRunResult:
        started_at = perf_counter()
        try:
            agent = self.agent_factory(case)
            if inspect.isawaitable(agent):
                agent = await agent
            run = agent.run(
                case.instruction,
                attachments=case.attachments,
                meta_data={
                    **case.metadata,
                    "evaluation_case_id": case.id,
                    "evaluation_category": case.category,
                    "evaluation_tags": case.tags,
                },
                task_id=f"{case.id}-run-{repeat}",
            )
            timeout = case.timeout_seconds or self.default_timeout_seconds
            agent_result = (
                await asyncio.wait_for(run, timeout=timeout)
                if timeout is not None
                else await run
            )
            if not isinstance(agent_result, GeneralAgentResult):
                raise TypeError("agent.run must return GeneralAgentResult")
        except Exception as exc:
            duration = perf_counter() - started_at
            return EvaluationRunResult(
                case_id=case.id,
                repeat=repeat,
                category=case.category,
                score=0.0,
                passed=False,
                answer="",
                agent_status="error",
                scorer_results=[],
                duration_seconds=duration,
                attempts=0,
                total_cost=0.0,
                input_tokens=0,
                output_tokens=0,
                case_weight=case.weight,
                error=f"{type(exc).__name__}: {exc}",
            )

        scored: list[ScoreResult] = []
        for spec in case.scorers:
            try:
                scored.append(await self.scorers.score(case, agent_result, spec))
            except Exception as exc:
                scored.append(
                    ScoreResult(
                        scorer=spec.type,
                        score=0.0,
                        passed=False,
                        reason=f"scorer failed: {type(exc).__name__}: {exc}",
                    )
                )
        total_scorer_weight = sum(spec.weight for spec in case.scorers)
        score = sum(
            result.score * spec.weight
            for result, spec in zip(scored, case.scorers)
        ) / total_scorer_weight
        input_tokens = agent_result.main_input_tokens + agent_result.sub_input_tokens
        output_tokens = agent_result.main_output_tokens + agent_result.sub_output_tokens
        return EvaluationRunResult(
            case_id=case.id,
            repeat=repeat,
            category=case.category,
            score=score,
            passed=score >= case.pass_threshold,
            answer=agent_result.answer,
            agent_status=agent_result.status,
            scorer_results=scored,
            duration_seconds=perf_counter() - started_at,
            attempts=agent_result.attempts,
            total_cost=agent_result.total_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            case_weight=case.weight,
            agent_result=agent_result.to_dict(),
        )
