from __future__ import annotations

import asyncio
import json
from pathlib import Path

from iflow_agent2.evaluation import (
    EvalCase,
    EvaluationRunner,
    ScorerRegistry,
    load_cases,
)
from iflow_agent2.evaluation.judges import LLMSemanticJudge
from iflow_agent2.general_agent import GeneralAgentResult


def make_result(answer: str, *, tool: str | None = None) -> GeneralAgentResult:
    entries = []
    if tool:
        entries = [
            {
                "is_parallel_batch": True,
                "subtask_results": [
                    {
                        "trace": [
                            {
                                "action": {"action": tool, "params": {}},
                                "observation": {},
                            }
                        ],
                        "input_tokens": 3,
                        "output_tokens": 2,
                    }
                ],
            }
        ]
    return GeneralAgentResult(
        answer=answer,
        status="done",
        attempts=2,
        main_cost=0.1,
        duration_seconds=0.01,
        main_input_tokens=5,
        main_output_tokens=4,
        sub_input_tokens=3 if tool else 0,
        sub_output_tokens=2 if tool else 0,
        subtask_entries=entries,
    )


class FakeAgent:
    def __init__(self, result: GeneralAgentResult) -> None:
        self.result = result

    async def run(self, instruction, **kwargs):
        return self.result


def test_load_cases_resolves_attachments_and_scorers(tmp_path: Path) -> None:
    suite = tmp_path / "suite.jsonl"
    suite.write_text(
        json.dumps(
            {
                "id": "case-1",
                "instruction": "inspect",
                "expected": "ok",
                "scorer": "contains",
                "attachments": [{"type": "image", "path": "assets/a.png"}],
                "expected_tools": ["ImageAnalysisAction"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cases = load_cases(suite)

    assert cases[0].attachments[0]["path"] == str(
        (tmp_path / "assets/a.png").resolve()
    )
    assert [spec.type for spec in cases[0].scorers] == ["contains", "tool_usage"]


def test_scorers_cover_answer_structure_and_tool_trace() -> None:
    async def scenario() -> None:
        registry = ScorerRegistry()
        result = make_result('{"value": 42, "extra": true}', tool="EchoAction")

        exact_case = EvalCase.from_dict(
            {"id": "exact", "instruction": "x", "expected": "answer", "scorer": "exact"}
        )
        assert (await registry.score(exact_case, make_result(" Answer "), exact_case.scorers[0])).passed

        numeric_case = EvalCase.from_dict(
            {"id": "numeric", "instruction": "x", "expected": 42.01,
             "scorer": {"type": "numeric", "tolerance": 0.02}}
        )
        assert (await registry.score(numeric_case, make_result("42"), numeric_case.scorers[0])).passed

        json_case = EvalCase.from_dict(
            {"id": "json", "instruction": "x", "expected": {"value": 42}, "scorer": "json"}
        )
        assert (await registry.score(json_case, result, json_case.scorers[0])).passed

        tool_case = EvalCase.from_dict(
            {"id": "tool", "instruction": "x", "expected_tools": ["EchoAction"]}
        )
        tool_score = await registry.score(tool_case, result, tool_case.scorers[-1])
        assert tool_score.passed
        assert tool_score.details["used_tools"] == ["EchoAction"]

    asyncio.run(scenario())


def test_evaluation_runner_repeats_aggregates_and_saves(tmp_path: Path) -> None:
    cases = [
        EvalCase.from_dict(
            {"id": "a", "instruction": "x", "expected": "ok", "scorer": "exact", "category": "text"}
        ),
        EvalCase.from_dict(
            {"id": "b", "instruction": "x", "expected": "needle", "scorer": "contains", "category": "search"}
        ),
    ]

    def factory(case: EvalCase) -> FakeAgent:
        return FakeAgent(make_result("ok" if case.id == "a" else "has needle"))

    report = asyncio.run(
        EvaluationRunner(factory, max_concurrency=2, suite_name="unit").run(
            cases, repeats=2
        )
    )
    summary = report.summary()
    assert summary["total_runs"] == 4
    assert summary["total_cases"] == 2
    assert summary["pass_rate"] == 1.0
    assert summary["agent_completion_rate"] == 1.0
    assert summary["total_input_tokens"] == 20
    assert summary["scorers"]["exact"]["pass_rate"] == 1.0
    paths = report.save(tmp_path / "report")
    assert all(path.exists() for path in paths.values())
    assert len(paths["results"].read_text(encoding="utf-8").splitlines()) == 4


def test_semantic_judge_is_explicit_and_parses_json() -> None:
    class JudgeLLM:
        async def __call__(self, prompt):
            return '{"score": 0.8, "passed": true, "reason": "correct"}'

    async def scenario() -> None:
        case = EvalCase.from_dict(
            {"id": "semantic", "instruction": "x", "expected": "y", "scorer": "semantic"}
        )
        missing = await ScorerRegistry().score(case, make_result("y"), case.scorers[0])
        assert not missing.passed
        registry = ScorerRegistry(semantic_judge=LLMSemanticJudge(JudgeLLM()))
        scored = await registry.score(case, make_result("y"), case.scorers[0])
        assert scored.score == 0.8
        assert scored.passed

    asyncio.run(scenario())


def test_evaluation_runner_isolates_timeout() -> None:
    class SlowAgent:
        async def run(self, instruction, **kwargs):
            await asyncio.sleep(0.05)
            return make_result("late")

    case = EvalCase.from_dict(
        {
            "id": "timeout",
            "instruction": "x",
            "expected": "late",
            "scorer": "exact",
            "timeout_seconds": 0.001,
        }
    )
    report = asyncio.run(EvaluationRunner(lambda _: SlowAgent()).run([case]))
    assert report.results[0].agent_status == "error"
    assert report.results[0].passed is False
    assert "TimeoutError" in report.results[0].error
    assert report.summary()["error_rate"] == 1.0
