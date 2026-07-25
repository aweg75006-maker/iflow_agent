from __future__ import annotations

import inspect
import json
import math
import re
from typing import Any, Awaitable, Callable, Dict, Iterable, Optional

from iflow_agent2.evaluation.models import EvalCase, ScoreResult, ScorerSpec
from iflow_agent2.general_agent import GeneralAgentResult

SemanticJudge = Callable[
    [EvalCase, GeneralAgentResult, ScorerSpec],
    Awaitable[ScoreResult | float | Dict[str, Any]] | ScoreResult | float | Dict[str, Any],
]


class ScorerRegistry:
    def __init__(self, semantic_judge: Optional[SemanticJudge] = None) -> None:
        self.semantic_judge = semantic_judge
        self._scorers = {
            "completion": self._completion,
            "exact": self._exact,
            "contains": self._contains,
            "numeric": self._numeric,
            "json": self._json,
            "tool_usage": self._tool_usage,
            "semantic": self._semantic,
        }

    async def score(
        self,
        case: EvalCase,
        result: GeneralAgentResult,
        spec: ScorerSpec,
    ) -> ScoreResult:
        scorer = self._scorers.get(spec.type)
        if scorer is None:
            raise ValueError(f"unknown scorer type: {spec.type}")
        value = scorer(case, result, spec)
        return await value if inspect.isawaitable(value) else value

    @staticmethod
    def _normalize(value: Any, *, case_sensitive: bool = False) -> str:
        text = " ".join(str(value if value is not None else "").split())
        return text if case_sensitive else text.casefold()

    def _completion(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        passed = result.status == "done" and bool(result.answer.strip())
        return ScoreResult(
            scorer="completion",
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="agent completed with a non-empty answer" if passed else "agent did not complete",
        )

    def _exact(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        case_sensitive = bool(spec.params.get("case_sensitive", False))
        actual = self._normalize(result.answer, case_sensitive=case_sensitive)
        expected_values = case.expected if isinstance(case.expected, list) else [case.expected]
        normalized = [
            self._normalize(item, case_sensitive=case_sensitive) for item in expected_values
        ]
        passed = actual in normalized
        return ScoreResult(
            scorer="exact",
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="exact match" if passed else "answer did not exactly match expected value",
            details={"actual": result.answer, "expected": case.expected},
        )

    def _contains(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        case_sensitive = bool(spec.params.get("case_sensitive", False))
        actual = self._normalize(result.answer, case_sensitive=case_sensitive)
        expected_values = case.expected if isinstance(case.expected, list) else [case.expected]
        needles = [
            self._normalize(item, case_sensitive=case_sensitive) for item in expected_values
        ]
        matches = [needle in actual for needle in needles if needle]
        match_mode = spec.params.get("match", "all")
        passed = bool(matches) and (any(matches) if match_mode == "any" else all(matches))
        score = sum(matches) / len(matches) if matches else 0.0
        if match_mode == "any" and passed:
            score = 1.0
        return ScoreResult(
            scorer="contains",
            score=score,
            passed=passed,
            reason=f"matched {sum(matches)}/{len(matches)} expected fragments" if matches else "no expected fragments",
            details={"match": match_mode, "expected": case.expected},
        )

    def _numeric(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        actual = self._first_number(result.answer)
        expected = self._first_number(case.expected)
        if actual is None or expected is None:
            return ScoreResult("numeric", 0.0, False, "could not parse numeric value")
        absolute = float(spec.params.get("absolute_tolerance", spec.params.get("tolerance", 0.0)))
        relative = float(spec.params.get("relative_tolerance", 0.0))
        allowed = max(absolute, abs(expected) * relative)
        difference = abs(actual - expected)
        passed = difference <= allowed
        return ScoreResult(
            scorer="numeric",
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"difference={difference:g}, allowed={allowed:g}",
            details={"actual": actual, "expected": expected},
        )

    @staticmethod
    def _first_number(value: Any) -> Optional[float]:
        match = re.search(r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        if not match:
            return None
        try:
            number = float(match.group(0).replace(",", ""))
        except ValueError:
            return None
        return number if math.isfinite(number) else None

    def _json(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        try:
            actual = self._parse_json_answer(result.answer)
        except (json.JSONDecodeError, ValueError) as exc:
            return ScoreResult("json", 0.0, False, f"invalid JSON answer: {exc}")
        mode = str(spec.params.get("mode", "subset"))
        required_keys = spec.params.get("required_keys", [])
        keys_ok = isinstance(actual, dict) and all(key in actual for key in required_keys)
        if not required_keys:
            keys_ok = True
        values_ok = (
            self._is_subset(case.expected, actual)
            if mode == "subset"
            else actual == case.expected
        )
        passed = keys_ok and values_ok
        return ScoreResult(
            scorer="json",
            score=1.0 if passed else 0.0,
            passed=passed,
            reason="JSON structure matched" if passed else "JSON structure/value mismatch",
            details={"actual": actual, "expected": case.expected, "mode": mode},
        )

    @staticmethod
    def _parse_json_answer(answer: str) -> Any:
        text = answer.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        return json.loads(text)

    @classmethod
    def _is_subset(cls, expected: Any, actual: Any) -> bool:
        if isinstance(expected, dict):
            return isinstance(actual, dict) and all(
                key in actual and cls._is_subset(value, actual[key])
                for key, value in expected.items()
            )
        if isinstance(expected, list):
            return isinstance(actual, list) and all(item in actual for item in expected)
        return expected == actual

    def _tool_usage(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        used = self.extract_tool_calls(result)
        required = set(spec.params.get("required_tools", []))
        forbidden = set(spec.params.get("forbidden_tools", []))
        required_score = len(required.intersection(used)) / len(required) if required else 1.0
        forbidden_score = 0.0 if forbidden.intersection(used) else 1.0
        components = [required_score]
        if forbidden:
            components.append(forbidden_score)
        score = sum(components) / len(components)
        passed = required.issubset(used) and not forbidden.intersection(used)
        return ScoreResult(
            scorer="tool_usage",
            score=score,
            passed=passed,
            reason="tool constraints satisfied" if passed else "tool constraints were not satisfied",
            details={
                "used_tools": sorted(used),
                "missing_tools": sorted(required - used),
                "forbidden_used": sorted(forbidden.intersection(used)),
            },
        )

    @staticmethod
    def extract_tool_calls(result: GeneralAgentResult) -> set[str]:
        names: set[str] = set()
        for entry in result.subtask_entries:
            candidates: Iterable[Dict[str, Any]] = (
                entry.get("subtask_results", [])
                if entry.get("is_parallel_batch")
                else [entry]
            )
            for candidate in candidates:
                for step in candidate.get("trace", []):
                    action = step.get("action", {}) if isinstance(step, dict) else {}
                    name = action.get("action") if isinstance(action, dict) else None
                    if name and name not in {"finish", "delegate_task", "complete"}:
                        names.add(str(name))
        return names

    async def _semantic(
        self, case: EvalCase, result: GeneralAgentResult, spec: ScorerSpec
    ) -> ScoreResult:
        if self.semantic_judge is None:
            return ScoreResult(
                "semantic", 0.0, False, "semantic scorer requires an injected judge"
            )
        judged = self.semantic_judge(case, result, spec)
        judged = await judged if inspect.isawaitable(judged) else judged
        if isinstance(judged, ScoreResult):
            return judged
        if isinstance(judged, dict):
            score = float(judged.get("score", 0.0))
            return ScoreResult(
                "semantic",
                score,
                bool(judged.get("passed", score >= 0.5)),
                str(judged.get("reason", "semantic judge result")),
                dict(judged.get("details") or {}),
            )
        score = float(judged)
        return ScoreResult("semantic", score, score >= 0.5, "semantic judge result")
