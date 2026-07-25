from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Dict, Iterable, List, Optional


@dataclass(frozen=True)
class ScorerSpec:
    type: str
    weight: float = 1.0
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_value(cls, value: str | Dict[str, Any]) -> "ScorerSpec":
        if isinstance(value, str):
            return cls(type=value)
        if not isinstance(value, dict) or not value.get("type"):
            raise ValueError(f"Invalid scorer specification: {value!r}")
        params = {
            key: item for key, item in value.items() if key not in {"type", "weight"}
        }
        weight = float(value.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("scorer weight must be positive")
        return cls(type=str(value["type"]), weight=weight, params=params)


@dataclass
class EvalCase:
    id: str
    instruction: str
    expected: Any = None
    category: str = "general"
    attachments: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    scorers: List[ScorerSpec] = field(default_factory=list)
    allowed_tools: Optional[List[str]] = None
    tags: List[str] = field(default_factory=list)
    pass_threshold: float = 0.5
    timeout_seconds: Optional[float] = None
    weight: float = 1.0

    @classmethod
    def from_dict(
        cls,
        data: Dict[str, Any],
        *,
        source_dir: Optional[Path] = None,
    ) -> "EvalCase":
        case_id = str(data.get("id", "")).strip()
        instruction = str(data.get("instruction", "")).strip()
        if not case_id or not instruction:
            raise ValueError("each evaluation case requires non-empty id and instruction")

        raw_scorers = data.get("scorers")
        if raw_scorers is None:
            raw_scorers = [data.get("scorer", "exact" if "expected" in data else "completion")]
        elif not isinstance(raw_scorers, list):
            raw_scorers = [raw_scorers]
        scorers = [ScorerSpec.from_value(item) for item in raw_scorers]

        expected_tools = data.get("expected_tools")
        forbidden_tools = data.get("forbidden_tools")
        if expected_tools is not None or forbidden_tools is not None:
            if not any(spec.type == "tool_usage" for spec in scorers):
                scorers.append(
                    ScorerSpec(
                        type="tool_usage",
                        weight=float(data.get("tool_score_weight", 1.0)),
                        params={
                            "required_tools": list(expected_tools or []),
                            "forbidden_tools": list(forbidden_tools or []),
                        },
                    )
                )

        attachments = cls._resolve_attachments(data.get("attachments", []), source_dir)
        threshold = float(data.get("pass_threshold", 0.5))
        if not 0 <= threshold <= 1:
            raise ValueError("pass_threshold must be between 0 and 1")
        weight = float(data.get("weight", 1.0))
        if weight <= 0:
            raise ValueError("case weight must be positive")

        return cls(
            id=case_id,
            instruction=instruction,
            expected=data.get("expected"),
            category=str(data.get("category", "general")),
            attachments=attachments,
            metadata=dict(data.get("metadata") or {}),
            scorers=scorers,
            allowed_tools=(
                list(data["allowed_tools"])
                if data.get("allowed_tools") is not None
                else None
            ),
            tags=[str(item) for item in data.get("tags", [])],
            pass_threshold=threshold,
            timeout_seconds=(
                float(data["timeout_seconds"])
                if data.get("timeout_seconds") is not None
                else None
            ),
            weight=weight,
        )

    @staticmethod
    def _resolve_attachments(items: Iterable[Any], source_dir: Optional[Path]) -> List[Any]:
        resolved = []
        for item in items or []:
            if not isinstance(item, dict) or not item.get("path") or source_dir is None:
                resolved.append(item)
                continue
            attachment = dict(item)
            path = Path(str(attachment["path"])).expanduser()
            if not path.is_absolute():
                path = (source_dir / path).resolve()
            attachment["path"] = str(path)
            resolved.append(attachment)
        return resolved


@dataclass
class ScoreResult:
    scorer: str
    score: float
    passed: bool
    reason: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.score = max(0.0, min(1.0, float(self.score)))


@dataclass
class EvaluationRunResult:
    case_id: str
    repeat: int
    category: str
    score: float
    passed: bool
    answer: str
    agent_status: str
    scorer_results: List[ScoreResult]
    duration_seconds: float
    attempts: int
    total_cost: float
    input_tokens: int
    output_tokens: int
    case_weight: float = 1.0
    error: Optional[str] = None
    agent_result: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    results: List[EvaluationRunResult]
    suite_name: str = "iflow-evaluation"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

    def summary(self) -> Dict[str, Any]:
        if not self.results:
            return {
                "suite_name": self.suite_name,
                "total_runs": 0,
                "total_cases": 0,
                "pass_rate": 0.0,
                "weighted_score": 0.0,
                "categories": {},
            }

        total_weight = sum(item.case_weight for item in self.results)
        weighted_score = sum(
            item.score * item.case_weight for item in self.results
        ) / total_weight
        categories: Dict[str, Dict[str, Any]] = {}
        for category in sorted({item.category for item in self.results}):
            subset = [item for item in self.results if item.category == category]
            category_weight = sum(item.case_weight for item in subset)
            categories[category] = {
                "runs": len(subset),
                "pass_rate": sum(item.passed for item in subset) / len(subset),
                "weighted_score": sum(
                    item.score * item.case_weight for item in subset
                ) / category_weight,
                "avg_latency_seconds": fmean(item.duration_seconds for item in subset),
            }

        scorer_groups: Dict[str, List[ScoreResult]] = {}
        for run in self.results:
            for scorer_result in run.scorer_results:
                scorer_groups.setdefault(scorer_result.scorer, []).append(scorer_result)
        scorer_summary = {
            name: {
                "runs": len(items),
                "pass_rate": sum(item.passed for item in items) / len(items),
                "avg_score": fmean(item.score for item in items),
            }
            for name, items in sorted(scorer_groups.items())
        }

        scores_by_case: Dict[str, List[float]] = {}
        for result in self.results:
            scores_by_case.setdefault(result.case_id, []).append(result.score)
        repeated = [scores for scores in scores_by_case.values() if len(scores) > 1]

        return {
            "suite_name": self.suite_name,
            "created_at": self.created_at,
            "total_runs": len(self.results),
            "total_cases": len(scores_by_case),
            "passed_runs": sum(item.passed for item in self.results),
            "pass_rate": sum(item.passed for item in self.results) / len(self.results),
            "agent_completion_rate": sum(
                item.agent_status == "done" for item in self.results
            ) / len(self.results),
            "weighted_score": weighted_score,
            "error_rate": sum(bool(item.error) for item in self.results) / len(self.results),
            "avg_latency_seconds": fmean(item.duration_seconds for item in self.results),
            "avg_attempts": fmean(item.attempts for item in self.results),
            "total_cost": sum(item.total_cost for item in self.results),
            "total_input_tokens": sum(item.input_tokens for item in self.results),
            "total_output_tokens": sum(item.output_tokens for item in self.results),
            "avg_input_tokens": fmean(item.input_tokens for item in self.results),
            "avg_output_tokens": fmean(item.output_tokens for item in self.results),
            "repeat_score_stddev": (
                fmean(pstdev(scores) for scores in repeated) if repeated else 0.0
            ),
            "categories": categories,
            "scorers": scorer_summary,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "summary": self.summary(),
            "results": [item.to_dict() for item in self.results],
        }

    def save(self, output_dir: str | Path) -> Dict[str, Path]:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        summary_path = destination / "summary.json"
        results_path = destination / "results.jsonl"
        csv_path = destination / "results.csv"

        summary_path.write_text(
            json.dumps(self.summary(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with results_path.open("w", encoding="utf-8") as file:
            for result in self.results:
                file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

        fields = [
            "case_id", "repeat", "category", "score", "passed", "agent_status",
            "duration_seconds", "attempts", "total_cost", "input_tokens",
            "output_tokens", "error", "answer",
        ]
        with csv_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for result in self.results:
                data = result.to_dict()
                writer.writerow({key: data.get(key) for key in fields})
        return {"summary": summary_path, "results": results_path, "csv": csv_path}
