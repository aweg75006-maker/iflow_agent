from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from iflow_agent2.evaluation.models import EvalCase


def load_cases(path: str | Path) -> List[EvalCase]:
    source = Path(path).expanduser().resolve()
    suffix = source.suffix.lower()
    if suffix == ".jsonl":
        raw_cases = _load_jsonl(source)
    elif suffix == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        raw_cases = raw.get("cases", []) if isinstance(raw, dict) else raw
    elif suffix in {".yaml", ".yml"}:
        raw = yaml.safe_load(source.read_text(encoding="utf-8")) or []
        raw_cases = raw.get("cases", []) if isinstance(raw, dict) else raw
    else:
        raise ValueError("evaluation suite must be .jsonl, .json, .yaml, or .yml")

    if not isinstance(raw_cases, list):
        raise ValueError("evaluation suite must contain a list of cases")
    cases = [EvalCase.from_dict(item, source_dir=source.parent) for item in raw_cases]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("evaluation case ids must be unique")
    return cases


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    cases = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} must be an object")
            cases.append(value)
    return cases
