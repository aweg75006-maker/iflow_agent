"""verl-compatible GRPO reward for IFlow MainAgent decisions.

The callback combines deterministic protocol checks with an optional LLM judge:

    score = 0.10 * format + 0.10 * action
          + 0.20 * tool_reasonableness + 0.60 * decision_quality

The public ``compute_score`` signature follows verl's custom reward contract.
"""
from __future__ import annotations

import json
import os
import random
import re
import time
import traceback
from typing import Any

import requests

from iflow_agent2.base.engine.async_llm import LLMsConfig

W_FORMAT = 0.10
W_ACTION = 0.10
W_TOOL = 0.20
W_DECISION = 0.60

VALID_ACTIONS = {"delegate_task", "complete"}

_JSON_BLOCK_RE = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_BRACE_RE = re.compile(r"\{.*\}", re.DOTALL)
_JUDGE_SCORE_RE = re.compile(
    r"^(TOOL_REASONABLE|DECISION_QUALITY)\s*[:=]\s*([0-3])\s*$",
    re.IGNORECASE | re.MULTILINE,
)

_JUDGE_PROMPT = """You evaluate one decision produced by a multi-agent task coordinator.

The coordinator may either delegate independent tasks or complete the user request.
Score only the two dimensions below with an integer from 0 to 3.

USER REQUEST:
{question}

EXPECTED ANSWER OR SUCCESS CRITERIA:
{ground_truth}

CURRENT SUBTASK HISTORY:
{subtask_history}

OPTIONAL REFERENCE DECISION:
{reference_decision}

CANDIDATE RAW OUTPUT:
{candidate_raw}

CANDIDATE PARSED DECISION:
{candidate_json}

TOOL_REASONABLE:
3 = tools, models, and subtasks are well selected and clearly scoped
2 = workable selection with minor omissions or inefficiency
1 = weak or mostly inappropriate selection
0 = irrelevant, unusable, or missing required delegation details

DECISION_QUALITY:
3 = makes excellent progress or provides the correct final answer
2 = reasonable progress with notable limitations
1 = partially relevant but unlikely to solve the request
0 = invalid, irrelevant, or clearly incorrect

Return exactly these two lines and no explanation:
TOOL_REASONABLE: <0-3>
DECISION_QUALITY: <0-3>"""


def parse_decision(text: str) -> dict[str, Any] | None:
    """Extract the first usable JSON decision from model output."""
    if not text:
        return None

    fenced = _JSON_BLOCK_RE.search(text)
    if fenced:
        try:
            value = json.loads(fenced.group(1))
            return value if isinstance(value, dict) else None
        except json.JSONDecodeError:
            pass

    braced = _BRACE_RE.search(text)
    if braced:
        candidate = braced.group(0)
        for end in range(len(candidate), 0, -1):
            if candidate[end - 1] != "}":
                continue
            try:
                value = json.loads(candidate[:end])
                return value if isinstance(value, dict) else None
            except json.JSONDecodeError:
                continue

    try:
        value = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def validate_decision(decision: dict[str, Any] | None) -> tuple[float, float]:
    """Return deterministic format and action-validity scores."""
    if not isinstance(decision, dict):
        return 0.0, 0.0

    action = decision.get("action")
    params = decision.get("params")
    format_score = float(isinstance(action, str) and isinstance(params, dict))
    if not format_score or action not in VALID_ACTIONS:
        return format_score, 0.0

    if action == "complete":
        answer = params.get("answer")
        return format_score, float(isinstance(answer, str) and bool(answer.strip()))

    tasks = params.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        return format_score, 0.0
    valid_tasks = all(
        isinstance(task, dict)
        and isinstance(task.get("task_instruction"), str)
        and bool(task["task_instruction"].strip())
        and isinstance(task.get("model"), str)
        and bool(task["model"].strip())
        and (
            "tools" not in task
            or isinstance(task.get("tools"), list)
        )
        for task in tasks
    )
    return format_score, float(valid_tasks)


def parse_judge_scores(response: str | None) -> dict[str, float]:
    """Parse and normalize the two 0-3 judge dimensions."""
    scores = {"tool_reasonable": 0.0, "decision_quality": 0.0}
    if not response:
        return scores

    key_map = {
        "TOOL_REASONABLE": "tool_reasonable",
        "DECISION_QUALITY": "decision_quality",
    }
    for match in _JUDGE_SCORE_RE.finditer(response):
        scores[key_map[match.group(1).upper()]] = int(match.group(2)) / 3.0
    return scores


def _judge_config() -> tuple[str, str, str]:
    """Resolve the judge from IFlow model config with optional RL overrides."""
    alias = os.getenv("IFLOW_RL_JUDGE_MODEL", "mimo-pro")
    config = LLMsConfig.default().get(alias)
    model = os.getenv("IFLOW_RL_JUDGE_REQUEST_MODEL", config.model)
    base_url = os.getenv("IFLOW_RL_JUDGE_BASE_URL", config.base_url)
    api_key = os.getenv("IFLOW_RL_JUDGE_API_KEY", config.key)
    return model, base_url, api_key


def _call_llm_judge(prompt: str) -> str | None:
    """Call the configured OpenAI-compatible judge with bounded retries."""
    model, base_url, api_key = _judge_config()
    timeout = float(os.getenv("IFLOW_RL_JUDGE_TIMEOUT", "60"))
    max_retries = int(os.getenv("IFLOW_RL_JUDGE_MAX_RETRIES", "3"))
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 128,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            if attempt >= max_retries:
                print(f"[iflow-grpo] judge failed: {exc}")
                return None
            delay = min(2.0**attempt, 30.0) + random.uniform(0.0, 0.5)
            time.sleep(delay)
    return None


def _format_reference(extra_info: dict[str, Any]) -> str:
    reference = extra_info.get("reference_decision")
    if not isinstance(reference, dict):
        return "No reference decision provided."
    return json.dumps(reference, ensure_ascii=False, indent=2)


def _evaluate_with_judge(
    solution_str: str,
    decision: dict[str, Any],
    ground_truth: str,
    extra_info: dict[str, Any],
) -> dict[str, float]:
    question = str(extra_info.get("question") or extra_info.get("prompt") or "N/A")
    expected = str(extra_info.get("answer") or ground_truth or "N/A")
    history = str(extra_info.get("subtask_history") or "No subtasks completed yet.")
    prompt = _JUDGE_PROMPT.format(
        question=question,
        ground_truth=expected,
        subtask_history=history,
        reference_decision=_format_reference(extra_info),
        candidate_raw=solution_str[:4000],
        candidate_json=json.dumps(decision, ensure_ascii=False, indent=2),
    )
    return parse_judge_scores(_call_llm_judge(prompt))


def _zero_score() -> dict[str, float]:
    return {
        "score": 0.0,
        "acc": 0.0,
        "format_correct": 0.0,
        "action_valid": 0.0,
        "tool_reasonable": 0.0,
        "decision_quality": 0.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score one rollout using the custom reward callback expected by verl."""
    del data_source, kwargs
    try:
        decision = parse_decision(solution_str)
        format_score, action_score = validate_decision(decision)
        if decision is None or action_score == 0.0:
            result = _zero_score()
            result["format_correct"] = format_score
            result["score"] = W_FORMAT * format_score
            return result

        try:
            judged = _evaluate_with_judge(
                solution_str,
                decision,
                ground_truth,
                extra_info or {},
            )
        except Exception as exc:
            print(f"[iflow-grpo] judge unavailable: {exc}")
            judged = {"tool_reasonable": 0.0, "decision_quality": 0.0}
        tool_score = judged["tool_reasonable"]
        decision_score = judged["decision_quality"]
        total = (
            W_FORMAT * format_score
            + W_ACTION * action_score
            + W_TOOL * tool_score
            + W_DECISION * decision_score
        )
        return {
            "score": max(0.0, min(1.0, total)),
            "acc": float(decision_score >= 2 / 3 and format_score == 1.0),
            "format_correct": format_score,
            "action_valid": action_score,
            "tool_reasonable": tool_score,
            "decision_quality": decision_score,
        }
    except Exception as exc:
        print(f"[iflow-grpo] reward computation failed: {exc}")
        traceback.print_exc()
        return _zero_score()
