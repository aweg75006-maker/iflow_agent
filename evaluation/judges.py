from __future__ import annotations

import json
import re
from typing import Any

from iflow_agent2.base.engine.async_llm import LLMsConfig, create_llm_instance
from iflow_agent2.evaluation.models import EvalCase, ScoreResult, ScorerSpec
from iflow_agent2.general_agent import GeneralAgentResult


class LLMSemanticJudge:
    """Rubric-based semantic judge using an injected AsyncLLM-compatible object."""

    def __init__(self, llm: Any) -> None:
        self.llm = llm

    @classmethod
    def from_model_name(cls, model_name: str) -> "LLMSemanticJudge":
        return cls(create_llm_instance(LLMsConfig.default().get(model_name)))

    async def __call__(
        self,
        case: EvalCase,
        result: GeneralAgentResult,
        spec: ScorerSpec,
    ) -> ScoreResult:
        rubric = spec.params.get("rubric") or case.metadata.get("rubric") or (
            "The answer must be correct, relevant, supported by the available evidence, "
            "and must not contradict the expected answer."
        )
        prompt = f"""You are evaluating one general-agent response.

Task:
{case.instruction}

Expected answer or reference:
{case.expected}

Agent answer:
{result.answer}

Rubric:
{rubric}

Return one JSON object only:
{{"score": <number from 0 to 1>, "passed": <boolean>, "reason": "brief explanation"}}
"""
        raw = await self.llm(prompt)
        data = self._parse_response(str(raw))
        score = float(data.get("score", 0.0))
        return ScoreResult(
            scorer="semantic",
            score=score,
            passed=bool(data.get("passed", score >= 0.5)),
            reason=str(data.get("reason", "semantic judge result")),
            details={"judge_raw_response": str(raw)},
        )

    @staticmethod
    def _parse_response(raw: str) -> dict:
        text = raw.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not match:
                raise ValueError("semantic judge did not return JSON")
            value = json.loads(match.group(0))
        if not isinstance(value, dict):
            raise ValueError("semantic judge JSON must be an object")
        return value
