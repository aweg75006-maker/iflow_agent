from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Callable, Dict

from iflow_agent2 import GeneralAgent
from iflow_agent2.evaluation.evaluator import EvaluationRunner
from iflow_agent2.evaluation.judges import LLMSemanticJudge
from iflow_agent2.evaluation.loaders import load_cases
from iflow_agent2.evaluation.scorers import ScorerRegistry
from iflow_agent2.tools import (
    ExecuteCodeAction,
    ExtractUrlContentAction,
    ImageAnalysisAction,
    ParseAudioAction,
    TavilySearchAction,
    VideoAnalysisAction,
)

TOOL_FACTORIES: Dict[str, Callable] = {
    "TavilySearchAction": TavilySearchAction,
    "ExtractUrlContentAction": ExtractUrlContentAction,
    "ImageAnalysisAction": ImageAnalysisAction,
    "ParseAudioAction": ParseAudioAction,
    "VideoAnalysisAction": VideoAnalysisAction,
    "ExecuteCodeAction": ExecuteCodeAction,
}
DEFAULT_TOOLS = [
    "TavilySearchAction",
    "ExtractUrlContentAction",
    "ImageAnalysisAction",
    "ParseAudioAction",
    "VideoAnalysisAction",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate IFlow Agent on a task suite")
    parser.add_argument("suite", help="JSONL, JSON, or YAML evaluation suite")
    parser.add_argument("--output", default="iflow_agent2/evaluation/results")
    parser.add_argument("--suite-name", default="iflow-evaluation")
    parser.add_argument("--main-model", default="mimo-pro")
    parser.add_argument("--sub-models", default="mimo")
    parser.add_argument("--tools", default=",".join(DEFAULT_TOOLS))
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--subagent-max-steps", type=int, default=20)
    return parser


async def run_from_args(args: argparse.Namespace) -> int:
    cases = load_cases(args.suite)
    selected_names = [item.strip() for item in args.tools.split(",") if item.strip()]
    unknown = sorted(set(selected_names) - set(TOOL_FACTORIES))
    if unknown:
        raise ValueError(f"unknown tools: {unknown}; available: {sorted(TOOL_FACTORIES)}")
    sub_models = [item.strip() for item in args.sub_models.split(",") if item.strip()]

    def create_agent(case):
        names = case.allowed_tools if case.allowed_tools is not None else selected_names
        disallowed = sorted(set(names) - set(selected_names))
        if disallowed:
            raise ValueError(
                f"case {case.id} requests tools not enabled by CLI: {disallowed}"
            )
        tools = [TOOL_FACTORIES[name]() for name in names]
        return GeneralAgent.from_model_names(
            main_model=args.main_model,
            sub_models=sub_models,
            tools=tools,
            max_attempts=args.max_attempts,
            subagent_max_steps=args.subagent_max_steps,
        )

    judge = (
        LLMSemanticJudge.from_model_name(args.judge_model)
        if args.judge_model
        else None
    )
    runner = EvaluationRunner(
        create_agent,
        scorers=ScorerRegistry(semantic_judge=judge),
        max_concurrency=args.max_concurrency,
        default_timeout_seconds=args.timeout,
        suite_name=args.suite_name,
    )
    report = await runner.run(cases, repeats=args.repeats)
    paths = report.save(Path(args.output))
    print(json.dumps(report.summary(), ensure_ascii=False, indent=2))
    print(f"Reports: {', '.join(f'{key}={value}' for key, value in paths.items())}")
    return 0 if report.summary()["pass_rate"] == 1.0 else 1


def main() -> int:
    return asyncio.run(run_from_args(build_parser().parse_args()))
