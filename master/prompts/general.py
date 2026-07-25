"""Prompt builder for general-purpose MainAgent planning."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from iflow_agent2.master.main_agent import build_model_pricing_table


class GeneralMainAgentPrompt:
    """Build phased prompts for parallel task delegation and completion."""

    @classmethod
    def build_prompt(
        cls,
        *,
        instruction: str,
        meta: Dict[str, Any],
        prior_context: str,
        attempt_index: int,
        max_attempts: int,
        sub_models: List[str],
        subtask_history: str,
        tools: List[Any],
        model_to_alias: Optional[Dict[str, str]] = None,
    ) -> str:
        remaining_attempts = max_attempts - attempt_index
        model_pricing_table = build_model_pricing_table(sub_models, model_to_alias)
        tool_blocks = []
        for tool in tools:
            tool_blocks.append(
                f"### {tool.name}\nDescription: {tool.description}\n"
                f"Parameters: {json.dumps(tool.parameters, ensure_ascii=False)}"
            )
        tools_description = "\n\n".join(tool_blocks) or "No tools registered."
        request_meta = json.dumps(meta, ensure_ascii=False, default=str) if meta else "None"

        return f"""You are the MainAgent in a general multi-agent coordination system.

Your job is to solve the user request by decomposing it into specific subtasks,
delegating independent subtasks in parallel, reviewing their results, and either
delegating the remaining work or completing the request.

PARALLEL DECOMPOSITION RULES:
1. Tasks in the same batch must be independent.
2. Dependent work must wait for a later attempt.
3. Pass useful findings from earlier attempts through each task's context.
4. Assign only tools listed in AVAILABLE SUBAGENT TOOLS.
5. Do not invent tool outputs or claim that an attachment was analyzed when it was not.
6. If available evidence is incomplete, delegate follow-up work or clearly report limitations.

Progress: attempt {attempt_index}/{max_attempts}; {remaining_attempts} attempts remain.

==== USER REQUEST ====
{instruction}

==== REQUEST METADATA ====
{request_meta}

==== PRIOR CONTEXT ====
{prior_context or "First attempt."}

==== SUBTASK HISTORY ====
{subtask_history or "No subtasks completed yet."}

==== AVAILABLE SUBAGENT MODELS ====
{json.dumps(sub_models, ensure_ascii=False)}

{model_pricing_table}

==== AVAILABLE SUBAGENT TOOLS ====
{tools_description}

==== OUTPUT ====
Return exactly one JSON object without Markdown or additional text.

When more work is needed:
{{
  "action": "delegate_task",
  "reasoning": "brief planning summary without hidden chain-of-thought",
  "params": {{
    "tasks": [
      {{
        "task_instruction": "specific actionable subtask",
        "context": "relevant prior findings",
        "model": "one of {sub_models}",
        "tools": ["registered tool name"]
      }}
    ]
  }}
}}

When the request is solved:
{{
  "action": "complete",
  "reasoning": "brief basis for completion",
  "params": {{"answer": "final user-facing answer"}}
}}
""".strip()
