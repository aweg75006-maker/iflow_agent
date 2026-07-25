"""Tool used by MainAgent to return the final answer."""
from __future__ import annotations

from typing import Any, Dict

from pydantic import Field

from iflow_agent2.base.agent.base_action import BaseAction
from iflow_agent2.base.engine.logs import logger


class CompleteTool(BaseAction):
    """Mark the current task as complete with a final answer."""
    
    name: str = "complete"
    description: str = "Mark the task as complete and provide the final answer"
    parameters: Dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "The final answer to the question"},
        },
        "required": ["answer"]
    })
    
    async def __call__(self, answer: str = "") -> Dict:
        """Execute complete action."""
        logger.info(f"[CompleteTool] Task completed with answer: {answer}")
        return {
            "success": True,
            "answer": answer,
            "done": True,
        }
