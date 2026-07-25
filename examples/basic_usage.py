from __future__ import annotations

import asyncio
from typing import Any, Dict

from iflow_agent2 import BaseAction, GeneralAgent


class AddAction(BaseAction):
    name: str = "AddAction"
    description: str = "Add two numbers."
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    async def __call__(self, a: float = 0, b: float = 0, **kwargs: Any):
        return {"success": True, "output": a + b}


async def main() -> None:
    agent = GeneralAgent.from_model_names(
        main_model="mimo-pro",
        sub_models=["mimo"],
        tools=[AddAction()],
        max_attempts=5,
        subagent_max_steps=8,
    )
    result = await agent.run("Calculate 19 + 23 and return the result.")
    print(result.answer)


if __name__ == "__main__":
    asyncio.run(main())
