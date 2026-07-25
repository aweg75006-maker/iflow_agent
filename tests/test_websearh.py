"""Live Tavily check using the ignored local plaintext model_config.yaml."""
from __future__ import annotations

import asyncio
import os

import pytest

from iflow_agent2.tools.tavily_search import TavilySearchAction

pytestmark = pytest.mark.skipif(
    os.getenv("IFLOW_RUN_LIVE_TESTS") != "1",
    reason="set IFLOW_RUN_LIVE_TESTS=1 to run live web tests",
)


def test_tavily_search() -> None:
    result = asyncio.run(
        TavilySearchAction()(query="小米 MiMo 大模型", max_results=3)
    )
    assert result["success"] is True
    assert len(result["output"]) == 3
    assert all(item["url"].startswith("http") for item in result["output"])
