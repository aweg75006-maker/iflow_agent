from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, Dict

import yaml

from iflow_agent2.base.agent.base_action import BaseAction


class TavilySearchAction(BaseAction):
    """Web search backed by Tavily's native API."""

    name: str = "TavilySearchAction"
    description: str = (
        "Search the live web with Tavily and return titles, URLs, and relevant snippets."
    )
    parameters: Dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 10,
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "basic",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    @staticmethod
    def _load_api_key() -> str:
        config_path = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
        try:
            with open(config_path, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file) or {}
            key = config.get("tools", {}).get("tavily", {}).get("api_key", "")
            if key:
                return str(key)
        except (OSError, yaml.YAMLError, AttributeError):
            pass
        return os.getenv("TAVILY_API_KEY", "")

    async def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        query = str(kwargs.get("query", "")).strip()
        if not query:
            return {"success": False, "output": None, "error": "query is required", "metrics": {}}

        api_key = self._load_api_key()
        if not api_key:
            return {
                "success": False,
                "output": None,
                "error": "Tavily API key is not configured in model_config.yaml",
                "metrics": {},
            }

        try:
            from tavily import TavilyClient
        except ImportError:
            return {
                "success": False,
                "output": None,
                "error": "tavily-python is not installed",
                "metrics": {},
            }

        max_results = max(1, min(int(kwargs.get("max_results", 5)), 10))
        search_depth = str(kwargs.get("search_depth", "basic"))
        client = TavilyClient(api_key=api_key)
        try:
            response = await asyncio.to_thread(
                client.search,
                query,
                max_results=max_results,
                search_depth=search_depth,
            )
        except Exception as exc:
            return {
                "success": False,
                "output": None,
                "error": f"Tavily search failed: {exc}",
                "metrics": {},
            }

        results = [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", ""),
                "score": item.get("score"),
            }
            for item in response.get("results", [])
        ]
        return {
            "success": True,
            "output": results,
            "error": None,
            "metrics": {"result_count": len(results), "provider": "tavily"},
        }
