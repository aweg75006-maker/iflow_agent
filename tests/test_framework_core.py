from __future__ import annotations

from iflow_agent2.base.agent.base_action import BaseAction
from iflow_agent2.base.agent.memory import Memory
from iflow_agent2.master.main_agent import MainAgent
from iflow_agent2.runtime.runner import Runner
from iflow_agent2.tools import (
    ExtractUrlContentAction,
    GoogleSearchAction,
    ImageAnalysisAction,
    ParseAudioAction,
    TavilySearchAction,
    VideoAnalysisAction,
)


def test_framework_core_classes_are_exposed() -> None:
    assert BaseAction.__name__ == "BaseAction"
    assert Memory.__name__ == "Memory"
    assert MainAgent.__name__ == "MainAgent"
    assert Runner.__name__ == "Runner"


def test_multimodal_tools_are_importable() -> None:
    tools = [
        GoogleSearchAction(),
        ExtractUrlContentAction(),
        ImageAnalysisAction(),
        ParseAudioAction(),
        TavilySearchAction(),
        VideoAnalysisAction(),
    ]
    assert [tool.name for tool in tools] == [
        "GoogleSearchAction",
        "ExtractUrlContentAction",
        "ImageAnalysisAction",
        "ParseAudioAction",
        "TavilySearchAction",
        "VideoAnalysisAction",
    ]
