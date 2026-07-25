"""Live MiMo checks using the ignored local plaintext model_config.yaml."""
from __future__ import annotations

import asyncio
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml
from openai import OpenAI

from iflow_agent2.tools import ImageAnalysisAction, VideoAnalysisAction

pytestmark = pytest.mark.skipif(
    os.getenv("IFLOW_RUN_LIVE_TESTS") != "1",
    reason="set IFLOW_RUN_LIVE_TESTS=1 to run live MiMo tests",
)


def client() -> OpenAI:
    config_path = Path(__file__).resolve().parents[1] / "config" / "model_config.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_config = config["models"]["mimo-pro"]
    return OpenAI(
        api_key=model_config["api_key"],
        base_url=model_config["base_url"],
    )


def test_available_models_and_text_completion() -> None:
    model_ids = {item.id for item in client().models.list().data}
    assert {"mimo-v2.5", "mimo-v2.5-pro", "mimo-v2.5-asr"} <= model_ids

    response = client().chat.completions.create(
        model="mimo-v2.5-pro",
        messages=[{"role": "user", "content": "只回答数字：1+1等于几？"}],
        max_tokens=100,
    )
    assert "2" in (response.choices[0].message.content or "")


def test_mimo_image_direct_video_and_frame_fallback(tmp_path) -> None:
    image_path = tmp_path / "green.png"
    green = np.zeros((64, 64, 3), dtype=np.uint8)
    green[:, :] = (0, 255, 0)
    assert cv2.imwrite(str(image_path), green)

    video_path = tmp_path / "blue.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5,
        (160, 120),
    )
    assert writer.isOpened()
    blue = np.zeros((120, 160, 3), dtype=np.uint8)
    blue[:, :] = (255, 0, 0)
    for _ in range(10):
        writer.write(blue)
    writer.release()

    async def verify() -> None:
        image = await ImageAnalysisAction()(
            query="图片是什么颜色？只回答颜色。",
            image_path=str(image_path),
        )
        assert image["success"] is True
        assert any(word in image["output"].lower() for word in ("绿", "green"))

        direct = await VideoAnalysisAction()(
            query="视频主要画面是什么颜色？只回答颜色。",
            video_path=str(video_path),
            analyze_audio=False,
        )
        assert direct["success"] is True
        assert direct["metrics"]["direct_video_analyzed"] is True
        assert any(word in direct["output"].lower() for word in ("蓝", "blue"))

        frames = await VideoAnalysisAction()(
            query="这些视频帧主要是什么颜色？只回答颜色。",
            video_path=str(video_path),
            analyze_audio=False,
            direct_video=False,
            max_frames=4,
        )
        assert frames["success"] is True
        assert frames["metrics"]["direct_video_analyzed"] is False
        assert frames["metrics"]["frames_analyzed"] > 0
        assert any(word in frames["output"].lower() for word in ("蓝", "blue"))

    asyncio.run(verify())
