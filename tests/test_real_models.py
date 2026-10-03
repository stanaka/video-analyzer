"""Opt-in acceptance tests; never download models or fixtures.

VIDEO_ANALYZER_MODEL_TEST_VIDEO=/absolute/path/two-speakers.mp4 uv run pytest -m models
The supplied fixture must have speech and visible content. Tests intentionally use
an empty general HF cache, exercising only the prepared model manifest/directory.
"""

import os
from pathlib import Path

import pytest

from video_analyzer import AnalysisConfig, analyze_video
from video_analyzer.models import local_models, resolve_profile
from video_analyzer.report import render_markdown
from video_analyzer.timeline import join_text
from video_analyzer.worker_client import Worker

pytestmark = pytest.mark.models


@pytest.mark.parametrize("asr", ["whisper", "qwen"])
def test_real_offline_pipeline(asr, monkeypatch, tmp_path):
    fixture = os.environ.get("VIDEO_ANALYZER_MODEL_TEST_VIDEO")
    if not fixture:
        pytest.skip("Set VIDEO_ANALYZER_MODEL_TEST_VIDEO to enable real-model tests")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = Path(__file__).resolve().parents[1]
    config = AnalysisConfig(
        asr=asr,
        model_dir=root / ".models",
        runtime_dir=root / ".runtime",
        work_dir=tmp_path / "work",
        report_language="ja",
        profile=os.environ.get("VIDEO_ANALYZER_TEST_PROFILE", "auto"),
    )
    result = analyze_video(Path(fixture), config)
    assert result.transcript and result.speakers
    assert result.scenes and all(scene.observations for scene in result.scenes)
    assert all(0 <= unit.start <= unit.end <= result.media.duration for unit in result.transcript)
    assert all(
        any("\u3040" <= char <= "\u9fff" for char in observation.description)
        for scene in result.scenes
        for observation in scene.observations
    )
    assert "動画解析レポート" in render_markdown(result)
    (tmp_path / f"{asr}-result.json").write_text(result.model_dump_json(indent=2), encoding="utf-8")
    (tmp_path / f"{asr}-result.md").write_text(render_markdown(result), encoding="utf-8")
    expected_languages = os.environ.get("VIDEO_ANALYZER_TEST_LANGUAGES", "")
    detected = {unit.language.lower() for unit in result.transcript if unit.language}
    assert set(filter(None, expected_languages.lower().split(","))) <= detected
    print(
        f"{asr}: {result.provenance['elapsed_seconds']:.2f}s; {len(result.speakers)} speakers; {join_text(result.transcript)}"
    )

    def unexpected_worker(*args, **kwargs):
        raise AssertionError("A complete resume must not load a worker")

    with monkeypatch.context() as patch:
        patch.setattr("video_analyzer.pipeline.Worker", unexpected_worker)
        resumed = analyze_video(Path(fixture), config.model_copy(update={"resume": True}))
    assert resumed.scenes == result.scenes
    assert resumed.transcript == result.transcript
    assert resumed.provenance["stage_metrics"] == result.provenance["stage_metrics"]


def test_molmo_maximum_frame_budget(monkeypatch, tmp_path):
    if not os.environ.get("VIDEO_ANALYZER_MODEL_TEST_VIDEO"):
        pytest.skip("Set VIDEO_ANALYZER_MODEL_TEST_VIDEO to enable real-model tests")
    import json

    from PIL import Image, ImageDraw

    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = Path(__file__).resolve().parents[1]
    profile = resolve_profile(os.environ.get("VIDEO_ANALYZER_TEST_PROFILE", "auto"))
    key = "molmo-mlx" if profile == "mac" else "molmo-torch"
    model = local_models(root / ".models", [key])[key]
    frames = []
    for i in range(8):
        frame = Image.new("RGB", (768, 576), "white")
        ImageDraw.Draw(frame).rectangle((30 + i * 75, 200, 150 + i * 75, 320), fill="red")
        path = tmp_path / f"frame-{i}.jpg"
        frame.save(path)
        frames.append(str(path))
    with Worker(root / ".runtime", "mlx" if profile == "mac" else "torch") as worker:
        worker.request("load", kind=key, model=model["directory"], device="auto")
        result = worker.request(
            "describe",
            frames=frames,
            timestamps=[i * 4 + 1 for i in range(8)],
            language="ja",
            max_new_tokens=448,
        )
    assert "赤" in result["description"]
    assert result["peak_device_bytes"] > 0
    (tmp_path / "molmo-budget.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))
