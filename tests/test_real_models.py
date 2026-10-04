"""Opt-in acceptance tests; never download models or fixtures.

VIDEO_ANALYZER_MODEL_TEST_VIDEO=/absolute/path/two-speakers.mp4 uv run pytest -m models
The supplied fixture must have speech and visible content. Tests intentionally use
an empty general HF cache, exercising only the prepared model manifest/directory.
"""

import os
import wave
from itertools import pairwise
from pathlib import Path

import pytest

from video_analyzer import AnalysisConfig, analyze_video
from video_analyzer.models import local_models, resolve_profile
from video_analyzer.report import render_markdown
from video_analyzer.timeline import join_text
from video_analyzer.worker_client import Worker

pytestmark = pytest.mark.models

DIARIZERS = ("pyannote", "sortformer")


def selected_diarizers():
    selected = os.environ.get("VIDEO_ANALYZER_TEST_DIARIZATION", "pyannote")
    return DIARIZERS if selected == "both" else (selected,)


@pytest.mark.parametrize("asr", ["whisper", "qwen"])
@pytest.mark.parametrize("diarizer", DIARIZERS)
def test_real_offline_pipeline(asr, diarizer, monkeypatch, tmp_path):
    fixture = os.environ.get("VIDEO_ANALYZER_MODEL_TEST_VIDEO")
    if not fixture:
        pytest.skip("Set VIDEO_ANALYZER_MODEL_TEST_VIDEO to enable real-model tests")
    if diarizer not in selected_diarizers():
        pytest.skip("Set VIDEO_ANALYZER_TEST_DIARIZATION=sortformer or both")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = Path(__file__).resolve().parents[1]
    config = AnalysisConfig(
        asr=asr,
        diarization_backend=diarizer,
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
        f"{diarizer}/{asr}: {result.provenance['elapsed_seconds']:.2f}s; {len(result.speakers)} speakers; {join_text(result.transcript)}"
    )

    def unexpected_worker(*args, **kwargs):
        raise AssertionError("A complete resume must not load a worker")

    with monkeypatch.context() as patch:
        patch.setattr("video_analyzer.pipeline.Worker", unexpected_worker)
        resumed = analyze_video(Path(fixture), config.model_copy(update={"resume": True}))
    assert resumed.scenes == result.scenes
    assert resumed.transcript == result.transcript
    assert resumed.provenance["stage_metrics"] == result.provenance["stage_metrics"]


def test_real_sortformer_short_silent_and_streaming_audio(monkeypatch, tmp_path, capfd):
    """Small edge cases plus a >2-chunk recording, using only prepared assets."""
    fixture = os.environ.get("VIDEO_ANALYZER_MODEL_TEST_VIDEO")
    if not fixture or "sortformer" not in selected_diarizers():
        pytest.skip("Select Sortformer and VIDEO_ANALYZER_MODEL_TEST_VIDEO for real-model tests")
    from video_analyzer.media import extract_audio, probe

    monkeypatch.setenv("HF_HOME", str(tmp_path / "empty-hf"))
    monkeypatch.delenv("HF_TOKEN", raising=False)
    root = Path(__file__).resolve().parents[1]
    model = local_models(root / ".models", ["sortformer"])["sortformer"]
    audio = tmp_path / "fixture.wav"
    extract_audio(probe(Path(fixture)), audio)
    with wave.open(str(audio), "rb") as source:
        samples = source.readframes(source.getnframes())
        assert source.getnchannels() == 1 and source.getframerate() == 16000
    assert samples
    # Enough audio to exercise cache/FIFO reuse across at least three 27.2s steps.
    long_samples = (samples * (int(65 * 32000 / len(samples)) + 1))[: 65 * 32000]
    device = (
        "cuda"
        if resolve_profile(os.environ.get("VIDEO_ANALYZER_TEST_PROFILE", "auto")) == "cuda"
        else "cpu"
    )
    with Worker(root / ".runtime", "sortformer") as worker:
        runtime = worker.request("load", kind="sortformer", model=model["directory"], device=device)
        assert runtime["diarization_backend"] == "sortformer"
        for name, pcm in (
            ("empty", b""),
            ("short", b"\0\0" * 160),
            ("silence", b"\0\0" * 16000),
            ("long", long_samples),
        ):
            target = tmp_path / f"{name}.wav"
            with wave.open(str(target), "wb") as wav:
                wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                wav.writeframes(pcm)
            result = worker.request("diarize", audio=str(target))
            duration = len(pcm) / 32000
            assert all(
                0 <= turn["start"] < turn["end"] <= duration
                for turn in result["regular"] + result["exclusive"]
            )
            assert all(a["end"] <= b["start"] for a, b in pairwise(result["exclusive"]))
            if name in ("empty", "short", "silence"):
                assert result["regular"] == result["exclusive"] == []
            else:
                assert result["regular"]
    assert "Sortformer streaming step 3" in capfd.readouterr().err


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
