from pathlib import Path

import pytest

import video_analyzer.pipeline as pipeline
from video_analyzer.cli import main
from video_analyzer.errors import AnalysisError
from video_analyzer.report import render_json, render_markdown
from video_analyzer.schema import AnalysisConfig, AnalysisResult


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setattr(
        pipeline,
        "local_models",
        lambda root, keys: {
            key: {
                "repo_id": key,
                "revision": "abc123",
                "quantization": None,
                "directory": str(root / key),
                "files": {"model": {"sha256": "fake", "size": 1}},
            }
            for key in keys
        },
    )
    monkeypatch.setattr(pipeline, "worker_python", lambda *_: Path("unused"))

    class Worker:
        descriptions = 0
        fail_at = None
        operations = []
        transcriptions = 0
        fail_transcribe_at = None
        turns = []

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def request(self, operation, **payload):
            self.operations.append(operation)
            if operation == "load":
                return {"device": "fixture"}
            if operation == "describe":
                Worker.descriptions += 1
                if Worker.descriptions == Worker.fail_at:
                    raise AnalysisError("Simulated interrupted worker")
                assert all(Path(frame).is_file() for frame in payload["frames"])
                return {
                    "description": "赤または青の画面です。",
                    "elapsed_seconds": 0,
                    "peak_rss_bytes": 0,
                    "peak_device_bytes": 0,
                }
            if operation == "diarize":
                return {"regular": Worker.turns, "exclusive": Worker.turns}
            if operation == "transcribe":
                Worker.transcriptions += 1
                if Worker.transcriptions == Worker.fail_transcribe_at:
                    raise AnalysisError("Simulated interrupted ASR chunk")
                return {
                    "units": [{"start": 5, "end": 6, "text": f"word{payload['start'] + 5}"}],
                    "device": "fixture",
                }
            raise AssertionError(operation)

    monkeypatch.setattr(pipeline, "Worker", Worker)
    return Worker


@pytest.mark.integration
def test_no_audio_reports_and_resume(video, tmp_path, fake_models):
    config = AnalysisConfig(profile="cpu", work_dir=tmp_path / "work")
    result = pipeline.analyze_video(video, config)
    assert len(result.scenes) == 2
    assert not result.transcript and not result.speakers
    assert "音声トラック" in result.warnings[0]
    assert "動画解析レポート" in render_markdown(result)
    restored = AnalysisResult.model_validate_json(render_json(result))
    assert restored == result
    assert fake_models.descriptions == 2
    result2 = pipeline.analyze_video(video, config.model_copy(update={"resume": True}))
    assert result2.scenes == result.scenes
    assert fake_models.descriptions == 2
    pipeline.analyze_video(
        video, config.model_copy(update={"resume": True, "report_language": "en"})
    )
    assert fake_models.descriptions == 4


@pytest.mark.integration
def test_failed_window_resumes_only_incomplete_work(video, tmp_path, fake_models):
    config = AnalysisConfig(profile="cpu", work_dir=tmp_path / "work")
    fake_models.fail_at = 2
    with pytest.raises(AnalysisError, match="interrupted"):
        pipeline.analyze_video(video, config)
    assert fake_models.descriptions == 2
    fake_models.fail_at = None
    result = pipeline.analyze_video(video, config.model_copy(update={"resume": True}))
    assert len(result.scenes) == 2
    assert fake_models.descriptions == 3


def test_cli_does_not_replace_report_on_failure(monkeypatch, tmp_path):
    import video_analyzer.cli as cli

    output = tmp_path / "report.md"
    output.write_text("old report")

    def fail(*_):
        raise AnalysisError("failed")

    monkeypatch.setattr(cli, "analyze_video", fail)
    assert main(["analyze", "input.mp4", "-o", str(output)]) == 1
    assert output.read_text() == "old report"


def test_cli_output_must_not_overwrite_video(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"original")
    assert main(["analyze", str(video), "-o", str(video)]) == 1
    assert video.read_bytes() == b"original"


@pytest.mark.integration
def test_silent_audio_skips_asr_but_keeps_visual_report(tmp_path, ffmpeg, fake_models):
    path = tmp_path / "silence.mp4"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=red:s=160x120:r=10:d=2",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=16000:cl=mono",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-t",
        "2",
        path,
    )
    result = pipeline.analyze_video(path, AnalysisConfig(profile="cpu", work_dir=tmp_path / "work"))
    assert result.media.audio_stream_index is not None
    assert result.transcript == [] and result.speakers == []
    assert "diarize" in fake_models.operations and "transcribe" not in fake_models.operations
    assert result.scenes[0].observations
    assert "発話は検出されません" in result.warnings[0]


@pytest.mark.integration
def test_interrupted_asr_reuses_completed_chunks(tmp_path, ffmpeg, fake_models):
    path = tmp_path / "long.mp4"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=red:s=160x120:r=10:d=55",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=16000:duration=55",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        path,
    )
    config = AnalysisConfig(profile="cpu", work_dir=tmp_path / "work")
    fake_models.turns = [{"start": 0, "end": 55, "speaker_id": "A"}]
    fake_models.fail_transcribe_at = 2
    with pytest.raises(AnalysisError, match="interrupted ASR"):
        pipeline.analyze_video(path, config)
    fake_models.fail_transcribe_at = None
    result = pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert fake_models.transcriptions == 4  # first chunk is recovered, two remain
    assert fake_models.operations.count("diarize") == 1
    assert [unit.start for unit in result.transcript] == [5, 20, 40]
    assert len(result.scenes) == 1 and len(result.scenes[0].observations) == 2
