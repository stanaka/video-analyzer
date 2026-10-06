import itertools
import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

import video_analyzer.pipeline as pipeline
from video_analyzer import LLMConfig, llm
from video_analyzer.cli import main, parser
from video_analyzer.errors import AnalysisError
from video_analyzer.report import render_json, render_markdown
from video_analyzer.schema import AnalysisConfig, AnalysisResult, MediaInfo, Scene


def test_llm_api_and_local_cache_independent_of_polishing_settings(
    simulated_media, fake_models, tmp_path, monkeypatch
):
    path, _, calls = simulated_media
    fake_models.turns = [{"start": 0, "end": 60, "speaker_id": "A"}]
    config = AnalysisConfig(profile="cpu", work_dir=tmp_path / "work", resume=True)
    raw = pipeline.analyze_video(path, config)
    previous = fake_models.requests.copy()
    extraction = calls.copy()
    llm_calls = []

    def complete(self, request_messages):
        llm_calls.append(request_messages)
        payload = json.loads(request_messages[1]["content"])
        if "core" in payload:
            return {
                "corrections": [
                    {"id": unit["id"], "text": unit["text"] + "!", "speaker_id": unit["speaker_id"]}
                    for unit in payload["core"]
                ]
            }
        return {"description": "Merged visual description"}

    monkeypatch.setattr(llm.LLMClient, "complete", complete)
    polished_config = config.model_copy(
        update={
            "llm_enabled": True,
            "llm": LLMConfig(base_url="http://localhost:8000/v1", model="test-model"),
        }
    )
    result = pipeline.analyze_video(path, polished_config)
    assert result.transcript[0].text == raw.transcript[0].text + "!"
    assert all(scene.description == "Merged visual description" for scene in result.scenes)
    assert fake_models.requests == previous and calls == extraction
    count = len(llm_calls)
    pipeline.analyze_video(path, polished_config)
    assert len(llm_calls) == count
    for change in (
        {"model": "another-model"},
        {"max_input_chars": 12000},
        {"transcript_overlap_units": 0},
    ):
        updated = polished_config.model_copy(
            update={"llm": polished_config.llm.model_copy(update=change)}
        )
        pipeline.analyze_video(path, updated)
        assert len(llm_calls) > count
        count = len(llm_calls)
        assert fake_models.requests == previous and calls == extraction


def test_llm_api_failure_resumes_without_local_workers(
    simulated_media, fake_models, tmp_path, monkeypatch
):
    path, _, _ = simulated_media
    config = AnalysisConfig(
        profile="cpu",
        work_dir=tmp_path / "work",
        resume=True,
        llm_enabled=True,
        llm={"base_url": "http://localhost:8000/v1", "model": "test-model"},
    )
    accepted = []

    def complete(self, request_messages):
        if accepted:
            raise AnalysisError("endpoint offline")
        accepted.append(request_messages)
        return {"description": "Merged description"}

    with monkeypatch.context() as patch:
        patch.setattr(llm.LLMClient, "complete", complete)
        with pytest.raises(AnalysisError, match="endpoint offline"):
            pipeline.analyze_video(path, config)
    local_requests = fake_models.requests.copy()
    resumed = []

    def recover(self, request_messages):
        resumed.append(request_messages)
        return {"description": "Merged description"}

    monkeypatch.setattr(llm.LLMClient, "complete", recover)
    result = pipeline.analyze_video(path, config)
    assert len(resumed) == 1
    assert all(scene.description == "Merged description" for scene in result.scenes)
    assert fake_models.requests == local_requests


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
        requests = []
        transcription_units = None

        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def request(self, operation, **payload):
            self.operations.append(operation)
            self.requests.append((operation, payload))
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
                    "units": (
                        Worker.transcription_units
                        if Worker.transcription_units is not None
                        else [{"start": 5, "end": 6, "text": f"word{payload['start'] + 5}"}]
                    ),
                    "device": "fixture",
                }
            raise AssertionError(operation)

    monkeypatch.setattr(pipeline, "Worker", Worker)
    return Worker


@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron"])
@pytest.mark.parametrize("diarization", [False, True])
@pytest.mark.parametrize("asr", ["whisper", "qwen"])
def test_selected_range_uses_local_audio_and_original_report_times(
    simulated_media, fake_models, tmp_path, monkeypatch, backend, diarization, asr
):
    path, media, calls = simulated_media
    fake_models.turns = [{"start": -1, "end": 100, "speaker_id": "A"}]
    audio_options = []
    original = pipeline.extract_audio

    def extract(info, target, **options):
        audio_options.append(options)
        return original(info, target, **options)

    monkeypatch.setattr(pipeline, "extract_audio", extract)
    config = AnalysisConfig(
        profile="cpu",
        work_dir=tmp_path / "work",
        start_time="00:10",
        end_time="00:40",
        diarization=diarization,
        diarization_backend=backend,
        asr=asr,
    )
    result = pipeline.analyze_video(path, config)
    assert result.media.duration == media.duration == 55
    assert result.provenance["analysis_range"] == {"start": 10, "end": 40}
    assert audio_options[0]["start_time"] == 10 and audio_options[0]["end_time"] == 40
    assert [(scene.start, scene.end) for scene in result.scenes] == [(10, 25), (25, 40)]
    requests = [payload for op, payload in fake_models.requests if op == "transcribe"]
    assert [(request["start"], request["end"]) for request in requests] == [(0, 25), (15, 30)]
    assert [unit.start for unit in result.transcript] == [15, 30]
    assert all(
        10 <= span.start <= span.end <= 40 for span in result.transcript + result.diarization
    )
    if diarization:
        assert [(turn.start, turn.end) for turn in result.diarization] == [(10, 40)]
    assert "00:00:10.000 – 00:00:40.000" in render_markdown(result)
    assert AnalysisResult.model_validate_json(render_json(result)) == result
    previous = list(fake_models.requests)
    equivalent = AnalysisConfig(
        **{**config.model_dump(), "start_time": 10, "end_time": 40, "resume": True}
    )
    pipeline.analyze_video(path, equivalent)
    assert fake_models.requests == previous and calls["audio"] == 1
    pipeline.analyze_video(path, equivalent.model_copy(update={"start_time": 20}))
    assert calls["audio"] == 2


def test_interrupted_segment_resumes_with_original_timestamps(
    simulated_media, fake_models, tmp_path
):
    path, _, calls = simulated_media
    config = AnalysisConfig(
        profile="cpu",
        work_dir=tmp_path / "work",
        start_time=10,
        end_time=40,
        diarization=False,
        resume=True,
    )
    fake_models.fail_transcribe_at = 2
    with pytest.raises(AnalysisError, match="interrupted ASR"):
        pipeline.analyze_video(path, config)
    fake_models.fail_transcribe_at = None
    result = pipeline.analyze_video(path, config)
    assert fake_models.transcriptions == 3 and calls["audio"] == 1 and calls["scenes"] == 1
    assert [unit.start for unit in result.transcript] == [15, 30]


@pytest.mark.parametrize("bounds", [{"start_time": 55}, {"start_time": 56}, {"end_time": 56}])
def test_range_duration_validation_precedes_models_and_checkpoints(
    simulated_media, monkeypatch, tmp_path, bounds
):
    path, _, calls = simulated_media

    def unexpected(*args, **kwargs):
        pytest.fail("Models must not be checked for an invalid range")

    monkeypatch.setattr(pipeline, "local_models", unexpected)
    work = tmp_path / "work"
    with pytest.raises(AnalysisError, match="video duration"):
        pipeline.analyze_video(path, AnalysisConfig(profile="cpu", work_dir=work, **bounds))
    assert not work.exists() and calls == {"scenes": 0, "audio": 0, "frames": 0}


@pytest.mark.integration
@pytest.mark.parametrize(
    "bounds,expected",
    [
        ({"start_time": 1}, (1, 4)),
        ({"end_time": 3}, (0, 3)),
        ({"start_time": 1, "end_time": 3}, (1, 3)),
    ],
)
def test_selected_range_without_audio(video, tmp_path, fake_models, bounds, expected):
    result = pipeline.analyze_video(
        video, AnalysisConfig(profile="cpu", work_dir=tmp_path / "work", **bounds)
    )
    assert result.scenes[0].start == expected[0] and result.scenes[-1].end == expected[1]
    assert result.media.duration == 4
    assert not result.transcript and not result.diarization
    assert all(
        expected[0] <= time < expected[1]
        for scene in result.scenes
        for obs in scene.observations
        for time in obs.frame_timestamps
    )


@pytest.mark.integration
def test_cli_segment_html_without_models(video, tmp_path):
    output = tmp_path / "range.html"
    assert (
        main(
            [
                "analyze",
                str(video),
                "--profile",
                "cpu",
                "--start-time",
                "00:01",
                "--end-time",
                "00:00:03",
                "--no-transcription",
                "--no-diarization",
                "--no-scene-description",
                "--format",
                "html",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    result = AnalysisResult.model_validate_json(output.with_suffix(".json").read_text())
    assert result.provenance["analysis_range"] == {"start": 1, "end": 3}
    assert result.media.duration == 4
    assert [(scene.start, scene.end) for scene in result.scenes] == [(1, 2), (2, 3)]
    assert result.provenance["models"] == {} and output.is_file()


@pytest.fixture
def simulated_media(monkeypatch, tmp_path):
    path = tmp_path / "input.mp4"
    path.write_bytes(b"fixture video")
    media = MediaInfo(
        path=str(path),
        duration=55,
        width=160,
        height=120,
        video_stream_index=0,
        video_start=0,
        audio_stream_index=1,
        audio_start=0,
    )
    calls = {"scenes": 0, "audio": 0, "frames": 0}

    def scenes(*_, start_time=0, end_time=55):
        calls["scenes"] += 1
        spans = [(max(start_time, start), min(end_time, end)) for start, end in [(0, 25), (25, 55)]]
        return [
            Scene(id=f"scene_{i + 1:04d}", start=start, end=end)
            for i, (start, end) in enumerate(span for span in spans if span[1] > span[0])
        ]

    def audio(_, target, **options):
        calls["audio"] += 1
        target.write_bytes(b"fixture audio")

    def frames(_, targets, directory, edge, **options):
        calls["frames"] += 1
        assert all(
            options.get("start_time", 0) <= time < options.get("end_time", 55) for time in targets
        )
        directory.mkdir(parents=True)
        frame = directory / "frame.jpg"
        frame.write_bytes(b"fixture frame")
        return [str(frame)], targets[:1]

    monkeypatch.setattr(pipeline, "probe", lambda *_: media)
    monkeypatch.setattr(pipeline, "detect_scenes", scenes)
    monkeypatch.setattr(pipeline, "extract_audio", audio)
    monkeypatch.setattr(pipeline, "extract_frames", frames)
    return path, media, calls


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
@pytest.mark.parametrize("diarization", [False, True])
def test_interrupted_asr_reuses_completed_chunks(tmp_path, ffmpeg, fake_models, diarization):
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
    config = AnalysisConfig(profile="cpu", diarization=diarization, work_dir=tmp_path / "work")
    fake_models.turns = [{"start": 0, "end": 55, "speaker_id": "A"}]
    fake_models.fail_transcribe_at = 2
    with pytest.raises(AnalysisError, match="interrupted ASR"):
        pipeline.analyze_video(path, config)
    fake_models.fail_transcribe_at = None
    result = pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert fake_models.transcriptions == 4  # first chunk is recovered, two remain
    assert fake_models.operations.count("diarize") == int(diarization)
    assert [unit.start for unit in result.transcript] == [5, 20, 40]
    assert len(result.scenes) == 1 and len(result.scenes[0].observations) == 2


FEATURES = ("transcription", "diarization", "scene_description")
COMBINATIONS = list(itertools.product((False, True), repeat=3))


@pytest.mark.parametrize("enabled", COMBINATIONS)
@pytest.mark.parametrize("asr", ["whisper", "qwen"])
@pytest.mark.parametrize("profile", ["cpu", "mac", "cuda"])
@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron"])
def test_independent_features_and_resume(
    enabled, asr, profile, backend, simulated_media, fake_models, monkeypatch, tmp_path
):
    path, media, calls = simulated_media
    flags = dict(zip(FEATURES, enabled, strict=True))
    transcription, diarization, vision = enabled
    config = AnalysisConfig(
        **flags,
        asr=asr,
        profile=profile,
        diarization_backend=backend,
        report_language="en",
        work_dir=tmp_path / "work",
    )
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    checked_models, checked_workers = [], []
    local = pipeline.local_models

    def check_models(root, keys):
        checked_models.extend(keys)
        return local(root, keys)

    monkeypatch.setattr(pipeline, "local_models", check_models)
    monkeypatch.setattr(pipeline, "worker_python", lambda _, kind: checked_workers.append(kind))
    monkeypatch.setattr(pipeline, "resolve_profile", lambda _: profile)
    result = pipeline.analyze_video(path, config)
    expected_models = set()
    if vision:
        expected_models.add("molmo-mlx" if profile == "mac" else "molmo-torch")
    if diarization:
        expected_models.add(pipeline.DIARIZATION_MODELS[backend])
    if transcription:
        expected_models.add(asr)
        if asr == "qwen":
            expected_models.add("aligner")
    assert set(checked_models) == expected_models
    assert set(result.provenance["models"]) == expected_models
    expected_workers = set()
    if transcription or (diarization and backend == "pyannote") or (vision and profile != "mac"):
        expected_workers.add("torch")
    if diarization and backend != "pyannote":
        expected_workers.add(backend)
    if vision and profile == "mac":
        expected_workers.add("mlx")
    assert set(checked_workers) == expected_workers
    assert len(checked_workers) == len(expected_workers)
    assert calls == {"scenes": 1, "audio": int(transcription or diarization), "frames": 2 * vision}
    assert bool(result.transcript) == transcription
    assert bool(result.speakers) == diarization
    assert bool(result.diarization) == diarization
    assert bool(result.exclusive_diarization) == diarization
    assert all(bool(scene.observations) == vision for scene in result.scenes)
    assert len(result.scenes) == 2
    assert result.warnings == []
    expected_stages = set()
    if transcription:
        expected_stages.add("asr")
    if diarization:
        expected_stages.add("diarization")
    if vision:
        expected_stages.add("vision")
    assert set(result.provenance["runtimes"]) == expected_stages
    assert set(result.provenance["stage_metrics"]) == expected_stages
    assert all(result.provenance["configuration"][key] == value for key, value in flags.items())
    loads = [payload for operation, payload in fake_models.requests if operation == "load"]
    assert {payload["kind"] for payload in loads} == expected_models - {"aligner"}
    assert all(payload["device"] == ("auto" if profile == "mac" else profile) for payload in loads)
    if transcription and asr == "qwen":
        assert next(payload for payload in loads if payload["kind"] == "qwen")["aligner"]
    assert fake_models.operations.count("diarize") == int(diarization)
    assert fake_models.operations.count("describe") == 2 * vision
    assert fake_models.operations.count("transcribe") == 3 * transcription
    if transcription:
        assert [unit.start for unit in result.transcript] == [5, 20, 40]
        assert all(
            unit.speaker_id == ("speaker_01" if diarization else None) for unit in result.transcript
        )
        assert sum(len(scene.transcript_ids) for scene in result.scenes) == 3
    assert AnalysisResult.model_validate_json(render_json(result)) == result
    markdown = render_markdown(result)
    assert ("### Speech" in markdown) == transcription
    assert ("### Visual description" in markdown) == vision
    assert ("- Speakers:" in markdown) == diarization
    if not diarization:
        assert "Unknown speaker" not in markdown and "speaker_01" not in markdown
    if not transcription:
        assert "No speech detected." not in markdown
    previous_calls = calls.copy()
    previous_requests = fake_models.requests.copy()
    resumed = pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert calls == previous_calls and fake_models.requests == previous_requests
    assert resumed.scenes == result.scenes and resumed.transcript == result.transcript
    assert resumed.diarization == result.diarization
    assert resumed.provenance["stage_metrics"] == result.provenance["stage_metrics"]


@pytest.mark.parametrize("feature", FEATURES)
def test_feature_changes_separate_checkpoints(feature, simulated_media, fake_models, tmp_path):
    path, media, calls = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    config = AnalysisConfig(profile="cpu", work_dir=tmp_path / "work")
    original = pipeline.analyze_video(path, config)
    changed = pipeline.analyze_video(
        path, config.model_copy(update={feature: False, "resume": True})
    )
    assert calls["scenes"] == 2 and calls["audio"] == 2
    assert fake_models.operations.count("diarize") == (1 if feature == "diarization" else 2)
    assert fake_models.operations.count("transcribe") == (3 if feature == "transcription" else 6)
    assert fake_models.operations.count("describe") == (2 if feature == "scene_description" else 4)
    assert changed.provenance["configuration"][feature] is False
    requests = fake_models.requests.copy()
    restored = pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert restored.scenes == original.scenes and restored.transcript == original.transcript
    assert fake_models.requests == requests


@pytest.mark.parametrize("change", ["backend", "revision"])
@pytest.mark.parametrize("backend", ["sortformer", "nemotron"])
def test_diarization_identity_separates_transcription_checkpoints(
    change, backend, simulated_media, fake_models, monkeypatch, tmp_path
):
    path, media, _ = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    config = AnalysisConfig(
        profile="cpu",
        diarization_backend=backend,
        scene_description=False,
        work_dir=tmp_path / "work",
    )
    pipeline.analyze_video(path, config)
    local = pipeline.local_models

    def changed_models(root, keys):
        models = local(root, keys)
        models[backend]["revision"] = "different-checkpoint"
        return models

    with monkeypatch.context() as patch:
        changed = config.model_copy(update={"resume": True})
        if change == "backend":
            changed.diarization_backend = "pyannote"
        else:
            patch.setattr(pipeline, "local_models", changed_models)
        pipeline.analyze_video(path, changed)
    assert fake_models.operations.count("diarize") == 2
    assert fake_models.operations.count("transcribe") == 6
    requests = fake_models.requests.copy()
    pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert fake_models.requests == requests


@pytest.mark.parametrize(
    "updates", [{"runtime_revision": "changed-source"}, {"speaker_cache": 272}]
)
def test_nemotron_runtime_and_settings_invalidate_checkpoints(
    updates, simulated_media, fake_models, monkeypatch, tmp_path
):
    path, media, _ = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "speaker_7"}]
    config = AnalysisConfig(
        profile="cpu",
        diarization_backend="nemotron",
        scene_description=False,
        work_dir=tmp_path / "work",
    )
    first = pipeline.analyze_video(path, config)
    spec = pipeline.NEMO_DIARIZERS["nemotron"]
    with monkeypatch.context() as patch:
        patch.setitem(pipeline.NEMO_DIARIZERS, "nemotron", replace(spec, **updates))
        pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert fake_models.operations.count("diarize") == 2
    assert fake_models.operations.count("transcribe") == 6
    requests = fake_models.requests.copy()
    resumed = pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert resumed.transcript == first.transcript
    assert fake_models.requests == requests


@pytest.mark.parametrize(
    "updates",
    [
        {"num_speakers": 2},
        {"min_speakers": 2},
        {"max_speakers": 4},
        {"pyannote": {"clustering": {"threshold": 0.65}}},
        {"pyannote": {"clustering": {"Fa": 0.1}}},
        {"pyannote": {"clustering": {"Fb": 1.2}}},
        {"pyannote": {"segmentation": {"min_duration_off": 0.1}}},
    ],
)
def test_pyannote_settings_invalidate_checkpoints(updates, simulated_media, fake_models, tmp_path):
    path, media, _ = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    config = AnalysisConfig(profile="cpu", scene_description=False, work_dir=tmp_path / "work")
    pipeline.analyze_video(path, config)
    asr_count = fake_models.operations.count("transcribe")
    updated = AnalysisConfig(**{**config.model_dump(), **updates, "resume": True})
    result = pipeline.analyze_video(path, updated)
    assert fake_models.operations.count("diarize") == 2
    assert fake_models.operations.count("transcribe") == 2 * asr_count
    for key in updates:
        assert result.provenance["configuration"][key] == updated.model_dump(mode="json")[key]
    requests = fake_models.requests.copy()
    pipeline.analyze_video(path, updated)
    pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert fake_models.requests == requests


@pytest.mark.parametrize(
    "backend,options,load_options,speaker_options",
    [
        ("pyannote", {}, {}, {}),
        ("sortformer", {}, {}, {}),
        ("nemotron", {}, {}, {}),
        (
            "pyannote",
            {"min_speakers": 2, "max_speakers": 4},
            {},
            {"min_speakers": 2, "max_speakers": 4},
        ),
        ("pyannote", {"num_speakers": 2, "min_speakers": 5, "max_speakers": 1}, {}, {}),
        (
            "pyannote",
            {"pyannote": {"clustering": {"Fa": 0.1}}},
            {"pyannote": {"clustering": {"Fa": 0.1}}},
            {},
        ),
    ],
)
def test_diarization_worker_settings(
    backend, options, load_options, speaker_options, simulated_media, fake_models, tmp_path
):
    path, _, _ = simulated_media
    config = AnalysisConfig(
        profile="cpu",
        diarization_backend=backend,
        transcription=False,
        scene_description=False,
        work_dir=tmp_path / "work",
        **options,
    )
    pipeline.analyze_video(path, config)
    [(load_operation, load), (operation, diarize)] = fake_models.requests
    assert load_operation == "load" and operation == "diarize"
    kind = pipeline.DIARIZATION_MODELS[backend]
    assert load == {
        "kind": kind,
        "model": str(config.model_dir / kind),
        "device": "cpu",
        **load_options,
    }
    assert {key: value for key, value in diarize.items() if key != "audio"} == {
        "num_speakers": config.num_speakers,
        **speaker_options,
    }


def test_equivalent_yaml_and_cli_reuse_checkpoints(simulated_media, fake_models, tmp_path, capsys):
    path, media, _ = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    work = tmp_path / "work"
    output = tmp_path / "report.json"
    settings = (
        f"profile: cpu\nscene_description: false\nresume: true\nwork_dir: {work}\n"
        "min_speakers: 2\nmax_speakers: 4\npyannote:\n  clustering: {threshold: 0.65}\n"
    )
    first = tmp_path / "first.yaml"
    second = tmp_path / "second.yaml"
    first.write_text(settings)
    second.write_text(settings)
    command = ["analyze", str(path), "--format", "json", "-o", str(output)]
    assert main([*command, "--config", str(first)]) == 0
    requests = fake_models.requests.copy()
    assert requests
    yaml_configuration = json.loads(output.read_text())["provenance"]["configuration"]
    assert main([*command, "--config", str(second), "--show-settings"]) == 0
    assert (
        main(
            [
                *command,
                "--profile",
                "cpu",
                "--no-scene-description",
                "--resume",
                "--work-dir",
                str(work),
                "--min-speakers",
                "2",
                "--max-speakers",
                "4",
                "--pyannote-clustering-threshold",
                "0.65",
            ]
        )
        == 0
    )
    assert fake_models.requests == requests
    assert json.loads(output.read_text())["provenance"]["configuration"] == yaml_configuration
    assert "show_settings" not in yaml_configuration
    terminal = capsys.readouterr()
    assert terminal.out == f"{output.resolve()}\n" * 3
    assert terminal.err.count("# Analysis settings\n") == 1


@pytest.mark.parametrize("enabled", COMBINATIONS)
@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron"])
def test_feature_selection_without_audio(
    enabled, backend, simulated_media, fake_models, tmp_path, monkeypatch
):
    path, media, calls = simulated_media
    media.audio_stream_index = None
    media.audio_start = None
    config = AnalysisConfig(
        **dict(zip(FEATURES, enabled, strict=True)),
        profile="cpu",
        work_dir=tmp_path / "work",
        diarization_backend=backend,
    )
    checked_workers = []
    monkeypatch.setattr(pipeline, "worker_python", lambda _, kind: checked_workers.append(kind))
    result = pipeline.analyze_video(path, config)
    assert calls["audio"] == 0
    assert "diarize" not in fake_models.operations and "transcribe" not in fake_models.operations
    assert not result.transcript and not result.speakers
    assert bool(result.warnings) == (enabled[0] or enabled[1])
    if result.warnings:
        assert "音声トラック" in result.warnings[0]
    assert set(result.provenance["models"]) == ({"molmo-torch"} if enabled[2] else set())
    assert checked_workers == (["torch"] if enabled[2] else [])


@pytest.mark.integration
def test_all_disabled_needs_no_models_or_workers(video, tmp_path, monkeypatch):
    def unexpected(*_, **__):
        pytest.fail("Disabled features must not check models, use workers, or extract audio/frames")

    for name in ("local_models", "worker_python", "Worker", "extract_audio", "extract_frames"):
        monkeypatch.setattr(pipeline, name, unexpected)
    result = pipeline.analyze_video(
        video,
        AnalysisConfig(
            transcription=False,
            diarization=False,
            scene_description=False,
            profile="cpu",
            model_dir=tmp_path / "missing-models",
            runtime_dir=tmp_path / "missing-runtime",
            work_dir=tmp_path / "work",
        ),
    )
    assert len(result.scenes) == 2 and result.warnings == []
    assert not result.provenance["models"] and not result.provenance["runtimes"]


@pytest.mark.parametrize("diarization,transcription", [(True, False), (True, True), (False, True)])
def test_empty_speech_warnings(diarization, transcription, simulated_media, fake_models, tmp_path):
    path, _, _ = simulated_media
    fake_models.transcription_units = []
    result = pipeline.analyze_video(
        path,
        AnalysisConfig(
            diarization=diarization,
            transcription=transcription,
            scene_description=False,
            profile="cpu",
            report_language="en",
            work_dir=tmp_path / "work",
        ),
    )
    assert result.transcript == []
    assert result.warnings == ["No speech was detected in the audio."]
    assert ("transcribe" in fake_models.operations) == (transcription and not diarization)


def test_mac_vision_device_check_only_for_enabled_vision(
    simulated_media, fake_models, monkeypatch, tmp_path
):
    path, _, _ = simulated_media
    monkeypatch.setattr(pipeline, "resolve_profile", lambda _: "mac")
    config = AnalysisConfig(
        profile="mac",
        vision_device="cpu",
        transcription=False,
        diarization=False,
        scene_description=False,
        work_dir=tmp_path / "work",
    )
    pipeline.analyze_video(path, config)
    with pytest.raises(AnalysisError, match="vision-device must be auto"):
        pipeline.analyze_video(path, config.model_copy(update={"scene_description": True}))


@pytest.mark.parametrize("language", ["ja", "en"])
def test_markdown_feature_summary_and_older_results(
    language, simulated_media, fake_models, tmp_path
):
    path, _, _ = simulated_media
    result = pipeline.analyze_video(
        path, AnalysisConfig(profile="cpu", report_language=language, work_dir=tmp_path / "work")
    )
    summary = "- 文字起こし: 有効" if language == "ja" else "- Transcription: Enabled"
    assert summary in render_markdown(result)
    legacy = result.model_copy(deep=True)
    for feature in FEATURES:
        del legacy.provenance["configuration"][feature]
    assert render_markdown(legacy) == render_markdown(result)
    result.provenance["configuration"].update(dict.fromkeys(FEATURES, False))
    markdown = render_markdown(result)
    assert ("- 文字起こし: 無効" if language == "ja" else "- Transcription: Disabled") in markdown
    assert "###" not in markdown
    assert ("- 話者数:" if language == "ja" else "- Speakers:") not in markdown


@pytest.mark.parametrize("enabled", COMBINATIONS)
def test_cli_feature_flags_reach_api(enabled, monkeypatch, tmp_path):
    import video_analyzer.cli as cli

    args = ["analyze", "input.mp4", "-o", str(tmp_path / "report.md")]
    for feature, value in zip(FEATURES, enabled, strict=True):
        args.append(f"--{'' if value else 'no-'}{feature.replace('_', '-')}")

    def capture(_, config):
        assert tuple(getattr(config, feature) for feature in FEATURES) == enabled
        raise AnalysisError("Stop after checking configuration")

    monkeypatch.setattr(cli, "analyze_video", capture)
    assert main(args) == 1


def test_cli_feature_defaults_and_help(capsys):
    args = parser().parse_args(["analyze", "input.mp4"])
    assert all(getattr(args, feature) is True for feature in FEATURES)
    assert all(getattr(AnalysisConfig(), feature) is True for feature in FEATURES)
    assert args.audio_normalization == AnalysisConfig().audio_normalization == "off"
    assert args.noise_reduction is AnalysisConfig().noise_reduction is False
    with pytest.raises(SystemExit) as exited:
        main(["analyze", "--help"])
    assert exited.value.code == 0
    help_text = capsys.readouterr().out
    for feature in FEATURES:
        flag = feature.replace("_", "-")
        assert f"--{flag}" in help_text and f"--no-{flag}" in help_text
    assert "--audio-normalization {off,peak,dynamic}" in help_text
    assert "--noise-reduction" in help_text and "--no-noise-reduction" in help_text


AUDIO_PROCESSING = list(itertools.product(("off", "peak", "dynamic"), (False, True)))


@pytest.mark.parametrize("normalization,noise_reduction", AUDIO_PROCESSING)
def test_cli_audio_processing_reaches_api(normalization, noise_reduction, monkeypatch, tmp_path):
    import video_analyzer.cli as cli

    def capture(_, config):
        assert config.audio_normalization == normalization
        assert config.noise_reduction is noise_reduction
        raise AnalysisError("Stop after checking configuration")

    monkeypatch.setattr(cli, "analyze_video", capture)
    assert (
        main(
            [
                "analyze",
                "input.mp4",
                "-o",
                str(tmp_path / "report.md"),
                "--audio-normalization",
                normalization,
                "--noise-reduction" if noise_reduction else "--no-noise-reduction",
            ]
        )
        == 1
    )


def test_invalid_audio_normalization():
    with pytest.raises(ValidationError, match="audio_normalization"):
        AnalysisConfig(audio_normalization="loudness")
    with pytest.raises(SystemExit) as exited:
        parser().parse_args(["analyze", "input.mp4", "--audio-normalization", "loudness"])
    assert exited.value.code == 2


@pytest.mark.parametrize("normalization,noise_reduction", AUDIO_PROCESSING)
@pytest.mark.parametrize("asr", ["whisper", "qwen"])
@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron"])
def test_shared_processed_audio_and_resume(
    normalization,
    noise_reduction,
    asr,
    backend,
    simulated_media,
    fake_models,
    monkeypatch,
    tmp_path,
):
    path, media, calls = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    extracted = []
    original_extract = pipeline.extract_audio

    def extract(media, target, **options):
        assert options["audio_normalization"] == normalization
        assert options["noise_reduction"] is noise_reduction
        extracted.append(str(target.resolve()))
        original_extract(media, target, **options)

    monkeypatch.setattr(pipeline, "extract_audio", extract)
    config = AnalysisConfig(
        profile="cpu",
        asr=asr,
        diarization_backend=backend,
        scene_description=False,
        work_dir=tmp_path / "work",
        audio_normalization=normalization,
        noise_reduction=noise_reduction,
    )
    result = pipeline.analyze_video(path, config)
    audio_requests = [
        payload["audio"]
        for operation, payload in fake_models.requests
        if operation in ("diarize", "transcribe")
    ]
    assert len(extracted) == 1 and audio_requests == extracted * 4
    assert result.provenance["configuration"]["audio_normalization"] == normalization
    assert result.provenance["configuration"]["noise_reduction"] is noise_reduction
    assert AnalysisResult.model_validate_json(render_json(result)) == result
    requests = fake_models.requests.copy()
    pipeline.analyze_video(path, config.model_copy(update={"resume": True}))
    assert calls["audio"] == 1 and fake_models.requests == requests


def test_audio_processing_modes_have_separate_resumable_caches(
    simulated_media, fake_models, tmp_path
):
    path, media, calls = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    work = tmp_path / "work"
    configs = [
        AnalysisConfig(
            profile="cpu",
            scene_description=False,
            work_dir=work,
            resume=True,
            audio_normalization=mode,
            noise_reduction=noise,
        )
        for mode, noise in AUDIO_PROCESSING
    ]
    for config in configs:
        pipeline.analyze_video(path, config)
    assert len(list(work.glob("*/audio.json"))) == len(configs)
    assert calls["audio"] == len(configs)
    requests = fake_models.requests.copy()
    for config in reversed(configs):
        pipeline.analyze_video(path, config)
    assert calls["audio"] == len(configs) and fake_models.requests == requests


@pytest.mark.parametrize("normalization,noise_reduction", AUDIO_PROCESSING)
@pytest.mark.parametrize("has_audio", [False, True])
def test_unused_audio_processing_is_skipped(
    normalization, noise_reduction, has_audio, simulated_media, fake_models, tmp_path
):
    path, media, calls = simulated_media
    if not has_audio:
        media.audio_stream_index = media.audio_start = None
    # Either audio is absent, or both audio consumers are disabled.
    config = AnalysisConfig(
        profile="cpu",
        scene_description=False,
        transcription=not has_audio,
        diarization=not has_audio,
        work_dir=tmp_path / "work",
        audio_normalization=normalization,
        noise_reduction=noise_reduction,
    )
    pipeline.analyze_video(path, config)
    assert calls["audio"] == 0 and not fake_models.requests


def test_audio_processing_failure_preserves_report_and_resumes(
    simulated_media, fake_models, monkeypatch, tmp_path
):
    path, media, calls = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    original_extract = pipeline.extract_audio
    attempts = 0

    def extract(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise AnalysisError("Audio processing failed")
        return original_extract(*args, **kwargs)

    monkeypatch.setattr(pipeline, "extract_audio", extract)
    output, work = tmp_path / "report.md", tmp_path / "work"
    output.write_text("Existing report")
    args = [
        "analyze",
        str(path),
        "--profile",
        "cpu",
        "--work-dir",
        str(work),
        "-o",
        str(output),
        "--audio-normalization",
        "dynamic",
        "--noise-reduction",
        "--no-scene-description",
    ]
    assert main(args) == 1
    assert output.read_text() == "Existing report"
    assert not list(work.glob("*/audio.json")) and not fake_models.requests
    assert main([*args, "--resume"]) == 0
    assert attempts == 2 and calls["scenes"] == 1
    assert len(list(work.glob("*/audio.json"))) == 1
    assert "word" in output.read_text()
