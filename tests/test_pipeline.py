import itertools
from pathlib import Path

import pytest
from pydantic import ValidationError

import video_analyzer.pipeline as pipeline
from video_analyzer.cli import main, parser
from video_analyzer.errors import AnalysisError
from video_analyzer.report import render_json, render_markdown
from video_analyzer.schema import AnalysisConfig, AnalysisResult, MediaInfo, Scene


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

    def scenes(*_):
        calls["scenes"] += 1
        return [Scene(id="scene_0001", start=0, end=25), Scene(id="scene_0002", start=25, end=55)]

    def audio(_, target, **options):
        calls["audio"] += 1
        target.write_bytes(b"fixture audio")

    def frames(_, targets, directory, edge):
        calls["frames"] += 1
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
@pytest.mark.parametrize("backend", ["pyannote", "sortformer"])
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
        expected_models.add("diarization" if backend == "pyannote" else "sortformer")
    if transcription:
        expected_models.add(asr)
        if asr == "qwen":
            expected_models.add("aligner")
    assert set(checked_models) == expected_models
    assert set(result.provenance["models"]) == expected_models
    expected_workers = set()
    if transcription or (diarization and backend == "pyannote") or (vision and profile != "mac"):
        expected_workers.add("torch")
    if diarization and backend == "sortformer":
        expected_workers.add("sortformer")
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
def test_diarization_identity_separates_transcription_checkpoints(
    change, simulated_media, fake_models, monkeypatch, tmp_path
):
    path, media, _ = simulated_media
    fake_models.turns = [{"start": 0, "end": media.duration, "speaker_id": "A"}]
    config = AnalysisConfig(
        profile="cpu",
        diarization_backend="sortformer",
        scene_description=False,
        work_dir=tmp_path / "work",
    )
    pipeline.analyze_video(path, config)
    local = pipeline.local_models

    def changed_models(root, keys):
        models = local(root, keys)
        models["sortformer"]["revision"] = "different-checkpoint"
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


@pytest.mark.parametrize("enabled", COMBINATIONS)
@pytest.mark.parametrize("backend", ["pyannote", "sortformer"])
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
@pytest.mark.parametrize("backend", ["pyannote", "sortformer"])
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
