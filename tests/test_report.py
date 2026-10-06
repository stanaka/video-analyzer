import itertools

import pytest

from video_analyzer.report import render_json, render_markdown
from video_analyzer.schema import AnalysisResult, MediaInfo, Observation, Scene

AUDIO_PROCESSING = list(itertools.product(("off", "peak", "dynamic"), (False, True)))
NORMALIZATION_STATUS = {
    "en": {"off": "Disabled", "peak": "Enabled (peak)", "dynamic": "Enabled (dynamic)"},
    "ja": {"off": "無効", "peak": "有効 (peak)", "dynamic": "有効 (dynamic)"},
}
NOISE_STATUS = {"en": {False: "Disabled", True: "Enabled"}, "ja": {False: "無効", True: "有効"}}


def test_markdown_prefers_safe_merged_description_and_keeps_json_observations(report_result):
    scene = report_result.scenes[0]
    scene.observations = [
        Observation(start=0, end=10, description="Original observation", frame_timestamps=[1])
    ]
    scene.description = "Merged <script>alert(1)</script> description."
    report_result.provenance["llm"] = {
        "model": "configured-model",
        "base_url": "http://localhost:8000/v1",
    }
    markdown = render_markdown(report_result)
    assert "Original observation" not in markdown
    assert "Merged &lt;script&gt;" in markdown
    assert "configured\\-model" in markdown
    assert "http://localhost:8000/v1" in markdown
    assert "Original observation" in render_json(report_result)
    scene.description = None
    assert "Original observation" in render_markdown(report_result)


@pytest.fixture
def report_result():
    return AnalysisResult(
        application_version="0.1.0",
        media=MediaInfo(
            path="input.mp4",
            duration=10,
            width=160,
            height=120,
            video_stream_index=0,
            video_start=0,
            audio_stream_index=1,
            audio_start=0,
        ),
        report_language="en",
        provenance={
            "configuration": {
                "transcription": True,
                "diarization": True,
                "audio_normalization": "off",
                "noise_reduction": False,
            },
            "models": {
                "whisper": {
                    "repo_id": "fixture/whisper",
                    "revision": "abc123",
                    "quantization": None,
                }
            },
            "runtimes": {"asr": {"device": "cpu"}},
        },
        scenes=[Scene(id="scene_0001", start=0, end=10)],
    )


@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("normalization,noise_reduction", AUDIO_PROCESSING)
def test_markdown_audio_preprocessing_settings(
    language, normalization, noise_reduction, report_result
):
    report_result.report_language = language
    report_result.provenance["configuration"].update(
        audio_normalization=normalization, noise_reduction=noise_reduction
    )
    original_json = render_json(report_result)
    lines = render_markdown(report_result).splitlines()
    heading = "## モデルと実行環境" if language == "ja" else "## Models and runtime"
    normalization_label = "音量正規化" if language == "ja" else "Audio normalization"
    noise_label = "ノイズ除去" if language == "ja" else "Noise reduction"
    section = lines.index(heading)
    assert lines[section + 1 : section + 6] == [
        "",
        f"- {normalization_label}: {NORMALIZATION_STATUS[language][normalization]}",
        f"- {noise_label}: {NOISE_STATUS[language][noise_reduction]}",
        "- whisper: fixture/whisper; `abc123`",
        "- asr: cpu",
    ]
    assert not any("スキップ" in line or "skipped" in line for line in lines)
    assert render_json(report_result) == original_json


@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("normalization,noise_reduction", AUDIO_PROCESSING)
@pytest.mark.parametrize(
    "has_audio,transcription,diarization,reason",
    [
        (False, True, True, "no_audio"),
        (True, False, False, "disabled"),
        (False, False, False, "no_audio"),
        (True, True, False, None),
        (True, False, True, None),
    ],
)
def test_markdown_audio_preprocessing_skip_reason(
    language,
    normalization,
    noise_reduction,
    has_audio,
    transcription,
    diarization,
    reason,
    report_result,
):
    report_result.report_language = language
    settings = report_result.provenance["configuration"]
    settings.update(audio_normalization=normalization, noise_reduction=noise_reduction)
    baseline_lines = render_markdown(report_result).splitlines()
    heading = "## モデルと実行環境" if language == "ja" else "## Models and runtime"
    section = baseline_lines.index(heading)
    configured_status = baseline_lines[section + 2 : section + 4]
    if not has_audio:
        report_result.media.audio_stream_index = report_result.media.audio_start = None
    settings.update(transcription=transcription, diarization=diarization)
    lines = render_markdown(report_result).splitlines()
    section = lines.index(heading)
    assert lines[section + 2 : section + 4] == configured_status
    skipped = [line for line in lines if "スキップ" in line or "skipped" in line]
    if reason is None or (normalization == "off" and not noise_reduction):
        assert not skipped
    else:
        explanations = {
            "en": {
                "no_audio": "- Audio preprocessing skipped: no audio track.",
                "disabled": "- Audio preprocessing skipped: transcription and diarization are disabled.",
            },
            "ja": {
                "no_audio": "- 音声前処理をスキップ: 音声トラックがありません。",
                "disabled": "- 音声前処理をスキップ: 文字起こしと話者区分が無効です。",
            },
        }
        assert skipped == [explanations[language][reason]]


@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize(
    "missing", ["audio_normalization", "noise_reduction", "both", "configuration"]
)
def test_markdown_older_results_default_to_disabled(language, missing, report_result):
    report_result.report_language = language
    expected = render_markdown(report_result)
    if missing == "configuration":
        del report_result.provenance["configuration"]
    else:
        keys = ("audio_normalization", "noise_reduction") if missing == "both" else (missing,)
        for key in keys:
            del report_result.provenance["configuration"][key]
    assert render_markdown(report_result) == expected
