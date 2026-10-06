from pathlib import Path

import pytest
import yaml

from video_analyzer import cli, llm
from video_analyzer.errors import AnalysisError
from video_analyzer.report import render_json
from video_analyzer.schema import AnalysisResult, MediaInfo, Observation, Scene, TranscriptUnit


@pytest.fixture
def source(tmp_path):
    return AnalysisResult(
        application_version="0.1.0",
        report_language="en",
        provenance={},
        media=MediaInfo(
            path=str(tmp_path / "missing.mp4"),
            duration=2,
            width=160,
            height=120,
            video_stream_index=0,
            video_start=0,
        ),
        transcript=[TranscriptUnit(id="t1", start=0, end=1, text="helo", scene_id="scene1")],
        scenes=[
            Scene(
                id="scene1",
                start=0,
                end=2,
                transcript_ids=["t1"],
                observations=[
                    Observation(start=0, end=2, description="A person waves.", frame_timestamps=[1])
                ],
            )
        ],
    )


@pytest.fixture
def endpoint_args():
    return ["--llm-base-url", "http://localhost:8000/v1", "--llm-model", "test-model"]


@pytest.fixture
def polish(monkeypatch):
    calls = []

    def run(result, config, **kwargs):
        calls.append((result.model_copy(deep=True), config, kwargs))
        result = result.model_copy(deep=True)
        result.transcript[0].text = "hello"
        result.scenes[0].description = "A waving person."
        return result

    monkeypatch.setattr(cli, "polish_analysis", run)
    return calls


def test_polish_saved_json_without_media_or_runtimes(
    source, endpoint_args, polish, monkeypatch, tmp_path, capsys
):
    path = tmp_path / "input.analysis.json"
    path.write_text(render_json(source))
    before = path.read_bytes()
    for name in ("analyze_video", "prerequisites", "prepare_models", "Worker", "write_html"):
        monkeypatch.setattr(
            cli, name, lambda *_: pytest.fail("Must not run local models or media tools")
        )
    monkeypatch.setenv("PATH", "")
    assert cli.main(["polish", str(path), *endpoint_args, "--resume"]) == 0
    output = path.with_suffix(".polished.json")
    result = AnalysisResult.model_validate_json(output.read_text())
    assert result.transcript[0].text == "hello"
    assert result.scenes[0].description == "A waving person."
    assert path.read_bytes() == before
    assert polish[0][2] == {"work_dir": tmp_path / ".video-analyzer", "resume": True}
    assert capsys.readouterr().out == f"{output.resolve()}\n"


@pytest.mark.parametrize("command", ["analyze", "polish"])
def test_nested_config_and_cli_precedence(
    source, endpoint_args, polish, monkeypatch, tmp_path, capsys, command
):
    config = tmp_path / "settings.yaml"
    config.write_text(
        "llm_enabled: true\nresume: true\nllm:\n  base_url: http://yaml/v1\n"
        "  model: yaml-model\n  max_input_chars: 3000\n  transcript_overlap_units: 4\n"
        "  scene_overlap_observations: 2\n  timeout: 45\n  api_key_env: CUSTOM_KEY\n"
    )
    path = tmp_path / "input.json"
    path.write_text(render_json(source))
    monkeypatch.setattr(cli, "analyze_video", lambda *_: source)
    args = [
        command,
        str(path),
        "--config",
        str(config),
        "--show-settings",
        "--llm-model",
        "cli-model",
        "--llm-transcript-overlap-units",
        "0",
        "--no-resume",
    ]
    assert cli.main(args) == 0
    selected = polish[0][1]
    assert selected.base_url == "http://yaml/v1"
    assert selected.model == "cli-model"
    assert selected.transcript_overlap_units == 0
    assert selected.scene_overlap_observations == 2
    assert selected.max_input_chars == 3000
    assert selected.timeout == 45
    assert selected.api_key_env == "CUSTOM_KEY"
    assert polish[0][2]["resume"] is False
    captured = capsys.readouterr()
    shown = yaml.safe_load(
        captured.err.split("Raw JSON result saved")[0].split("JSON result saved")[0]
    )
    assert shown["llm"]["transcript_overlap_units"] == 0
    assert shown["llm_enabled"] is True


@pytest.mark.parametrize("format_option", ["json", "markdown", "html"])
def test_analyze_saves_raw_before_polishing(
    source, endpoint_args, monkeypatch, tmp_path, format_option
):
    output = tmp_path / f"report.{format_option}"
    json_output = output if format_option == "json" else output.with_suffix(".json")
    raw_output = json_output.with_suffix(".raw.json")
    supplied_configs = []

    def analyze(path, config):
        supplied_configs.append(config)
        return source

    def polish(result, config, **kwargs):
        assert AnalysisResult.model_validate_json(raw_output.read_text()) == source
        assert not json_output.exists()
        corrected = result.model_copy(deep=True)
        corrected.transcript[0].text = "hello"
        return corrected

    monkeypatch.setattr(cli, "analyze_video", analyze)
    monkeypatch.setattr(cli, "polish_analysis", polish)
    monkeypatch.setattr(cli, "write_html", lambda result, output: output.write_text("HTML"))
    assert (
        cli.main(
            [
                "analyze",
                source.media.path,
                "--llm",
                *endpoint_args,
                "--format",
                format_option,
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert not supplied_configs[0].llm_enabled
    assert AnalysisResult.model_validate_json(raw_output.read_text()).transcript[0].text == "helo"
    assert AnalysisResult.model_validate_json(json_output.read_text()).transcript[0].text == "hello"


@pytest.mark.parametrize("format_option", ["json", "markdown", "html"])
def test_polish_failure_preserves_selected_outputs_and_raw(
    source, endpoint_args, monkeypatch, tmp_path, format_option
):
    output = tmp_path / f"report.{format_option}"
    json_output = output if format_option == "json" else output.with_suffix(".json")
    output.write_text("old report")
    json_output.write_text("old JSON")
    before = {path: path.read_bytes() for path in (output, json_output)}
    monkeypatch.setattr(cli, "analyze_video", lambda *_: source)

    def fail(*args, **kwargs):
        raise AnalysisError("endpoint unavailable")

    monkeypatch.setattr(cli, "polish_analysis", fail)
    assert (
        cli.main(
            [
                "analyze",
                source.media.path,
                "--llm",
                *endpoint_args,
                "--format",
                format_option,
                "-o",
                str(output),
            ]
        )
        == 1
    )
    assert {path: path.read_bytes() for path in before} == before
    assert (
        AnalysisResult.model_validate_json(json_output.with_suffix(".raw.json").read_text())
        == source
    )


def test_raw_write_failure_does_not_contact_endpoint(source, endpoint_args, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "analyze_video", lambda *_: source)
    monkeypatch.setattr(
        cli, "polish_analysis", lambda *_: pytest.fail("Cannot polish before raw publication")
    )

    def fail(*_):
        raise OSError("raw write failed")

    monkeypatch.setattr(cli, "atomic_text", fail)
    assert (
        cli.main(
            [
                "analyze",
                source.media.path,
                "--llm",
                *endpoint_args,
                "-o",
                str(tmp_path / "result.md"),
            ]
        )
        == 1
    )


@pytest.mark.parametrize("target", ["input", "video", "raw"])
@pytest.mark.parametrize("symlink", [False, True])
def test_saved_polish_output_collisions(source, endpoint_args, polish, tmp_path, target, symlink):
    path = tmp_path / "input.json"
    path.write_text(render_json(source))
    video = Path(source.media.path)
    video.write_bytes(b"original video")
    raw = path.with_suffix(".raw.json")
    raw.write_text("original raw")
    output = {"input": path, "video": video, "raw": raw}[target]
    before = {item: item.read_bytes() for item in (path, video, raw)}
    if symlink:
        alias = tmp_path / "alias.json"
        alias.symlink_to(output)
        output = alias
    assert cli.main(["polish", str(path), *endpoint_args, "-o", str(output)]) == 1
    assert {item: item.read_bytes() for item in before} == before
    assert not polish


def test_analyze_raw_output_alias_is_rejected(source, endpoint_args, monkeypatch, tmp_path):
    output = tmp_path / "report.md"
    raw = output.with_suffix(".raw.json")
    raw.symlink_to(output)
    monkeypatch.setattr(
        cli, "analyze_video", lambda *_: pytest.fail("Path validation must come first")
    )
    assert cli.main(["analyze", source.media.path, "--llm", *endpoint_args, "-o", str(output)]) == 1


def test_saved_polish_failure_preserves_input_and_existing_output(
    source, endpoint_args, monkeypatch, tmp_path
):
    path = tmp_path / "source.json"
    path.write_text(render_json(source))
    output = path.with_suffix(".polished.json")
    output.write_text("old output")
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise AnalysisError("invalid corrections")

    monkeypatch.setattr(cli, "polish_analysis", fail)
    assert cli.main(["polish", str(path), *endpoint_args]) == 1
    assert path.read_bytes() == before
    assert output.read_text() == "old output"


def test_disabled_polishing_never_contacts_endpoint(source, endpoint_args, monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "analyze_video", lambda *_: source)
    monkeypatch.setattr(
        llm.LLMClient, "complete", lambda *_: pytest.fail("Polishing must be opt-in")
    )
    output = tmp_path / "report.json"
    assert (
        cli.main(
            [
                "analyze",
                source.media.path,
                *endpoint_args,
                "--no-llm",
                "--format",
                "json",
                "-o",
                str(output),
            ]
        )
        == 0
    )
    assert not output.with_suffix(".raw.json").exists()


def test_llm_secret_does_not_appear_in_settings_outputs_or_checkpoints(
    source, endpoint_args, monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv("VIDEO_ANALYZER_LLM_API_KEY", "VERY-SECRET-KEY")
    monkeypatch.setattr(llm.LLMClient, "complete", lambda *_: {"description": "Scene summary"})
    source.transcript = []
    path = tmp_path / "source.json"
    path.write_text(render_json(source))
    assert cli.main(["polish", str(path), *endpoint_args, "--show-settings"]) == 0
    captured = capsys.readouterr()
    assert "VERY-SECRET-KEY" not in captured.out + captured.err
    assert all("VERY-SECRET-KEY" not in item.read_text() for item in tmp_path.rglob("*.json"))


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--llm-timeout", "0"),
        ("--llm-max-input-chars", "0"),
        ("--llm-transcript-overlap-units", "-1"),
        ("--llm-scene-overlap-observations", "-1"),
    ],
)
def test_invalid_llm_flags_fail_before_analysis(source, endpoint_args, monkeypatch, flag, value):
    monkeypatch.setattr(
        cli, "analyze_video", lambda *_: pytest.fail("Invalid settings must fail early")
    )
    assert cli.main(["analyze", source.media.path, "--llm", *endpoint_args, flag, value]) == 1


def test_polish_help_documents_batching_and_no_models(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["polish", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    for flag in (
        "--llm-max-input-chars",
        "--llm-transcript-overlap-units",
        "--llm-scene-overlap-observations",
        "--resume",
    ):
        assert flag in help_text
    assert "No original video, FFmpeg, or model runtimes are required" in help_text
