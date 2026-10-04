from pathlib import Path

import pytest

from video_analyzer import cli
from video_analyzer.errors import AnalysisError
from video_analyzer.media import probe
from video_analyzer.report import render_json, render_markdown
from video_analyzer.schema import (
    AnalysisResult,
    MediaInfo,
    Observation,
    Scene,
    Speaker,
    TranscriptUnit,
)


@pytest.fixture
def result(video, monkeypatch):
    result = AnalysisResult(
        application_version="0.1.0",
        media=probe(video),
        report_language="en",
        provenance={},
        scenes=[Scene(id="scene_0001", start=0, end=4)],
    )
    monkeypatch.setattr(cli, "analyze_video", lambda *_: result)
    return result


@pytest.mark.integration
@pytest.mark.parametrize("format_option", [None, "markdown", "json", "html"])
@pytest.mark.parametrize("custom_output", [False, True])
def test_analyze_always_saves_json(video, tmp_path, result, capsys, format_option, custom_output):
    selected_format = format_option or "markdown"
    extension = {"markdown": "md", "json": "json", "html": "html"}[selected_format]
    output = (
        tmp_path / "reports" / f"review.{extension}"
        if custom_output
        else video.with_suffix(f".analysis.{extension}")
    )
    args = ["analyze", str(video)]
    if format_option:
        args.extend(["--format", format_option])
    if custom_output:
        args.extend(["-o", str(output)])

    assert cli.main(args) == 0

    json_output = output.with_suffix(".json")
    assert AnalysisResult.model_validate_json(json_output.read_text()) == result
    assert list(tmp_path.rglob("*.json")) == [json_output]
    if selected_format == "markdown":
        assert output.read_text().startswith("#")
    elif selected_format == "html":
        assert output.read_text().startswith("<!doctype html>")
    captured = capsys.readouterr()
    assert captured.out == f"{output.resolve()}\n"
    if selected_format == "json":
        assert captured.err == ""
    else:
        assert f"JSON result saved to {json_output.resolve()}" in captured.err


@pytest.mark.integration
@pytest.mark.parametrize("format_option", ["markdown", "json", "html"])
def test_extensionless_report_path(video, tmp_path, result, format_option):
    output = tmp_path / "review"
    assert cli.main(["analyze", str(video), "--format", format_option, "-o", str(output)]) == 0
    json_output = output if format_option == "json" else output.with_suffix(".json")
    assert AnalysisResult.model_validate_json(json_output.read_text()) == result
    assert output.is_file()
    if format_option == "json":
        assert not output.with_suffix(".json").exists()


@pytest.mark.parametrize("format_option", ["markdown", "html"])
@pytest.mark.parametrize(
    "collision", ["json_extension", "uppercase_extension", "report_alias", "video_alias"]
)
def test_output_collisions_are_rejected_before_analysis(
    tmp_path, monkeypatch, capsys, format_option, collision
):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"original video")
    output = tmp_path / f"review.{format_option}"
    json_output = output.with_suffix(".json")
    json_output.write_text("old JSON")
    if collision == "json_extension":
        output = json_output
    elif collision == "uppercase_extension":
        output = output.with_suffix(".JSON")
    elif collision == "report_alias":
        output.symlink_to(json_output)
    elif collision == "video_alias":
        json_output.unlink()
        json_output.symlink_to(video)
    monkeypatch.setattr(
        cli, "analyze_video", lambda *_: pytest.fail("Analysis must not run for a path collision")
    )

    assert cli.main(["analyze", str(video), "--format", format_option, "-o", str(output)]) == 1
    assert video.read_bytes() == b"original video"
    if collision != "video_alias":
        assert json_output.read_text() == "old JSON"
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err


@pytest.mark.parametrize("format_option", ["markdown", "json", "html"])
def test_analysis_failure_preserves_both_outputs(tmp_path, monkeypatch, format_option):
    video = tmp_path / "input.mp4"
    output = tmp_path / f"review.{format_option}"
    json_output = output.with_suffix(".json")
    output.write_text("old selected report")
    json_output.write_text("old JSON")
    before = {path: path.read_bytes() for path in {output, json_output}}

    def fail(*_):
        raise AnalysisError("Analysis failed")

    monkeypatch.setattr(cli, "analyze_video", fail)
    assert cli.main(["analyze", str(video), "--format", format_option, "-o", str(output)]) == 1
    assert {path: path.read_bytes() for path in before} == before


@pytest.mark.integration
@pytest.mark.parametrize("format_option", ["markdown", "html"])
def test_presentation_failure_preserves_new_json_for_export(
    video, tmp_path, result, monkeypatch, capsys, format_option
):
    output = tmp_path / f"review.{format_option}"
    json_output = output.with_suffix(".json")
    output.write_text("old selected report")
    json_output.write_text("old JSON")

    def fail(*_):
        raise AnalysisError("Presentation failed")

    with monkeypatch.context() as patch:
        patch.setattr(cli, "write_html" if format_option == "html" else "render_markdown", fail)
        assert cli.main(["analyze", str(video), "--format", format_option, "-o", str(output)]) == 1

    assert output.read_text() == "old selected report"
    assert AnalysisResult.model_validate_json(json_output.read_text()) == result
    captured = capsys.readouterr()
    assert captured.out == ""
    assert str(json_output.resolve()) in captured.err
    assert "Presentation failed" in captured.err

    monkeypatch.setattr(
        cli, "analyze_video", lambda *_: pytest.fail("Export must not rerun analysis")
    )
    exported = tmp_path / ("retry.html" if format_option == "html" else "retry.md")
    assert cli.main([f"export-{format_option}", str(json_output), "-o", str(exported)]) == 0
    if format_option == "html":
        assert exported.read_text().startswith("<!doctype html>")
    else:
        assert exported.read_text(encoding="utf-8") == render_markdown(result)


@pytest.mark.integration
@pytest.mark.parametrize("format_option", ["markdown", "html"])
def test_json_write_failure_does_not_publish_selected_report(
    video, tmp_path, result, monkeypatch, format_option
):
    output = tmp_path / f"review.{format_option}"
    json_output = output.with_suffix(".json")
    output.write_text("old selected report")
    json_output.write_text("old JSON")

    def fail(*_):
        raise OSError("Cannot write JSON")

    monkeypatch.setattr(cli, "atomic_text", fail)
    monkeypatch.setattr(cli, "write_html", lambda *_: pytest.fail("Must save JSON first"))
    monkeypatch.setattr(cli, "render_markdown", lambda *_: pytest.fail("Must save JSON first"))
    assert cli.main(["analyze", str(video), "--format", format_option, "-o", str(output)]) == 1
    assert output.read_text() == "old selected report"
    assert json_output.read_text() == "old JSON"


@pytest.fixture
def markdown_result(tmp_path):
    return AnalysisResult(
        application_version="0.1.0",
        media=MediaInfo(
            path=str(tmp_path / "missing video 日本語.mp4"),
            duration=4,
            width=160,
            height=120,
            video_stream_index=0,
            video_start=0,
        ),
        report_language="en",
        provenance={"runtimes": {"asr": {"device": "cpu"}}},
        speakers=[Speaker(id="speaker_01")],
        transcript=[
            TranscriptUnit(id="t1", start=1, end=2, text="こんにちは", speaker_id="speaker_01")
        ],
        scenes=[
            Scene(
                id="scene_0001",
                start=0,
                end=4,
                transcript_ids=["t1"],
                observations=[
                    Observation(start=0, end=4, description="青い空", frame_timestamps=[1])
                ],
            )
        ],
        warnings=["Fixture warning"],
    )


@pytest.mark.parametrize("language", ["ja", "en"])
@pytest.mark.parametrize("output_kind", ["default", "nested", "extensionless", "existing"])
def test_export_markdown_from_json_without_media_or_models(
    tmp_path, markdown_result, monkeypatch, capsys, language, output_kind
):
    markdown_result.report_language = language
    source = tmp_path / "input.analysis.json"
    source.write_text(render_json(markdown_result), encoding="utf-8")
    before = source.read_bytes()
    output = source.with_suffix(".md")
    args = ["export-markdown", source.name]
    monkeypatch.chdir(tmp_path)
    if output_kind != "default":
        output = (
            tmp_path / "reports" / ("review" if output_kind == "extensionless" else "review.md")
        )
        args.extend(["--output", str(output.relative_to(tmp_path))])
    if output_kind == "existing":
        output.parent.mkdir()
        output.write_text("old report")

    def unexpected(*args, **kwargs):
        pytest.fail("Markdown export must not use analysis, media tools, or models")

    for name in ("analyze_video", "write_html", "prerequisites", "prepare_models", "Worker"):
        monkeypatch.setattr(cli, name, unexpected)
    monkeypatch.setenv("PATH", "")
    assert not Path(markdown_result.media.path).exists()

    assert cli.main(args) == 0

    assert output.read_text(encoding="utf-8") == render_markdown(markdown_result)
    assert source.read_bytes() == before
    assert list(tmp_path.rglob("*.json")) == [source]
    captured = capsys.readouterr()
    assert captured.out == f"{output.resolve()}\n"
    assert captured.err == ""


@pytest.mark.parametrize("input_kind", ["missing", "malformed", "invalid", "unsupported_schema"])
def test_export_markdown_invalid_input_preserves_report(tmp_path, capsys, input_kind):
    source = tmp_path / "input.json"
    output = source.with_suffix(".md")
    output.write_text("existing report")
    values = {
        "malformed": "not JSON",
        "invalid": "{}",
        "unsupported_schema": '{"schema_version":"2.0"}',
    }
    if input_kind != "missing":
        source.write_text(values[input_kind])
    before = source.read_bytes() if source.exists() else None

    assert cli.main(["export-markdown", str(source)]) == 1

    assert output.read_text() == "existing report"
    assert (source.read_bytes() if source.exists() else None) == before
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error:" in captured.err


@pytest.mark.parametrize("target", ["json", "video"])
@pytest.mark.parametrize("alias", [False, True])
def test_export_markdown_rejects_output_collisions(
    tmp_path, markdown_result, capsys, target, alias
):
    video = Path(markdown_result.media.path)
    video.write_bytes(b"original video")
    source = tmp_path / "input.json"
    source.write_text(render_json(markdown_result), encoding="utf-8")
    before = source.read_bytes()
    output = source if target == "json" else video
    if alias:
        link = tmp_path / "alias.md"
        link.symlink_to(output)
        output = link

    assert cli.main(["export-markdown", str(source), "-o", str(output)]) == 1

    assert source.read_bytes() == before
    assert video.read_bytes() == b"original video"
    if alias:
        assert output.is_symlink()
    captured = capsys.readouterr()
    assert captured.out == ""
    label = "JSON" if target == "json" else "video"
    assert f"Output must not overwrite the input {label}" in captured.err


@pytest.mark.parametrize("failure", ["render", "write"])
def test_export_markdown_failure_preserves_existing_report(
    tmp_path, markdown_result, monkeypatch, capsys, failure
):
    source = tmp_path / "input.json"
    source.write_text(render_json(markdown_result), encoding="utf-8")
    before = source.read_bytes()
    output = source.with_suffix(".md")
    output.write_text("existing report")

    def fail(*args):
        raise OSError("Export failed")

    if failure == "render":
        monkeypatch.setattr(cli, "render_markdown", fail)
    else:
        monkeypatch.setattr("video_analyzer.storage.os.replace", fail)

    assert cli.main(["export-markdown", str(source)]) == 1

    assert output.read_text() == "existing report"
    assert source.read_bytes() == before
    assert set(tmp_path.iterdir()) == {source, output}
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error: Export failed" in captured.err


def test_export_markdown_help(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["export-markdown", "--help"])
    assert exc.value.code == 0
    help_text = " ".join(capsys.readouterr().out.split())
    assert "--output" in help_text
    assert ".md extension" in help_text
    assert "No original video, FFmpeg, or models are required" in help_text
