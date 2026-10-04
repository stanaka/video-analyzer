import pytest

from video_analyzer import cli
from video_analyzer.errors import AnalysisError
from video_analyzer.media import probe
from video_analyzer.schema import AnalysisResult, Scene


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
    exported = tmp_path / "retry.html"
    assert cli.main(["export-html", str(json_output), "-o", str(exported)]) == 0
    assert exported.read_text().startswith("<!doctype html>")


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
