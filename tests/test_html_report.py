import itertools
import json
import re
import shutil
import wave
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pytest

from video_analyzer import cli, write_html
from video_analyzer.errors import AnalysisError
from video_analyzer.html_report import render_html
from video_analyzer.media import extract_audio, probe
from video_analyzer.report import render_json
from video_analyzer.schema import AnalysisResult, Scene
from video_analyzer.storage import file_hash


def embedded(html):
    return json.loads(
        re.search(r'<script id="analysis-data" type="application/json">(.*?)</script>', html, re.S)[
            1
        ]
    )


def asset_path(output, url):
    return (output.parent / unquote(url)).resolve()


@pytest.mark.integration
def test_segment_export_regenerates_matching_audio(viewer_result, tmp_path):
    result = viewer_result
    result.provenance["analysis_range"] = {"start": 2.25, "end": 5.75}
    output = write_html(result, tmp_path / "segment.html")
    data = embedded(output.read_text())
    playback = data["playback"]
    assert playback["video_start"] == 1 and playback["audio_start"] == 2.25
    assert data["result"]["media"]["duration"] == 12
    for name, rate in [("original", 48000), ("processed", 16000)]:
        with wave.open(str(asset_path(output, playback[name]))) as audio:
            assert audio.getnframes() / rate == pytest.approx(3.5, abs=1 / rate)
    expected = tmp_path / "expected.wav"
    extract_audio(
        result.media,
        expected,
        start_time=2.25,
        end_time=5.75,
        audio_normalization="dynamic",
        noise_reduction=True,
    )
    assert asset_path(output, playback["processed"]).read_bytes() == expected.read_bytes()


@pytest.mark.integration
@pytest.mark.parametrize(
    "bounds",
    [
        {"start": -1, "end": 5},
        {"start": 3, "end": 13},
        {"start": 3, "end": 2},
        {"start": 2},
        {"start": True, "end": 5},
    ],
)
def test_export_rejects_invalid_range_before_publication(viewer_result, tmp_path, bounds):
    viewer_result.provenance["analysis_range"] = bounds
    output = tmp_path / "bad.html"
    with pytest.raises(AnalysisError, match="range|time"):
        write_html(viewer_result, output)
    assert not output.exists() and not output.with_suffix(".media").exists()


def no_audio_result(video):
    return AnalysisResult(
        application_version="0.1.0",
        media=probe(video),
        report_language="ja",
        provenance={},
        scenes=[Scene(id="scene_0001", start=0, end=4)],
    )


@pytest.mark.integration
def test_no_audio_legacy_export_and_safe_urls(video, tmp_path):
    video = video.rename(video.with_name('local # 50% ? "日本語".mp4'))
    result = no_audio_result(video)
    output = tmp_path / "reports" / "view.html"
    before = render_json(result)
    assert write_html(result, output) == output
    report = output.read_text()
    data = embedded(report)
    assert data["result"] == json.loads(before)
    assert data["playback"]["original"] is data["playback"]["processed"] is None
    assert "%23" in data["playback"]["video"] and "%25" in data["playback"]["video"]
    assert asset_path(output, data["playback"]["video"]) == video
    assert not output.with_suffix(".media").exists()
    assert report.count("<script") == 2
    assert '<html lang="ja">' in report
    assert not re.search(r"<(?:script|link)[^>]+(?:src|href)=", report)
    assert render_json(result) == before


@pytest.mark.integration
def test_untrusted_text_cannot_escape_json_or_template(video):
    result = no_audio_result(video)
    payload = "</script><script>window.injected=true</script>&\u2028@@JS@@"
    result.warnings = [payload]
    result.media.path = '@@CSS@@ <img src=x onerror="alert(1)">.mp4'
    report = render_html(result, {"video": "./safe.mp4"})
    assert report.count("<script") == 2
    assert payload not in report
    assert embedded(report)["result"]["warnings"] == [payload]
    assert "<img src=x" not in report
    assert "@@CSS@@ &lt;img" in report


@pytest.mark.integration
@pytest.mark.parametrize(
    "normalization,denoise", list(itertools.product(("off", "peak", "dynamic"), (False, True)))
)
def test_export_audio_tracks_match_processing(viewer_result, tmp_path, normalization, denoise):
    result = viewer_result
    result.provenance["configuration"].update(
        audio_normalization=normalization, noise_reduction=denoise
    )
    output = write_html(result, tmp_path / "report.html")
    data = embedded(output.read_text())["playback"]
    original = asset_path(output, data["original"])
    with wave.open(str(original)) as stream:
        assert stream.getframerate() == 48000
        assert stream.getnchannels() == 2
        assert stream.getnframes() / 48000 == pytest.approx(result.media.duration, abs=0.001)
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
        assert np.max(np.abs(samples)) > 1000  # Selected tone, not the silent default track.
    if normalization == "off" and not denoise:
        assert data["processed"] is None
    else:
        actual = asset_path(output, data["processed"])
        expected = tmp_path / "expected.wav"
        extract_audio(
            result.media, expected, audio_normalization=normalization, noise_reduction=denoise
        )
        assert actual.read_bytes() == expected.read_bytes()
        with wave.open(str(actual)) as stream:
            assert stream.getframerate() == 16000 and stream.getnchannels() == 1
            assert stream.getnframes() / 16000 == pytest.approx(result.media.duration, abs=0.001)
    assert data["video_start"] == 1
    assert asset_path(output, data["video"]) == Path(result.media.path)


@pytest.mark.integration
def test_single_delayed_audio_uses_native_original(tmp_path, ffmpeg):
    path = tmp_path / "delayed.mp4"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=red:s=160x120:r=10:d=4",
        "-itsoffset",
        "1",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=2",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        path,
    )
    result = AnalysisResult(
        application_version="0.1.0",
        media=probe(path),
        report_language="en",
        provenance={"configuration": {"noise_reduction": True}},
        scenes=[Scene(id="scene_0001", start=0, end=4)],
    )
    output = write_html(result, tmp_path / "delayed.html")
    data = embedded(output.read_text())["playback"]
    assert data["original"] is None
    with wave.open(str(asset_path(output, data["processed"]))) as stream:
        samples = np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2")
    assert len(samples) == 4 * 16000
    assert not np.any(samples[:12800])
    assert not np.any(samples[-8000:])


@pytest.mark.integration
def test_source_validation_and_relocation(video, tmp_path):
    result = no_audio_result(video)
    result.provenance["input_sha256"] = file_hash(video)
    output = tmp_path / "report.html"
    relocated = video.rename(tmp_path / "moved.mp4")
    with pytest.raises(AnalysisError, match="--video"):
        write_html(result, output)
    write_html(result, output, video_path=relocated)
    assert asset_path(output, embedded(output.read_text())["playback"]["video"]) == relocated
    result.provenance["input_sha256"] = "0" * 64
    with pytest.raises(AnalysisError, match="SHA-256"):
        write_html(result, output, video_path=relocated)
    del result.provenance["input_sha256"]
    result.media.duration += 1
    with pytest.raises(AnalysisError, match="metadata"):
        write_html(result, output, video_path=relocated)


@pytest.mark.integration
@pytest.mark.parametrize("failure", ["audio", "render", "publish", "interrupt"])
def test_failed_export_preserves_previous_generation(viewer_result, tmp_path, monkeypatch, failure):
    import video_analyzer.html_report as report

    output = write_html(viewer_result, tmp_path / "report.html")
    old_html = output.read_bytes()
    media = tmp_path / "report.media"
    old_assets = {p: p.read_bytes() for p in media.rglob("*.wav")}
    old_directories = set(media.iterdir())

    def fail(*args, **kwargs):
        if failure == "interrupt":
            raise KeyboardInterrupt
        raise OSError("Simulated failure")

    monkeypatch.setattr(
        report,
        {
            "audio": "extract_audio",
            "render": "render_html",
            "publish": "atomic_text",
            "interrupt": "extract_audio",
        }[failure],
        fail,
    )
    with pytest.raises(KeyboardInterrupt if failure == "interrupt" else OSError):
        write_html(viewer_result, output)
    assert output.read_bytes() == old_html
    assert set(media.iterdir()) == old_directories
    assert {p: p.read_bytes() for p in media.rglob("*.wav")} == old_assets


@pytest.mark.integration
def test_reexport_keeps_assets_for_open_report(viewer_result, tmp_path):
    output = write_html(viewer_result, tmp_path / "report.html")
    old_audio = embedded(output.read_text())["playback"]["processed"]
    write_html(viewer_result, output)
    new_audio = embedded(output.read_text())["playback"]["processed"]
    assert new_audio != old_audio
    assert asset_path(output, old_audio).is_file() and asset_path(output, new_audio).is_file()


@pytest.mark.integration
def test_interrupt_after_atomic_publish_keeps_referenced_assets(
    viewer_result, tmp_path, monkeypatch
):
    import video_analyzer.html_report as report
    from video_analyzer.storage import atomic_text

    output = write_html(viewer_result, tmp_path / "report.html")
    old = output.read_text()

    def publish_then_interrupt(path, text):
        atomic_text(path, text)
        raise KeyboardInterrupt

    monkeypatch.setattr(report, "atomic_text", publish_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        write_html(viewer_result, output)
    assert output.read_text() != old
    for name in ("original", "processed"):
        assert asset_path(output, embedded(output.read_text())["playback"][name]).is_file()


@pytest.mark.integration
def test_export_cli_and_analyze_html(video, tmp_path, monkeypatch, capsys):
    result = no_audio_result(video)
    json_path = tmp_path / "report.json"
    json_path.write_text(render_json(result))
    monkeypatch.setattr(cli, "analyze_video", lambda *_: pytest.fail("Must not run inference"))
    assert cli.main(["export-html", str(json_path)]) == 0
    assert capsys.readouterr().out == str(json_path.with_suffix(".html")) + "\n"
    relocated = tmp_path / "moved.mp4"
    shutil.copyfile(video, relocated)
    custom = tmp_path / "other" / "custom.html"
    assert (
        cli.main(["export-html", str(json_path), "--video", str(relocated), "-o", str(custom)]) == 0
    )
    assert capsys.readouterr().out == str(custom) + "\n"
    assert asset_path(custom, embedded(custom.read_text())["playback"]["video"]) == relocated
    original = json_path.read_bytes()
    assert cli.main(["export-html", str(json_path), "-o", str(json_path)]) == 1
    assert json_path.read_bytes() == original
    video_bytes = video.read_bytes()
    assert cli.main(["export-html", str(json_path), "-o", str(video)]) == 1
    assert video.read_bytes() == video_bytes
    monkeypatch.setattr(cli, "analyze_video", lambda *_: result)
    assert cli.main(["analyze", str(video), "--format", "html"]) == 0
    assert video.with_suffix(".analysis.html").is_file()
    assert cli.main(["analyze", str(video), "--format", "html", "-o", str(custom)]) == 0


def test_export_cli_invalid_json_does_not_replace_report(tmp_path):
    source = tmp_path / "invalid.json"
    output = source.with_suffix(".html")
    output.write_text("existing")
    for value in ("not JSON", "{}", '{"schema_version":"2.0"}'):
        source.write_text(value)
        assert cli.main(["export-html", str(source)]) == 1
        assert output.read_text() == "existing"
