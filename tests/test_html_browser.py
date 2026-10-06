"""Opt-in real file:// playback tests; no server, models, or network resources."""

import os
from pathlib import Path

import pytest

from video_analyzer import write_html
from video_analyzer.schema import TranscriptUnit

pytestmark = [pytest.mark.browser, pytest.mark.integration]


@pytest.fixture(scope="module")
def browser():
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        kind = os.environ.get("VIDEO_ANALYZER_TEST_BROWSER", "chromium")
        instance = getattr(playwright, kind).launch()
        yield instance
        instance.close()


@pytest.fixture
def page(browser):
    page = browser.new_page(viewport={"width": 1280, "height": 1000})
    errors, remote = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "request",
        lambda request: (
            remote.append(request.url) if request.url.startswith(("http:", "https:")) else None
        ),
    )
    page.set_default_timeout(10000)
    yield page
    page.close()
    assert not errors
    assert not remote


def load(page, result, tmp_path):
    output = write_html(result, tmp_path / "report.html")
    page.goto(output.as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("document.querySelector('#video').readyState >= 2")
    assert page.locator("#media-error").is_hidden()
    return output


def test_merged_scene_description_and_polished_text_render_safely(page, viewer_result, tmp_path):
    summary = '<img src=x onerror="window.injected=true"> Merged description'
    viewer_result.scenes[0].description = summary
    viewer_result.transcript[0].text = "Polished passage "
    viewer_result.provenance["llm"] = {
        "model": "test-model",
        "base_url": "http://localhost:8000/v1",
    }
    load(page, viewer_result, tmp_path)
    assert page.locator(".scene-entry p").count() == 1
    assert page.locator(".scene-entry p").inner_text() == summary
    assert "Test pattern." not in page.locator(".scene-entry").inner_text()
    assert page.locator(".scene-entry img").count() == 0
    assert page.evaluate("window.injected") is None
    assert "Polished passage" in page.locator("#transcript").inner_text()
    assert "test-model" in page.locator("#metadata").text_content()
    assert "http://localhost:8000/v1" in page.locator("#metadata").text_content()


def slider(page, selector, value):
    page.locator(selector).evaluate(
        "(el, value) => { el.value = value; el.dispatchEvent(new Event('input', {bubbles: true})); }",
        str(value),
    )


def seek_paused(page, seconds):
    slider(page, "#seek", seconds)
    page.wait_for_function("!document.querySelector('#video').seeking")


def assert_sync(page, offset=1):
    page.wait_for_function(
        """offset => {
            const video = document.querySelector('#video');
            const mode = document.querySelector('#audio-mode').value;
            const audio = document.querySelector('#' + mode + '-audio');
            return audio && !video.paused && !audio.paused && !video.seeking && !audio.seeking &&
                Math.abs(video.currentTime - offset - audio.currentTime) < 0.15;
        }""",
        arg=offset,
    )


@pytest.mark.parametrize("mode", ["original", "processed"])
def test_segment_playback_offsets_limits_and_replay(page, viewer_result, tmp_path, mode):
    result = viewer_result
    start, end = 2.25, 5.75
    result.provenance["analysis_range"] = {"start": start, "end": end}
    result.scenes[0].start = start
    result.scenes[0].end = end
    for observation in result.scenes[0].observations:
        observation.start, observation.end = start, end
        observation.frame_timestamps = [3]
    result.transcript = [unit for unit in result.transcript if start <= unit.start < end]
    result.diarization = [
        turn.model_copy(update={"start": max(start, turn.start), "end": min(end, turn.end)})
        for turn in result.diarization
        if turn.start < end and turn.end > start
    ]
    load(page, result, tmp_path)
    assert page.locator("#seek").get_attribute("min") == str(start)
    assert page.locator("#seek").get_attribute("max") == str(end)
    assert "Analysis range 00:02.250 – 00:05.750" in page.locator("#summary").inner_text()
    assert page.locator(".tick").first.inner_text() == "00:02.250"
    page.wait_for_function("Math.abs(document.querySelector('#video').currentTime - 3.25) < 0.01")
    page.locator("#audio-mode").select_option(mode)
    page.locator("#play").click()
    assert_sync(page, offset=3.25)
    page.locator("#play").click()
    seek_paused(page, -10)
    assert page.locator("#video").evaluate("v => v.currentTime") == pytest.approx(3.25, abs=0.01)
    lane = page.locator(".track-body").first.bounding_box()
    page.mouse.click(lane["x"] + lane["width"] / 2, lane["y"] + lane["height"] / 2)
    assert_sync(page, offset=3.25)
    assert page.locator("#video").evaluate("v => v.currentTime") == pytest.approx(5, abs=0.35)
    page.locator("#play").click()
    seek_paused(page, end - 0.15)
    page.locator("#play").click()
    page.wait_for_function("document.querySelector('#video').paused")
    assert page.locator("#video").evaluate("v => v.currentTime") == pytest.approx(end + 1, abs=0.04)
    assert page.locator("#" + mode + "-audio").evaluate("a => a.paused")
    page.locator("#play").click()
    assert_sync(page, offset=3.25)
    assert page.locator("#video").evaluate("v => v.currentTime") < start + 1.75


@pytest.mark.parametrize("mode", ["original", "processed"])
def test_click_video_toggles_playback_with_selected_audio(page, viewer_result, tmp_path, mode):
    load(page, viewer_result, tmp_path)
    page.locator("#audio-mode").select_option(mode)
    page.locator("#video").click()
    assert_sync(page)
    assert page.locator("#play").inner_text() == "Pause"
    # Toolbar controls must not bubble into the video click action.
    page.locator("#mute").click()
    assert_sync(page)
    page.locator("#video").click()
    assert page.locator("#play").inner_text() == "Play"
    assert page.locator("#video").evaluate("v => v.paused")
    assert page.locator("#" + mode + "-audio").evaluate("a => a.paused")
    position = page.locator("#video").evaluate("v => v.currentTime")
    page.locator("#video").click()
    assert_sync(page)
    assert page.locator("#video").evaluate("v => v.currentTime") >= position


def test_timeline_click_scrolls_transcript_to_matching_passage(page, viewer_result, tmp_path):
    viewer_result.transcript = [
        TranscriptUnit(
            id=f"t{index}",
            start=index / 4,
            end=index / 4 + 0.125,
            text=f"Passage {index}",
            speaker_id=f"speaker_{index % 2 + 1:02d}",
        )
        for index in range(40)
    ]
    # This other speaker's passage also contains the clicked time, but the
    # clicked speaker's passage should be centered in the transcript pane.
    viewer_result.transcript.append(
        TranscriptUnit(
            id="overlap", start=3.9, end=4.2, text="Overlapping speech", speaker_id="speaker_02"
        )
    )
    load(page, viewer_result, tmp_path)
    assert page.locator("#transcript").evaluate("p => p.scrollHeight > p.clientHeight")
    slider(page, "#zoom", 4)
    page.locator("#timeline-scroll").evaluate("el => el.scrollLeft = 900")
    lane = page.locator(".track-body").nth(1).bounding_box()
    page_y = page.evaluate("scrollY")
    page.mouse.click(lane["x"] + lane["width"] * 4.05 / 12, lane["y"] + 22)
    assert_sync(page)
    row = page.locator(".transcript-entry").filter(has=page.get_by_text("Passage 16", exact=True))
    pane_rect = page.locator("#transcript").bounding_box()
    row_rect = row.bounding_box()
    assert row_rect["y"] + row_rect["height"] / 2 == pytest.approx(
        pane_rect["y"] + pane_rect["height"] / 2, abs=2
    )
    assert page.locator("#transcript").evaluate("p => p.scrollTop") > 0
    assert page.evaluate("scrollY") == page_y
    page.locator("#play").click()

    page.locator("#fit").click()
    ruler = page.locator("#ruler").bounding_box()
    for position, text in [(5.2, "Passage 21"), (11, "Passage 39"), (0.05, "Passage 0")]:
        page.mouse.click(ruler["x"] + ruler["width"] * position / 12, ruler["y"] + 10)
        assert_sync(page)
        row = page.locator(".transcript-entry").filter(has=page.get_by_text(text, exact=True))
        row_rect = row.bounding_box()
        assert row_rect["y"] >= pane_rect["y"] - 1
        assert row_rect["y"] + row_rect["height"] <= pane_rect["y"] + pane_rect["height"] + 1
        page.locator("#play").click()


def test_hover_overlap_seek_zoom_and_keyboard(page, viewer_result, tmp_path):
    viewer_result.transcript.append(
        TranscriptUnit(id="next", start=5.5, end=6, text="Following turn", speaker_id="speaker_01")
    )
    load(page, viewer_result, tmp_path)
    assert page.locator(".track").count() == 4  # Scenes, two speakers, unassigned.
    first = page.locator(".turn").first
    first.hover()
    assert page.locator("#tooltip").is_visible()
    assert page.locator("#tooltip-text").inner_text() == "こんにちは世界。"
    assert "Other speaker" not in page.locator("#tooltip-text").inner_text()
    bounds = page.locator("#tooltip").bounding_box()
    assert 0 <= bounds["x"] and bounds["x"] + bounds["width"] <= 1280
    page.locator(".turn").nth(2).hover()
    assert page.locator("#tooltip-text").inner_text() == "Other speaker"

    # The point timestamp remains visible and can be reached by keyboard.
    point = page.locator(".turn").last
    point.focus()
    assert page.locator("#tooltip-text").inner_text() == "Unassigned word"
    point.press("Enter")
    assert_sync(page)
    page.locator("#play").click()
    assert 10 <= page.locator("#video").evaluate("v => v.currentTime") < 10.8

    slider(page, "#zoom", 4)
    page.locator("#timeline-scroll").evaluate("el => el.scrollLeft = 900")
    body = page.locator(".track-body").nth(1)
    rect = body.bounding_box()
    target = rect["x"] + rect["width"] * 4 / 12
    page.mouse.click(target, rect["y"] + 22)
    assert_sync(page)
    page.locator("#play").click()
    assert page.locator("#video").evaluate("v => v.currentTime") == pytest.approx(5, abs=0.3)
    page.locator("#fit").click()
    assert page.locator("#timeline-scroll").evaluate("el => el.scrollLeft") == 0
    assert page.locator("#zoom").input_value() == "1"


def test_audio_switch_pause_rate_volume_buffering_and_end(page, viewer_result, tmp_path):
    load(page, viewer_result, tmp_path)
    seek_paused(page, 3)
    page.locator("#audio-mode").select_option("processed")
    page.wait_for_function("!document.querySelector('#processed-audio').seeking")
    assert page.locator("#video").evaluate("v => v.paused && v.muted")
    assert page.locator("#processed-audio").evaluate(
        "a => a.paused && Math.abs(a.currentTime - 3) < .1"
    )
    page.locator("#speed").select_option("1.5")
    slider(page, "#volume", 0.4)
    page.locator("#mute").click()
    page.locator("#play").click()
    assert_sync(page)
    assert page.locator("#processed-audio").evaluate(
        "a => a.muted && a.volume === 0.4 && a.playbackRate === 1.5"
    )
    page.locator("#mute").click()
    assert not page.locator("#processed-audio").evaluate("a => a.muted")
    for mode in ["original", "processed", "original", "processed"]:
        page.locator("#audio-mode").select_option(mode)
    assert_sync(page)
    assert page.locator("#original-audio").evaluate("a => a.paused && a.muted")
    assert page.locator("#video").evaluate("v => v.muted")
    # Buffering notifications preserve intent and do not strand the other stream.
    page.locator("#processed-audio").evaluate("a => a.dispatchEvent(new Event('waiting'))")
    assert_sync(page)
    # Simulate drift in a real running media element and await correction.
    page.locator("#processed-audio").evaluate(
        "a => a.currentTime = Math.max(0, a.currentTime - .7)"
    )
    assert_sync(page)
    # Browser/OS media controls may pause a media element independently.
    page.locator("#video").evaluate("v => v.pause()")
    page.wait_for_function("document.querySelector('#play').textContent === 'Play'")
    assert page.locator("#video").evaluate("v => v.paused")
    assert page.locator("#processed-audio").evaluate("a => a.paused")
    page.locator("#speed").select_option("1")
    seek_paused(page, 11.6)
    page.locator("#play").click()
    page.wait_for_function("document.querySelector('#play').textContent === 'Play'")
    assert page.locator("#processed-audio").evaluate("a => a.paused")
    assert page.locator("#video").evaluate("v => v.currentTime") >= 12.9


def test_native_original_and_actual_buffer_recovery(page, viewer_result, tmp_path, ffmpeg):
    from video_analyzer.media import probe
    from video_analyzer.storage import file_hash

    source = tmp_path / "single.mp4"
    ffmpeg("-i", viewer_result.media.path, "-map", "0:v:0", "-map", "0:a:1", "-c", "copy", source)
    viewer_result.media = probe(source)
    viewer_result.provenance["input_sha256"] = file_hash(source)
    load(page, viewer_result, tmp_path)
    assert page.locator("#original-audio").count() == 0
    page.locator("#play").click()
    page.wait_for_function("!document.querySelector('#video').paused")
    assert not page.locator("#video").evaluate("v => v.muted")
    page.locator("#audio-mode").select_option("processed")
    assert_sync(page, viewer_result.media.video_start)
    # Hold a media element below HAVE_FUTURE_DATA during a buffering event.
    page.locator("#processed-audio").evaluate("""a => {
        Object.defineProperty(a, 'readyState', {configurable: true, value: 2});
        a.dispatchEvent(new Event('waiting'));
    }""")
    assert page.locator("#video").evaluate("v => v.paused")
    assert page.locator("#processed-audio").evaluate("a => a.paused")
    page.locator("#processed-audio").evaluate("""a => {
        delete a.readyState; a.dispatchEvent(new Event('canplay'));
    }""")
    assert_sync(page, viewer_result.media.video_start)
    page.locator("#audio-mode").select_option("original")
    page.wait_for_function("!document.querySelector('#video').paused")
    assert not page.locator("#video").evaluate("v => v.muted")
    assert page.locator("#processed-audio").evaluate("a => a.paused && a.muted")
    # Rejected play promises must not leave the other media element running.
    page.locator("#processed-audio").evaluate(
        "a => a.play = () => Promise.reject(new Error('denied'))"
    )
    page.locator("#audio-mode").select_option("processed")
    page.wait_for_function("!document.querySelector('#media-error').hidden")
    assert "could not start" in page.locator("#media-error").inner_text()
    assert page.locator("#video").evaluate("v => v.paused")


def test_missing_video_reconnect_and_missing_audio(page, viewer_result, tmp_path):
    output = write_html(viewer_result, tmp_path / "report.html")
    source = Path(viewer_result.media.path)
    moved = source.rename(source.with_name("relocated.mp4"))
    page.goto(output.as_uri(), wait_until="domcontentloaded")
    page.wait_for_function("!document.querySelector('#media-error').hidden")
    assert "Locate" in page.locator("#media-error").inner_text()
    page.locator("#video-file").set_input_files(moved)
    page.wait_for_function("document.querySelector('#video').readyState >= 2")
    assert page.locator("#media-error").is_hidden()
    page.locator("#play").click()
    assert_sync(page)
    page.locator("#play").click()
    # Reload a selected unavailable sidecar: fail visibly and stop all sources.
    page.locator("#processed-audio").evaluate("a => { a.src = './missing-audio.wav'; a.load(); }")
    page.locator("#audio-mode").select_option("processed")
    page.wait_for_function("!document.querySelector('#media-error').hidden")
    assert ".media" in page.locator("#media-error").inner_text()
    assert page.locator("#video").evaluate("v => v.paused")
    assert page.locator("#original-audio").evaluate("a => a.paused")


def test_safe_text_and_disabled_features(page, viewer_result, tmp_path):
    viewer_result.report_language = "ja"
    viewer_result.provenance["configuration"].update(diarization=False, scene_description=False)
    payload = '</script><img src=x onerror="window.injected=true">'
    viewer_result.transcript = [TranscriptUnit(id="t1", start=1, end=3, text=payload)]
    load(page, viewer_result, tmp_path)
    assert page.locator(".track").count() == 2
    assert page.locator(".track-label").last.inner_text() == "文字起こし"
    page.locator(".turn").hover()
    assert page.locator("#tooltip-text").inner_text() == payload
    assert page.evaluate("window.injected") is None
    assert page.locator("img").count() == 0
    assert "無効" in page.locator("#scenes").inner_text()


def test_silent_video_without_audio(page, video, tmp_path):
    from video_analyzer.media import probe
    from video_analyzer.schema import AnalysisResult, Scene

    result = AnalysisResult(
        application_version="0.1.0",
        media=probe(video),
        report_language="en",
        provenance={"configuration": {"transcription": False, "diarization": False}},
        scenes=[Scene(id="scene_0001", start=0, end=4)],
    )
    load(page, result, tmp_path)
    assert page.locator("#audio-mode").is_disabled()
    assert page.locator("#volume").is_disabled()
    assert "disabled" in page.locator("#transcript").inner_text()
    # Timeline navigation also works when there are no transcript rows to scroll.
    page.locator(".scene-marker").click()
    page.wait_for_function("document.querySelector('#video').currentTime > .2")
    assert page.locator("#media-error").is_hidden()
