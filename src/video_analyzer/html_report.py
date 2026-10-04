"""Offline HTML rendering and transactional publication of playback assets."""

import html
import json
import math
import os
import re
import shutil
import tempfile
from importlib.resources import files
from pathlib import Path
from urllib.parse import quote

from .errors import AnalysisError
from .media import extract_audio, extract_original_audio, probe_streams
from .schema import AnalysisResult
from .storage import atomic_text, file_hash


def _relative_url(path: Path, output: Path) -> str:
    # Encode even ':' and '#' so file names cannot become schemes/fragments.
    return "./" + quote(Path(os.path.relpath(path, output.parent)).as_posix(), safe="/")


def render_html(result: AnalysisResult, playback: dict) -> str:
    """Render using already prepared URLs; no filesystem writes or media work."""
    assets = files("video_analyzer").joinpath("viewer")
    data = json.dumps(
        {"result": result.model_dump(mode="json"), "playback": playback},
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    # application/json script elements still recognize an HTML closing script tag.
    for character in "<>&\u2028\u2029":
        data = data.replace(character, f"\\u{ord(character):04x}")
    template = assets.joinpath("index.html").read_text(encoding="utf-8")
    replacements = {
        "LANG": result.report_language,
        "TITLE": html.escape(Path(result.media.path).name),
        "CSS": assets.joinpath("viewer.css").read_text(encoding="utf-8"),
        "JS": assets.joinpath("viewer.js").read_text(encoding="utf-8"),
        "DATA": data,
    }
    # One pass: user text containing a template marker must remain ordinary text.
    return re.sub(r"@@(LANG|TITLE|CSS|JS|DATA)@@", lambda m: replacements[m[1]], template)


def _validate_source(result: AnalysisResult, source: Path) -> int:
    data = probe_streams(source)
    expected_hash = result.provenance.get("input_sha256")
    if expected_hash and file_hash(source) != expected_hash:
        raise AnalysisError("Video does not match the analysis input SHA-256")
    streams = {stream["index"]: stream for stream in data.get("streams", [])}
    video = streams.get(result.media.video_stream_index, {})
    duration = float(video.get("duration") or data.get("format", {}).get("duration", 0))
    if (
        video.get("codec_type") != "video"
        or video.get("width") != result.media.width
        or video.get("height") != result.media.height
        or not math.isclose(duration, result.media.duration, abs_tol=0.05)
        or not math.isclose(
            float(video.get("start_time", 0)), result.media.video_start, abs_tol=0.001
        )
    ):
        raise AnalysisError("Video metadata does not match the analysis; use the original MP4")
    if result.media.audio_stream_index is not None:
        audio = streams.get(result.media.audio_stream_index, {})
        if audio.get("codec_type") != "audio" or not math.isclose(
            float(audio.get("start_time", 0)), result.media.audio_start or 0, abs_tol=0.001
        ):
            raise AnalysisError("The analyzed audio track is missing or has different timestamps")
    return sum(s.get("codec_type") == "audio" for s in streams.values())


def write_html(
    result: AnalysisResult, output: str | Path, *, video_path: str | Path | None = None
) -> Path:
    """Write a local HTML viewer and audio sidecars, without loading model runtimes.

    The source MP4 is referenced, never copied. Audio is regenerated from recorded
    settings, independent of analysis checkpoints. Existing asset generations are
    retained so an already-open report remains usable after another export.
    """
    output = Path(output).expanduser().resolve()
    source = (
        Path(video_path if video_path is not None else result.media.path).expanduser().resolve()
    )
    if output == source:
        raise AnalysisError("Output must not overwrite the input video")
    if not source.is_file():
        raise AnalysisError(f"Source video is missing: {source}. Use --video to locate it.")
    audio_count = _validate_source(result, source)
    settings = result.provenance.get("configuration", {})
    if not isinstance(settings, dict):
        raise AnalysisError("Invalid configuration in the analysis result")
    normalization = settings.get("audio_normalization", "off")
    denoise = settings.get("noise_reduction", False)
    if normalization not in ("off", "peak", "dynamic") or not isinstance(denoise, bool):
        raise AnalysisError("Invalid audio preprocessing settings in the analysis result")
    media = result.media.model_copy(update={"path": str(source)})
    has_audio = media.audio_stream_index is not None
    original_sidecar = has_audio and audio_count > 1
    processed_sidecar = has_audio and (normalization != "off" or denoise)
    playback = {
        "video": _relative_url(source, output),
        # MP4 edit lists retain the delay before the first video sample in the
        # browser media timeline. Sidecars instead start at analysis time zero.
        "video_start": media.video_start,
        "original": None,
        "processed": None,
        "normalization": normalization,
        "denoise": denoise,
    }
    generation = None
    document = None
    try:
        if original_sidecar or processed_sidecar:
            directory = output.with_name(output.stem + ".media")
            directory.mkdir(parents=True, exist_ok=True)
            generation = Path(tempfile.mkdtemp(prefix="audio-", dir=directory))
            for name, needed in (("original", original_sidecar), ("processed", processed_sidecar)):
                if not needed:
                    continue
                audio_path = generation / f"{name}.wav"
                if name == "original":
                    extract_original_audio(media, audio_path)
                else:
                    extract_audio(
                        media,
                        audio_path,
                        audio_normalization=normalization,
                        noise_reduction=denoise,
                    )
                playback[name] = _relative_url(audio_path, output)
        document = render_html(result, playback)
        atomic_text(output, document)
    except BaseException:
        if generation is not None:
            # An interrupt may arrive just after os.replace publishes the HTML.
            # Never remove assets that the committed report already references.
            try:
                committed = document is not None and output.read_text(encoding="utf-8") == document
            except FileNotFoundError:
                committed = False
            except (OSError, UnicodeError):
                committed = True  # If publication cannot be checked, retain assets.
            if not committed:
                shutil.rmtree(generation)
        raise
    return output
