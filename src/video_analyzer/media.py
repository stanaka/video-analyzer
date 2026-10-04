import json
import math
import re
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from .errors import AnalysisError
from .schema import MediaInfo, Scene


def run_media(args: list[str], *, timeout=600, return_stderr=False) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
        return result.stderr if return_stderr else result.stdout
    except FileNotFoundError as exc:
        raise AnalysisError(f"{args[0]} is required. Install FFmpeg including ffprobe.") from exc
    except subprocess.CalledProcessError as exc:
        raise AnalysisError(f"{args[0]} failed: {exc.stderr[-2000:]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise AnalysisError(f"{args[0]} exceeded its {timeout}s timeout") from exc


def probe(path: Path, audio_stream: int = 0) -> MediaInfo:
    if not path.is_file() or path.suffix.lower() != ".mp4":
        raise AnalysisError(f"Input must be an existing local MP4 file: {path}")
    data = json.loads(
        run_media(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            timeout=120,
        )
    )
    streams = data.get("streams", [])
    videos = [
        s
        for s in streams
        if s.get("codec_type") == "video" and not s.get("disposition", {}).get("attached_pic")
    ]
    audios = [s for s in streams if s.get("codec_type") == "audio"]
    if not videos:
        raise AnalysisError("The MP4 has no decodable video stream")
    video = videos[0]
    video_start = float(video.get("start_time", 0))
    duration = float(video.get("duration") or data.get("format", {}).get("duration", 0))
    if not math.isfinite(duration) or duration <= 0:
        raise AnalysisError("Cannot determine a positive video duration")
    if audios and audio_stream >= len(audios):
        raise AnalysisError(
            f"Audio stream {audio_stream} does not exist; found {len(audios)} streams"
        )
    if not audios and audio_stream:
        raise AnalysisError("An audio stream was selected but the video has no audio")
    audio = audios[audio_stream] if audios else None
    return MediaInfo(
        path=str(path.resolve()),
        duration=duration,
        width=video["width"],
        height=video["height"],
        video_stream_index=video["index"],
        video_start=video_start,
        audio_stream_index=audio["index"] if audio else None,
        audio_start=float(audio.get("start_time", 0)) if audio else None,
    )


def _peak_gain(args: list[str], filters: str, timeout: float) -> float:
    """Measure the same floating-point signal that will receive the final gain."""
    statistics = run_media(
        [
            *args,
            "-hide_banner",
            "-nostats",
            "-v",
            "info",
            "-af",
            f"{filters},astats=reset=0:measure_perchannel=none:measure_overall=Peak_level",
            "-c:a",
            "pcm_f32le",
            "-f",
            "null",
            "-",
        ],
        timeout=timeout,
        return_stderr=True,
    )
    peaks = re.findall(r"Peak level dB:\s*(\S+)", statistics)
    try:
        peak = float(peaks[0]) if len(peaks) == 1 else math.nan
    except ValueError:
        peak = math.nan
    if peak == -math.inf:
        return 0.0  # Digital silence must stay silent, without an infinite gain.
    if not math.isfinite(peak):
        raise AnalysisError("Cannot normalize audio: FFmpeg did not report a valid peak level")
    return -3.0 - peak


def extract_audio(
    media: MediaInfo,
    target: Path,
    *,
    audio_normalization: Literal["off", "peak", "dynamic"] = "off",
    noise_reduction: bool = False,
    progress: Callable[[str], None] | None = None,
) -> None:
    """Make optionally processed PCM aligned to video time zero, in bounded memory."""
    if audio_normalization not in ("off", "peak", "dynamic"):
        raise AnalysisError(f"Unknown audio normalization mode: {audio_normalization}")
    temporary = target.with_suffix(".partial.wav")
    args = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-copyts",
        "-i",
        media.path,
        "-map",
        f"0:{media.audio_stream_index}",
        "-vn",
    ]
    filters = [f"asetpts=PTS-({media.video_start:.9f})/TB"]
    processing = []
    if noise_reduction or audio_normalization != "off":
        # Downmix and resample in floating point before measuring or changing gain.
        filters.append("aformat=sample_fmts=fltp:channel_layouts=mono")
    filters.extend(
        ["aresample=16000:async=1:first_pts=0", "apad", f"atrim=end={media.duration:.9f}"]
    )
    if noise_reduction:
        processing.append("noise reduction")
        # afftdn delays samples by two 1/80-second hops (400 samples at 16 kHz).
        # Flush its tail before discarding that delay, including partial final hops.
        filters.extend(
            [
                "highpass=f=80",
                "apad=pad_len=800",
                "afftdn=nr=10:nf=-50:tn=1:gs=5",
                "atrim=start_sample=400",
                "asetpts=PTS-STARTPTS",
                f"atrim=end={media.duration:.9f}",
            ]
        )
    if audio_normalization != "off":
        processing.append(f"{audio_normalization} normalization")
    timeout = max(600, media.duration * 2)
    try:
        if audio_normalization == "peak":
            if progress:
                progress(
                    "Measuring audio peak" + (" after noise reduction" if noise_reduction else "")
                )
            gain = _peak_gain(args, ",".join(filters), timeout)
            filters.append(f"volume={gain:.9f}dB:precision=double")
        elif audio_normalization == "dynamic":
            # Keep one filter instance over the finite recording and let EOF drain
            # its lookahead. Its buffering does not shift the output samples.
            filters.append("dynaudnorm=f=500:g=31:p=0.707945784384:m=10:r=0:s=0:t=0.001")
        if progress:
            progress(
                "Extracting audio" + (f" with {' and '.join(processing)}" if processing else "")
            )
        run_media(
            [
                *args,
                "-v",
                "error",
                "-af",
                ",".join(filters),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            timeout=timeout,
        )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def detect_scenes(media: MediaInfo, threshold: float, minimum: float) -> list[Scene]:
    import av
    from scenedetect import SceneManager
    from scenedetect.backends.pyav import VideoStreamAv
    from scenedetect.detectors import AdaptiveDetector

    # PyAV supplies presentation timecodes in PySceneDetect 0.7, including VFR.
    av.logging.set_level(av.logging.ERROR)
    # Open the PTS-aware backend directly; never silently fall back to frame/FPS times.
    video = VideoStreamAv(media.path, suppress_output=True, threading_mode="NONE")
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector(adaptive_threshold=threshold, min_scene_len=minimum))
    try:
        count = manager.detect_scenes(video=video, show_progress=False)
        if count == 0:
            raise AnalysisError("No video frames could be decoded")
        detected = manager.get_scene_list(start_in_scene=True)
    finally:
        # VideoStreamAv exposes its container, but not a public close() in 0.7.1.
        container = getattr(video, "_container", None)
        if container is not None:
            container.close()
        video._io.close()
    starts = [0.0] + [
        float(start.seconds) for start, _ in detected if 0 < start.seconds < media.duration
    ]
    starts = sorted(set(starts))
    ends = starts[1:] + [media.duration]
    return [
        Scene(id=f"scene_{i + 1:04d}", start=start, end=end)
        for i, (start, end) in enumerate(zip(starts, ends, strict=True))
    ]


def windows(scene: Scene, seconds: float, frames: int):
    start = scene.start
    while start < scene.end - 1e-7:
        end = min(scene.end, start + seconds)
        count = max(1, min(frames, math.ceil(end - start)))
        targets = [start + (i + 0.5) * (end - start) / count for i in range(count)]
        yield start, end, targets
        start = end


def extract_frames(
    media: MediaInfo, targets: list[float], directory: Path, edge: int
) -> tuple[list[str], list[float]]:
    """Decode a bounded window, selecting actual frames using presentation times."""
    import av
    from PIL import Image

    av.logging.set_level(av.logging.ERROR)
    directory.mkdir(parents=True, exist_ok=True)
    paths, timestamps = [], []
    with av.open(media.path) as container:
        stream = next(s for s in container.streams.video if s.index == media.video_stream_index)
        container.seek(
            int((targets[0] + media.video_start) / stream.time_base), stream=stream, backward=True
        )
        pending = 0
        previous = None
        previous_time = 0.0

        def save(frame, timestamp):
            if timestamps and abs(timestamps[-1] - timestamp) < 1e-7:
                return
            image = frame.to_image()
            image.thumbnail((edge, edge), Image.Resampling.LANCZOS)
            destination = directory / f"frame-{len(paths):02d}.jpg"
            image.save(destination, quality=90)
            paths.append(str(destination.resolve()))
            timestamps.append(max(0.0, timestamp))

        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            timestamp = float(frame.pts * frame.time_base) - media.video_start
            while pending < len(targets) and timestamp > targets[pending]:
                save(
                    previous if previous is not None else frame,
                    previous_time if previous is not None else timestamp,
                )
                pending += 1
            if pending == len(targets):
                break
            previous, previous_time = frame, timestamp
        while pending < len(targets) and previous is not None:
            save(previous, previous_time)
            pending += 1
    if not paths:
        raise AnalysisError(f"No frames found near {targets[0]:.3f}s")
    return paths, timestamps


def prerequisites() -> dict:
    return {name: shutil.which(name) for name in ("ffmpeg", "ffprobe")}
