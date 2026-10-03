import json
import math
import shutil
import subprocess
from pathlib import Path

from .errors import AnalysisError
from .schema import MediaInfo, Scene


def run_media(args: list[str], *, timeout=600) -> str:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=True)
        return result.stdout
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


def extract_audio(media: MediaInfo, target: Path) -> None:
    """Make PCM whose sample zero is video time zero, padding/trimming PTS offsets."""
    temporary = target.with_suffix(".partial.wav")
    try:
        run_media(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-copyts",
                "-i",
                media.path,
                "-map",
                f"0:{media.audio_stream_index}",
                "-vn",
                "-af",
                (
                    f"asetpts=PTS-({media.video_start:.9f})/TB,"
                    "aresample=16000:async=1:first_pts=0,apad,"
                    f"atrim=end={media.duration:.9f}"
                ),
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_s16le",
                str(temporary),
            ],
            timeout=max(600, media.duration * 2),
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
