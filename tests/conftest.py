import shutil
import subprocess

import pytest


@pytest.fixture
def ffmpeg():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("FFmpeg and ffprobe are required")

    def run(*args):
        return subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", *map(str, args)],
            check=True,
            capture_output=True,
            timeout=60,
        )

    return run


@pytest.fixture
def video(tmp_path, ffmpeg):
    path = tmp_path / "test video 日本語.mp4"
    ffmpeg(
        "-f",
        "lavfi",
        "-i",
        "color=red:s=160x120:r=10:d=2",
        "-f",
        "lavfi",
        "-i",
        "color=blue:s=160x120:r=10:d=2",
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0[v]",
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        path,
    )
    return path


@pytest.fixture
def viewer_result(tmp_path, ffmpeg):
    """A real two-track MP4; selected stereo audio begins before the video."""
    from video_analyzer.media import probe
    from video_analyzer.schema import (
        AnalysisResult,
        DiarizationTurn,
        Observation,
        Scene,
        Speaker,
        TranscriptUnit,
    )
    from video_analyzer.storage import file_hash

    path = tmp_path / "source #日本語 %.mp4"
    ffmpeg(
        "-itsoffset",
        "1",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=s=320x180:r=10:d=12",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo:d=14",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=440:sample_rate=48000:duration=14",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-ac",
        "2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        path,
    )
    return AnalysisResult(
        application_version="0.1.0",
        media=probe(path, audio_stream=1),
        report_language="en",
        provenance={
            "input_sha256": file_hash(path),
            "configuration": {
                "audio_stream": 1,
                "audio_normalization": "dynamic",
                "noise_reduction": True,
            },
        },
        speakers=[Speaker(id="speaker_01"), Speaker(id="speaker_02")],
        diarization=[
            DiarizationTurn(start=0.5, end=5.5, speaker_id="speaker_01"),
            DiarizationTurn(start=2, end=8, speaker_id="speaker_02"),
            DiarizationTurn(start=10, end=11, speaker_id="speaker_01"),
        ],
        transcript=[
            TranscriptUnit(id="t1", start=1, end=2, text="こんにちは", speaker_id="speaker_01"),
            TranscriptUnit(id="t2", start=2, end=2, text="世界。", speaker_id="speaker_01"),
            TranscriptUnit(id="t3", start=3, end=4, text="Other speaker", speaker_id="speaker_02"),
            TranscriptUnit(id="t4", start=9, end=9, text="Unassigned word"),
        ],
        scenes=[
            Scene(
                id="scene_0001",
                start=0,
                end=12,
                transcript_ids=["t1", "t2", "t3", "t4"],
                observations=[
                    Observation(start=0, end=12, description="Test pattern.", frame_timestamps=[1])
                ],
            )
        ],
        warnings=["Fixture warning"],
    )
