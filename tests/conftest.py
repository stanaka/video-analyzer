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
