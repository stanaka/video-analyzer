import wave

import av
import numpy as np
import pytest

from video_analyzer.media import detect_scenes, extract_audio, extract_frames, probe, windows
from video_analyzer.schema import Scene

pytestmark = pytest.mark.integration


def test_cuts_and_frame_sampling(video, tmp_path):
    media = probe(video)
    assert media.audio_stream_index is None
    scenes = detect_scenes(media, 3, 0.5)
    assert len(scenes) == 2
    assert scenes[1].start == pytest.approx(2, abs=0.11)
    assert scenes[-1].end == 4
    for scene in scenes:
        start, end, targets = next(windows(scene, 30, 8))
        paths, pts = extract_frames(media, targets, tmp_path / scene.id, 768)
        assert paths
        assert all(start <= time < end for time in pts)


def test_audio_pts_offset_and_no_cuts(tmp_path, ffmpeg):
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
        "sine=frequency=440:sample_rate=16000:duration=2",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        path,
    )
    media = probe(path)
    assert media.audio_start > 0.8
    assert len(detect_scenes(media, 3, 0.5)) == 1
    wav = tmp_path / "audio.wav"
    extract_audio(media, wav)
    with wave.open(str(wav)) as stream:
        assert stream.getframerate() == 16000
        assert stream.getnchannels() == 1
        data = np.frombuffer(stream.readframes(stream.getnframes()), dtype=np.int16)
    assert len(data) / 16000 == pytest.approx(4, abs=0.01)
    assert np.max(np.abs(data[:12800])) == 0
    assert np.max(np.abs(data[19000:24000])) > 500
    assert np.max(np.abs(data[-8000:])) == 0


def test_vfr_cuts_use_presentation_times(tmp_path, ffmpeg):
    path = tmp_path / "vfr.mp4"
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
        "[0:v][1:v]concat=n=2:v=1:a=0,setpts='if(lt(N,20),PTS,PTS+1/TB)'[v]",
        "-map",
        "[v]",
        "-fps_mode",
        "vfr",
        "-c:v",
        "libx264",
        path,
    )
    with av.open(str(path)) as container:
        pts = [float(frame.pts * frame.time_base) for frame in container.decode(video=0)]
    assert max(b - a for a, b in zip(pts, pts[1:], strict=False)) > 1
    scenes = detect_scenes(probe(path), 3, 0.5)
    assert len(scenes) == 2
    assert scenes[1].start == pytest.approx(pts[20], abs=0.01)


def test_long_scene_windows_preserve_boundaries():
    scene = Scene(id="s", start=5, end=96)
    result = list(windows(scene, 30, 8))
    assert len(result) == 4
    assert result[0][0] == 5 and result[-1][1] == 96
    assert all(len(targets) <= 8 and end - start <= 30 for start, end, targets in result)
