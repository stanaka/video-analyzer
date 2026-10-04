import itertools
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

import video_analyzer.media as audio_media
from video_analyzer.errors import AnalysisError
from video_analyzer.media import extract_audio, probe
from video_analyzer.schema import MediaInfo

PROCESSING = list(itertools.product(("off", "peak", "dynamic"), (False, True)))
RATE = 16000
TARGET_PEAK = 10 ** (-3 / 20)


def read_pcm(path):
    with wave.open(str(path)) as stream:
        assert stream.getframerate() == RATE
        assert stream.getnchannels() == 1
        assert stream.getsampwidth() == 2
        return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2") / 32768


def rms(samples):
    return np.sqrt(np.mean(samples**2))


@pytest.mark.integration
@pytest.mark.parametrize("normalization,noise_reduction", PROCESSING)
@pytest.mark.parametrize("count", [1, 199, 16123])
def test_segment_processing_matches_processing_an_isolated_clip(
    pcm_audio, tmp_path, normalization, noise_reduction, count
):
    # Loud excluded samples must not affect the segment's normalization or filters.
    samples = 0.9 * np.sin(2 * np.pi * 440 * np.arange(3 * RATE) / RATE)
    clip = np.zeros(count)
    clip[0] = clip[-1] = 0.05
    if count > 1000:
        clip[count // 2] = 0.05
    samples[RATE : RATE + count] = clip
    selected, isolated = tmp_path / "selected.wav", tmp_path / "isolated.wav"
    options = {"audio_normalization": normalization, "noise_reduction": noise_reduction}
    extract_audio(pcm_audio(samples), selected, start_time=1, end_time=1 + count / RATE, **options)
    extract_audio(pcm_audio(clip), isolated, **options)
    np.testing.assert_array_equal(read_pcm(selected), read_pcm(isolated))


@pytest.mark.integration
@pytest.mark.parametrize("normalization,noise_reduction", PROCESSING)
@pytest.mark.parametrize("start,end", [(0.5, 2.5), (0.1, 0.5), (3, 3.5)])
def test_segment_alignment_and_ranges_outside_audio(
    ffmpeg, tmp_path, normalization, noise_reduction, start, end
):
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
        "sine=frequency=440:sample_rate=16000:duration=1",
        "-c:v",
        "libx264",
        "-c:a",
        "alac",
        path,
    )
    output = tmp_path / "segment.wav"
    extract_audio(
        probe(path),
        output,
        start_time=start,
        end_time=end,
        audio_normalization=normalization,
        noise_reduction=noise_reduction,
    )
    samples = read_pcm(output)
    assert len(samples) == round((end - start) * RATE)
    if start == 0.5:
        assert not np.any(samples[: int(0.45 * RATE)])
        assert np.max(np.abs(samples[int(0.8 * RATE) : int(1.2 * RATE)])) > 0.01
        assert not np.any(samples[int(1.8 * RATE) :])
    else:
        assert not np.any(samples)


def media_info(path, duration=4):
    return MediaInfo(
        path=str(path),
        duration=duration,
        width=160,
        height=120,
        video_stream_index=1,
        video_start=0,
        audio_stream_index=0,
        audio_start=0,
    )


@pytest.fixture
def pcm_audio(tmp_path, ffmpeg):
    """Use lossless PCM sources to measure processing independently of AAC artifacts."""
    numbers = itertools.count()

    def create(samples, rate=RATE):
        path = tmp_path / f"source-{next(numbers)}.wav"
        with wave.open(str(path), "wb") as stream:
            stream.setnchannels(1 if samples.ndim == 1 else samples.shape[1])
            stream.setsampwidth(4)
            stream.setframerate(rate)
            values = np.rint(np.clip(samples, -1, 1 - 2**-31) * 2**31)
            stream.writeframes(values.astype("<i4").tobytes())
        return media_info(path, len(samples) / rate)

    return create


@pytest.mark.integration
def test_disabled_processing_matches_original_extraction(pcm_audio, tmp_path, ffmpeg):
    samples = np.random.default_rng(1).normal(0, 0.04, (48000, 2))
    media = pcm_audio(samples, rate=48000)
    original, actual = tmp_path / "original.wav", tmp_path / "actual.wav"
    ffmpeg(
        "-copyts",
        "-i",
        media.path,
        "-map",
        "0:0",
        "-vn",
        "-af",
        "asetpts=PTS-(0.000000000)/TB,aresample=16000:async=1:first_pts=0,apad,atrim=end=1.000000000",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        original,
    )
    extract_audio(media, actual)
    np.testing.assert_array_equal(read_pcm(actual), read_pcm(original))


@pytest.mark.integration
@pytest.mark.parametrize("noise_reduction", [False, True])
def test_peak_normalization_preserves_relative_levels(pcm_audio, tmp_path, noise_reduction):
    samples = np.sin(2 * np.pi * 440 * np.arange(4 * RATE) / RATE)
    samples[: 2 * RATE] *= 0.08
    samples[2 * RATE :] *= 0.4
    media = pcm_audio(samples)
    baseline, normalized = tmp_path / "baseline.wav", tmp_path / "normalized.wav"
    extract_audio(media, baseline, noise_reduction=noise_reduction)
    extract_audio(media, normalized, audio_normalization="peak", noise_reduction=noise_reduction)
    before, after = read_pcm(baseline), read_pcm(normalized)
    assert len(after) == len(samples)
    assert 20 * np.log10(np.max(np.abs(after))) == pytest.approx(-3, abs=0.1)
    gains = [
        rms(after[start:end]) / rms(before[start:end])
        for start, end in [(RATE // 2, RATE), (3 * RATE, 7 * RATE // 2)]
    ]
    assert gains[0] == pytest.approx(gains[1], rel=0.002)


@pytest.mark.integration
@pytest.mark.parametrize("amplitude", [0.000001, 0.01, 0.99])
def test_peak_normalization_handles_very_quiet_and_loud_audio(pcm_audio, tmp_path, amplitude):
    samples = amplitude * np.sin(2 * np.pi * 440 * np.arange(RATE) / RATE)
    media = pcm_audio(samples)
    target = tmp_path / "normalized.wav"
    extract_audio(media, target, audio_normalization="peak")
    result = read_pcm(target)
    assert 20 * np.log10(np.max(np.abs(result))) == pytest.approx(-3, abs=0.1)


@pytest.mark.integration
@pytest.mark.parametrize("normalization,noise_reduction", PROCESSING)
def test_processing_preserves_silence(pcm_audio, tmp_path, normalization, noise_reduction):
    samples = np.zeros(16123)
    media = pcm_audio(samples)
    target = tmp_path / "silence.wav"
    extract_audio(media, target, audio_normalization=normalization, noise_reduction=noise_reduction)
    np.testing.assert_array_equal(read_pcm(target), samples)


@pytest.mark.integration
@pytest.mark.parametrize("noise_reduction", [False, True])
def test_dynamic_normalization_balances_passages(pcm_audio, tmp_path, noise_reduction):
    samples = np.zeros(100 * RATE)
    for start, end, amplitude in [(5, 35, 0.05), (40, 70, 0.5), (75, 95, 0.0005)]:
        samples[start * RATE : end * RATE] = amplitude * np.sin(
            2 * np.pi * 440 * np.arange((end - start) * RATE) / RATE
        )
    media = pcm_audio(samples)
    target = tmp_path / "dynamic.wav"
    extract_audio(media, target, audio_normalization="dynamic", noise_reduction=noise_reduction)
    result = read_pcm(target)
    assert len(result) == len(samples)
    quiet, loud = slice(15 * RATE, 25 * RATE), slice(50 * RATE, 60 * RATE)
    before_difference = 20 * np.log10(rms(samples[loud]) / rms(samples[quiet]))
    after_difference = 20 * np.log10(rms(result[loud]) / rms(result[quiet]))
    assert before_difference - after_difference > 10
    assert 1 < rms(result[quiet]) / rms(samples[quiet]) <= 10
    assert np.max(np.abs(result)) <= TARGET_PEAK + 1 / 32768
    # No new discontinuities at frame or passage boundaries; this sine's maximum
    # sample-to-sample step at the target level is approximately 0.123.
    assert np.max(np.abs(np.diff(result))) < 0.14
    assert not np.any(result[: 4 * RATE]) and not np.any(result[96 * RATE :])
    assert rms(result[82 * RATE : 88 * RATE]) <= rms(samples[82 * RATE : 88 * RATE]) * 1.1


@pytest.mark.integration
def test_dynamic_normalization_attenuates_loud_input(pcm_audio, tmp_path):
    samples = 0.99 * np.sin(2 * np.pi * 440 * np.arange(20 * RATE) / RATE)
    target = tmp_path / "dynamic.wav"
    extract_audio(pcm_audio(samples), target, audio_normalization="dynamic")
    result = read_pcm(target)
    assert np.max(np.abs(result)) <= TARGET_PEAK + 1 / 32768
    assert rms(result[8 * RATE : 12 * RATE]) < rms(samples[8 * RATE : 12 * RATE])


@pytest.mark.integration
def test_noise_reduction_reduces_noise_and_preserves_foreground(pcm_audio, tmp_path):
    samples = np.random.default_rng(11).normal(0, 0.006, 4 * RATE)
    samples[RATE : 3 * RATE] += 0.2 * np.sin(2 * np.pi * 440 * np.arange(2 * RATE) / RATE)
    target = tmp_path / "denoised.wav"
    extract_audio(pcm_audio(samples), target, noise_reduction=True)
    result = read_pcm(target)
    for start, end in [(8000, 14000), (56000, 62000)]:
        assert 20 * np.log10(rms(result[start:end]) / rms(samples[start:end])) < -1
    assert rms(result[24000:40000]) == pytest.approx(rms(samples[24000:40000]), rel=0.05)


@pytest.mark.integration
@pytest.mark.parametrize("normalization,noise_reduction", PROCESSING)
@pytest.mark.parametrize("count", [1, 199, 16123, 48001])
def test_processing_preserves_short_inputs_and_boundary_signals(
    pcm_audio, tmp_path, normalization, noise_reduction, count
):
    samples = np.zeros(count)
    samples[0] = samples[-1] = 0.2
    if count > 1000:
        samples[count // 2] = 0.2
    target = tmp_path / "boundaries.wav"
    extract_audio(
        pcm_audio(samples),
        target,
        audio_normalization=normalization,
        noise_reduction=noise_reduction,
    )
    result = read_pcm(target)
    assert len(result) == count
    assert abs(result[0]) > 0.02 and abs(result[-1]) > 0.02
    if count > 1000:
        start = count // 2 - 100
        assert start + np.argmax(np.abs(result[start : start + 200])) == count // 2


@pytest.mark.integration
@pytest.mark.parametrize("normalization,noise_reduction", PROCESSING)
def test_early_stereo_audio_and_stream_selection(
    pcm_audio, ffmpeg, tmp_path, normalization, noise_reduction
):
    silence = pcm_audio(np.zeros(3 * RATE))
    rate = 48000
    samples = np.zeros((3 * rate, 2))
    for timestamp in (0.4, 1.4, 2.7):
        samples[round(timestamp * rate)] = [0.15, 0.25]
    stereo = pcm_audio(samples, rate=rate)
    path = tmp_path / "early.mp4"
    ffmpeg(
        "-itsoffset",
        "1",
        "-f",
        "lavfi",
        "-i",
        "color=red:s=160x120:r=10:d=2",
        "-i",
        silence.path,
        "-i",
        stereo.path,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-c:a",
        "alac",
        path,
    )
    media = probe(path, audio_stream=1)
    assert media.video_start == 1 and media.audio_start == 0
    assert media.audio_stream_index == 2
    target = tmp_path / "selected.wav"
    extract_audio(media, target, audio_normalization=normalization, noise_reduction=noise_reduction)
    result = read_pcm(target)
    assert len(result) == 2 * RATE
    assert not np.any(result[: RATE // 4])
    for timestamp in (0.4, 1.7):
        center = round(timestamp * RATE)
        start = center - 100
        assert start + np.argmax(np.abs(result[start : start + 200])) == center
        assert abs(result[center]) > 0.01
    segment = tmp_path / "selected-segment.wav"
    extract_audio(
        media,
        segment,
        start_time=0.2,
        end_time=1.9,
        audio_normalization=normalization,
        noise_reduction=noise_reduction,
    )
    selected = read_pcm(segment)
    assert len(selected) == round(1.7 * RATE)
    for timestamp in (0.2, 1.5):
        center = round(timestamp * RATE)
        assert center - 100 + np.argmax(np.abs(selected[center - 100 : center + 100])) == center


@pytest.mark.parametrize("measurement", ["", "nan", "inf", "invalid", "-20\nPeak level dB: -21"])
def test_invalid_peak_measurement_preserves_output_and_cleans_partial(
    monkeypatch, tmp_path, measurement
):
    target = tmp_path / "audio.wav"
    target.write_bytes(b"previous audio")
    partial = target.with_suffix(".partial.wav")
    partial.write_bytes(b"interrupted audio")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["return_stderr"] is True
        return f"[astats] Peak level dB: {measurement}"

    monkeypatch.setattr(audio_media, "run_media", run)
    with pytest.raises(AnalysisError, match="valid peak level"):
        extract_audio(media_info("input.mp4"), target, audio_normalization="peak")
    assert len(calls) == 1
    assert target.read_bytes() == b"previous audio"
    assert not partial.exists()


@pytest.mark.parametrize("failure", ["process", "timeout", "missing"])
@pytest.mark.parametrize(
    "normalization,failed_call", [("off", 1), ("peak", 1), ("peak", 2), ("dynamic", 1)]
)
def test_ffmpeg_failure_is_atomic(monkeypatch, tmp_path, failure, normalization, failed_call):
    target = tmp_path / "audio.wav"
    target.write_bytes(b"previous audio")
    partial = target.with_suffix(".partial.wav")
    calls = []

    def run(args, **kwargs):
        calls.append(args)
        assert kwargs["timeout"] == 7200
        if len(calls) < failed_call:
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="Peak level dB: -20")
        partial.write_bytes(b"incomplete")
        if failure == "process":
            raise subprocess.CalledProcessError(1, args, stderr="filter failed")
        if failure == "timeout":
            raise subprocess.TimeoutExpired(args, kwargs["timeout"])
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(audio_media.subprocess, "run", run)
    with pytest.raises(AnalysisError, match="ffmpeg"):
        extract_audio(
            media_info("input.mp4", duration=3600),
            target,
            audio_normalization=normalization,
            noise_reduction=True,
        )
    assert len(calls) == failed_call
    assert target.read_bytes() == b"previous audio"
    assert not partial.exists()


@pytest.mark.parametrize("normalization,expected_calls", [("off", 1), ("peak", 2), ("dynamic", 1)])
def test_processing_progress_and_streaming_passes(
    monkeypatch, tmp_path, normalization, expected_calls
):
    calls, progress = [], []

    def run(args, **kwargs):
        calls.append(args)
        if kwargs.get("return_stderr"):
            return "Peak level dB: -20"
        Path(args[-1]).write_bytes(b"completed audio")
        return ""

    monkeypatch.setattr(audio_media, "run_media", run)
    extract_audio(
        media_info("input.mp4"),
        tmp_path / "audio.wav",
        audio_normalization=normalization,
        noise_reduction=True,
        progress=progress.append,
    )
    assert len(calls) == expected_calls
    assert "noise reduction" in progress[-1]
    if normalization != "off":
        assert f"{normalization} normalization" in progress[-1]
    assert any("Measuring audio peak" in message for message in progress) == (
        normalization == "peak"
    )
