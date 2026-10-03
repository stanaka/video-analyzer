import sys
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from video_analyzer.errors import AnalysisError
from video_analyzer.workers import audio

MPS_ERRORS = [
    (NotImplementedError, "operation not implemented for MPS"),
    (RuntimeError, "MPS backend out of memory (MPS allocated: 10.63 GiB)"),
]


class ModelState:
    pass


@pytest.fixture
def audio_file(monkeypatch):
    waveform = [0.0] * 20
    file = MagicMock(samplerate=10)
    file.read.return_value = waveform
    soundfile = MagicMock()
    soundfile.SoundFile.return_value.__enter__.return_value = file
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    return waveform


@pytest.mark.parametrize("device,retry", [("auto", True), ("mps", False)])
@pytest.mark.parametrize("error_type,message", MPS_ERRORS)
def test_mps_load_retry_only_for_automatic_device(
    monkeypatch, capsys, device, retry, error_type, message
):
    devices = []
    resources = []
    cleared = []
    monkeypatch.setattr(audio, "select_device", lambda *a, **kw: "mps")

    def empty_cache():
        assert resources and all(ref() is None for ref in resources)
        cleared.append(True)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(empty_cache=empty_cache))
    )

    def load(backend):
        devices.append(backend.device)
        if backend.device == "mps":
            for name in ("model", "aligner", "processor"):
                # Keep a local reference as well as the backend attribute, as
                # third-party loading code does when allocating model weights.
                resource = ModelState()
                setattr(backend, name, resource)
                resources.append(weakref.ref(resource))
            raise error_type(message)
        assert cleared == [True]

    monkeypatch.setattr(audio.AudioBackend, "_load", load)
    if retry:
        backend = audio.AudioBackend({"kind": "whisper", "device": device})
        assert devices == ["mps", "cpu"]
        assert backend.warnings and backend.device == "cpu"
        if "out of memory" in message:
            assert "out of memory" in backend.warnings[0]
        assert "retrying ASR on CPU" in capsys.readouterr().err
    else:
        with pytest.raises(error_type) as caught:
            audio.AudioBackend({"kind": "whisper", "device": device})
        assert str(caught.value) == message
        assert devices == ["mps"]
        assert cleared == []


@pytest.mark.parametrize(
    "device,message",
    [
        ("cpu", "MPS backend out of memory"),
        ("cuda", "CUDA out of memory"),
        ("mps", "CPU allocator out of memory"),
        ("mps", "unexpected model failure"),
    ],
)
@pytest.mark.parametrize("stage", ["load", "transcribe"])
def test_unrelated_errors_are_not_hidden_by_retry(monkeypatch, audio_file, device, message, stage):
    monkeypatch.setattr(audio, "select_device", lambda *a, **kw: device)
    error = RuntimeError(message)

    def fail(*_):
        raise error

    monkeypatch.setattr(audio.AudioBackend, "_load", fail if stage == "load" else lambda _: None)
    monkeypatch.setattr(audio.AudioBackend, "_transcribe", fail)
    with pytest.raises(RuntimeError) as caught:
        backend = audio.AudioBackend({"kind": "whisper", "device": "auto"})
        backend.transcribe("audio.wav", 0, 2)
    assert caught.value is error


@pytest.mark.parametrize("device,retry", [("auto", True), ("mps", False)])
@pytest.mark.parametrize("error_type,message", MPS_ERRORS)
def test_mps_transcription_retry_releases_tensors_and_stays_on_cpu(
    monkeypatch, audio_file, device, retry, error_type, message
):
    loads, calls, resources, cleared = [], [], [], []
    monkeypatch.setattr(audio, "select_device", lambda *a, **kw: "mps")

    def empty_cache():
        assert all(ref() is None for ref in resources)
        cleared.append(True)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(empty_cache=empty_cache))
    )

    def load(backend):
        loads.append(backend.device)
        if backend.device == "cpu":
            assert cleared == [True]
        backend.model = ModelState()

    def transcribe(backend, waveform, sample_rate, language):
        calls.append((backend.device, waveform, sample_rate, language))
        # Exercise an OOM after an earlier chunk has already succeeded.
        if len(calls) == 2:
            tensor = ModelState()
            resources.extend([weakref.ref(tensor), weakref.ref(backend.model)])
            raise error_type(message)
        return [{"start": -0.1, "end": 2.1, "text": "hello", "language": language}]

    monkeypatch.setattr(audio.AudioBackend, "_load", load)
    monkeypatch.setattr(audio.AudioBackend, "_transcribe", transcribe)
    backend = audio.AudioBackend({"kind": "whisper", "device": device})
    assert backend.transcribe("audio.wav", 0, 2, "en")["device"] == "mps"
    if retry:
        result = backend.transcribe("audio.wav", 2, 4, "en")
        assert result == {
            "units": [{"start": 0.0, "end": 2.0, "text": "hello", "language": "en"}],
            "device": "cpu",
            "warnings": backend.warnings,
        }
        assert len(result["warnings"]) == 1
        if "out of memory" in message:
            assert "out of memory" in result["warnings"][0]
        assert backend.transcribe("audio.wav", 4, 6, "en") == result
        assert loads == ["mps", "cpu"]
        assert [call[0] for call in calls] == ["mps", "mps", "cpu", "cpu"]
        assert all(call[1:] == (audio_file, 10, "en") for call in calls)
    else:
        with pytest.raises(error_type) as caught:
            backend.transcribe("audio.wav", 2, 4, "en")
        assert str(caught.value) == message
        assert loads == ["mps"]
        assert [call[0] for call in calls] == ["mps", "mps"]
        assert cleared == []


@pytest.mark.parametrize("stage", ["load", "transcribe"])
def test_cpu_retry_failure_propagates_without_another_retry(monkeypatch, audio_file, stage):
    loads, calls = [], []
    monkeypatch.setattr(audio, "select_device", lambda *a, **kw: "mps")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: None))
    )

    def load(backend):
        loads.append(backend.device)
        if backend.device == "cpu" and stage == "load":
            raise RuntimeError("CPU load failed")

    def transcribe(backend, *_):
        calls.append(backend.device)
        if backend.device == "mps":
            raise RuntimeError("MPS backend out of memory")
        raise RuntimeError("CPU transcribe failed")

    monkeypatch.setattr(audio.AudioBackend, "_load", load)
    monkeypatch.setattr(audio.AudioBackend, "_transcribe", transcribe)
    backend = audio.AudioBackend({"kind": "whisper", "device": "auto"})
    with pytest.raises(RuntimeError, match=f"CPU {stage} failed"):
        backend.transcribe("audio.wav", 0, 2)
    assert loads == ["mps", "cpu"]
    assert calls == (["mps"] if stage == "load" else ["mps", "cpu"])


def test_qwen_unsupported_alignment_language_recommends_whisper():
    backend = audio.AudioBackend.__new__(audio.AudioBackend)
    backend.kind = "qwen"
    with pytest.raises(AnalysisError, match="use --asr whisper"):
        backend._transcribe(None, 16000, "pl")


def test_qwen_restores_japanese_punctuation_from_original_transcript():
    backend = audio.AudioBackend.__new__(audio.AudioBackend)
    backend.kind = "qwen"
    backend.model = SimpleNamespace(
        transcribe=lambda **kw: [SimpleNamespace(text="今日は、晴れです。", language="Japanese")]
    )
    backend.aligner = SimpleNamespace(
        align=lambda **kw: [
            [
                SimpleNamespace(text=text, start_time=i, end_time=i + 1)
                for i, text in enumerate(["今日", "は", "晴れ", "です"])
            ]
        ]
    )
    units = backend._transcribe(None, 16000, "ja")
    assert "".join(unit["text"] for unit in units) == "今日は、晴れです。"


def test_diarization_reports_real_progress_and_keeps_speaker_options(monkeypatch):
    import numpy as np

    waveform = np.zeros(32000, dtype=np.float32)
    soundfile = SimpleNamespace(read=lambda *a, **kw: (waveform, 16000))
    monkeypatch.setitem(sys.modules, "soundfile", soundfile)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(from_numpy=MagicMock()))
    backend = audio.AudioBackend.__new__(audio.AudioBackend)
    backend.device = "cpu"

    def model(file, *, num_speakers, hook):
        assert file["sample_rate"] == 16000
        assert num_speakers == 2
        hook("segmentation", completed=0, total=1)
        hook("segmentation", completed=1, total=1)
        hook("embeddings", None, completed=0, total=1, file=file)
        hook("embeddings", None, completed=1, total=1, file=file)
        hook("embeddings", object(), file=file)
        annotation = [(SimpleNamespace(start=0.2, end=1.8), "speaker")]
        return SimpleNamespace(
            speaker_diarization=annotation, exclusive_speaker_diarization=annotation
        )

    backend.model = model
    messages = []
    result = backend.diarize(
        "input.wav", num_speakers=2, progress=lambda message, **_: messages.append(message)
    )
    assert messages[0] == "Diarization on cpu: 2.0s of audio"
    assert messages[1] == "Diarization segmentation: 0/1"
    assert messages[2] == "Diarization segmentation: 1/1"
    assert messages[-1] == "Diarization embeddings: complete; clustering speakers"
    assert (
        result["regular"]
        == result["exclusive"]
        == [{"start": 0.2, "end": 1.8, "speaker_id": "speaker"}]
    )


def test_diarization_progress_throttles_advancing_batches_without_hiding_completion(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(audio.time, "monotonic", lambda: now[0])
    messages = []
    visible = []

    def report(message, *, display):
        messages.append(message)
        if display:
            visible.append(message)

    hook = audio.DiarizationProgress(report)
    hook("segmentation", completed=0, total=100)
    now[0] = 1.0
    hook("segmentation", completed=20, total=100)
    assert len(visible) == 1
    assert messages[-1] == "Diarization segmentation: 20/100"
    now[0] = 10.0
    hook("segmentation", completed=40, total=100)
    assert messages[-1] == "Diarization segmentation: 40/100"
    now[0] = 30.0
    hook("segmentation", completed=40, total=100)
    assert len(messages) == 3  # no new progress must not keep a stalled worker alive
    hook("segmentation", completed=120, total=100)
    assert messages[-1] == "Diarization segmentation: 100/100"
    hook("embeddings", completed=0, total=10)
    hook("embeddings", completed=10, total=10)
    assert messages[-2:] == ["Diarization embeddings: 0/10", "Diarization embeddings: 10/10"]
    assert visible[-3:] == messages[-3:]
