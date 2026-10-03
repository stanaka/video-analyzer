import sys
from types import SimpleNamespace

import pytest

from video_analyzer.errors import AnalysisError
from video_analyzer.workers import audio


@pytest.mark.parametrize("device,retry", [("auto", True), ("mps", False)])
def test_mps_load_retry_only_for_automatic_device(monkeypatch, device, retry):
    devices = []
    monkeypatch.setattr(audio, "select_device", lambda *a, **kw: "mps")
    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(mps=SimpleNamespace(empty_cache=lambda: None))
    )

    def load(backend):
        devices.append(backend.device)
        if backend.device == "mps":
            raise NotImplementedError("operation not implemented for MPS")

    monkeypatch.setattr(audio.AudioBackend, "_load", load)
    if retry:
        backend = audio.AudioBackend({"kind": "whisper", "device": device})
        assert devices == ["mps", "cpu"]
        assert backend.warnings and backend.device == "cpu"
    else:
        with pytest.raises(NotImplementedError):
            audio.AudioBackend({"kind": "whisper", "device": device})
        assert devices == ["mps"]


def test_mps_out_of_memory_is_not_hidden_by_retry(monkeypatch):
    monkeypatch.setattr(audio, "select_device", lambda *a, **kw: "mps")

    def fail(_):
        raise RuntimeError("MPS backend out of memory")

    monkeypatch.setattr(audio.AudioBackend, "_load", fail)
    with pytest.raises(RuntimeError, match="out of memory"):
        audio.AudioBackend({"kind": "whisper", "device": "auto"})


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
