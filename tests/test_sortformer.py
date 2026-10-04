import contextlib
import sys
from itertools import pairwise
from types import SimpleNamespace

import numpy as np
import pytest

from video_analyzer.errors import AnalysisError
from video_analyzer.models import SORTFORMER_CHECKPOINT
from video_analyzer.workers import sortformer


def turn(start, end, speaker=0):
    return {"start": start, "end": end, "speaker_id": f"speaker_{speaker}"}


def test_exclusive_overlap_ties_silence_and_partial_frames():
    regular = [turn(0.02, 0.15), turn(0.06, 0.20, 1), turn(0.23, 0.27, 2)]
    # An inactive speaker with a higher probability must never create speech.
    probabilities = [
        [0.6, 0.8, 0.1, 0.99],
        [0.7, 0.7, 0.1, 0.99],
        [0.9, 0.6, 0.8, 0.99],
        [0.1, 0.1, 0.6, 0.99],
    ]
    assert sortformer.exclusive_turns(regular, probabilities, 0.27) == [
        turn(0.02, 0.06),
        turn(0.06, 0.08, 1),
        turn(0.08, 0.15),
        turn(0.15, 0.20, 1),
        turn(0.23, 0.27, 2),
    ]


def test_exclusive_union_matches_regular_speech_without_overlaps():
    rng = np.random.default_rng(42)
    raw = sorted(rng.uniform(0, 3.17, size=120).reshape(-1, 2).tolist())
    regular = [turn(min(a, b), max(a, b), i % 4) for i, (a, b) in enumerate(raw)]
    probabilities = rng.random((40, 4))
    exclusive = sortformer.exclusive_turns(regular, probabilities, 3.17)
    assert all(0 <= t["start"] < t["end"] <= 3.17 for t in exclusive)
    assert all(a["end"] <= b["start"] for a, b in pairwise(exclusive))
    # Check every elementary interval, not a few sample timestamps.
    boundaries = sorted({0.0, 3.17} | {t[k] for t in regular + exclusive for k in ("start", "end")})
    for start, end in pairwise(boundaries):
        midpoint = (start + end) / 2
        active = {t["speaker_id"] for t in regular if t["start"] <= midpoint < t["end"]}
        chosen = [t["speaker_id"] for t in exclusive if t["start"] <= midpoint < t["end"]]
        assert bool(active) == bool(chosen)
        assert not chosen or (len(chosen) == 1 and chosen[0] in active)


def test_silence_and_short_segments():
    assert sortformer.exclusive_turns([], np.zeros((2, 4)), 0.1) == []
    assert sortformer.exclusive_turns([], np.zeros((0, 4)), 0) == []
    assert sortformer.exclusive_turns([turn(0, 0.01, 3)], [[0, 0, 0, 1]], 0.01) == [
        turn(0, 0.01, 3)
    ]
    assert sortformer.parse_segments(["-0.01 0.3 speaker_0", "0.3 0.5 speaker_1"], 0.27) == [
        turn(0, 0.27)
    ]


@pytest.mark.parametrize(
    "segment", ["bad", "0 nan speaker_0", "1 0 speaker_1", "0 1 speaker_4", "0 1 person"]
)
def test_invalid_segments_fail(segment):
    with pytest.raises(AnalysisError, match="Invalid Sortformer segment"):
        sortformer.parse_segments([segment], 1)


@pytest.mark.parametrize(
    "probabilities", [[], [[0, 0]], [[0, 0, 0, np.nan]], [[0, 0, 0, 1.1]], np.empty((0, 4))]
)
def test_invalid_probabilities_fail(probabilities):
    with pytest.raises(AnalysisError, match="probabilities"):
        sortformer.exclusive_turns([turn(0, 0.1)], probabilities, 0.1)


def test_missing_tail_probabilities_cannot_be_invented():
    with pytest.raises(AnalysisError, match="do not cover"):
        sortformer.exclusive_turns([turn(0, 1)], [[0.7, 0, 0, 0]], 1)


@pytest.fixture
def reference_model(monkeypatch, tmp_path):
    """A small stateful reference double; no torch or NeMo in normal tests."""

    class Tensor:
        def __init__(self, values):
            self.values = values

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Model:
        calls = 0
        state_values = []
        fail = False
        segments = ["0.02 0.28 speaker_0", "0.1 0.5 speaker_1"]

        def __init__(self):
            self.sortformer_modules = SimpleNamespace(_check_streaming_parameters=lambda: None)

        def eval(self):
            pass

        def to(self, device):
            assert device == "cpu"

        def forward_streaming_step(self, state, frame_count):
            state["step"] += 1
            self.state_values.append(state["step"])
            return state, SimpleNamespace(shape=(1, frame_count, 4))

        def diarize(self, **kwargs):
            self.calls += 1
            self.arguments = kwargs
            state = {"step": 0}
            for frame_count in (2, 4, 5):
                state, _ = self.forward_streaming_step(state, frame_count)
                if self.fail:
                    raise RuntimeError("failed streaming step")
            return [self.segments], [Tensor(np.full((1, 5, 4), 0.75))]

    model = Model()
    restored = []

    def restore(**kwargs):
        restored.append(kwargs)
        return model

    monkeypatch.setattr(
        sortformer, "import_sortformer", lambda: SimpleNamespace(restore_from=restore)
    )
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: False),
            backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: True)),
            inference_mode=contextlib.nullcontext,
        ),
    )
    waveform = np.zeros(6400, dtype=np.float32)
    monkeypatch.setitem(
        sys.modules, "soundfile", SimpleNamespace(read=lambda *a, **kw: (waveform, 16000))
    )
    (tmp_path / SORTFORMER_CHECKPOINT).write_bytes(b"checkpoint")
    backend = sortformer.SortformerBackend({"model": str(tmp_path), "device": "auto"})
    return backend, model, restored


def test_reference_restore_settings_state_and_progress(reference_model):
    backend, model, restored = reference_model
    assert backend.device == "cpu"  # Never selects the available Mac GPU.
    assert restored[0]["map_location"] == "cpu"
    assert restored[0]["restore_path"].endswith(SORTFORMER_CHECKPOINT)
    assert model.streaming_mode is True
    assert {
        k: getattr(model.sortformer_modules, k) for k in sortformer.STREAMING_SETTINGS
    } == sortformer.STREAMING_SETTINGS
    messages = []
    result = backend.diarize("audio.wav", progress=lambda message, **kw: messages.append(message))
    assert model.calls == 1 and model.state_values == [1, 2, 3]
    assert len([message for message in messages if "streaming step" in message]) == 3
    assert "forward_streaming_step" not in vars(model)
    assert model.arguments["batch_size"] == 1 and model.arguments["num_workers"] == 0
    assert model.arguments["include_tensor_outputs"] is True
    assert result["regular"] == [turn(0.02, 0.28), turn(0.1, 0.4, 1)]
    assert result["exclusive"] == [turn(0.02, 0.28), turn(0.28, 0.4, 1)]
    assert backend.runtime_details["exclusive_timeline_method"] == sortformer.EXCLUSIVE_METHOD


@pytest.mark.parametrize("instance_override", [False, True])
def test_progress_restored_on_failure(reference_model, instance_override):
    backend, model, _ = reference_model
    original = model.forward_streaming_step
    if instance_override:
        model.forward_streaming_step = original
    model.fail = True
    with pytest.raises(RuntimeError, match="failed streaming"):
        backend.diarize("audio.wav", progress=lambda *a, **kw: None)
    assert model.forward_streaming_step == original
    assert ("forward_streaming_step" in vars(model)) == instance_override


@pytest.mark.parametrize("samples", [0, 1, 1599])
def test_short_waveforms_are_padded_and_results_clamped(reference_model, monkeypatch, samples):
    backend, model, _ = reference_model
    monkeypatch.setattr(
        sys.modules["soundfile"], "read", lambda *a, **kw: (np.zeros(samples), 16000)
    )
    result = backend.diarize("short.wav")
    if not samples:
        assert model.calls == 0
        assert result["regular"] == result["exclusive"] == []
    else:
        assert len(model.arguments["audio"][0]) == 1600
        assert all(
            0 <= t["start"] < t["end"] <= samples / 16000
            for t in result["regular"] + result["exclusive"]
        )


def test_speaker_count_and_unavailable_device_fail(reference_model, tmp_path):
    backend, model, _ = reference_model
    with pytest.raises(AnalysisError, match="omit it or select pyannote"):
        backend.diarize("audio.wav", num_speakers=2)
    assert model.calls == 0
    with pytest.raises(AnalysisError, match="cuda is unavailable"):
        sortformer.SortformerBackend({"model": str(tmp_path), "device": "cuda"})
    with pytest.raises(AnalysisError, match="Missing Sortformer checkpoint"):
        sortformer.SortformerBackend({"model": str(tmp_path / "missing"), "device": "cpu"})
