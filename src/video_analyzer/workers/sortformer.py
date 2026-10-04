"""NVIDIA's reference Sortformer v2.1, loaded only in its optional worker."""

import contextlib
import math
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from ..errors import AnalysisError
from ..models import SORTFORMER_CHECKPOINT
from .audio import select_device

FRAME_SECONDS = 0.08
STREAMING_SETTINGS = {
    "chunk_len": 340,
    "chunk_right_context": 40,
    "fifo_len": 40,
    "spkcache_len": 188,
    "spkcache_update_period": 300,
}
EXCLUSIVE_METHOD = "highest_posterior_active_speaker_per_80ms_frame"


def import_sortformer():
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:
        raise AnalysisError(
            "Cannot import the Sortformer reference runtime. Run mise run setup:sortformer."
        ) from exc
    return SortformerEncLabelModel


def parse_segments(segments, duration):
    """Parse NeMo's start/end/speaker strings without rounding away short turns."""
    result = []
    for segment in segments:
        try:
            start, end, speaker = segment.split()
            start, end = float(start), float(end)
            index = int(speaker.removeprefix("speaker_"))
            if (
                not speaker.startswith("speaker_")
                or not 0 <= index < 4
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end < start
            ):
                raise ValueError("Invalid interval or speaker")
        except (ValueError, AttributeError) as exc:
            raise AnalysisError(f"Invalid Sortformer segment: {segment!r}") from exc
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            result.append({"start": start, "end": end, "speaker_id": f"speaker_{index}"})
    return sorted(result, key=lambda turn: (turn["start"], turn["end"], turn["speaker_id"]))


def exclusive_turns(regular, probabilities, duration):
    """Project overlaps onto one speaker, preserving the union of regular speech.

    Split at both frame and turn boundaries, including sub-frame speech/silence.
    An event sweep avoids scanning every turn for each frame on long recordings.
    """
    import numpy as np

    probabilities = np.asarray(probabilities)
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != 4
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise AnalysisError(
            "Invalid Sortformer frame probabilities; expected finite [frames, 4] values in [0, 1]"
        )
    if not regular:
        return []
    if not len(probabilities):
        raise AnalysisError("Sortformer returned speech without frame probabilities")
    if len(probabilities) * FRAME_SECONDS + 1e-9 < max(turn["end"] for turn in regular):
        raise AnalysisError(
            "Sortformer frame probabilities do not cover the regular speech timeline"
        )

    events = defaultdict(list)
    for turn in regular:
        index = int(turn["speaker_id"].removeprefix("speaker_"))
        events[turn["start"]].append((index, 1))
        events[turn["end"]].append((index, -1))
    boundaries = sorted(
        set(events)
        | {0.0, duration}
        | {i * FRAME_SECONDS for i in range(1, math.ceil(duration / FRAME_SECONDS))}
    )
    active = [0] * 4
    result = []
    for start, end in pairwise(boundaries):
        for index, change in events.get(start, ()):
            active[index] += change
        candidates = [index for index, count in enumerate(active) if count > 0]
        if not candidates or end <= start:
            continue
        frame = min(int((start + end) / 2 / FRAME_SECONDS), len(probabilities) - 1)
        # candidates are ordered by speaker index, so max breaks ties consistently.
        winner = max(candidates, key=lambda index: probabilities[frame, index])
        speaker = f"speaker_{winner}"
        if result and result[-1]["speaker_id"] == speaker and result[-1]["end"] == start:
            result[-1]["end"] = end
        else:
            result.append({"start": start, "end": end, "speaker_id": speaker})
    return result


class SortformerBackend:
    def __init__(self, options):
        self.device = select_device(options.get("device", "auto"), prefer_mps=False)
        if self.device not in ("cpu", "cuda"):
            raise AnalysisError("Sortformer supports CPU and CUDA devices")
        checkpoint = Path(options["model"]) / SORTFORMER_CHECKPOINT
        if not checkpoint.is_file():
            raise AnalysisError(
                f"Missing Sortformer checkpoint: {checkpoint}. "
                "Run prepare-models --diarization-backend sortformer."
            )
        self.model = import_sortformer().restore_from(
            restore_path=str(checkpoint), map_location=self.device
        )
        self.model.eval()
        self.model.to(self.device)
        self.model.streaming_mode = True
        for name, value in STREAMING_SETTINGS.items():
            setattr(self.model.sortformer_modules, name, value)
        self.model.sortformer_modules._check_streaming_parameters()
        self.runtime_details = {
            "diarization_backend": "sortformer",
            "inference_settings": {
                **STREAMING_SETTINGS,
                "batch_size": 1,
                "num_workers": 0,
                "frame_seconds": FRAME_SECONDS,
                "effective_spkcache_update_period": max(
                    STREAMING_SETTINGS["chunk_len"], STREAMING_SETTINGS["spkcache_update_period"]
                ),
                "postprocessing": "NeMo defaults: onset=offset=0.5, no padding or duration filtering",
            },
            "exclusive_timeline_method": EXCLUSIVE_METHOD,
            "exclusive_timeline_settings": {
                "active_regions": "regular_timeline",
                "tie_break": "lowest_model_speaker_index",
                "preserve_silence": True,
                "clamp_to_audio_duration": True,
            },
        }

    @contextlib.contextmanager
    def streaming_progress(self, report, duration):
        if report is None:
            yield
            return
        # Wrap only this instance. NeMo still owns the complete streaming state and
        # FIFO/cache updates; a completed step is real work, not a timer heartbeat.
        original = self.model.forward_streaming_step
        was_overridden = "forward_streaming_step" in vars(self.model)
        reported_at = 0.0
        steps = 0

        def step(*args, **kwargs):
            nonlocal reported_at, steps
            result = original(*args, **kwargs)
            steps += 1
            completed = min(duration, result[1].shape[1] * FRAME_SECONDS)
            now = time.monotonic()
            display = steps == 1 or completed >= duration or now - reported_at >= 5
            if display:
                reported_at = now
            report(
                f"Sortformer streaming step {steps}: {completed:.1f}/{duration:.1f}s",
                display=display,
            )
            return result

        self.model.forward_streaming_step = step
        try:
            yield
        finally:
            if was_overridden:
                self.model.forward_streaming_step = original
            else:
                del self.model.forward_streaming_step

    def diarize(self, audio: str, num_speakers=None, *, progress=None):
        import numpy as np
        import soundfile as sf
        import torch

        if num_speakers is not None:
            raise AnalysisError(
                "Sortformer cannot honor --num-speakers; omit it or select pyannote"
            )
        waveform, sample_rate = sf.read(audio, dtype="float32")
        if sample_rate != 16000 or waveform.ndim != 1:
            raise AnalysisError("Sortformer requires 16 kHz mono audio")
        if not np.isfinite(waveform).all():
            raise AnalysisError("Sortformer audio contains non-finite samples")
        duration = len(waveform) / sample_rate
        if progress:
            progress(
                f"Sortformer v2.1 on {self.device}: {duration:.1f}s of audio (up to four speakers)"
            )
        if not len(waveform):
            return {"regular": [], "exclusive": [], "device": self.device}
        # Very short clips need enough samples for NeMo's STFT reflect padding.
        if len(waveform) < 1600:
            waveform = np.pad(waveform, (0, 1600 - len(waveform)))
        with torch.inference_mode(), self.streaming_progress(progress, duration):
            segments, predictions = self.model.diarize(
                audio=[waveform],
                sample_rate=sample_rate,
                batch_size=1,
                num_workers=0,
                include_tensor_outputs=True,
                verbose=False,
            )
        if len(segments) != 1 or len(predictions) != 1:
            raise AnalysisError("Sortformer returned an unexpected number of recordings")
        probabilities = predictions[0].detach().cpu().numpy()
        if probabilities.ndim == 3 and probabilities.shape[0] == 1:
            probabilities = probabilities[0]
        regular = parse_segments(segments[0], duration)
        exclusive = exclusive_turns(regular, probabilities, duration)
        if progress:
            progress("Sortformer diarization: complete")
        return {"regular": regular, "exclusive": exclusive, "device": self.device}
