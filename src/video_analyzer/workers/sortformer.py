"""NVIDIA's Sortformer and Nemotron diarizers, loaded in isolated workers."""

import contextlib
import importlib.metadata
import json
import math
import time
from collections import defaultdict
from itertools import pairwise
from pathlib import Path

from ..diarization import NEMO_DIARIZERS, NEMOTRON_SOURCE_URL
from ..errors import AnalysisError
from .audio import select_device

FRAME_SECONDS = NEMO_DIARIZERS["sortformer"].frame_seconds
STREAMING_SETTINGS = NEMO_DIARIZERS["sortformer"].streaming_settings
EXCLUSIVE_METHOD = NEMO_DIARIZERS["sortformer"].exclusive_method


def import_sortformer(backend="sortformer"):
    label = "Sortformer" if backend == "sortformer" else "Nemotron 3"
    try:
        from nemo.collections.asr.models import SortformerEncLabelModel
    except ImportError as exc:
        raise AnalysisError(
            f"Cannot import the {label} reference runtime. Run mise run setup:{backend}."
        ) from exc
    if backend == "nemotron":
        # Version 3.0.0 alone is insufficient: its release predates the model.
        # Verify both the APIs and the immutable archive used by our hash lock.
        try:
            direct_url = importlib.metadata.distribution("nemo-toolkit").read_text(
                "direct_url.json"
            )
            pinned = json.loads(direct_url or "{}").get("url") == NEMOTRON_SOURCE_URL
        except (importlib.metadata.PackageNotFoundError, ValueError, AttributeError) as exc:
            raise AnalysisError(
                "Cannot verify the NeMo source. Run mise run setup:nemotron."
            ) from exc
        if not pinned or not all(
            callable(getattr(SortformerEncLabelModel, name, None))
            for name in (
                "_resolve_output_resolution",
                "_check_streaming_parameters",
                "forward_streaming_step",
            )
        ):
            raise AnalysisError(
                "Nemotron 3 requires the pinned NeMo source with high-resolution output support. "
                "Run mise run setup:nemotron."
            )
    return SortformerEncLabelModel


def parse_segments(segments, duration, *, speakers=4, label="Sortformer"):
    """Parse NeMo's start/end/speaker strings without rounding away short turns."""
    result = []
    for segment in segments:
        try:
            start, end, speaker = segment.split()
            start, end = float(start), float(end)
            index = int(speaker.removeprefix("speaker_"))
            if (
                not speaker.startswith("speaker_")
                or not 0 <= index < speakers
                or not math.isfinite(start)
                or not math.isfinite(end)
                or end < start
            ):
                raise ValueError("Invalid interval or speaker")
        except (ValueError, AttributeError) as exc:
            raise AnalysisError(f"Invalid {label} segment: {segment!r}") from exc
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            result.append({"start": start, "end": end, "speaker_id": f"speaker_{index}"})
    return sorted(result, key=lambda turn: (turn["start"], turn["end"], turn["speaker_id"]))


def exclusive_turns(
    regular, probabilities, duration, *, speakers=4, frame_seconds=FRAME_SECONDS, label="Sortformer"
):
    """Project overlaps onto one speaker, preserving the union of regular speech.

    Split at both frame and turn boundaries, including sub-frame speech/silence.
    An event sweep avoids scanning every turn for each frame on long recordings.
    """
    import numpy as np

    probabilities = np.asarray(probabilities)
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != speakers
        or not np.isfinite(probabilities).all()
        or np.any((probabilities < 0) | (probabilities > 1))
    ):
        raise AnalysisError(
            f"Invalid {label} frame probabilities; expected finite [frames, {speakers}] values in [0, 1]"
        )
    if not regular:
        return []
    if not len(probabilities):
        raise AnalysisError(f"{label} returned speech without frame probabilities")
    if len(probabilities) * frame_seconds + 1e-9 < max(turn["end"] for turn in regular):
        raise AnalysisError(f"{label} frame probabilities do not cover the regular speech timeline")

    events = defaultdict(list)
    for turn in regular:
        index = int(turn["speaker_id"].removeprefix("speaker_"))
        events[turn["start"]].append((index, 1))
        events[turn["end"]].append((index, -1))
    boundaries = sorted(
        set(events)
        | {0.0, duration}
        | {i * frame_seconds for i in range(1, math.ceil(duration / frame_seconds))}
    )
    active = [0] * speakers
    result = []
    for start, end in pairwise(boundaries):
        for index, change in events.get(start, ()):
            active[index] += change
        candidates = [index for index, count in enumerate(active) if count > 0]
        if not candidates or end <= start:
            continue
        frame = min(int((start + end) / 2 / frame_seconds), len(probabilities) - 1)
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
        self.kind = options.get("kind", "sortformer")
        if self.kind not in NEMO_DIARIZERS:
            raise AnalysisError(f"Unknown NeMo diarization backend: {self.kind}")
        self.spec = NEMO_DIARIZERS[self.kind]
        self.label = "Sortformer" if self.kind == "sortformer" else self.spec.label
        self.frame_seconds = self.spec.frame_seconds
        settings = self.spec.streaming_settings
        self.device = select_device(options.get("device", "auto"), prefer_mps=False)
        if self.device not in ("cpu", "cuda"):
            raise AnalysisError(f"{self.label} supports CPU and CUDA devices")
        checkpoint = Path(options["model"]) / self.spec.checkpoint
        if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
            raise AnalysisError(
                f"Missing {self.label} checkpoint: {checkpoint}. "
                f"Run prepare-models --diarization-backend {self.kind}."
            )
        model_class = (
            import_sortformer() if self.kind == "sortformer" else import_sortformer(self.kind)
        )
        self.model = model_class.restore_from(
            restore_path=str(checkpoint),
            map_location=self.device,
            **({"strict": False} if self.kind == "nemotron" else {}),
        )
        if self.kind == "nemotron":
            self.validate_nemotron()
            self.model.async_streaming = False
        self.model.eval()
        self.model.to(self.device)
        self.model.streaming_mode = True
        for name, value in settings.items():
            setattr(self.model.sortformer_modules, name, value)
        if self.kind == "nemotron":
            self.model._check_streaming_parameters()
        else:
            self.model.sortformer_modules._check_streaming_parameters()
        self.runtime_details = {
            "diarization_backend": self.kind,
            "inference_settings": {
                **settings,
                "batch_size": 1,
                "num_workers": 0,
                "frame_seconds": self.frame_seconds,
                "effective_spkcache_update_period": max(
                    settings["chunk_len"], settings["spkcache_update_period"]
                ),
                "postprocessing": "NeMo defaults: onset=offset=0.5, no padding or duration filtering",
            },
            "exclusive_timeline_method": self.spec.exclusive_method,
            "exclusive_timeline_settings": {
                "active_regions": "regular_timeline",
                "tie_break": "lowest_model_speaker_index",
                "preserve_silence": True,
                "clamp_to_audio_duration": True,
            },
        }
        if self.kind == "nemotron":
            self.runtime_details["nemo_source_revision"] = self.spec.runtime_revision
            self.runtime_details["inference_settings"].update(
                encoder_frame_seconds=0.08, max_speakers=self.spec.speakers, async_streaming=False
            )

    def validate_nemotron(self):
        try:
            frame_seconds = (
                float(self.model._cfg.preprocessor.window_stride)
                * self.model.output_subsampling_factor
            )
            valid = (
                self.model._cfg.max_num_of_spks == self.spec.speakers
                and self.model.high_resolution is True
                and math.isclose(frame_seconds, self.spec.frame_seconds, rel_tol=0, abs_tol=1e-9)
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise AnalysisError(
                "Cannot validate Nemotron 3 speaker capacity and output stride"
            ) from exc
        if not valid:
            raise AnalysisError(
                "Nemotron 3 requires an eight-speaker checkpoint with 10 ms output frames"
            )
        self.frame_seconds = frame_seconds

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
            completed = min(duration, result[1].shape[1] * self.frame_seconds)
            now = time.monotonic()
            display = steps == 1 or completed >= duration or now - reported_at >= 5
            if display:
                reported_at = now
            report(
                f"{self.label} streaming step {steps}: {completed:.1f}/{duration:.1f}s",
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
                f"{self.label} cannot honor --num-speakers; omit it or select pyannote"
            )
        waveform, sample_rate = sf.read(audio, dtype="float32")
        if sample_rate != 16000 or waveform.ndim != 1:
            raise AnalysisError(f"{self.label} requires 16 kHz mono audio")
        if not np.isfinite(waveform).all():
            raise AnalysisError(f"{self.label} audio contains non-finite samples")
        duration = len(waveform) / sample_rate
        if progress:
            progress(
                f"{self.spec.label} on {self.device}: {duration:.1f}s of audio (up to {self.spec.speakers} speakers)"
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
            raise AnalysisError(f"{self.label} returned an unexpected number of recordings")
        probabilities = predictions[0].detach().cpu().numpy()
        if probabilities.ndim == 3 and probabilities.shape[0] == 1:
            probabilities = probabilities[0]
        regular = parse_segments(
            segments[0], duration, speakers=self.spec.speakers, label=self.label
        )
        exclusive = exclusive_turns(
            regular,
            probabilities,
            duration,
            speakers=self.spec.speakers,
            frame_seconds=self.frame_seconds,
            label=self.label,
        )
        if progress:
            progress(f"{self.label} diarization: complete")
        return {"regular": regular, "exclusive": exclusive, "device": self.device}
