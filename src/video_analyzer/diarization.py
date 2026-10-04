"""Lightweight diarizer metadata shared by the CLI, pipeline, and NeMo worker."""

from dataclasses import dataclass

NEMOTRON_RUNTIME_REVISION = "1688cc3d6a9ade854f544987810c53f605dc86fc"
NEMOTRON_SOURCE_URL = (
    f"https://github.com/NVIDIA-NeMo/Speech/archive/{NEMOTRON_RUNTIME_REVISION}.tar.gz"
)
DIARIZATION_MODELS = {"pyannote": "diarization", "sortformer": "sortformer", "nemotron": "nemotron"}
DIARIZATION_RUNTIMES = {"pyannote": "torch", "sortformer": "sortformer", "nemotron": "nemotron"}


def selected_diarizers(backend):
    # Preserve the public meaning of "both" when new backends are added.
    return ("pyannote", "sortformer") if backend == "both" else (backend,)


@dataclass(frozen=True)
class NemoDiarizer:
    label: str
    checkpoint: str
    speakers: int
    frame_seconds: float
    speaker_cache: int
    runtime_revision: str | None = None

    @property
    def streaming_settings(self):
        # NeMo measures these in encoder frames (80 ms), independently of the
        # returned probabilities' frame stride.
        return {
            "chunk_len": 340,
            "chunk_right_context": 40,
            "fifo_len": 40,
            "spkcache_len": self.speaker_cache,
            "spkcache_update_period": 300,
        }

    @property
    def exclusive_method(self):
        return f"highest_posterior_active_speaker_per_{self.frame_seconds * 1000:g}ms_frame"

    @property
    def cache_settings(self):
        return {
            "runtime_revision": self.runtime_revision,
            "speakers": self.speakers,
            "frame_seconds": self.frame_seconds,
            "streaming_settings": self.streaming_settings,
            "exclusive_timeline_method": self.exclusive_method,
        }


NEMO_DIARIZERS = {
    "sortformer": NemoDiarizer(
        "Sortformer v2.1", "diar_streaming_sortformer_4spk-v2.1.nemo", 4, 0.08, 188
    ),
    "nemotron": NemoDiarizer(
        "Nemotron 3", "Nemotron-3-Diarization.nemo", 8, 0.01, 264, NEMOTRON_RUNTIME_REVISION
    ),
}
