from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class AnalysisConfig(Record):
    asr: Literal["whisper", "qwen"] = "whisper"
    profile: Literal["auto", "mac", "cuda", "cpu"] = "auto"
    model_dir: Path = Path(".models")
    runtime_dir: Path = Path(".runtime")
    work_dir: Path | None = None
    resume: bool = False
    language: str | None = None
    report_language: Literal["ja", "en"] = "ja"
    num_speakers: int | None = Field(default=None, ge=1)
    audio_stream: int = Field(default=0, ge=0, description="Zero-based audio stream ordinal")
    scene_threshold: float = Field(default=3.0, gt=0)
    min_scene_seconds: float = Field(default=0.5, gt=0)
    audio_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    diarization_device: Literal["auto", "cpu", "cuda"] = "auto"
    vision_device: Literal["auto", "cpu", "cuda"] = "auto"
    window_seconds: float = Field(default=30.0, gt=0, le=30)
    frames_per_window: int = Field(default=8, ge=1, le=8)
    frame_long_edge: int = Field(default=768, ge=64, le=768)
    max_new_tokens: int = Field(default=448, ge=32, le=2048)
    worker_timeout: float = Field(default=3600, gt=0)


class Span(Record):
    start: float = Field(ge=0)
    end: float = Field(ge=0)

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("end must not precede start")
        return self


class MediaInfo(Record):
    path: str
    duration: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    video_stream_index: int
    video_start: float
    audio_stream_index: int | None = None
    audio_start: float | None = None


class Speaker(Record):
    id: str


class DiarizationTurn(Span):
    speaker_id: str


class TranscriptUnit(Span):
    id: str = ""
    text: str
    language: str | None = None
    speaker_id: str | None = None
    scene_id: str | None = None


class Observation(Span):
    description: str
    frame_timestamps: list[float]


class Scene(Span):
    id: str
    observations: list[Observation] = Field(default_factory=list)
    transcript_ids: list[str] = Field(default_factory=list)


class AnalysisResult(Record):
    schema_version: Literal["1.0"] = "1.0"
    application_version: str
    media: MediaInfo
    report_language: Literal["ja", "en"]
    provenance: dict
    speakers: list[Speaker] = Field(default_factory=list)
    diarization: list[DiarizationTurn] = Field(default_factory=list)
    exclusive_diarization: list[DiarizationTurn] = Field(default_factory=list)
    transcript: list[TranscriptUnit] = Field(default_factory=list)
    scenes: list[Scene]
    warnings: list[str] = Field(default_factory=list)
