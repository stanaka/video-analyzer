import math
import sys
import time
import traceback

from ..errors import AnalysisError

ALIGN_LANGUAGES = {
    "Chinese",
    "English",
    "Cantonese",
    "French",
    "German",
    "Italian",
    "Japanese",
    "Korean",
    "Portuguese",
    "Russian",
    "Spanish",
}
LANGUAGE_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "yue": "Cantonese",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "pt": "Portuguese",
    "ru": "Russian",
    "es": "Spanish",
}


def select_device(requested: str, *, prefer_mps=True) -> str:
    import torch

    available = {
        "cpu": True,
        "cuda": torch.cuda.is_available(),
        "mps": torch.backends.mps.is_available(),
    }
    if requested != "auto":
        if not available.get(requested):
            raise AnalysisError(f"Requested device {requested} is unavailable")
        return requested
    if available["cuda"]:
        return "cuda"
    return "mps" if prefer_mps and available["mps"] else "cpu"


def mps_unsupported(exc: Exception) -> bool:
    message = str(exc).lower()
    return "mps" in message and any(
        word in message for word in ("not implemented", "not supported", "unsupported", "float64")
    )


def mps_out_of_memory(exc: Exception) -> bool:
    message = str(exc).lower()
    return "mps" in message and "out of memory" in message


class DiarizationProgress:
    """Report real pyannote progress without retaining large intermediate tensors."""

    def __init__(self, report):
        self.report = report
        self.step = None
        self.completed = None
        self.reported_at = 0.0

    def __call__(self, step_name, _artifact=None, *, completed=None, total=None, **_):
        now = time.monotonic()
        display = True
        if completed is not None and total is not None:
            # Pyannote may report a full batch beyond the final partial batch.
            completed = min(completed, total)
            if step_name == self.step and self.completed is not None:
                if completed <= self.completed:
                    return
                if completed < total and now - self.reported_at < 10:
                    display = False
            message = f"Diarization {step_name}: {completed}/{total}"
        else:
            message = f"Diarization {step_name}: complete"
            if step_name == "embeddings":
                message += "; clustering speakers"
        # Every advancing batch resets the watchdog, including when the user sets
        # a timeout below the log interval. Only terminal output is throttled.
        self.report(message, display=display)
        self.step, self.completed = step_name, completed
        if display:
            self.reported_at = now


class AudioBackend:
    def __init__(self, options):
        self.options = options
        self.kind = options["kind"]
        self.auto = options.get("device", "auto") == "auto"
        self.warnings = []
        self.device = select_device(
            options.get("device", "auto"), prefer_mps=self.kind != "diarization"
        )
        try:
            self._load()
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            self._retry_cpu(exc)

    def _retry_cpu(self, exc):
        out_of_memory = mps_out_of_memory(exc)
        if not (self.auto and self.device == "mps" and (out_of_memory or mps_unsupported(exc))):
            raise exc
        import gc

        import torch

        reason = "MPS ran out of memory" if out_of_memory else "MPS operation unsupported"
        print(f"{reason}; retrying ASR on CPU", file=sys.stderr, flush=True)
        # Failed model/generation frames retain tensors even after deleting the
        # backend's references. Release them before clearing MPS and loading CPU weights.
        traceback.clear_frames(exc.__traceback__)
        for name in ("model", "aligner", "processor"):
            if hasattr(self, name):
                delattr(self, name)
        gc.collect()
        torch.mps.empty_cache()
        self.device = "cpu"
        self._load()
        self.warnings.append(
            "ASR used CPU after MPS ran out of memory."
            if out_of_memory
            else "ASR used CPU after an unsupported MPS operation."
        )

    def _load(self):
        import torch

        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        if self.device == "mps":
            # Keep working-set pressure bounded on 24 GB Macs. Never disable the
            # allocator high-water mark to force a model onto an undersized device.
            torch.mps.set_per_process_memory_fraction(0.6)
        if self.kind == "diarization":
            import warnings

            # We always pass a waveform, so TorchCodec's optional file decoder is
            # unused (and may not link against the host's newer FFmpeg).
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="(?s).*torchcodec is not installed correctly.*",
                    category=UserWarning,
                )
                from pyannote.audio import Pipeline

            self.model = Pipeline.from_pretrained(self.options["model"])
            if self.model is None:
                raise AnalysisError("Cannot load the local pyannote pipeline")
            self.model.to(torch.device(self.device))
        elif self.kind == "whisper":
            from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

            self.model = (
                AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.options["model"],
                    local_files_only=True,
                    dtype=dtype,
                    attn_implementation="eager",  # word timestamps need attention weights
                )
                .to(self.device)
                .eval()
            )
            self.processor = AutoProcessor.from_pretrained(
                self.options["model"], local_files_only=True
            )
        elif self.kind == "qwen":
            from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner

            self.model = Qwen3ASRModel.from_pretrained(
                self.options["model"],
                local_files_only=True,
                dtype=dtype,
                low_cpu_mem_usage=True,
                device_map=self.device,
                max_inference_batch_size=1,
                max_new_tokens=1024,
            )
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                self.options["aligner"],
                local_files_only=True,
                dtype=dtype,
                device_map=self.device,
            )
        else:
            raise AnalysisError(f"Unknown audio backend: {self.kind}")

    def diarize(self, audio: str, num_speakers: int | None = None, *, progress=None):
        import soundfile as sf
        import torch

        waveform, sample_rate = sf.read(audio, dtype="float32")
        if progress:
            progress(f"Diarization on {self.device}: {len(waveform) / sample_rate:.1f}s of audio")
        output = self.model(
            {"waveform": torch.from_numpy(waveform).unsqueeze(0), "sample_rate": sample_rate},
            hook=DiarizationProgress(progress) if progress else None,
            **({"num_speakers": num_speakers} if num_speakers else {}),
        )

        def turns(annotation):
            return [
                {
                    "start": max(0.0, turn.start),
                    "end": min(len(waveform) / sample_rate, turn.end),
                    "speaker_id": speaker,
                }
                for turn, speaker in annotation
                if turn.end > turn.start
            ]

        return {
            "regular": turns(output.speaker_diarization),
            "exclusive": turns(output.exclusive_speaker_diarization),
            "device": self.device,
        }

    def transcribe(self, audio: str, start: float, end: float, language: str | None = None):
        import soundfile as sf

        with sf.SoundFile(audio) as file:
            file.seek(round(start * file.samplerate))
            waveform = file.read(round((end - start) * file.samplerate), dtype="float32")
            sample_rate = file.samplerate
        try:
            units = self._transcribe(waveform, sample_rate, language)
        except (RuntimeError, NotImplementedError, TypeError) as exc:
            self._retry_cpu(exc)
            units = self._transcribe(waveform, sample_rate, language)
        duration = len(waveform) / sample_rate
        for unit in units:
            if not all(math.isfinite(unit[key]) for key in ("start", "end")):
                raise AnalysisError("ASR returned a non-finite timestamp")
            unit["start"] = max(0.0, min(duration, unit["start"]))
            unit["end"] = max(unit["start"], min(duration, unit["end"]))
        return {"units": units, "warnings": self.warnings, "device": self.device}

    def _transcribe(self, waveform, sample_rate, language):
        if self.kind == "whisper":
            import torch

            kwargs = {
                "task": "transcribe",
                "do_sample": False,
                "condition_on_prev_tokens": False,
                "num_beams": 1,
                # Leave room for Whisper's decoder prompt within its 448 positions.
                "max_new_tokens": 440,
            }
            if language:
                kwargs["language"] = language
            # The generic pipeline imports TorchCodec even for numpy input.
            # Decode audio ourselves and use the pinned model/processor directly.
            inputs = self.processor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
                return_attention_mask=True,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
            from transformers.models.whisper.tokenization_whisper import TO_LANGUAGE_CODE

            with torch.inference_mode():
                if language:
                    code = TO_LANGUAGE_CODE.get(language.lower(), language.lower())
                    language_token = f"<|{code}|>"
                    language_id = self.model.generation_config.lang_to_id.get(language_token)
                    if language_id is None:
                        raise AnalysisError(f"Whisper does not support language {language!r}")
                else:
                    language_id = int(
                        self.model.detect_language(
                            input_features=inputs["input_features"],
                            generation_config=self.model.generation_config,
                        )[0].item()
                    )
                    language_token = next(
                        token
                        for token, value in self.model.generation_config.lang_to_id.items()
                        if value == language_id
                    )
                kwargs["language"] = language_token
                generated = self.model.generate(
                    **inputs,
                    **kwargs,
                    return_timestamps=True,
                    return_token_timestamps=True,
                    return_segments=True,
                )
            if "segments" in generated:
                times = torch.cat(
                    [segment["token_timestamps"] for segment in generated["segments"][0]]
                )[None, :]
            else:
                times = generated["token_timestamps"]
            # generate(return_segments=True) strips decoder prompt tokens. Restore
            # the language marker so Japanese is split on Unicode, not on spaces.
            tokens = torch.cat([torch.tensor([[language_id]]), generated["sequences"].cpu()], dim=1)
            times = torch.cat([torch.zeros((1, 1)), times.cpu()], dim=1)
            _, result = self.processor.tokenizer._decode_asr(
                [
                    {
                        "tokens": tokens.numpy(),
                        "token_timestamps": times.cpu().numpy(),
                    }
                ],
                return_timestamps="word",
                return_language=True,
                time_precision=self.processor.feature_extractor.chunk_length
                / self.model.config.max_source_positions,
            )
            units = []
            for chunk in result.get("chunks", []):
                start, end = chunk["timestamp"]
                if start is None or end is None:
                    raise AnalysisError(
                        "Whisper returned missing word timestamps; try a different ASR backend"
                    )
                units.append(
                    {
                        "start": float(start),
                        "end": float(end),
                        "text": chunk["text"],
                        "language": chunk.get("language", language),
                    }
                )
            return units
        requested = LANGUAGE_NAMES.get(language, language.title() if language else None)
        if requested and requested not in ALIGN_LANGUAGES:
            raise AnalysisError(f"Qwen alignment does not support {language}; use --asr whisper")
        result = self.model.transcribe(audio=(waveform, sample_rate), language=requested)[0]
        if not result.text.strip():
            return []
        detected = result.language
        if detected not in ALIGN_LANGUAGES:
            raise AnalysisError(
                f"Qwen alignment does not support detected language {detected!r}; use --asr whisper"
            )
        alignment = self.aligner.align(
            audio=(waveform, sample_rate), text=result.text, language=detected
        )[0]
        # Restore punctuation and whitespace discarded by forced alignment tokenization.
        import unicodedata

        def normalized(char):
            return unicodedata.normalize("NFKC", char).casefold()

        text = result.text
        normalized_text = ""
        offsets = []
        for index, char in enumerate(text):
            part = normalized(char)
            normalized_text += part
            offsets.extend([index] * len(part))
        cursor, original_cursor, units = 0, 0, []
        for item in alignment:
            token = normalized(item.text)
            pos = normalized_text.find(token, cursor)
            if pos < 0:
                raise AnalysisError(
                    "Qwen alignment text could not be reconciled with its transcript"
                )
            stop = offsets[pos + len(token) - 1] + 1
            units.append(
                {
                    "start": float(item.start_time),
                    "end": float(item.end_time),
                    "text": text[original_cursor:stop],
                    "language": detected,
                }
            )
            original_cursor, cursor = stop, pos + len(token)
        if units:
            units[-1]["text"] += text[original_cursor:]
        return units
