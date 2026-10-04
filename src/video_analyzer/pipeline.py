import contextlib
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .diarization import DIARIZATION_MODELS, DIARIZATION_RUNTIMES, NEMO_DIARIZERS
from .errors import AnalysisError
from .media import detect_scenes, extract_audio, extract_frames, probe, resolve_range, windows
from .models import local_models, required_models, resolve_profile
from .schema import AnalysisConfig, AnalysisResult, Observation, Scene
from .storage import Checkpoints, file_hash
from .timeline import assign_units, audio_chunks, normalize_speakers, owned_units, speech_chunks
from .worker_client import Worker, worker_python


def progress(message: str):
    print(message, file=sys.stderr, flush=True)


def analyze_video(path: str | Path, config: AnalysisConfig) -> AnalysisResult:
    started = time.monotonic()
    path = Path(path).expanduser().resolve()
    profile = resolve_profile(config.profile)
    media = probe(path, config.audio_stream)
    range_start, range_end = resolve_range(media, config.start_time, config.end_time)
    duration = range_end - range_start
    analysis_range = {"start": range_start, "end": range_end}
    range_options = (
        {"start_time": range_start, "end_time": range_end}
        if range_start or range_end != media.duration
        else {}
    )
    has_audio = media.audio_stream_index is not None
    audio_enabled = config.transcription or config.diarization
    required = required_models(
        profile,
        config.asr,
        audio=has_audio,
        transcription=config.transcription,
        diarization=config.diarization,
        diarization_backend=config.diarization_backend,
        scene_description=config.scene_description,
    )
    models = local_models(config.model_dir, required) if required else {}
    vision_kind = "molmo-mlx" if profile == "mac" else "molmo-torch"
    diarization_kind = DIARIZATION_MODELS[config.diarization_backend]
    diarization_runtime = DIARIZATION_RUNTIMES[config.diarization_backend]
    if (
        has_audio
        and (config.transcription or (config.diarization and diarization_runtime == "torch"))
    ) or (config.scene_description and profile != "mac"):
        worker_python(config.runtime_dir, "torch")
    if has_audio and config.diarization and diarization_runtime != "torch":
        worker_python(config.runtime_dir, diarization_runtime)
    if config.scene_description and profile == "mac":
        worker_python(config.runtime_dir, "mlx")
        if config.vision_device != "auto":
            raise AnalysisError(
                "The mac profile uses MLX/Metal vision; --vision-device must be auto"
            )

    def device(requested):
        return (
            ("cpu" if profile == "cpu" else "cuda")
            if requested == "auto" and profile != "mac"
            else requested
        )

    model_provenance = {
        key: {k: entry[k] for k in ("repo_id", "revision", "quantization")}
        for key, entry in models.items()
    }
    # Paths and output-only settings do not affect analysis, but report language does
    # affect the vision prompt and therefore must be part of the cache identity.
    settings = config.model_dump(
        mode="json", exclude={"resume", "work_dir", "model_dir", "runtime_dir", "worker_timeout"}
    )
    manifest_hash = hashlib.sha256(
        json.dumps({key: entry["files"] for key, entry in models.items()}, sort_keys=True).encode()
    ).hexdigest()
    identity = {
        "input_sha256": file_hash(path),
        "application_version": __version__,
        "configuration": settings,
        "analysis_range": analysis_range,
        "profile": profile,
        "models": model_provenance,
        "model_files_sha256": manifest_hash,
    }
    if has_audio and config.diarization and config.diarization_backend == "nemotron":
        identity["diarization_settings"] = NEMO_DIARIZERS["nemotron"].cache_settings
    identity["implementation_sha256"] = hashlib.sha256(
        "".join(
            file_hash(source) for source in sorted(Path(__file__).parent.rglob("*.py"))
        ).encode()
    ).hexdigest()
    root = config.work_dir or path.parent / ".video-analyzer"
    warnings, runtimes, metrics = [], {}, {}

    def record_metrics(stage, response):
        # Inference measurements belong to the completed work, including work
        # recovered from checkpoints. They exclude model loading and parent work.
        measured = metrics.setdefault(stage, {"inference_seconds": 0.0})
        measured["inference_seconds"] += response.get("elapsed_seconds", 0.0)
        for key in ("peak_rss_bytes", "peak_device_bytes"):
            if key in response:
                measured[key] = max(measured.get(key, 0), response[key])

    with Checkpoints(root, identity, config.resume) as cache:
        progress(f"Work directory: {cache.path}")
        saved = cache.get("scenes")
        if saved is None:
            progress("Detecting scenes")
            scenes = detect_scenes(
                media, config.scene_threshold, config.min_scene_seconds, **range_options
            )
            cache.put("scenes", [scene.model_dump() for scene in scenes])
        else:
            scenes = [Scene.model_validate(scene) for scene in saved]
        speakers, regular, exclusive, units = [], [], [], []
        if has_audio and audio_enabled:
            audio_path = cache.path / "audio.wav"
            saved_audio = cache.get("audio")
            if (
                not saved_audio
                or not audio_path.is_file()
                or audio_path.stat().st_size != saved_audio["size"]
            ):
                extract_audio(
                    media,
                    audio_path,
                    audio_normalization=config.audio_normalization,
                    noise_reduction=config.noise_reduction,
                    progress=progress,
                    **range_options,
                )
                cache.put("audio", {"size": audio_path.stat().st_size})
            if config.diarization:
                diarization = cache.get("diarization")
                if diarization is None:
                    progress(f"Identifying speakers with {config.diarization_backend}")
                    with Worker(
                        config.runtime_dir, diarization_runtime, config.worker_timeout
                    ) as worker:
                        load_options, speaker_options = {}, {}
                        if config.diarization_backend == "pyannote":
                            if overrides := config.pyannote.overrides():
                                load_options["pyannote"] = overrides
                            if config.num_speakers is None:
                                speaker_options = {
                                    key: value
                                    for key, value in (
                                        ("min_speakers", config.min_speakers),
                                        ("max_speakers", config.max_speakers),
                                    )
                                    if value is not None
                                }
                        runtime = worker.request(
                            "load",
                            kind=diarization_kind,
                            model=models[diarization_kind]["directory"],
                            device=device(config.diarization_device),
                            **load_options,
                        )
                        diarization = worker.request(
                            "diarize",
                            audio=str(audio_path.resolve()),
                            num_speakers=config.num_speakers,
                            **speaker_options,
                        )
                        diarization["runtime"] = runtime
                    cache.put("diarization", diarization)
                runtimes["diarization"] = diarization["runtime"]
                record_metrics("diarization", diarization)

                def bounded_turns(turns):
                    return [
                        {**turn, "start": max(0, turn["start"]), "end": min(duration, turn["end"])}
                        for turn in turns
                        if min(duration, turn["end"]) > max(0, turn["start"])
                    ]

                speakers, regular, exclusive = normalize_speakers(
                    bounded_turns(diarization["regular"]), bounded_turns(diarization["exclusive"])
                )
            # Keep one ASR worker alive for all chunks, but skip loading it when all
            # chunks are cached or the diarizer detected no speech.
            chunks = ()
            if config.transcription:
                chunks = (
                    speech_chunks(duration, regular)
                    if config.diarization
                    else audio_chunks(duration)
                )
            with contextlib.ExitStack() as stack:
                worker = None
                for index, core_start, core_end, clip_start, clip_end in chunks:
                    name = f"asr/{index:06d}"
                    chunk = cache.get(name)
                    if chunk is None:
                        if worker is None:
                            progress(f"Loading {config.asr} transcription model")
                            worker = stack.enter_context(
                                Worker(config.runtime_dir, "torch", config.worker_timeout)
                            )
                            options = {
                                "kind": config.asr,
                                "model": models[config.asr]["directory"],
                                "device": device(config.audio_device),
                            }
                            if config.asr == "qwen":
                                options["aligner"] = models["aligner"]["directory"]
                            runtime = worker.request("load", **options)
                        progress(
                            f"Transcribing {core_start + range_start:.1f}–{core_end + range_start:.1f}s"
                        )
                        chunk = worker.request(
                            "transcribe",
                            audio=str(audio_path.resolve()),
                            start=clip_start,
                            end=clip_end,
                            language=config.language,
                        )
                        chunk["runtime"] = {**runtime, "device": chunk["device"]}
                        cache.put(name, chunk)
                    runtimes["asr"] = chunk["runtime"]
                    record_metrics("asr", chunk)
                    warnings.extend(chunk.get("warnings", []))
                    units.extend(
                        owned_units(chunk["units"], clip_start, core_start, core_end, duration)
                    )
            if (config.diarization and not regular) or (config.transcription and not units):
                warnings.append(
                    "音声から発話は検出されませんでした。"
                    if config.report_language == "ja"
                    else "No speech was detected in the audio."
                )
        elif not has_audio and audio_enabled:
            warnings.append(
                "音声トラックがないため、映像のみを解析しました。"
                if config.report_language == "ja"
                else "No audio track; analyzed video only."
            )
        for span in [*units, *regular, *exclusive]:
            span.start += range_start
            span.end += range_start
        transcript = list(assign_units(units, exclusive, scenes))
        with contextlib.ExitStack() as stack:
            worker = None
            for scene in scenes if config.scene_description else ():
                for index, (start, end, targets) in enumerate(
                    windows(scene, config.window_seconds, config.frames_per_window)
                ):
                    name = f"vision/{scene.id}-{index:06d}"
                    saved = cache.get(name)
                    if saved is None:
                        if worker is None:
                            progress("Loading Molmo2 scene description model")
                            worker = stack.enter_context(
                                Worker(
                                    config.runtime_dir,
                                    "mlx" if profile == "mac" else "torch",
                                    config.worker_timeout,
                                )
                            )
                            runtime = worker.request(
                                "load",
                                kind=vision_kind,
                                model=models[vision_kind]["directory"],
                                device=device(config.vision_device),
                            )
                        progress(f"Describing {scene.id} {start:.1f}–{end:.1f}s")
                        frame_dir = cache.path / "frames" / f"{scene.id}-{index:06d}"
                        frames, timestamps = extract_frames(
                            media, targets, frame_dir, config.frame_long_edge, **range_options
                        )
                        response = worker.request(
                            "describe",
                            frames=frames,
                            timestamps=timestamps,
                            language=config.report_language,
                            max_new_tokens=config.max_new_tokens,
                        )
                        saved = {
                            "observation": Observation(
                                start=start,
                                end=end,
                                description=response["description"],
                                frame_timestamps=timestamps,
                            ).model_dump(),
                            "runtime": runtime,
                            "metrics": {
                                k: response[k]
                                for k in ("elapsed_seconds", "peak_rss_bytes", "peak_device_bytes")
                            },
                        }
                        cache.put(name, saved)
                        shutil.rmtree(frame_dir)
                    runtimes["vision"] = saved["runtime"]
                    record_metrics("vision", saved["metrics"])
                    scene.observations.append(Observation.model_validate(saved["observation"]))
        # Report creation is deliberately outside this API: callers may render either
        # format without rerunning inference, and errors never publish a partial report.
        return AnalysisResult(
            application_version=__version__,
            media=media,
            report_language=config.report_language,
            provenance={
                "models": model_provenance,
                "runtimes": runtimes,
                "stage_metrics": metrics,
                "profile": profile,
                "input_sha256": identity["input_sha256"],
                "configuration": settings,
                "analysis_range": analysis_range,
                "elapsed_seconds": time.monotonic() - started,
            },
            speakers=speakers,
            diarization=regular,
            exclusive_diarization=exclusive,
            transcript=transcript,
            scenes=scenes,
            warnings=list(dict.fromkeys(warnings)),
        )
