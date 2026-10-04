import fnmatch
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .errors import AnalysisError
from .storage import atomic_json, file_hash, read_json

MODEL_REPOS = {
    "whisper": "openai/whisper-large-v3",
    "qwen": "Qwen/Qwen3-ASR-1.7B",
    "aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
    "diarization": "pyannote/speaker-diarization-community-1",
    "sortformer": "nvidia/diar_streaming_sortformer_4spk-v2.1",
    "molmo-mlx": "mlx-community/Molmo2-8B-4bit",
    "molmo-torch": "allenai/Molmo2-8B",
}
SORTFORMER_CHECKPOINT = "diar_streaming_sortformer_4spk-v2.1.nemo"
DIARIZATION_MODELS = {"pyannote": "diarization", "sortformer": "sortformer"}
PATTERNS = [
    "*.json",
    "*.yaml",
    "*.yml",
    "*.safetensors",
    "*.txt",
    "*.model",
    "*.jinja",
    "*.py",
    "*.tiktoken",
]


def resolve_profile(profile: str) -> str:
    if profile == "auto":
        return "mac" if platform.system() == "Darwin" and platform.machine() == "arm64" else "cuda"
    if profile == "mac" and (platform.system() != "Darwin" or platform.machine() != "arm64"):
        raise AnalysisError("The mac profile requires Apple Silicon macOS")
    return profile


def required_models(
    profile: str,
    asr: str,
    *,
    audio=True,
    transcription=True,
    diarization=True,
    diarization_backend="pyannote",
    scene_description=True,
) -> list[str]:
    result = ["molmo-mlx" if profile == "mac" else "molmo-torch"] if scene_description else []
    if audio and diarization:
        result.extend(
            DIARIZATION_MODELS.values()
            if diarization_backend == "both"
            else [DIARIZATION_MODELS[diarization_backend]]
        )
    if audio and transcription:
        if asr in ("whisper", "both"):
            result += ["whisper"]
        if asr in ("qwen", "both"):
            result += ["qwen", "aligner"]
    return result


def prepare_models(
    root: Path, profile: str, asr: str, progress=print, *, diarization_backend="pyannote"
) -> dict:
    """The only application path allowed to contact the Hugging Face Hub.

    hf uses its saved login (or HF_TOKEN). Tokens are never command arguments or
    stored in the manifest. Each complete local snapshot records its resolved SHA.
    """
    from huggingface_hub import HfApi
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    bundled_hf = Path(sys.executable).with_name("hf")
    hf = str(bundled_hf) if bundled_hf.is_file() else shutil.which("hf")
    if not hf or not Path(hf).is_file():
        raise AnalysisError("hf CLI is missing. Install the project's locked dependencies.")
    manifest_path = root / "manifest.json"
    manifest = (
        read_json(manifest_path) if manifest_path.exists() else {"schema_version": 1, "models": {}}
    )
    for key in required_models(
        resolve_profile(profile), asr, diarization_backend=diarization_backend
    ):
        repo = MODEL_REPOS[key]
        progress(f"Preparing {repo}")
        try:
            info = HfApi().model_info(repo, files_metadata=True)
            patterns = PATTERNS + (
                ["*.bin", "*.npz", "*.pt", "*.ckpt"] if key == "diarization" else []
            )
            if key == "sortformer":
                patterns = [SORTFORMER_CHECKPOINT, "README.md"]
            expected = [
                s
                for s in info.siblings
                if any(fnmatch.fnmatch(s.rfilename, p) for p in patterns)
                and ".fp32" not in s.rfilename
                and not (key != "diarization" and "pytorch_model.bin.index" in s.rfilename)
            ]
            if not expected:
                raise AnalysisError(f"No model files found for {repo}")
            if key == "sortformer" and SORTFORMER_CHECKPOINT not in {s.rfilename for s in expected}:
                raise AnalysisError(f"Missing {SORTFORMER_CHECKPOINT} in {repo}")
            # On updates use a fresh directory. Existing checkpoints can still resolve
            # their old snapshot and an interrupted download cannot corrupt it.
            previous = manifest["models"].get(key)
            relative = previous["path"] if previous and previous["revision"] == info.sha else key
            if previous and previous["revision"] != info.sha:
                relative = f"{key}-{info.sha[:12]}"
            destination = root / relative
            needed = sum(s.size or 0 for s in expected if not (destination / s.rfilename).is_file())
            if shutil.disk_usage(root).free < needed + 512 * 1024**2:
                raise AnalysisError(
                    f"Insufficient disk space to download {repo}: need {needed / 1024**3:.1f} GiB plus headroom"
                )
            subprocess.run(
                [
                    hf,
                    "download",
                    repo,
                    *[s.rfilename for s in expected],
                    "--revision",
                    info.sha,
                    "--local-dir",
                    str(destination),
                    "--quiet",
                ],
                check=True,
                stdout=sys.stderr,
            )
        except GatedRepoError as exc:
            raise AnalysisError(
                f"Accept access at https://huggingface.co/{repo} and run `hf auth login` with a read token."
            ) from exc
        except (HfHubHTTPError, subprocess.CalledProcessError) as exc:
            raise AnalysisError(
                f"Cannot prepare {repo}. Check network access and model access permissions."
            ) from exc
        files = {}
        for entry in expected:
            path = destination / entry.rfilename
            if not path.is_file() or (entry.size is not None and path.stat().st_size != entry.size):
                raise AnalysisError(f"Incomplete download: {path}")
            files[entry.rfilename] = {"size": path.stat().st_size, "sha256": file_hash(path)}
        manifest["models"][key] = {
            "repo_id": repo,
            "revision": info.sha,
            "path": relative,
            "quantization": "4-bit MLX" if key == "molmo-mlx" else None,
            "files": files,
        }
        atomic_json(manifest_path, manifest)
    return manifest


def local_models(root: Path, required: list[str], *, verify_hashes=False) -> dict:
    root = root.resolve()
    if not (root / "manifest.json").is_file():
        raise AnalysisError(
            f"Missing {root / 'manifest.json'} for models: {', '.join(required)}. "
            "Run video-analyzer prepare-models with the selected --diarization-backend first."
        )
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 1:
        raise AnalysisError("Unsupported model manifest version")
    result = {}
    for key in required:
        entry = manifest.get("models", {}).get(key)
        if not entry or not entry.get("files"):
            raise AnalysisError(
                f"Model {key} is not prepared. Run prepare-models with the requested ASR/profile "
                "and --diarization-backend."
            )
        if key == "sortformer" and (
            entry.get("repo_id") != MODEL_REPOS[key]
            or SORTFORMER_CHECKPOINT not in entry["files"]
            or entry["files"][SORTFORMER_CHECKPOINT].get("size", 0) <= 0
        ):
            raise AnalysisError(
                f"Sortformer requires the NVIDIA v2.1 checkpoint {SORTFORMER_CHECKPOINT}. "
                "Run prepare-models --diarization-backend sortformer."
            )
        directory = (root / entry["path"]).resolve()
        if not directory.is_relative_to(root):
            raise AnalysisError("Model manifest contains a path outside the model directory")
        for name, expected in entry["files"].items():
            path = (directory / name).resolve()
            if (
                not path.is_relative_to(directory)
                or not path.is_file()
                or path.stat().st_size != expected["size"]
            ):
                raise AnalysisError(
                    f"Missing or incomplete model file: {path}. Run prepare-models again."
                )
            if verify_hashes and file_hash(path) != expected["sha256"]:
                raise AnalysisError(f"Model checksum mismatch: {path}")
        # Validate indexed weight shards as well as recorded files.
        for name in entry["files"]:
            if name.endswith(".index.json"):
                for shard in set(read_json(directory / name).get("weight_map", {}).values()):
                    if shard not in entry["files"]:
                        raise AnalysisError(f"Untracked or missing weight shard: {shard}")
        result[key] = {**entry, "directory": str(directory)}
    return result
