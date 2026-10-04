import argparse
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError

from . import __version__, analyze_video
from .config import load_settings, merge_settings
from .diarization import NEMO_DIARIZERS, selected_diarizers
from .errors import AnalysisError
from .media import prerequisites
from .models import local_models, prepare_models, required_models, resolve_profile
from .report import render_json, render_markdown
from .schema import AnalysisConfig
from .storage import atomic_text
from .worker_client import Worker

PYANNOTE_PARAMETERS = {
    "pyannote_clustering_threshold": ("clustering", "threshold"),
    "pyannote_clustering_fa": ("clustering", "Fa"),
    "pyannote_clustering_fb": ("clustering", "Fb"),
    "pyannote_segmentation_min_duration_off": ("segmentation", "min_duration_off"),
}


def parser(*, sparse=False):
    cli = argparse.ArgumentParser(
        prog="video-analyzer",
        description="Analyze local MP4 videos entirely offline after model preparation.",
        epilog="Run video-analyzer COMMAND --help for command options and examples.",
    )
    cli.add_argument(
        "--version",
        action="version",
        version=__version__,
        help="Show the application version and exit",
    )
    commands = cli.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-models",
        help="Explicitly download models using your hf login",
        description="Download and verify model assets using your saved hf login or HF_TOKEN.\n"
        "This command requires network access; subsequent analysis runs offline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  video-analyzer prepare-models --profile mac\n"
        "  video-analyzer prepare-models --profile cuda --asr both --model-dir /data/models",
    )
    prepare.add_argument(
        "--profile",
        choices=("mac", "cuda"),
        required=True,
        help="Target platform: mac for Apple Silicon, cuda for NVIDIA GPUs (required)",
    )
    prepare.add_argument(
        "--asr",
        choices=("whisper", "qwen", "both"),
        default="whisper",
        help="Speech models to download alongside diarization and vision models (default: %(default)s)",
    )
    prepare.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".models"),
        help="Directory for downloaded models and their manifest (default: %(default)s)",
    )
    doctor = commands.add_parser(
        "doctor",
        help="Check local dependencies and model assets without network access",
        description="Check FFmpeg, worker environments, accelerators, and prepared models offline.\n"
        "Print a JSON diagnostic report; exit with status 1 if any check fails.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  video-analyzer doctor\n"
        "  video-analyzer doctor --profile mac --asr both --verify-hashes",
    )
    doctor.add_argument(
        "--profile",
        choices=("auto", "mac", "cuda", "cpu"),
        default="auto",
        help="Runtime profile; auto selects mac on Apple Silicon, cuda elsewhere (default: %(default)s)",
    )
    doctor.add_argument(
        "--asr",
        choices=("whisper", "qwen", "both"),
        default="whisper",
        help="Speech model assets to check (default: %(default)s)",
    )
    doctor.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".models"),
        help="Directory containing prepared models and their manifest (default: %(default)s)",
    )
    doctor.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path(".runtime"),
        help="Parent directory of the torch, mlx, and optional sortformer/nemotron worker environments (default: %(default)s)",
    )
    doctor.add_argument(
        "--verify-hashes",
        action="store_true",
        help="Read model files and verify their SHA-256 checksums",
    )
    analyze = commands.add_parser(
        "analyze",
        help="Analyze one local MP4",
        description="Transcribe speech, identify speakers, detect scenes, and describe one local MP4\n"
        "using prepared models offline. Progress goes to stderr; stdout prints the\n"
        "report path.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
        "  video-analyzer analyze input.mp4\n"
        "  video-analyzer analyze input.mp4 --config settings.yaml --max-speakers 4\n"
        "  video-analyzer analyze input.mp4 --asr qwen --format json -o report.json\n"
        "  video-analyzer analyze input.mp4 --resume --language ja --num-speakers 2\n"
        "  video-analyzer analyze input.mp4 --audio-normalization dynamic --noise-reduction\n"
        "  video-analyzer analyze input.mp4 --no-diarization --no-scene-description\n"
        "  video-analyzer analyze input.mp4 --no-transcription --no-diarization",
    )
    analyze.add_argument("input", type=Path, help="Path to the local MP4 video")
    analyze.add_argument(
        "--config",
        type=Path,
        help="YAML analysis settings; explicit CLI options override individual settings; paths are relative to the current directory",
    )
    analyze.add_argument(
        "--show-settings",
        action="store_true",
        help="Print merged analysis settings as YAML to stderr before starting analysis (default: disabled)",
    )
    analyze.add_argument(
        "--transcription",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Transcribe speech (default: enabled)",
    )
    analyze.add_argument(
        "--diarization",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Identify speakers; when disabled, transcription uses fixed audio windows without speaker labels (default: enabled)",
    )
    analyze.add_argument(
        "--scene-description",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Describe scene content; scene detection always runs (default: enabled)",
    )
    analyze.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Report output format (default: %(default)s)",
    )
    analyze.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Report path; defaults to INPUT.analysis.md or INPUT.analysis.json beside the video",
    )
    analyze.add_argument(
        "--asr",
        choices=("whisper", "qwen"),
        default="whisper",
        help="Speech recognition backend (default: %(default)s)",
    )
    analyze.add_argument(
        "--profile",
        choices=("auto", "mac", "cuda", "cpu"),
        default="auto",
        help="Runtime profile; auto selects mac on Apple Silicon, cuda elsewhere (default: %(default)s)",
    )
    analyze.add_argument(
        "--model-dir",
        type=Path,
        default=Path(".models"),
        help="Directory containing prepared models and their manifest (default: %(default)s)",
    )
    analyze.add_argument(
        "--runtime-dir",
        type=Path,
        default=Path(".runtime"),
        help="Parent directory of the torch, mlx, and optional sortformer/nemotron worker environments (default: %(default)s)",
    )
    analyze.add_argument(
        "--work-dir",
        type=Path,
        help="Checkpoint and temporary audio directory (default: .video-analyzer beside the video)",
    )
    analyze.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Reuse completed checkpoints matching the input, settings, and models",
    )
    analyze.add_argument("--language", help="Spoken language; omitted means automatic detection")
    analyze.add_argument(
        "--report-language",
        choices=("ja", "en"),
        default="ja",
        help="Language of report headings and visual descriptions (default: %(default)s)",
    )
    analyze.add_argument(
        "--num-speakers",
        type=int,
        help="Known speaker count for pyannote, at least 1; overrides min/max-speakers; omit when using Sortformer or Nemotron",
    )
    for bound in ("min", "max"):
        analyze.add_argument(
            f"--{bound}-speakers",
            type=int,
            help=f"{'Minimum' if bound == 'min' else 'Maximum'} pyannote speaker count, at least 1; ignored with --num-speakers",
        )
    for name, (group, parameter) in PYANNOTE_PARAMETERS.items():
        analyze.add_argument(
            f"--{name.replace('_', '-')}",
            type=float,
            help=f"Override pyannote {group}.{parameter}; omitted uses the loaded model's default",
        )
    analyze.add_argument(
        "--audio-stream",
        type=int,
        default=0,
        help="Zero-based audio stream ordinal, not container stream index (default: %(default)s)",
    )
    analyze.add_argument(
        "--audio-normalization",
        choices=("off", "peak", "dynamic"),
        default="off",
        help="Normalize audio before transcription and diarization: peak adjusts the whole recording to -3 dBFS; dynamic gently balances quiet and loud passages (default: %(default)s)",
    )
    analyze.add_argument(
        "--noise-reduction",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Conservatively reduce rumble and background noise before audio normalization and analysis (default: disabled)",
    )
    analyze.add_argument(
        "--scene-threshold",
        type=float,
        default=3.0,
        help="Positive adaptive scene cut detection threshold (default: %(default)s)",
    )
    analyze.add_argument(
        "--min-scene-seconds",
        type=float,
        default=0.5,
        help="Positive minimum spacing between scene cuts in seconds (default: %(default)s)",
    )
    analyze.add_argument(
        "--audio-device",
        choices=("auto", "cpu", "mps", "cuda"),
        default="auto",
        help="Speech recognition device; auto allows CPU fallback for MPS failures (default: %(default)s)",
    )
    analyze.add_argument(
        "--diarization-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Speaker diarization device; auto uses CPU on Mac, CUDA on the cuda profile (default: %(default)s)",
    )
    analyze.add_argument(
        "--vision-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Vision device; auto uses MLX/Metal on Mac, CPU/CUDA overrides use PyTorch (default: %(default)s)",
    )
    analyze.add_argument(
        "--frames-per-window",
        type=int,
        default=8,
        help="Maximum chronological frames per description window, from 1 to 8 (default: %(default)s)",
    )
    analyze.add_argument(
        "--window-seconds",
        type=float,
        default=30,
        help="Positive description window length in seconds, at most 30 (default: %(default)s)",
    )
    analyze.add_argument(
        "--worker-timeout",
        type=float,
        default=3600,
        help="Positive seconds allowed without worker progress (default: %(default)s)",
    )
    for command in (prepare, doctor, analyze):
        command.add_argument(
            "--diarization-backend",
            choices=("pyannote", "sortformer", "nemotron")
            if command is analyze
            else ("pyannote", "sortformer", "nemotron", "both"),
            default="pyannote",
            help="Speaker diarization backend; sortformer is NVIDIA v2.1 (up to four speakers), nemotron is NVIDIA Nemotron 3 (up to eight). Both NVIDIA backends estimate counts automatically; omit speaker counts and pyannote tuning. On prepare-models/doctor, both selects pyannote and Sortformer v2.1 (default: %(default)s)",
        )
    if sparse:
        # Only supplied analysis flags should override YAML. Keep the public
        # parser's defaults and help unchanged for ordinary callers.
        for action in analyze._actions:
            if action.dest in AnalysisConfig.model_fields or action.dest in PYANNOTE_PARAMETERS:
                if action.help:
                    action.help = action.help.replace("%(default)s", str(action.default))
                action.default = argparse.SUPPRESS
    return cli


def diagnose(args) -> int:
    profile = resolve_profile(args.profile)
    report = {
        "profile": profile,
        "executables": prerequisites(),
        "workers": {},
        "models": {},
        "errors": [],
    }
    for name, path in report["executables"].items():
        if not path:
            report["errors"].append(f"Missing {name}")
    worker_kinds = ["torch", "mlx"] if profile == "mac" else ["torch"]
    worker_kinds.extend(
        backend
        for backend in selected_diarizers(args.diarization_backend)
        if backend in NEMO_DIARIZERS
    )
    for kind in worker_kinds:
        try:
            with Worker(args.runtime_dir, kind, timeout=120) as worker:
                options = {"check_backend": kind} if kind in NEMO_DIARIZERS else {}
                report["workers"][kind] = worker.request("capabilities", **options)
        except AnalysisError as exc:
            report["errors"].append(str(exc))
    if profile == "cuda" and not report["workers"].get("torch", {}).get("cuda"):
        report["errors"].append("CUDA is unavailable")
    if profile == "mac" and not report["workers"].get("mlx", {}).get("metal"):
        report["errors"].append("MLX Metal is unavailable")
    for kind in worker_kinds:
        if (
            profile == "cuda"
            and kind in NEMO_DIARIZERS
            and not report["workers"].get(kind, {}).get("cuda")
        ):
            label = "Sortformer" if kind == "sortformer" else "Nemotron"
            report["errors"].append(f"CUDA is unavailable in the {label} worker")
    for key in required_models(profile, args.asr, diarization_backend=args.diarization_backend):
        try:
            model = local_models(args.model_dir, [key], verify_hashes=args.verify_hashes)[key]
            report["models"][key] = {"revision": model["revision"], "directory": model["directory"]}
        except AnalysisError as exc:
            report["errors"].append(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-models":
            prepare_models(
                args.model_dir,
                args.profile,
                args.asr,
                progress=lambda text: print(text, file=sys.stderr, flush=True),
                diarization_backend=args.diarization_backend,
            )
            print(f"Models prepared in {args.model_dir.resolve()}", file=sys.stderr)
            return 0
        if args.command == "doctor":
            return diagnose(args)
        explicit = parser(sparse=True).parse_args(argv)
        options = {
            key: value
            for key, value in vars(explicit).items()
            if key in AnalysisConfig.model_fields
        }
        for name, (group, parameter) in PYANNOTE_PARAMETERS.items():
            if hasattr(explicit, name):
                options.setdefault("pyannote", {}).setdefault(group, {})[parameter] = getattr(
                    explicit, name
                )
        settings = load_settings(args.config) if args.config is not None else {}
        try:
            config = AnalysisConfig.model_validate(merge_settings(settings, options))
        except ValidationError as exc:
            if args.config is not None:
                raise AnalysisError(f"Invalid configuration {args.config}: {exc}") from exc
            raise
        suffix = ".analysis.json" if args.format == "json" else ".analysis.md"
        output = args.output or args.input.with_suffix(suffix)
        if output.resolve() == args.input.resolve():
            raise AnalysisError("Output must not overwrite the input video")
        if args.show_settings:
            print(
                "# Analysis settings\n"
                "# Null pyannote tuning values use the loaded model's defaults.\n"
                + yaml.safe_dump(
                    config.model_dump(mode="json"),
                    allow_unicode=True,
                    sort_keys=False,
                    default_flow_style=False,
                ),
                end="",
                file=sys.stderr,
                flush=True,
            )
        result = analyze_video(args.input, config)
        atomic_text(
            output, render_json(result) if args.format == "json" else render_markdown(result)
        )
        print(str(output.resolve()))
        return 0
    except (AnalysisError, ValidationError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted. Completed checkpoints are preserved; rerun with --resume.",
            file=sys.stderr,
        )
        return 130


if __name__ == "__main__":
    sys.exit(main())
