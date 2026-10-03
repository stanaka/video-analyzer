import argparse
import json
import sys
from pathlib import Path

from pydantic import ValidationError

from . import __version__, analyze_video
from .errors import AnalysisError
from .media import prerequisites
from .models import local_models, prepare_models, required_models, resolve_profile
from .report import render_json, render_markdown
from .schema import AnalysisConfig
from .storage import atomic_text
from .worker_client import Worker


def parser():
    cli = argparse.ArgumentParser(
        description="Analyze local MP4 videos entirely offline after model preparation."
    )
    cli.add_argument("--version", action="version", version=__version__)
    commands = cli.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser(
        "prepare-models", help="Explicitly download models using your hf login"
    )
    prepare.add_argument("--profile", choices=("mac", "cuda"), required=True)
    prepare.add_argument("--asr", choices=("whisper", "qwen", "both"), default="whisper")
    prepare.add_argument("--model-dir", type=Path, default=Path(".models"))
    doctor = commands.add_parser(
        "doctor", help="Check local dependencies and model assets without network access"
    )
    doctor.add_argument("--profile", choices=("auto", "mac", "cuda", "cpu"), default="auto")
    doctor.add_argument("--asr", choices=("whisper", "qwen", "both"), default="whisper")
    doctor.add_argument("--model-dir", type=Path, default=Path(".models"))
    doctor.add_argument("--runtime-dir", type=Path, default=Path(".runtime"))
    doctor.add_argument("--verify-hashes", action="store_true")
    analyze = commands.add_parser("analyze", help="Analyze one local MP4")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--format", choices=("markdown", "json"), default="markdown")
    analyze.add_argument("-o", "--output", type=Path)
    analyze.add_argument("--asr", choices=("whisper", "qwen"), default="whisper")
    analyze.add_argument("--profile", choices=("auto", "mac", "cuda", "cpu"), default="auto")
    analyze.add_argument("--model-dir", type=Path, default=Path(".models"))
    analyze.add_argument("--runtime-dir", type=Path, default=Path(".runtime"))
    analyze.add_argument("--work-dir", type=Path)
    analyze.add_argument("--resume", action="store_true")
    analyze.add_argument("--language", help="Spoken language; omitted means automatic detection")
    analyze.add_argument("--report-language", choices=("ja", "en"), default="ja")
    analyze.add_argument("--num-speakers", type=int)
    analyze.add_argument("--audio-stream", type=int, default=0)
    analyze.add_argument("--scene-threshold", type=float, default=3.0)
    analyze.add_argument("--min-scene-seconds", type=float, default=0.5)
    analyze.add_argument("--audio-device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    analyze.add_argument("--diarization-device", choices=("auto", "cpu", "cuda"), default="auto")
    analyze.add_argument("--vision-device", choices=("auto", "cpu", "cuda"), default="auto")
    analyze.add_argument("--frames-per-window", type=int, default=8)
    analyze.add_argument("--window-seconds", type=float, default=30)
    analyze.add_argument(
        "--worker-timeout",
        type=float,
        default=3600,
        help="Seconds allowed without worker progress (default: 3600)",
    )
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
    for kind in ["torch", "mlx"] if profile == "mac" else ["torch"]:
        try:
            with Worker(args.runtime_dir, kind, timeout=120) as worker:
                report["workers"][kind] = worker.request("capabilities")
        except AnalysisError as exc:
            report["errors"].append(str(exc))
    if profile == "cuda" and not report["workers"].get("torch", {}).get("cuda"):
        report["errors"].append("CUDA is unavailable")
    if profile == "mac" and not report["workers"].get("mlx", {}).get("metal"):
        report["errors"].append("MLX Metal is unavailable")
    for key in required_models(profile, args.asr):
        try:
            model = local_models(args.model_dir, [key], verify_hashes=args.verify_hashes)[key]
            report["models"][key] = {"revision": model["revision"], "directory": model["directory"]}
        except AnalysisError as exc:
            report["errors"].append(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare-models":
            prepare_models(
                args.model_dir,
                args.profile,
                args.asr,
                progress=lambda text: print(text, file=sys.stderr, flush=True),
            )
            print(f"Models prepared in {args.model_dir.resolve()}", file=sys.stderr)
            return 0
        if args.command == "doctor":
            return diagnose(args)
        options = {
            key: value
            for key, value in vars(args).items()
            if key not in {"input", "command", "format", "output"}
        }
        config = AnalysisConfig(**options)
        suffix = ".analysis.json" if args.format == "json" else ".analysis.md"
        output = args.output or args.input.with_suffix(suffix)
        if output.resolve() == args.input.resolve():
            raise AnalysisError("Output must not overwrite the input video")
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
