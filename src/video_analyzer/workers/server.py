import contextlib
import importlib.metadata
import json
import resource
import sys
import time
import traceback

from ..offline import enforce_worker_offline


def capabilities():
    result = {"python": sys.version.split()[0], "packages": {}}
    for name in ("torch", "transformers", "pyannote.audio", "qwen-asr", "mlx", "mlx-vlm"):
        try:
            result["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    if "torch" in result["packages"]:
        import torch

        result.update(cuda=torch.cuda.is_available(), mps=torch.backends.mps.is_available())
    if "mlx" in result["packages"]:
        import mlx.core as mx

        result["metal"] = mx.metal.is_available()
    return result


def main():
    enforce_worker_offline()
    backend = None
    # Third-party progress/import prints must never contaminate the JSONL channel.
    protocol = sys.stdout

    def send(response):
        protocol.write(json.dumps(response, ensure_ascii=False, allow_nan=False) + "\n")
        protocol.flush()

    def progress(message, *, display=True):
        send(
            {
                "version": 1,
                "id": request["id"],
                "event": "progress",
                "message": message,
                "display": display,
            }
        )

    with contextlib.redirect_stdout(sys.stderr):
        for line in sys.stdin:
            started = time.monotonic()
            request = {}
            try:
                request = json.loads(line)
                if request.get("version") != 1:
                    raise ValueError("Unsupported worker protocol version")
                operation, payload = request["operation"], request["payload"]
                if operation == "capabilities":
                    result = capabilities()
                elif operation == "load":
                    if backend is not None:
                        raise ValueError("Use a fresh worker for each model stage")
                    if payload["kind"] == "molmo-mlx":
                        from .vision import MlxVision

                        backend = MlxVision(payload)
                    elif payload["kind"] == "molmo-torch":
                        from .vision import TorchVision

                        backend = TorchVision(payload)
                    else:
                        from .audio import AudioBackend

                        backend = AudioBackend(payload)
                    result = {"runtime": capabilities(), "device": backend.device}
                elif backend is None:
                    raise ValueError("Load a model before requesting inference")
                elif operation == "diarize":
                    result = backend.diarize(**payload, progress=progress)
                elif operation == "transcribe":
                    result = backend.transcribe(**payload)
                elif operation == "describe":
                    result = backend.describe(**payload)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                response = {
                    "version": 1,
                    "id": request["id"],
                    "ok": True,
                    "result": {
                        **result,
                        "elapsed_seconds": time.monotonic() - started,
                        "peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
                    },
                }
            except Exception as exc:
                traceback.print_exc(file=sys.stderr)
                response = {
                    "version": 1,
                    "id": request.get("id"),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            send(response)


if __name__ == "__main__":
    main()
