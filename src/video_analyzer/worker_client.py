import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

from .errors import AnalysisError
from .offline import offline_environment

PROTOCOL_VERSION = 1


def worker_python(runtime_dir: Path, kind: str) -> Path:
    path = runtime_dir.resolve() / kind / "bin" / "python"
    if not path.is_file():
        setup = (
            f"mise run setup:{kind}"
            if kind in ("sortformer", "nemotron")
            else "scripts/setup-mac.sh or scripts/setup-cuda.sh"
        )
        raise AnalysisError(f"Worker environment is missing: {path}. Run {setup}.")
    return path


class Worker:
    def __init__(self, runtime_dir: Path, kind: str, timeout: float = 3600):
        self.python = worker_python(runtime_dir, kind)
        self.timeout = timeout
        self.process = None
        self.responses = queue.Queue()

    def __enter__(self):
        env = {**os.environ, **offline_environment()}
        env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
        env["MPLCONFIGDIR"] = str(self.python.parent.parent / ".matplotlib")
        env["NUMBA_CACHE_DIR"] = str(self.python.parent.parent / ".numba")
        # Analysis never needs authentication. Do not expose a token to model code.
        env.pop("HF_TOKEN", None)
        env.pop("HUGGING_FACE_HUB_TOKEN", None)
        self.process = subprocess.Popen(
            [str(self.python), "-u", "-m", "video_analyzer.workers.server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,
            text=True,
            encoding="utf-8",
            env=env,
            bufsize=1,
        )

        stdout = self.process.stdout

        def read():
            for line in stdout:
                self.responses.put(line)
            self.responses.put(None)

        self.reader = threading.Thread(target=read, daemon=True)
        self.reader.start()
        return self

    def request(self, operation: str, **payload):
        request_id = uuid.uuid4().hex
        started = time.monotonic()
        try:
            self.process.stdin.write(
                json.dumps(
                    {
                        "version": PROTOCOL_VERSION,
                        "id": request_id,
                        "operation": operation,
                        "payload": payload,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            self.process.stdin.flush()
        except BrokenPipeError as exc:
            detail = self._exit_detail()
            self.close()
            raise AnalysisError(
                f"Worker pipe closed during {operation}{detail}; see diagnostics above"
            ) from exc
        last_progress = None
        while True:
            try:
                # Progress is emitted by completed model batches, not a heartbeat.
                # Long recordings may run for more than one timeout interval.
                line = self.responses.get(timeout=self.timeout)
            except queue.Empty as exc:
                elapsed = time.monotonic() - started
                detail = f" Last progress: {last_progress}." if last_progress else ""
                self.close()
                raise AnalysisError(
                    f"Worker timed out during {operation}: no progress for {self.timeout:g}s "
                    f"({elapsed:.1f}s elapsed).{detail} "
                    "Rerun with --resume and a larger --worker-timeout if needed."
                ) from exc
            if line is None:
                detail = self._exit_detail()
                self.close()
                raise AnalysisError(
                    f"Worker exited during {operation}{detail}; see diagnostics above"
                )
            try:
                response = json.loads(line)
            except ValueError as exc:
                raise AnalysisError(f"Invalid worker protocol response during {operation}") from exc
            if not isinstance(response, dict):
                raise AnalysisError(f"Invalid worker protocol response during {operation}")
            if response.get("version") != PROTOCOL_VERSION or response.get("id") != request_id:
                raise AnalysisError("Worker protocol version or request ID mismatch")
            if response.get("event") == "progress":
                last_progress = response.get("message")
                if not isinstance(last_progress, str) or not last_progress:
                    raise AnalysisError(f"Invalid worker progress response during {operation}")
                if response.get("display", True):
                    print(last_progress, file=sys.stderr, flush=True)
                continue
            if not response.get("ok"):
                raise AnalysisError(response.get("error", "Unknown worker error"))
            return response["result"]

    def _exit_detail(self):
        try:
            code = self.process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            return " (exit status unavailable)"
        if code < 0:
            try:
                name = signal.Signals(-code).name
            except ValueError:
                name = str(-code)
            return f" (signal {name})"
        return f" (exit code {code})"

    def close(self):
        if self.process is None:
            return
        # EOF lets Python/MLX clean up semaphores on normal completion. Closing a
        # buffered pipe can itself raise when the worker has already exited.
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        self.reader.join(timeout=1)
        self.process.stdout.close()
        self.process = None

    def __exit__(self, *_):
        self.close()
