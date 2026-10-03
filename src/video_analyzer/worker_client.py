import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from .errors import AnalysisError
from .offline import offline_environment

PROTOCOL_VERSION = 1


def worker_python(runtime_dir: Path, kind: str) -> Path:
    path = runtime_dir.resolve() / kind / "bin" / "python"
    if not path.is_file():
        raise AnalysisError(
            f"Worker environment is missing: {path}. Run scripts/setup-mac.sh or scripts/setup-cuda.sh."
        )
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
            line = self.responses.get(timeout=self.timeout)
        except (BrokenPipeError, queue.Empty) as exc:
            self.close()
            raise AnalysisError(f"Worker failed or timed out during {operation}") from exc
        if line is None:
            raise AnalysisError(f"Worker exited during {operation}; see diagnostics above")
        try:
            response = json.loads(line)
        except ValueError as exc:
            raise AnalysisError(f"Invalid worker protocol response during {operation}") from exc
        if response.get("version") != PROTOCOL_VERSION or response.get("id") != request_id:
            raise AnalysisError("Worker protocol version or request ID mismatch")
        if not response.get("ok"):
            raise AnalysisError(response.get("error", "Unknown worker error"))
        return response["result"]

    def close(self):
        if self.process is None:
            return
        if self.process.poll() is None:
            # EOF lets Python/MLX clean up semaphores on normal completion.
            self.process.stdin.close()
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
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
        self.process = None

    def __exit__(self, *_):
        self.close()
