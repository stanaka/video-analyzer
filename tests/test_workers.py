import subprocess
import sys
from pathlib import Path

import pytest

import video_analyzer.worker_client as client
from video_analyzer.errors import AnalysisError


def test_offline_guard_blocks_connections_without_touching_parent():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import socket
from video_analyzer.offline import enforce_worker_offline
from video_analyzer.errors import AnalysisError
enforce_worker_offline()
try:
    socket.create_connection(('huggingface.co', 443))
except AnalysisError:
    pass
else:
    raise AssertionError('Network access was allowed')
""",
        ],
        check=True,
    )


def test_worker_errors_are_propagated(monkeypatch, tmp_path):
    monkeypatch.setattr(client, "worker_python", lambda *_: Path(sys.executable))
    with client.Worker(tmp_path, "torch", timeout=10) as worker:
        with pytest.raises(AnalysisError, match="Load a model"):
            worker.request("describe")


def test_worker_timeout_terminates_child(monkeypatch, tmp_path):
    executable = tmp_path / "sleeper"
    executable.write_text(f"#!{sys.executable}\nimport time\ntime.sleep(60)\n")
    executable.chmod(0o755)
    monkeypatch.setattr(client, "worker_python", lambda *_: executable)
    with client.Worker(tmp_path, "torch", timeout=0.1) as worker:
        process = worker.process
        with pytest.raises(AnalysisError, match="timed out"):
            worker.request("capabilities")
        assert process.poll() is not None
