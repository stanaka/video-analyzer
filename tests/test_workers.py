import io
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import video_analyzer.worker_client as client
from video_analyzer.errors import AnalysisError
from video_analyzer.workers import audio, server


@pytest.fixture
def scripted_worker(monkeypatch, tmp_path):
    def create(source, timeout=10):
        executable = tmp_path / "worker"
        executable.write_text(f"#!{sys.executable}\n{source}\n")
        executable.chmod(0o755)
        monkeypatch.setattr(client, "worker_python", lambda *_: executable)
        return client.Worker(tmp_path, "torch", timeout=timeout)

    return create


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


def test_worker_progress_extends_timeout_and_preserves_next_request(scripted_worker, capfd):
    with scripted_worker(
        """
import json, sys, time
for line in sys.stdin:
    request = json.loads(line)
    response = {'version': 1, 'id': request['id']}
    for i in range(6):
        print(json.dumps({**response, 'event': 'progress', 'message': f'embeddings {i}/6',
                          'display': i in (0, 5)}), flush=True)
        time.sleep(0.15)
    print(json.dumps({**response, 'ok': True, 'result': {'done': True}}), flush=True)
""",
        timeout=0.6,
    ) as worker:
        assert worker.request("diarize") == {"done": True}
        assert worker.request("diarize") == {"done": True}
    output = capfd.readouterr()
    assert output.out == ""
    assert output.err.count("embeddings 5/6") == 2
    assert "embeddings 3/6" not in output.err


def test_worker_stall_after_progress_times_out(scripted_worker):
    with scripted_worker(
        """
import json, sys
request = json.loads(sys.stdin.readline())
print(json.dumps({'version': 1, 'id': request['id'], 'event': 'progress',
                  'message': 'embeddings complete; clustering speakers', 'display': False}), flush=True)
sys.stdin.read()
""",
        timeout=0.5,
    ) as worker:
        process = worker.process
        with pytest.raises(AnalysisError, match="timed out during diarize") as caught:
            worker.request("diarize")
        assert "no progress for 0.5s" in str(caught.value)
        assert "Last progress: embeddings complete; clustering speakers" in str(caught.value)
        assert "--worker-timeout" in str(caught.value)
        assert process.poll() is not None
        assert worker.process is None


@pytest.mark.parametrize(
    "exit_statement,diagnostic",
    [("sys.exit(7)", "exit code 7"), ("os.kill(os.getpid(), signal.SIGTERM)", "signal SIGTERM")],
)
def test_worker_exit_reports_status(scripted_worker, exit_statement, diagnostic):
    with scripted_worker(
        f"import sys, os, signal\nsys.stdin.readline()\n{exit_statement}"
    ) as worker:
        with pytest.raises(AnalysisError, match=f"Worker exited during diarize.*{diagnostic}"):
            worker.request("diarize")
        assert worker.process is None


def test_worker_broken_pipe_is_not_reported_as_timeout(scripted_worker):
    with scripted_worker("import sys\nsys.exit(8)") as worker:
        worker.process.wait(timeout=5)
        with pytest.raises(AnalysisError, match="Worker pipe closed during diarize.*exit code 8"):
            worker.request("diarize")
        assert worker.process is None


@pytest.mark.parametrize(
    "overrides,diagnostic",
    [
        ({"id": "wrong"}, "request ID mismatch"),
        ({"version": 99}, "protocol version"),
        ({"message": None}, "Invalid worker progress"),
    ],
)
def test_worker_validates_progress_events(scripted_worker, overrides, diagnostic):
    with scripted_worker(
        f"""
import json, sys
request = json.loads(sys.stdin.readline())
event = {{'version': 1, 'id': request['id'], 'event': 'progress', 'message': 'segmentation'}}
event.update(json.loads({json.dumps(overrides)!r}))
print(json.dumps(event), flush=True)
"""
    ) as worker:
        with pytest.raises(AnalysisError, match=diagnostic):
            worker.request("diarize")


def test_server_progress_keeps_protocol_ids_and_redirects_model_prints(monkeypatch, capsys):
    def diarize(*, audio, progress):
        assert audio == "input.wav"
        print("model diagnostic")
        progress("segmentation 1/2")
        progress("segmentation 2/2")
        return {"regular": [], "exclusive": [], "device": "cpu"}

    backend = SimpleNamespace(device="cpu", diarize=diarize)
    monkeypatch.setattr(audio, "AudioBackend", lambda _: backend)
    monkeypatch.setattr(server, "enforce_worker_offline", lambda: None)
    monkeypatch.setattr(server, "capabilities", lambda: {})
    requests = [
        {"version": 1, "id": "load", "operation": "load", "payload": {"kind": "diarization"}},
        {"version": 1, "id": "run", "operation": "diarize", "payload": {"audio": "input.wav"}},
    ]
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(map(json.dumps, requests))))
    server.main()
    output = capsys.readouterr()
    responses = [json.loads(line) for line in output.out.splitlines()]
    assert [response["id"] for response in responses] == ["load", "run", "run", "run"]
    assert all(response["version"] == 1 for response in responses)
    assert [response["message"] for response in responses[1:3]] == [
        "segmentation 1/2",
        "segmentation 2/2",
    ]
    assert all(response["event"] == "progress" for response in responses[1:3])
    assert responses[-1]["ok"] is True
    assert "elapsed_seconds" in responses[-1]["result"]
    assert output.err == "model diagnostic\n"
