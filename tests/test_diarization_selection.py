import json

import pytest
from pydantic import ValidationError

from video_analyzer import AnalysisConfig, cli
from video_analyzer.errors import AnalysisError
from video_analyzer.worker_client import worker_python


def test_default_and_invalid_backend():
    assert AnalysisConfig().diarization_backend == "pyannote"
    for command in (["analyze", "input.mp4"], ["prepare-models", "--profile", "cuda"], ["doctor"]):
        assert cli.parser().parse_args(command).diarization_backend == "pyannote"
    with pytest.raises(ValidationError):
        AnalysisConfig(diarization_backend="both")
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["analyze", "input.mp4", "--diarization-backend", "both"])


@pytest.mark.parametrize("backend", ["sortformer", "nemotron"])
def test_speaker_count_rejection_precedes_inference(monkeypatch, capsys, backend):
    with pytest.raises(ValidationError, match="select the pyannote"):
        AnalysisConfig(diarization_backend=backend, num_speakers=2)
    assert AnalysisConfig(diarization_backend=backend, num_speakers=2, diarization=False)
    assert AnalysisConfig(num_speakers=2)
    monkeypatch.setattr(
        cli, "analyze_video", lambda *args: pytest.fail("Should reject before inference")
    )
    assert (
        cli.main(["analyze", "input.mp4", "--diarization-backend", backend, "--num-speakers", "2"])
        == 1
    )
    assert "Omit --num-speakers" in capsys.readouterr().err


@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron"])
def test_analyze_selection_reaches_api(backend, monkeypatch):
    def capture(_, config):
        assert config.diarization_backend == backend
        raise AnalysisError("Stop after checking config")

    monkeypatch.setattr(cli, "analyze_video", capture)
    assert cli.main(["analyze", "input.mp4", "--diarization-backend", backend]) == 1


@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron", "both"])
def test_preparation_selection_reaches_api(backend, monkeypatch):
    selections = []
    monkeypatch.setattr(
        cli, "prepare_models", lambda *a, **kw: selections.append(kw["diarization_backend"])
    )
    assert cli.main(["prepare-models", "--profile", "cuda", "--diarization-backend", backend]) == 0
    assert selections == [backend]


@pytest.mark.parametrize("backend", ["pyannote", "sortformer", "nemotron", "both"])
@pytest.mark.parametrize("cuda", [False, True])
def test_doctor_checks_selected_workers_models_and_devices(
    backend, cuda, monkeypatch, tmp_path, capsys
):
    requests, keys = [], []

    class Worker:
        def __init__(self, root, kind, **kwargs):
            self.kind = kind

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def request(self, operation, **payload):
            requests.append((self.kind, operation, payload))
            return {"cuda": cuda, "metal": True}

    def local(root, required, **kwargs):
        assert kwargs["verify_hashes"] is True
        keys.extend(required)
        return {key: {"revision": "abc", "directory": str(root / key)} for key in required}

    monkeypatch.setattr(cli, "Worker", Worker)
    monkeypatch.setattr(cli, "local_models", local)
    monkeypatch.setattr(cli, "prerequisites", lambda: {"ffmpeg": "/ffmpeg"})
    assert cli.main(
        ["doctor", "--profile", "cuda", "--diarization-backend", backend, "--verify-hashes"]
    ) == (0 if cuda else 1)
    report = json.loads(capsys.readouterr().out)
    optional = "sortformer" if backend == "both" else backend if backend != "pyannote" else None
    assert ("sortformer" in keys) == (optional == "sortformer")
    assert ("nemotron" in keys) == (optional == "nemotron")
    assert ("diarization" in keys) == (backend in ("pyannote", "both"))
    assert [kind for kind, _, _ in requests] == ["torch"] + ([optional] if optional else [])
    if optional:
        assert requests[-1] == (optional, "capabilities", {"check_backend": optional})
        label = "Sortformer" if optional == "sortformer" else "Nemotron"
        assert (f"CUDA is unavailable in the {label} worker" in report["errors"]) == (not cuda)


@pytest.mark.parametrize("backend", ["sortformer", "nemotron"])
def test_missing_optional_runtime_has_actionable_error(tmp_path, backend):
    with pytest.raises(AnalysisError, match=f"mise run setup:{backend}"):
        worker_python(tmp_path, backend)
