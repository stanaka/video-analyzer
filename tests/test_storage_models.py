import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from video_analyzer.errors import AnalysisError
from video_analyzer.models import (
    MODEL_REPOS,
    SORTFORMER_CHECKPOINT,
    local_models,
    prepare_models,
    required_models,
)
from video_analyzer.storage import Checkpoints, atomic_json, file_hash


def test_atomic_checkpoints_and_identity(tmp_path):
    identity = {"input": "one", "model": "revision-a"}
    with Checkpoints(tmp_path, identity, False) as run:
        run.put("chunks/1", {"complete": True})
        assert run.get("chunks/1") is None
        with pytest.raises(AnalysisError, match="Another process"):
            with Checkpoints(tmp_path, identity, True):
                pass
    with Checkpoints(tmp_path, identity, True) as run:
        assert run.get("chunks/1") == {"complete": True}
        (run.path / "chunks/2.json.partial").write_text('{"incomplete":')
        assert run.get("chunks/2") is None
    for changed in (
        {"input": "two", "model": "revision-a"},
        {"input": "one", "model": "revision-b"},
    ):
        with Checkpoints(tmp_path, changed, True) as run:
            assert run.get("chunks/1") is None


def test_models_fail_locally_on_missing_or_corrupt_files(tmp_path):
    weights = tmp_path / "whisper" / "weights.safetensors"
    weights.parent.mkdir()
    weights.write_bytes(b"model")
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "models": {
                "whisper": {
                    "path": "whisper",
                    "files": {weights.name: {"size": 5, "sha256": file_hash(weights)}},
                }
            },
        },
    )
    assert local_models(tmp_path, ["whisper"])
    weights.write_bytes(b"other")
    with pytest.raises(AnalysisError, match="checksum"):
        local_models(tmp_path, ["whisper"], verify_hashes=True)
    weights.unlink()
    with pytest.raises(AnalysisError, match="Missing or incomplete"):
        local_models(tmp_path, ["whisper"])


def test_fresh_interrupted_run_cannot_reuse_stale_chunks(tmp_path):
    identity = {"input": "same"}
    with Checkpoints(tmp_path, identity, False) as run:
        run.put("asr/0", {"text": "old"})
        run.put("asr/1", {"text": "stale"})
    with Checkpoints(tmp_path, identity, False) as run:
        run.put("asr/0", {"text": "fresh"})
        # A failed second chunk leaves no completed checkpoint.
    with Checkpoints(tmp_path, identity, True) as run:
        assert run.get("asr/0") == {"text": "fresh"}
        assert run.get("asr/1") is None


def test_missing_manifest_and_escaping_paths(tmp_path):
    with pytest.raises(AnalysisError, match="prepare-models"):
        local_models(tmp_path, ["whisper"])
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "models": {"whisper": {"path": "..", "files": {"anything": {"size": 1}}}},
        },
    )
    with pytest.raises(AnalysisError, match="outside"):
        local_models(tmp_path, ["whisper"])


def test_missing_indexed_shard(tmp_path):
    directory = tmp_path / "model"
    directory.mkdir()
    index = directory / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"weight": "missing.safetensors"}}))
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "models": {
                "whisper": {"path": "model", "files": {index.name: {"size": index.stat().st_size}}}
            },
        },
    )
    with pytest.raises(AnalysisError, match="weight shard"):
        local_models(tmp_path, ["whisper"])


@pytest.mark.parametrize("profile,vision", [("mac", "molmo-mlx"), ("cuda", "molmo-torch")])
@pytest.mark.parametrize(
    "asr,speech",
    [
        ("whisper", ["whisper"]),
        ("qwen", ["qwen", "aligner"]),
        ("both", ["whisper", "qwen", "aligner"]),
    ],
)
def test_required_models_preserves_defaults(profile, vision, asr, speech):
    assert required_models(profile, asr) == [vision, "diarization", *speech]
    assert required_models(profile, asr, audio=False) == [vision]
    assert required_models(profile, asr, transcription=False) == [vision, "diarization"]


@pytest.mark.parametrize(
    "backend,keys",
    [
        ("pyannote", ["diarization"]),
        ("sortformer", ["sortformer"]),
        ("both", ["diarization", "sortformer"]),
    ],
)
def test_selected_diarization_models(backend, keys):
    options = dict(scene_description=False, transcription=False, diarization_backend=backend)
    assert required_models("cpu", "whisper", **options) == keys
    assert required_models("cpu", "whisper", audio=False, **options) == []
    assert required_models("cpu", "whisper", diarization=False, **options) == []


@pytest.mark.parametrize("files", [{"README.md": b"metadata only"}, {SORTFORMER_CHECKPOINT: b""}])
def test_sortformer_requires_nonempty_checkpoint(tmp_path, files):
    for name, content in files.items():
        (tmp_path / name).write_bytes(content)
    atomic_json(
        tmp_path / "manifest.json",
        {
            "schema_version": 1,
            "models": {
                "sortformer": {
                    "repo_id": MODEL_REPOS["sortformer"],
                    "revision": "abc",
                    "path": ".",
                    "files": {
                        name: {"size": len(content), "sha256": file_hash(tmp_path / name)}
                        for name, content in files.items()
                    },
                }
            },
        },
    )
    with pytest.raises(AnalysisError, match="NVIDIA v2.1 checkpoint"):
        local_models(tmp_path, ["sortformer"])


@pytest.fixture
def hub(monkeypatch, tmp_path):
    import huggingface_hub

    import video_analyzer.models as models

    executable = tmp_path / "hf"
    executable.touch()
    monkeypatch.setattr(models.shutil, "which", lambda _: str(executable))
    sources = {}
    requests = []
    for key, repo in MODEL_REPOS.items():
        filename = SORTFORMER_CHECKPOINT if key == "sortformer" else "model.safetensors"
        sources[repo] = {filename: b"weights", "README.md": b"model card"}

    def info(repo, **kwargs):
        requests.append(repo)
        return SimpleNamespace(
            sha="resolved-immutable-sha",
            siblings=[
                SimpleNamespace(rfilename=name, size=len(content))
                for name, content in sources[repo].items()
            ],
        )

    def download(command, **kwargs):
        repo = command[2]
        end = command.index("--revision")
        assert command[end + 1] == "resolved-immutable-sha"
        destination = Path(command[command.index("--local-dir") + 1])
        destination.mkdir(parents=True, exist_ok=True)
        for name in command[3:end]:
            (destination / name).write_bytes(sources[repo][name])

    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(model_info=info))
    monkeypatch.setattr(models.subprocess, "run", download)
    return sources, requests


def test_preparation_coexists_preserves_models_and_hashes_checkpoint(tmp_path, hub):
    root = tmp_path / "models"
    first = prepare_models(root, "cuda", "both", diarization_backend="pyannote")
    assert "sortformer" not in first["models"]
    second = prepare_models(root, "cuda", "whisper", diarization_backend="sortformer")
    assert all(second["models"][key] == value for key, value in first["models"].items())
    checkpoint = second["models"]["sortformer"]
    assert checkpoint["revision"] == "resolved-immutable-sha"
    assert checkpoint["files"][SORTFORMER_CHECKPOINT] == {
        "size": 7,
        "sha256": file_hash(root / "sortformer" / SORTFORMER_CHECKPOINT),
    }
    both = prepare_models(root, "cuda", "both", diarization_backend="both")
    assert both == second
    assert set(local_models(root, list(both["models"]), verify_hashes=True)) == set(both["models"])
    (root / "sortformer" / SORTFORMER_CHECKPOINT).unlink()
    with pytest.raises(AnalysisError, match="Missing or incomplete.*sortformer"):
        local_models(root, ["sortformer"])


def test_upstream_sortformer_metadata_without_checkpoint_fails(tmp_path, hub):
    sources, _ = hub
    del sources[MODEL_REPOS["sortformer"]][SORTFORMER_CHECKPOINT]
    with pytest.raises(AnalysisError, match="Missing diar_streaming_sortformer"):
        prepare_models(tmp_path / "models", "cuda", "whisper", diarization_backend="sortformer")


def test_download_failure_names_selected_backend(tmp_path, hub, monkeypatch):
    import subprocess

    import video_analyzer.models as models

    download = models.subprocess.run

    def fail_sortformer(command, **kwargs):
        if MODEL_REPOS["sortformer"] in command:
            raise subprocess.CalledProcessError(1, command)
        return download(command, **kwargs)

    monkeypatch.setattr(models.subprocess, "run", fail_sortformer)
    with pytest.raises(
        AnalysisError, match="Cannot prepare nvidia/diar_streaming_sortformer_4spk-v2.1"
    ):
        prepare_models(tmp_path / "models", "cuda", "whisper", diarization_backend="sortformer")
