import json

import pytest

from video_analyzer.errors import AnalysisError
from video_analyzer.models import local_models
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
