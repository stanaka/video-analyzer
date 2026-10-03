import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from .errors import AnalysisError


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def atomic_json(path: Path, data) -> None:
    atomic_text(path, json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"Cannot read {path}: {exc}") from exc


class Checkpoints:
    """Content-addressed run; atomic writes and a process lock protect resumability."""

    def __init__(self, root: Path, identity: dict, resume: bool):
        serialized = json.dumps(identity, sort_keys=True, ensure_ascii=False)
        self.path = root / hashlib.sha256(serialized.encode()).hexdigest()[:24]
        self.path.mkdir(parents=True, exist_ok=True)
        self.resume = resume
        self.lock = None
        self.identity = identity

    def __enter__(self):
        import fcntl

        self.lock = (self.path / "run.lock").open("a+")
        try:
            fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.lock.close()
            raise AnalysisError(f"Another process is using {self.path}") from exc
        try:
            if not self.resume:
                # A fresh run must not leave old chunks available to a later
                # resume: rerun diarization can change chunk boundaries.
                for child in self.path.iterdir():
                    if child.name == "run.lock":
                        continue
                    if child.is_dir() and not child.is_symlink():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            atomic_json(self.path / "identity.json", self.identity)
        except BaseException:
            self.lock.close()
            raise
        return self

    def __exit__(self, *_):
        if self.lock:
            self.lock.close()

    def get(self, name: str):
        path = self.path / f"{name}.json"
        return read_json(path) if self.resume and path.exists() else None

    def put(self, name: str, data):
        atomic_json(self.path / f"{name}.json", data)
