"""Durable, intentionally small JSONL batch primitives.

Final batches are directories so data and its checksum become visible together.  A
partially-written sibling directory is deliberately disposable on resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


class ValidationError(RuntimeError):
    """Raised when immutable experiment evidence does not validate."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(handle: Any) -> None:
    handle.flush()
    os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory sync; Windows does not permit opening directories."""
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def strict_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def atomic_write_json(path: str | Path, value: Any, *, overwrite: bool = False) -> Path:
    """Write one strict JSON document atomically, refusing accidental replacement."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError("immutable file already exists: %s" % destination)
    fd, temporary_name = tempfile.mkstemp(prefix=".%s." % destination.name,
                                        suffix=".tmp", dir=str(destination.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(strict_json_bytes(value))
            _fsync_file(handle)
        if destination.exists() and not overwrite:
            raise FileExistsError("immutable file already exists: %s" % destination)
        os.replace(str(temporary), str(destination))
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Parse JSONL by literal LF, never treating U+2028/U+2029 as delimiters."""
    raw = Path(path).read_text(encoding="utf-8")
    lines = raw.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    for number, line in enumerate(lines, 1):
        if not line:
            raise ValidationError("blank JSONL record at %s:%d" % (path, number))
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValidationError("invalid JSONL at %s:%d: %s" % (path, number, exc)) from exc
        if not isinstance(item, dict):
            raise ValidationError("JSONL record at %s:%d is not an object" % (path, number))
        yield item


def write_jsonl_fsynced(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> tuple[int, str]:
    """Write complete JSONL and sync it once.  Callers publish only after this returns."""
    destination = Path(path)
    count = 0
    with destination.open("wb") as handle:
        for row in rows:
            handle.write(strict_json_bytes(row))
            count += 1
        _fsync_file(handle)
    return count, sha256_file(destination)


def _validate_row(row: Mapping[str, Any], required_keys: Sequence[str] | None) -> None:
    if required_keys is not None and set(row) != set(required_keys):
        raise ValidationError("row schema mismatch; expected %s, got %s" %
                              (sorted(required_keys), sorted(row)))


def publish_batch(
    batches_dir: str | Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    key: Callable[[Mapping[str, Any]], str],
    required_keys: Sequence[str] | None = None,
    extra_manifest: Mapping[str, Any] | None = None,
) -> Path:
    """Publish an immutable batch directory; no existing final is ever replaced."""
    root = Path(batches_dir)
    root.mkdir(parents=True, exist_ok=True)
    final = root / name
    if final.exists():
        raise FileExistsError("immutable batch already exists: %s" % final)
    temporary = Path(tempfile.mkdtemp(prefix=".%s.tmp-" % name, dir=str(root)))
    try:
        data = temporary / "data.jsonl"
        count, checksum = write_jsonl_fsynced(data, rows)
        seen: set[str] = set()
        for row in iter_jsonl(data):
            _validate_row(row, required_keys)
            row_key = key(row)
            if not row_key or row_key in seen:
                raise ValidationError("duplicate or empty batch key: %r" % row_key)
            seen.add(row_key)
        manifest: dict[str, Any] = {
            "format": "immutable-jsonl-batch-v1",
            "data_file": "data.jsonl",
            "sha256": checksum,
            "row_count": count,
            "keys": sorted(seen),
            "required_keys": list(required_keys) if required_keys is not None else None,
        }
        if extra_manifest:
            if set(extra_manifest).intersection(manifest):
                raise ValueError("extra manifest cannot replace immutable batch metadata")
            manifest.update(extra_manifest)
        # It is still temporary, so atomic replacement inside it is safe.
        atomic_write_json(temporary / "manifest.json", manifest)
        _fsync_directory(temporary)
        if final.exists():
            raise FileExistsError("immutable batch already exists: %s" % final)
        os.replace(str(temporary), str(final))
        _fsync_directory(root)
    except BaseException:
        # A temporary directory is explicitly expendable.  Leave it if cleanup itself fails.
        if temporary.exists():
            for child in temporary.iterdir():
                child.unlink(missing_ok=True)
            temporary.rmdir()
        raise
    return final


def finalized_batches(batches_dir: str | Path) -> list[Path]:
    root = Path(batches_dir)
    if not root.exists():
        return []
    finals: list[Path] = []
    for path in root.iterdir():
        if not path.is_dir():
            raise ValidationError("unexpected file in batch directory: %s" % path)
        # Only the exact private temporary naming convention is expendable.
        if path.name.startswith(".") and ".tmp-" in path.name:
            continue
        if not (path / "manifest.json").is_file() or not (path / "data.jsonl").is_file():
            raise ValidationError("incomplete non-temporary batch directory: %s" % path)
        finals.append(path)
    return sorted(finals)


def validate_batches(
    batches_dir: str | Path,
    *,
    key: Callable[[Mapping[str, Any]], str],
    required_keys: Sequence[str] | None = None,
    expected_keys: Iterable[str] | None = None,
) -> list[dict[str, Any]]:
    """Validate every final and global coverage before any expensive initialization."""
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for batch in finalized_batches(batches_dir):
        try:
            manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValidationError("invalid batch manifest: %s" % batch) from exc
        if manifest.get("format") != "immutable-jsonl-batch-v1":
            raise ValidationError("unknown batch format: %s" % batch)
        data = batch / manifest.get("data_file", "")
        if data.name != "data.jsonl" or not data.is_file():
            raise ValidationError("invalid batch data path: %s" % batch)
        if manifest.get("sha256") != sha256_file(data):
            raise ValidationError("batch checksum mismatch: %s" % batch)
        batch_rows = list(iter_jsonl(data))
        if manifest.get("row_count") != len(batch_rows):
            raise ValidationError("batch row count mismatch: %s" % batch)
        stored_schema = manifest.get("required_keys")
        if stored_schema is not None and (not isinstance(stored_schema, list) or
                                          not all(isinstance(name, str) for name in stored_schema)):
            raise ValidationError("invalid stored row schema: %s" % batch)
        if required_keys is None:
            schema = stored_schema
        else:
            if stored_schema != list(required_keys):
                raise ValidationError("batch row schema manifest mismatch: %s" % batch)
            schema = required_keys
        manifest_keys = manifest.get("keys")
        if not isinstance(manifest_keys, list) or not all(isinstance(name, str) for name in manifest_keys):
            raise ValidationError("invalid batch key manifest: %s" % batch)
        actual_keys: list[str] = []
        for row in batch_rows:
            _validate_row(row, schema)
            row_key = key(row)
            if not row_key or row_key in seen:
                raise ValidationError("duplicate or empty global key: %r" % row_key)
            seen.add(row_key)
            actual_keys.append(row_key)
        if sorted(actual_keys) != manifest_keys:
            raise ValidationError("batch key manifest mismatch: %s" % batch)
        rows.extend(batch_rows)
    if expected_keys is not None:
        expected = set(expected_keys)
        if seen != expected:
            missing, unexpected = sorted(expected - seen), sorted(seen - expected)
            raise ValidationError("coverage mismatch (missing=%s, unexpected=%s)" %
                                  (missing[:5], unexpected[:5]))
    return rows


def assert_run_mutable(run_dir: str | Path) -> Path:
    directory = Path(run_dir)
    done, crashed = directory / "DONE", directory / "CRASHED"
    if done.exists() and crashed.exists():
        raise ValidationError("conflicting terminal markers: %s" % directory)
    if done.exists() or crashed.exists():
        raise ValidationError("terminal run cannot be reused: %s" % directory)
    return directory


class RunHeartbeat:
    """Fresh heartbeat plus fsynced operational metrics for one mutable run."""

    def __init__(self, run_dir: str | Path, interval: float = 10.0):
        self.run_dir = Path(run_dir)
        self.interval = interval
        self.heartbeat = self.run_dir / "HEARTBEAT"
        self.metrics = self.run_dir / "metrics.jsonl"
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            self.heartbeat.touch()

    def __enter__(self) -> "RunHeartbeat":
        self.run_dir.mkdir(parents=True, exist_ok=True)
        assert_run_mutable(self.run_dir)
        self.heartbeat.touch()
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def write_metric(self, **values: Any) -> None:
        with self.metrics.open("ab") as handle:
            handle.write(strict_json_bytes(values))
            _fsync_file(handle)

    def __exit__(self, exc_type: Any, exc_value: Any, exc_tb: Any) -> bool:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        if exc_type is KeyboardInterrupt:
            self.write_metric(event="interrupted", status="RESUMABLE", reason="KeyboardInterrupt")
            return False
        if exc_type is not None and not (self.run_dir / "DONE").exists() and not (self.run_dir / "CRASHED").exists():
            mark_crashed(self.run_dir, {"status": "CRASHED", "error_type": exc_type.__name__,
                                        "message": str(exc_value)})
        return False


def write_terminal_marker(run_dir: str | Path, marker: str, detail: Mapping[str, Any] | None = None) -> Path:
    if marker not in {"DONE", "CRASHED"}:
        raise ValueError("marker must be DONE or CRASHED")
    directory = Path(run_dir)
    directory.mkdir(parents=True, exist_ok=True)
    assert_run_mutable(directory)
    other = directory / ("CRASHED" if marker == "DONE" else "DONE")
    if other.exists():
        raise ValidationError("conflicting terminal marker: %s" % other)
    return atomic_write_json(directory / marker, detail or {"status": marker})


def mark_done(run_dir: str | Path, detail: Mapping[str, Any] | None = None) -> Path:
    return write_terminal_marker(run_dir, "DONE", detail)


def mark_crashed(run_dir: str | Path, detail: Mapping[str, Any] | None = None) -> Path:
    return write_terminal_marker(run_dir, "CRASHED", detail)
