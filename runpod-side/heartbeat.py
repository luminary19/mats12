"""Durable heartbeat, JSONL metrics, and exclusive terminal markers."""
import json
import os
import pathlib
import re
import threading
import time
import traceback

RUNS_ROOT = os.environ.get("RUNS_ROOT", "/workspace/runs")
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class Heartbeat:
    def __init__(self, run_id, interval=10, runs_root=RUNS_ROOT):
        run_id = str(run_id)
        if not RUN_ID_RE.fullmatch(run_id) or run_id in {".", ".."}:
            raise ValueError("run_id must be one safe path component")
        self.run_id = run_id
        self.dir = pathlib.Path(runs_root) / run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        if (self.dir / "DONE").exists() or (self.dir / "CRASHED").exists():
            raise RuntimeError("run directory already has a terminal marker")
        self.file = self.dir / "HEARTBEAT"
        self.metrics = self.dir / "metrics.jsonl"
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None

    def _beat(self):
        while not self._stop.is_set():
            try:
                self.file.touch()
            except OSError as exc:
                print(f"[heartbeat] touch failed (continuing): {exc}", flush=True)
            self._stop.wait(self.interval)

    def start(self):
        try:
            self.file.touch()
        except OSError as exc:
            print(f"[heartbeat] initial touch failed (continuing): {exc}", flush=True)
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def _write_terminal(self, name, text):
        other = self.dir / ("CRASHED" if name == "DONE" else "DONE")
        if other.exists():
            raise RuntimeError("conflicting terminal marker exists")
        final = self.dir / name
        temp = self.dir / f".{name}.{os.getpid()}.{time.time_ns()}.tmp"
        try:
            with temp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, final)
        finally:
            try:
                temp.unlink()
            except FileNotFoundError:
                pass

    def stop(self, exc_info=None):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        try:
            if exc_info and exc_info[0] is not None:
                self._write_terminal("CRASHED", stamp + "\n" + "".join(traceback.format_exception(*exc_info)))
            else:
                self._write_terminal("DONE", stamp + "\n")
        except OSError as exc:
            print(f"[heartbeat] could not write status marker (continuing): {exc}", flush=True)

    def write_metric(self, **values):
        try:
            with self.metrics.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(values, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        except (OSError, TypeError, ValueError) as exc:
            print(f"[heartbeat] write_metric failed (continuing): {exc}", flush=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.stop((exc_type, exc_value, exc_tb))
        return False
