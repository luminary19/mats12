import json
import pathlib
import tempfile
import unittest
from unittest.mock import patch

import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "runpod-side"))
from heartbeat import Heartbeat


class HeartbeatTests(unittest.TestCase):
    def test_clean_exit_writes_done_and_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            with Heartbeat("clean", interval=0.01, runs_root=root) as heartbeat:
                heartbeat.write_metric(iteration=1, loss=0.5)
            run_dir = pathlib.Path(root) / "clean"
            self.assertTrue((run_dir / "HEARTBEAT").exists())
            self.assertTrue((run_dir / "DONE").exists())
            self.assertFalse((run_dir / "CRASHED").exists())
            self.assertEqual(json.loads((run_dir / "metrics.jsonl").read_text(encoding="utf-8")), {"iteration": 1, "loss": 0.5})

    def test_metric_write_oserror_does_not_escape(self):
        with tempfile.TemporaryDirectory() as root:
            heartbeat = Heartbeat("write-failure", runs_root=root)
            with patch.object(pathlib.Path, "open", side_effect=OSError("temporary storage failure")):
                heartbeat.write_metric(loss=1.0)

    def test_rejects_unsafe_ids_and_existing_terminal_marker(self):
        with tempfile.TemporaryDirectory() as root:
            for run_id in ("../escape", "a/b", "a\\b", ".", ".."):
                with self.assertRaises(ValueError):
                    Heartbeat(run_id, runs_root=root)
            run_dir = pathlib.Path(root) / "complete"
            run_dir.mkdir()
            (run_dir / "DONE").write_text("done", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                Heartbeat("complete", runs_root=root)

    def test_crash_writes_marker_and_preserves_exception(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError, "expected failure"):
                with Heartbeat("crash", interval=0.01, runs_root=root):
                    raise RuntimeError("expected failure")
            text = (pathlib.Path(root) / "crash" / "CRASHED").read_text(encoding="utf-8")
            self.assertIn("RuntimeError: expected failure", text)
            self.assertFalse((pathlib.Path(root) / "crash" / "DONE").exists())


if __name__ == "__main__":
    unittest.main()
