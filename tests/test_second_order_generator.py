from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import generate_second_order_20k as subject
from experiment.batch_io import ValidationError, atomic_write_json, iter_jsonl, publish_batch, sha256_file, sha256_text

ROOT = Path(__file__).resolve().parents[1]


def raw(authority: dict, start: int, size: int = 1) -> dict:
    response = "response-%d" % authority["global_index"]
    return {**authority, "response": response, "response_sha256": sha256_text(response), "model": subject.MODEL_LABEL,
            "adapter": {"checkpoint_manifest_sha256": subject.evaluation.CHECKPOINT_MANIFEST_SHA256,
                        "adapter_model_sha256": subject.evaluation.ADAPTER_SHA256,
                        "adapter_config_sha256": subject.evaluation.ADAPTER_CONFIG_SHA256},
            "batch_start": start, "batch_size": size, "batch_seed": subject._batch_seed(start),
            "prompt_tokens": 3, "padded_input_tokens": 3, "output_tokens": 2, "termination": "eos", "is_blank": False}


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, count=1, name=subject.GPU_NAME):
        self.count, self.name = count, name

    def is_available(self): return True
    def device_count(self): return self.count
    def get_device_name(self, index): return self.name


class FakeTorch:
    def __init__(self, count=1, name=subject.GPU_NAME): self.cuda = FakeCuda(count, name)


class SecondOrderContractTests(unittest.TestCase):
    def test_exact_authoritative_direct_stream_and_amendment(self):
        rows = subject._load_source(ROOT / subject.INPUT_RELATIVE)
        self.assertEqual(len(rows), 20000)
        self.assertEqual([row["global_index"] for row in rows], list(range(20000)))
        self.assertEqual(set(rows[0]), set(subject.PROMPT_KEYS))
        self.assertNotIn("response", rows[0])
        self.assertEqual(sha256_file(ROOT / subject.INPUT_RELATIVE), subject.INPUT_SHA256)
        self.assertEqual(subject.validate_amendment(ROOT / subject.AMENDMENT_RELATIVE)["sha256"], subject.AMENDMENT_SHA256)

    def test_no_sharding_or_four_gpu_compatibility_path_remains(self):
        source = (ROOT / "experiment" / "generate_second_order_20k.py").read_text(encoding="utf-8")
        for forbidden in ("SHARDS", "shard_index", "coordinator", "CUDA_VISIBLE_DEVICES", "RTX PRO 4500", "worker_gpu_processes"):
            self.assertNotIn(forbidden, source)
        self.assertIn('GPU_NAME = "NVIDIA B200"', source)
        self.assertIn('"stream": "direct ordered global indices 0..19999"', source)
        self.assertNotIn("--coordinator-start", source)

    def test_left_padding_continuation_and_b200_enforcement(self):
        class Tensor:
            def __init__(self, value): self.value = value
        class Torch:
            def tensor(self, value, device=None): return Tensor(value)
        inputs, masks, width = subject._left_pad(Torch(), [{"input_ids": [7]}, {"input_ids": [8, 9]}], 0)
        self.assertEqual((inputs.value, masks.value, width), ([[0, 7], [8, 9]], [[0, 1], [1, 1]], 2))
        self.assertEqual(subject._trim_completion([4, 2, 9], 2), ([4, 2], "eos"))
        with patch.object(subject, "_packages", return_value=subject.RUNTIME_PACKAGES):
            subject._runtime(FakeTorch())
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(count=2))
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(name="NVIDIA B100"))

    def test_smoke_starts_at_512_and_converges_by_descending_or_bracket(self):
        def write_attempt(smoke, ordinal, size, recommendation, successful, oom=None, accepted=None):
            attempt = smoke / ("attempt-%04d-batch-%04d" % (ordinal, size)); attempt.mkdir(parents=True)
            report = {"format": "second-order-smoke-report-v2", "plan_sha256": "p", "attempted_batch_size": size,
                      "recommended_next_batch_size": recommendation, "successful_batch_size": successful,
                      "accepted_batch_size": accepted, "oom_evidence": oom}
            atomic_write_json(attempt / "smoke-report.json", report)
            atomic_write_json(attempt / "DONE", {"status": "DONE", "report_sha256": sha256_file(attempt / "smoke-report.json")})
            return attempt
        with tempfile.TemporaryDirectory() as temp:
            root, smoke = Path(temp), Path(temp) / "smoke"
            upper = write_attempt(smoke, 1, 512, 256, None, oom={"error_type": "OutOfMemoryError"})
            lower = write_attempt(smoke, 2, 256, 256, 256, accepted=256)
            gate = subject._smoke_gate(root, "p")
            self.assertEqual(gate["batch_size"], 256)
            self.assertEqual(gate["attempt"], lower.name)
            with self.assertRaises(ValidationError): subject._smoke_gate(root, "p", 128)
            self.assertEqual(upper.name, "attempt-0001-batch-0512")
        with tempfile.TemporaryDirectory() as temp:
            root, smoke = Path(temp), Path(temp) / "smoke"
            lower = write_attempt(smoke, 1, 512, 1024, 512)
            upper = write_attempt(smoke, 2, 1024, 512, 1024)
            gate = subject._smoke_gate(root, "p")
            self.assertEqual((gate["batch_size"], gate["attempt"], gate["bracket_attempt"]), (512, lower.name, upper.name))

    def test_direct_raw_binding_and_immutable_prefix_resume(self):
        authority = subject._load_source(ROOT / subject.INPUT_RELATIVE)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            rows = [raw(authority[0], 0, 2), raw(authority[1], 0, 2)]
            publish_batch(run / "raw" / "batches", "batch-00000", rows, key=lambda row: str(row["global_index"]),
                required_keys=subject.RAW_KEYS, extra_manifest={"plan_sha256": "p", "batch_start": 0, "batch_seed": 42, "actual_batch_size": 2})
            loaded = subject._worker_rows(run, authority, "p")
            self.assertEqual([row["global_index"] for row in loaded], [0, 1])
            drift = dict(rows[0]); drift["id"] = "wrong"
            with self.assertRaises(ValidationError): subject._validate_raw([drift], batch_start=0, authority=authority)
            self.assertEqual(loaded[-1]["batch_size"], 2)
            with self.assertRaises(FileExistsError):
                publish_batch(run / "raw" / "batches", "batch-00000", rows, key=lambda row: str(row["global_index"]), required_keys=subject.RAW_KEYS)

    def test_raw_memory_recommendation_and_tensor_cleanup_contract(self):
        self.assertEqual(subject.recommend_batch_size(512, .93, .1), 256)
        self.assertEqual(subject.recommend_batch_size(512, .1, .1), 1024)
        self.assertEqual(subject.recommend_batch_size(512, .7, .79), 512)
        self.assertEqual(subject.recommend_batch_size(512, .1, .1, oom=True), 256)
        with self.assertRaises(ValidationError): subject.recommend_batch_size(3, .1, .1)
        source = (ROOT / "experiment" / "generate_second_order_20k.py").read_text(encoding="utf-8")
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn("oom_before_publish", source)
        self.assertIn("write_jsonl_fsynced", source)
        self.assertIn("_fsync_directory(destination.parent)", source)

    def test_single_start_duplicate_refusal_and_monitor_done(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs" / "run"; root.parent.mkdir(parents=True); root.mkdir()
            atomic_write_json(root / "plan.json", {"plan": True})
            args = type("Args", (), {"run_root": root, "runs_root": root.parent, "input": root.parent / "input",
                "checkpoint": root.parent / "checkpoint", "staging_manifest": root.parent / "staging", "batch_size": 512})()
            manifest = {"repository": {"head": "x", "dirty": False}, "runtime_source_sha256": {}}
            class Process:
                pid = 123
                def poll(self): return None
            with patch.dict(sys.modules, {"torch": object()}), patch.object(subject, "plan", return_value={"manifest": manifest}), \
                 patch.object(subject, "_clean_live"), patch.object(subject, "_smoke_gate", return_value={"batch_size": 512}), \
                 patch.object(subject, "_runtime"), patch.object(subject.subprocess, "Popen", return_value=Process()), \
                 patch.object(subject, "_process_start_identity", return_value="identity"):
                result = subject.start(args)
                self.assertEqual(result["started"], 1)
                with self.assertRaises(ValidationError): subject.start(args)
            formal = root / "formal"; formal.mkdir()
            atomic_write_json(formal / "DONE", {"status": "DONE"})
            with patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_clean_live"), \
                 patch.object(subject, "_smoke_gate", return_value={"batch_size": 512}):
                self.assertEqual(subject.monitor(args)["state"], "DONE")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs" / "rollback"; root.parent.mkdir(parents=True); root.mkdir()
            atomic_write_json(root / "plan.json", {"plan": True})
            args = type("Args", (), {"run_root": root, "runs_root": root.parent, "input": root.parent / "input",
                "checkpoint": root.parent / "checkpoint", "staging_manifest": root.parent / "staging", "batch_size": 512})()
            with patch.dict(sys.modules, {"torch": object()}), patch.object(subject, "plan", return_value={"manifest": manifest}), \
                 patch.object(subject, "_clean_live"), patch.object(subject, "_smoke_gate", return_value={"batch_size": 512}), \
                 patch.object(subject, "_runtime"), patch.object(subject.subprocess, "Popen", side_effect=OSError("launch failed")):
                with self.assertRaises(OSError): subject.start(args)
            rollback = json.loads((root / "launch" / "rollback.json").read_text(encoding="utf-8"))
            self.assertTrue(rollback["supervisor_terminated"])
            self.assertFalse((root / "launch" / "supervisor.json").exists())

    def test_controller_is_ascii_ps51_remote_only_and_inputpath_safe(self):
        data = (ROOT / "scripts" / "generate-second-order-20k.ps1").read_bytes()
        text = data.decode("ascii")
        self.assertIn("#requires -Version 5.1", text)
        self.assertIn("[Alias('Input')][string]$InputPath", text)
        self.assertIn("'--input',$InputPath", text)
        self.assertIn("RemoteCode + '/mats12'", text)
        self.assertIn("/root/mats12-second-order-venv/bin/python", text)
        self.assertIn("2>&1", text)
        self.assertIn("('--start')", text)
        self.assertNotIn("coordinator-start", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("pod-down", text)
        self.assertIn("Specify exactly one action", text)


if __name__ == "__main__":
    unittest.main()
