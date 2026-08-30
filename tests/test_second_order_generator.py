from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import generate_second_order_20k as subject
from experiment.batch_io import ValidationError, atomic_write_json, iter_jsonl, publish_batch, sha256_text

ROOT = Path(__file__).resolve().parents[1]


def prompt(index: int) -> dict:
    value = "prompt-%d" % index
    return {"global_index": index, "id": "id-%d" % index, "source": "source", "prompt": value,
            "prompt_sha256": sha256_text(value)}


def raw(index: int, shard: int, start: int, size: int = 1) -> dict:
    item = prompt(index); response = "response-%d" % index
    return {**item, "response": response, "response_sha256": sha256_text(response), "model": subject.MODEL_LABEL,
            "adapter": {"checkpoint_manifest_sha256": subject.evaluation.CHECKPOINT_MANIFEST_SHA256,
                        "adapter_model_sha256": subject.evaluation.ADAPTER_SHA256,
                        "adapter_config_sha256": subject.evaluation.ADAPTER_CONFIG_SHA256},
            "shard_index": shard, "batch_start": start, "batch_size": size, "batch_seed": subject._batch_seed(start),
            "prompt_tokens": 3, "padded_input_tokens": 3, "output_tokens": 2, "termination": "eos", "is_blank": False}


class FakeTensor:
    def __init__(self, value): self.value = value

class FakeTorch:
    def __init__(self): self.seeds = []
    def tensor(self, value, device=None): return FakeTensor(value)
    def manual_seed(self, seed): self.seeds.append(("torch", seed))
    class cuda:
        values = []
        @classmethod
        def manual_seed_all(cls, seed): cls.values.append(seed)


class SecondOrderContractTests(unittest.TestCase):
    def test_exact_source_is_projected_prompt_only_and_sharded(self):
        rows = subject._load_source(ROOT / subject.INPUT_RELATIVE)
        self.assertEqual(len(rows), 20000)
        self.assertEqual(set(rows[0]), set(subject.PROMPT_KEYS))
        self.assertNotIn("response", rows[0])
        self.assertEqual([(a, b) for a, b in subject.SHARDS], [(0, 5000), (5000, 10000), (10000, 15000), (15000, 20000)])
        all_indices = [index for start, end in subject.SHARDS for index in range(start, end)]
        self.assertEqual(all_indices, list(range(20000)))

    def test_amendment_and_checkpoint_are_exactly_bound(self):
        amendment = subject.validate_amendment(ROOT / subject.AMENDMENT_RELATIVE)
        self.assertEqual(amendment["sha256"], subject.AMENDMENT_SHA256)
        self.assertEqual(subject.evaluation.validate_checkpoint(ROOT / subject.CHECKPOINT_RELATIVE)["adapter_model_sha256"], subject.evaluation.ADAPTER_SHA256)

    def test_left_padding_continuation_and_first_eos(self):
        torch = FakeTorch(); inputs, masks, width = subject._left_pad(torch, [{"input_ids": [7]}, {"input_ids": [8, 9]}], 0)
        self.assertEqual((inputs.value, masks.value, width), ([[0, 7], [8, 9]], [[0, 1], [1, 1]], 2))
        self.assertEqual(subject._trim_completion([4, 2, 9], 2), ([4, 2], "eos"))
        self.assertEqual(subject._trim_completion([4] * 4096, 2)[1], "max_new_tokens")

    def test_stable_batch_seed_and_recommendation_boundaries(self):
        self.assertEqual(subject._batch_seed(0), 42); self.assertEqual(subject._batch_seed(5000), 5042)
        self.assertEqual(subject.recommend_batch_size(256, .93, .1), 128)
        self.assertEqual(subject.recommend_batch_size(256, .1, .1), 512)
        self.assertEqual(subject.recommend_batch_size(256, .7, .79), 256)
        self.assertEqual(subject.recommend_batch_size(256, .1, .1, oom=True), 128)
        with self.assertRaises(ValidationError): subject.recommend_batch_size(3, .1, .1)

    def test_oom_is_classified_before_any_batch_publication(self):
        class Cuda:
            class OutOfMemoryError(RuntimeError): pass
        class Torch: cuda = Cuda
        self.assertTrue(subject._is_oom(Torch, Cuda.OutOfMemoryError("oom")))
        self.assertTrue(subject._is_oom(Torch, RuntimeError("CUDA out of memory")))
        self.assertFalse(subject._is_oom(Torch, RuntimeError("other failure")))

    def test_safe_run_root_rejects_nested_or_input_overlapping_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            runs = Path(temp) / "runs"; runs.mkdir(); source = Path(temp) / "source"; source.write_text("x")
            with self.assertRaises(ValidationError): subject._safe_run_root(runs / "nested" / "run", runs, [source])
            with self.assertRaises(ValidationError): subject._safe_run_root(source, runs, [source])

    def test_smoke_state_machine_self_accepts_and_brackets_without_oscillation(self):
        def write_attempt(smoke, ordinal, size, recommendation, successful, oom=None, accepted=None):
            attempt = smoke / ("attempt-%04d-batch-%04d" % (ordinal, size)); attempt.mkdir(parents=True)
            report = {"format":"second-order-smoke-report-v2","plan_sha256":"p",
                      "attempted_batch_size":size,"recommended_next_batch_size":recommendation,
                      "successful_batch_size":successful,"accepted_batch_size":accepted,"oom_evidence":oom}
            atomic_write_json(attempt / "smoke-report.json", report)
            atomic_write_json(attempt / "DONE", {"status":"DONE", "report_sha256":subject.sha256_file(attempt / "smoke-report.json")})
            return attempt
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); smoke = root / "smoke"
            attempt = write_attempt(smoke, 1, 256, 256, 256, accepted=256)
            gate = {"attempt":attempt.name,"batch_size":256,"plan_sha256":"p","report_sha256":subject.sha256_file(attempt / "smoke-report.json")}
            atomic_write_json(smoke / "accepted.json", gate); atomic_write_json(smoke / "DONE", {"status":"DONE", **gate})
            self.assertEqual(subject._smoke_gate(root, "p", required=True), gate)
            with self.assertRaises(ValidationError): subject._smoke_gate(root, "p", 256)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); smoke = root / "smoke"
            upper = write_attempt(smoke, 1, 256, 128, None, oom={"error_type":"OutOfMemoryError"})
            lower = write_attempt(smoke, 2, 128, 256, 128)
            gate = subject._smoke_gate(root, "p")
            self.assertEqual(gate["batch_size"], 128); self.assertEqual(gate["attempt"], lower.name)
            self.assertEqual(gate["bracket_attempt"], upper.name)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); smoke = root / "smoke"
            lower = write_attempt(smoke, 1, 256, 512, 256)
            upper = write_attempt(smoke, 2, 512, 256, 512)
            gate = subject._smoke_gate(root, "p")
            self.assertEqual(gate["batch_size"], 256); self.assertEqual(gate["attempt"], lower.name)
            self.assertEqual(gate["bracket_attempt"], upper.name)
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValidationError): subject._smoke_gate(Path(temp), "p", 128)

    def test_deterministic_smoke_selection_is_representative_and_stress(self):
        rows = [prompt(i) for i in range(300)]
        first = subject._smoke_selection(rows, 256); second = subject._smoke_selection(rows, 256)
        self.assertEqual(first, second); self.assertEqual(len(first), 256)
        self.assertEqual(len({item["global_index"] for item in first}), 256)

    def test_immutable_batches_reject_semantic_drift_and_resume_coverage(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); entries = [prompt(i) for i in range(3)]
            batch = [raw(0, 0, 0, 2), raw(1, 0, 0, 2)]
            publish_batch(run / "raw" / "batches", "batch-00000", batch, key=lambda row: str(row["global_index"]), required_keys=subject.RAW_KEYS, extra_manifest={"shard_index": 0, "plan_sha256": "p", "batch_start": 0, "batch_seed": 42, "actual_batch_size": 2})
            loaded = subject._worker_rows(run, 0, entries, "p")
            self.assertEqual([row["global_index"] for row in loaded], [0, 1])
            bad = dict(batch[0]); bad["response_sha256"] = "0" * 64
            with self.assertRaises(ValidationError): subject._validate_raw([bad], shard=0, start=0, end=5000, batch_start=0)
            with self.assertRaises(FileExistsError): publish_batch(run / "raw" / "batches", "batch-00000", batch, key=lambda row: str(row["global_index"]), required_keys=subject.RAW_KEYS)

    def test_final_merge_preserves_original_order_and_checksum(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "runs" / "second-order"; root.parent.mkdir(parents=True); root.mkdir()
            atomic_write_json(root / "plan.json", {"plan": True})
            for shard in range(4):
                location = root / ("shard-%d" % shard); location.mkdir(); atomic_write_json(location / "DONE", {"status": "DONE"})
            manifest = {"run_id": "second-order"}
            shard_rows = [[prompt(index) for index in range(start, end)] for start, end in subject.SHARDS]
            def worker_rows(run, shard, prompts, digest):
                start, end = subject.SHARDS[shard]
                return [raw(index, shard, index) for index in range(start, end)]
            args = type("Args", (), {"run_root": root})()
            def fake_json(path):
                shard_dir = path.parent.parent if path.name == "record.json" else path.parent
                shard = int(shard_dir.name.split("-")[1])
                rows = worker_rows(None, shard, None, None)
                record = {"format":"second-order-worker-record-v2","plan_sha256":subject.sha256_file(root / "plan.json"),"accepted_smoke":{"batch_size":256},"shard_index":shard,"row_count":5000,"raw_sha256":subject.sha256_text(json.dumps(rows, ensure_ascii=False, separators=(",", ":"))),"blank_count":0,"termination_counts":{"eos":5000}}
                return record if path.name == "record.json" else {"status":"DONE", **record}
            with patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_clean_live"), patch.object(subject, "_smoke_gate", return_value={"batch_size":256}), patch.object(subject, "_authoritative_shards", return_value=shard_rows), patch.object(subject, "_worker_rows", side_effect=worker_rows), patch.object(subject, "_json", side_effect=fake_json):
                outcome = subject.finalise(args)
            rows = list(iter_jsonl(root / "final" / "output" / "rollouts.jsonl"))
            self.assertEqual(outcome["row_count"], 20000); self.assertEqual([row["id"] for row in rows[:2]], ["id-0", "id-1"])
            self.assertEqual(rows[-1]["id"], "id-19999")
            self.assertEqual(outcome["sha256"], subject.sha256_file(root / "final" / "output" / "rollouts.jsonl"))

    def test_launcher_is_ascii_ps51_and_binds_every_worker_gpu(self):
        data = (ROOT / "scripts" / "generate-second-order-20k.ps1").read_bytes()
        text = data.decode("ascii")
        self.assertIn("#requires -Version 5.1", text)
        self.assertIn("RemoteCode + '/mats12'", text)
        self.assertIn("Resolve-RunpodPodOrThrow", text)
        self.assertIn("Invoke-PodSsh", text)
        self.assertIn("coordinator-start", text)
        self.assertNotIn("nvidia-smi", text)
        self.assertNotIn("pod-down", text)
        self.assertIn("/root/mats12-second-order-venv/bin/python", text)
        self.assertIn("Specify exactly one action", text)
        source = (ROOT / "experiment" / "generate_second_order_20k.py").read_text(encoding="utf-8")
        self.assertEqual(source.count("start_new_session=True"), 1)

if __name__ == "__main__":
    unittest.main()
