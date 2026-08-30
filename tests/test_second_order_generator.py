from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import generate_second_order_20k as subject
from experiment import generate_teacher_20k as teacher
from experiment.batch_io import ValidationError, atomic_write_json, publish_batch, sha256_file, sha256_text

ROOT = Path(__file__).resolve().parents[1]


def prompt(index: int, tokens: int) -> dict:
    value = "prompt-%d" % index
    return {"global_index": index, "id": "id-%d" % index, "source": "test", "prompt": value,
            "prompt_sha256": sha256_text(value), "input_tokens": tokens, "prompt_tokens": tokens,
            "input_ids": list(range(tokens)), "prompt_ids_sha256": "x" * 64}


def raw(authority: dict, ordinal: int, size: int, padded: int) -> dict:
    response = " response-%d " % authority["global_index"]
    return {"global_index": authority["global_index"], "id": authority["id"], "source": authority["source"],
            "prompt": authority["prompt"], "prompt_sha256": authority["prompt_sha256"], "response": response,
            "response_sha256": sha256_text(response), "model": subject.MODEL_LABEL,
            "adapter": {"checkpoint_manifest_sha256": subject.evaluation.CHECKPOINT_MANIFEST_SHA256,
                        "adapter_model_sha256": subject.evaluation.ADAPTER_SHA256,
                        "adapter_config_sha256": subject.evaluation.ADAPTER_CONFIG_SHA256},
            "batch_ordinal": ordinal, "batch_size": size, "batch_seed": 42, "prompt_tokens": authority["input_tokens"],
            "padded_input_tokens": padded, "output_tokens": 2, "generated_tokens": 2, "termination": "eos",
            "hit_token_cap": False, "is_blank": False}


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, count=1, name=subject.GPU_NAME): self.count, self.name = count, name
    def is_available(self): return True
    def device_count(self): return self.count
    def get_device_name(self, index): return self.name


class FakeTorch:
    def __init__(self, count=1, name=subject.GPU_NAME): self.cuda = FakeCuda(count, name)


class SecondOrderContractTests(unittest.TestCase):
    def test_authoritative_input_amendment_and_single_b200_contract(self):
        rows = subject._load_source(ROOT / subject.INPUT_RELATIVE)
        self.assertEqual(len(rows), 20_000)
        self.assertEqual([row["global_index"] for row in rows], list(range(20_000)))
        self.assertEqual(subject.validate_amendment(ROOT / subject.AMENDMENT_RELATIVE)["sha256"], subject.AMENDMENT_SHA256)
        self.assertEqual((subject.MAX_BATCH_SIZE, subject.CONV_INDEX_BUDGET, subject.MEMORY_PRESSURE_THRESHOLD), (512, 131072, .92))
        source = (ROOT / "experiment/generate_second_order_20k.py").read_text(encoding="utf-8")
        for forbidden in ("_smoke_selection", "SHARDS", "shard_index", "CUDA_VISIBLE_DEVICES", "tensor_parallel=True", "42 + global"):
            self.assertNotIn(forbidden, source)
        self.assertIn('GPU_NAME = "NVIDIA B200"', source)
        self.assertIn("generate_teacher_20k as teacher", source)

    def test_exact_teacher_schedule_and_decode_parity(self):
        work = [prompt(3, 11), prompt(2, 11), prompt(0, 7217), prompt(1, 7217)]
        for maximum in (512, 18, 17):
            self.assertEqual(subject._schedule_batch(work, maximum, 131072), teacher._schedule_batch(work, maximum, 131072))
        group, padded = subject._schedule_batch([prompt(i, 7217) for i in range(50)], 512, 131072)
        self.assertEqual((len(group), padded, len(group) * padded), (18, 7217, 129906))

        class Tokenizer:
            eos_token_id, pad_token_id = 2, 0
            def decode(self, ids, **kwargs):
                self.kwargs = kwargs
                return "<%s>" % ",".join(map(str, ids))
        tokenizer = Tokenizer()
        fixture = ([9, 9, 4, 0, 5, 2, 7], 2, 4096)
        self.assertEqual(subject._decode_completion(tokenizer, *fixture), teacher._decode_completion(tokenizer, *fixture))
        self.assertEqual(subject._decode_completion(tokenizer, *fixture)[0], "<4,5>")
        self.assertFalse(tokenizer.kwargs["clean_up_tokenization_spaces"])

    def test_short_sorted_groups_and_exact_20k_offline_coverage(self):
        ordered = subject._sorted_work([prompt(4, 12), prompt(1, 3), prompt(0, 3)])
        self.assertEqual([row["global_index"] for row in ordered], [0, 1, 4])
        short = [prompt(i, 10) for i in range(512)]
        group, padded = subject._schedule_batch(short, 512, 131072)
        self.assertEqual((len(group), padded), (512, 10))
        authority = subject._load_source(ROOT / subject.INPUT_RELATIVE)
        work = sorted(({**row, "input_tokens": 1} for row in authority), key=lambda row: (row["input_tokens"], row["global_index"]))
        groups = subject._simulate_schedule(work, 512)
        flattened = [item for group in groups for item in group["original_indices"]]
        self.assertEqual(flattened, list(range(20_000)))
        self.assertTrue(all(group["product"] <= 131072 for group in groups))

    def test_seed_pressure_and_oom_reductions_are_monotonic(self):
        self.assertEqual(subject._next_scheduler_max_after_success(512, .919), 512)
        self.assertEqual(subject._next_scheduler_max_after_success(512, .92), 256)
        self.assertEqual(subject._next_scheduler_max_after_success(18, .99), 9)
        self.assertEqual(subject.MASTER_SEED, 42)
        source = (ROOT / "experiment/generate_second_order_20k.py").read_text(encoding="utf-8")
        self.assertIn("torch.manual_seed(MASTER_SEED)", source)
        self.assertIn("event=\"oom_before_publish\"", source)
        self.assertIn("_release_cuda_allocator_cache(torch)", source)
        self.assertIn("clean_up_tokenization_spaces=False", (ROOT / "experiment/generate_teacher_20k.py").read_text(encoding="utf-8"))

    def test_sorted_prefix_resume_allows_noncontiguous_original_indices(self):
        authority = [prompt(index, 20) for index in range(10)]
        authority[2], authority[7], authority[9] = prompt(2, 50_000), prompt(7, 50_000), prompt(9, 50_000)
        work = [authority[2], authority[9], authority[7]]
        rows = [raw(authority[2], 0, 2, 50_000), raw(authority[9], 0, 2, 50_000)]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            manifest = {"plan_sha256": "p", "batch_ordinal": 0, "actual_size": 2, "padded_input_tokens": 50_000,
                        "budget_product": 100_000, "scheduler_max_before": 512, "scheduler_max_after": 512,
                        "batch_seed": 42, "original_indices": [2, 9]}
            publish_batch(run / "raw" / "batches", "batch-00000", rows, key=lambda row: str(row["global_index"]),
                          required_keys=subject.RAW_KEYS, extra_manifest=manifest)
            subject._append_scheduler_event(run, event="attempt", batch_ordinal=0, scheduler_max=512, actual_size=2,
                                            padded_input_tokens=50_000, original_indices=[2, 9], seed=42)
            subject._append_scheduler_event(run, event="published", batch="batch-00000", batch_ordinal=0,
                                            scheduler_max_before=512, scheduler_max_after=512, actual_size=2)
            loaded, scheduler = subject._worker_rows(run, authority, "p", 512, work)
            self.assertEqual(([row["global_index"] for row in loaded], scheduler), ([2, 9], 512))
            with self.assertRaises(ValidationError):
                subject._worker_rows(run, authority, "p", 512, [authority[9], authority[2], authority[7]])

    def test_oom_is_journaled_before_publication_and_reconstructs_next_maximum(self):
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            work = [prompt(index, 1) for index in range(512)]
            subject._append_scheduler_event(run, event="attempt", batch_ordinal=0, scheduler_max=256, actual_size=256,
                                            padded_input_tokens=1, original_indices=list(range(256)), seed=42)
            subject._append_scheduler_event(run, event="oom_before_publish", batch_ordinal=0, scheduler_max_before=256,
                                            scheduler_max_after=128, actual_size=256, padded_input_tokens=1,
                                            original_indices=list(range(256)), error_type="OutOfMemoryError")
            rows, maximum = subject._worker_rows(run, work, "p", 256, work)
            self.assertEqual((rows, maximum), ([], 128))
            no_rows, accepted_maximum = subject._worker_rows(Path(temp) / "accepted-256", work, "p", 256, [])
            self.assertEqual((no_rows, accepted_maximum), ([], 256))
            with self.assertRaises(ValidationError):
                subject._worker_rows(Path(temp) / "invalid", work, "p", 257, [])

    def test_smoke_attempts_have_distinct_schedules_and_gate_binds_accepted_256(self):
        with tempfile.TemporaryDirectory() as temp:
            root, smoke = Path(temp), Path(temp) / "smoke"
            def write_attempt(ordinal, maximum, after, accepted):
                attempt = smoke / ("attempt-%04d-max-%04d" % (ordinal, maximum)); attempt.mkdir(parents=True)
                atomic_write_json(attempt / "schedule.json", {"scheduler_max": maximum, "layout_sha256": "a" * 64,
                                                                "attempt": ordinal})
                report = {"plan_sha256": "p", "scheduler_max_before": maximum, "scheduler_max_after": after,
                          "actual_size": maximum, "padded_input_tokens": 10, "prompt_layout_sha256": "a" * 64,
                          "schedule_sha256": sha256_file(attempt / "schedule.json"),
                          "oom_evidence": None if accepted else {"error_type": "OutOfMemoryError"}, "accepted": accepted}
                atomic_write_json(attempt / "smoke-report.json", report)
                atomic_write_json(attempt / "DONE", {"status": "DONE", "report_sha256": sha256_file(attempt / "smoke-report.json")})
                return attempt
            first = write_attempt(1, 512, 256, False)
            second = write_attempt(2, 256, 256, True)
            gate = subject._smoke_gate(root, "p")
            self.assertEqual((gate["scheduler_max"], gate["actual_size"]), (256, 256))
            self.assertEqual(gate["schedule_sha256"], sha256_file(second / "schedule.json"))
            self.assertNotEqual(sha256_file(first / "schedule.json"), gate["schedule_sha256"])
            self.assertFalse((root / "formal-schedule.json").exists())
            formal_work = [prompt(index, 1) for index in range(subject.EXPECTED_ROWS)]
            subject._write_schedule_simulation(root / "formal-schedule.json", formal_work, gate["scheduler_max"], "a" * 64)
            self.assertEqual(subject._json(root / "formal-schedule.json")["scheduler_max"], 256)
            with self.assertRaises(ValidationError): subject._smoke_gate(root, "p", 128)

    def test_raw_decode_preserves_whitespace_and_final_order_is_original_index(self):
        first, second = prompt(1, 1), prompt(0, 1)
        decoded = {1: {**raw(first, 0, 1, 1), "response": "  later\n", "response_sha256": sha256_text("  later\n")},
                   0: {**raw(second, 1, 1, 1), "response": " first ", "response_sha256": sha256_text(" first ")}}
        output = [{key: decoded[index][key] for key in subject.ROW_KEYS} for index in range(2)]
        self.assertEqual([row["id"] for row in output], ["id-0", "id-1"])
        self.assertEqual(output[1]["response"], "  later\n")

    def test_controller_is_ascii_ps51_and_runtime_refuses_multiple_gpus(self):
        data = (ROOT / "scripts/generate-second-order-20k.ps1").read_bytes()
        text = data.decode("ascii")
        self.assertIn("#requires -Version 5.1", text)
        self.assertIn("[Alias('Input')][string]$InputPath", text)
        self.assertNotIn("coordinator", text)
        with patch.object(subject, "_packages", return_value=subject.RUNTIME_PACKAGES):
            subject._runtime(FakeTorch())
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(count=2))


if __name__ == "__main__":
    unittest.main()
