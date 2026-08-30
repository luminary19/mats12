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
    text = "prompt-%d" % index
    return {"global_index": index, "id": "id-%d" % index, "source": "test", "prompt": text,
            "prompt_sha256": sha256_text(text), "input_tokens": tokens, "prompt_tokens": tokens,
            "input_ids": [index], "prompt_ids_sha256": "x" * 64}


def raw(item: dict, ordinal: int, size: int, padded: int) -> dict:
    response = " response-%d " % item["global_index"]
    return {"global_index": item["global_index"], "id": item["id"], "source": item["source"],
            "prompt": item["prompt"], "prompt_sha256": item["prompt_sha256"], "response": response,
            "response_sha256": sha256_text(response), "model": subject.MODEL_LABEL,
            "adapter": {"checkpoint_manifest_sha256": subject.evaluation.CHECKPOINT_MANIFEST_SHA256,
                        "adapter_model_sha256": subject.evaluation.ADAPTER_SHA256,
                        "adapter_config_sha256": subject.evaluation.ADAPTER_CONFIG_SHA256},
            "batch_ordinal": ordinal, "batch_size": size, "batch_seed": 42, "prompt_tokens": item["input_tokens"],
            "padded_input_tokens": padded, "output_tokens": 2, "generated_tokens": 2, "termination": "eos",
            "hit_token_cap": False, "is_blank": False}


class FakeCuda:
    class OutOfMemoryError(RuntimeError): pass
    def __init__(self, count=1, name=subject.GPU_NAME): self.count, self.name = count, name
    def is_available(self): return True
    def device_count(self): return self.count
    def get_device_name(self, index): return self.name


class FakeTorch:
    def __init__(self, count=1, name=subject.GPU_NAME): self.cuda = FakeCuda(count, name)


class SecondOrderContractTests(unittest.TestCase):
    def test_amendment_authoritative_input_and_no_smoke_or_budget_scheduler(self):
        rows = subject._load_source(ROOT / subject.INPUT_RELATIVE)
        self.assertEqual(len(rows), 20_000)
        self.assertEqual(subject.validate_amendment(ROOT / subject.AMENDMENT_RELATIVE)["sha256"], subject.AMENDMENT_SHA256)
        source = (ROOT / "experiment/generate_second_order_20k.py").read_text(encoding="utf-8").lower()
        controller = (ROOT / "scripts/generate-second-order-20k.ps1").read_text(encoding="ascii").lower()
        for forbidden in ("smoke", "conv_index", "_schedule_batch", "accepted_smoke", "formal-schedule"):
            self.assertNotIn(forbidden, source)
            self.assertNotIn(forbidden, controller)
        self.assertIn('gpu_name = "nvidia rtx pro 6000 blackwell workstation edition"', source)

    def test_decode_is_exact_teacher_parity_and_preserves_raw_whitespace(self):
        class Tokenizer:
            eos_token_id, pad_token_id = 2, 0
            def decode(self, ids, **kwargs): self.kwargs = kwargs; return " <%s> " % ",".join(map(str, ids))
        tokenizer = Tokenizer()
        fixture = ([9, 9, 4, 0, 5, 2], 2, 4096)
        self.assertEqual(subject._decode_completion(tokenizer, *fixture), teacher._decode_completion(tokenizer, *fixture))
        self.assertEqual(subject._decode_completion(tokenizer, *fixture)[0], " <4,5> ")
        self.assertFalse(tokenizer.kwargs["clean_up_tokenization_spaces"])

    def test_shortest_first_and_first_selected_group_is_exactly_256_regardless_of_lengths(self):
        work = subject._sorted_work([prompt(index, 10_000 - index) for index in range(600)])
        self.assertEqual([row["global_index"] for row in work[:3]], [599, 598, 597])
        first = list(work[:subject.MAX_BATCH_SIZE])
        self.assertEqual(len(first), 256)
        self.assertEqual(max(row["input_tokens"] for row in first), 10_000 - 344)

    def test_three_quarter_oom_transitions_and_no_pressure_reduction(self):
        current = 256
        seen = []
        for _ in range(4):
            seen.append(current); current = subject._next_batch_size_after_oom(current)
        self.assertEqual(seen + [current], [256, 192, 144, 108, 81])
        self.assertEqual(subject._next_batch_size_after_oom(1), 1)
        source = (ROOT / "experiment/generate_second_order_20k.py").read_text(encoding="utf-8")
        self.assertNotIn("allocated_memory_pressure >=", source)
        self.assertIn("current = _next_batch_size_after_oom(before)", source)

    def test_exact_journal_resume_validates_same_prefix_and_arbitrary_current(self):
        authority = [prompt(index, 1) for index in range(1024)]
        work = list(authority)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            first = authority[:256]
            first_rows = [raw(item, 0, 256, 1) for item in first]
            manifest = {"plan_sha256": "p", "batch_ordinal": 0, "actual_size": 256, "padded_input_tokens": 1,
                        "batch_size_before": 256, "batch_size_after": 256, "batch_seed": 42,
                        "original_indices": list(range(256))}
            publish_batch(run / "raw" / "batches", "batch-00000", first_rows, key=lambda row: str(row["global_index"]),
                          required_keys=subject.RAW_KEYS, extra_manifest=manifest)
            subject._append_scheduler_event(run, event="attempt", batch_ordinal=0, current_batch_size=256,
                                            actual_size=256, padded_input_tokens=1, original_indices=list(range(256)), seed=42)
            subject._append_scheduler_event(run, event="published", batch="batch-00000", batch_ordinal=0,
                                            current_batch_size=256, actual_size=256, padded_input_tokens=1,
                                            original_indices=list(range(256)), seed=42)
            subject._append_scheduler_event(run, event="attempt", batch_ordinal=1, current_batch_size=256,
                                            actual_size=256, padded_input_tokens=1, original_indices=list(range(256, 512)), seed=42)
            subject._append_scheduler_event(run, event="oom_before_publish", batch_ordinal=1,
                                            current_batch_size_before=256, current_batch_size_after=192, actual_size=256,
                                            padded_input_tokens=1, original_indices=list(range(256, 512)), error_type="OutOfMemoryError")
            rows, current = subject._worker_rows(run, authority, "p", work)
            self.assertEqual((len(rows), current), (256, 192))
            with self.assertRaises(ValidationError):
                subject._worker_rows(run, authority, "p", [*work[256:], *work[:256]])

    def test_dangling_attempt_is_exactly_retryable_after_abrupt_interruption(self):
        work = [prompt(index, 1) for index in range(600)]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            event = {"event": "attempt", "batch_ordinal": 0, "current_batch_size": 256, "actual_size": 256,
                     "padded_input_tokens": 1, "original_indices": list(range(256)), "seed": 42}
            subject._append_scheduler_event(run, **event)
            rows, current = subject._worker_rows(run, work, "p", work)
            self.assertEqual((rows, current), ([], 256))
            subject._append_scheduler_event(run, **event)
            rows, current = subject._worker_rows(run, work, "p", work)
            self.assertEqual((rows, current), ([], 256))
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp)
            subject._append_scheduler_event(run, **event)
            changed = {**event, "actual_size": 255, "original_indices": list(range(255))}
            subject._append_scheduler_event(run, **changed)
            with self.assertRaises(ValidationError):
                subject._worker_rows(run, work, "p", work)

    def test_mocked_formal_worker_oom_retries_same_prefix_then_publishes(self):
        prompts = [prompt(index, 1) for index in range(4)]
        layout = list(prompts)
        with tempfile.TemporaryDirectory() as temp, patch.object(subject, "EXPECTED_ROWS", 4):
            root = Path(temp) / "run"; root.mkdir(); atomic_write_json(root / "plan.json", {"plan": True})
            args = type("Args", (), {"run_root": root, "runs_root": root.parent, "batch_size": 256,
                                        "input": root / "input", "checkpoint": root / "checkpoint",
                                        "staging_manifest": root / "staging", "tokenizer_path": "t"})()
            calls = []
            def generate(torch, tokenizer, model, group, padded):
                calls.append([item["global_index"] for item in group])
                if len(calls) == 1: raise FakeCuda.OutOfMemoryError("out of memory")
                return [raw(item, 0, len(group), padded) for item in group], {"elapsed_seconds": 1.0,
                    "peak_allocated_bytes": 1, "peak_reserved_bytes": 1, "total_vram_bytes": 10,
                    "allocated_memory_pressure": .1, "reserved_memory_pressure": .1}
            manifest = {"repository": {"head": "x", "dirty": False}, "runtime_source_sha256": {}}
            with patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_load_source", return_value=prompts):
                subject.prepare(args)
            class Process:
                pid = 123
                def poll(self): return None
            with patch.dict(sys.modules, {"torch": object()}), patch.object(subject, "plan", return_value={"manifest": manifest}), \
                 patch.object(subject, "_clean_live"), patch.object(subject, "_runtime"), \
                 patch.object(subject.subprocess, "Popen", return_value=Process()), patch.object(subject, "_process_start_identity", return_value="identity"):
                subject.start(args)
            with patch.dict(sys.modules, {"torch": object()}), patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_clean_live"), \
                 patch.object(subject, "_authoritative_prompts", return_value=prompts), patch.object(subject, "_runtime"), \
                 patch.object(subject, "_staging", return_value={}), patch.object(subject, "verify_staged_snapshot"), \
                 patch.object(subject, "_load_tokenizer", return_value=object()), patch.object(subject, "_load_model", return_value=object()), \
                 patch.object(subject, "_layout", return_value=layout), patch.object(subject, "_prepare_layout", return_value=(layout, "layout")), \
                 patch.object(subject, "_attempt_after_allocator_cleanup", side_effect=lambda torch, operation: operation()), \
                 patch.object(subject, "_generate_attempt", side_effect=generate), patch.object(subject, "_is_oom", return_value=True), \
                 patch.object(subject, "_release_cuda_allocator_cache"):
                subject.worker(args)
            self.assertEqual(calls, [[0, 1, 2, 3], [0, 1, 2, 3]])
            record = json.loads((root / "formal" / "raw" / "record.json").read_text(encoding="utf-8"))
            self.assertTrue(record["first_published_batch_is_launch_probe"])
            self.assertEqual(record["current_batch_size"], 192)
            subject._write_no_clobber_jsonl(root / "prompt-layout.jsonl", subject._layout_evidence(layout))
            with patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_clean_live"), \
                 patch.object(subject, "_authoritative_prompts", return_value=prompts):
                final = subject.finalise(args)
            self.assertEqual(final["row_count"], 4)
            final_rows = [json.loads(line) for line in (root / "final" / "output" / "rollouts.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["id"] for row in final_rows], ["id-0", "id-1", "id-2", "id-3"])

    def test_start_rejects_wrong_batch_dirty_or_wrong_gpu_and_controller_is_ascii(self):
        data = (ROOT / "scripts/generate-second-order-20k.ps1").read_bytes(); text = data.decode("ascii")
        self.assertIn("#requires -Version 5.1", text)
        self.assertIn("[Alias('Input')][string]$InputPath", text)
        self.assertIn("$BatchSize -ne 256", text)
        self.assertNotIn("Smoke", text)
        with patch.object(subject, "_git_state", return_value={"head": "x", "dirty": True}):
            with self.assertRaises(ValidationError):
                subject._clean_live({"repository": {"head": "x", "dirty": False}, "runtime_source_sha256": {}})
        with patch.object(subject, "_packages", return_value=subject.RUNTIME_PACKAGES):
            subject._runtime(FakeTorch())
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(count=2))
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(name="NVIDIA B100"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "run"; root.mkdir(); atomic_write_json(root / "plan.json", {"plan": True})
            args = type("Args", (), {"run_root": root, "runs_root": root.parent, "batch_size": 128,
                                        "input": root / "input", "checkpoint": root / "checkpoint", "staging_manifest": root / "staging"})()
            with patch.object(subject, "plan", return_value={"manifest": {}}), patch.object(subject, "_clean_live"):
                with self.assertRaises(ValidationError): subject.start(args)

    def test_final_order_is_original_index_order(self):
        rows = {1: raw(prompt(1, 1), 0, 1, 1), 0: raw(prompt(0, 1), 1, 1, 1)}
        output = [{key: rows[index][key] for key in subject.ROW_KEYS} for index in range(2)]
        self.assertEqual([row["id"] for row in output], ["id-0", "id-1"])


if __name__ == "__main__":
    unittest.main()
