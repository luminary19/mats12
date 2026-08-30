from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import generate_second_order_20k as subject
from experiment import generate_teacher_20k as teacher
from experiment.batch_io import ValidationError, atomic_write_json, publish_batch, sha256_text

ROOT = Path(__file__).resolve().parents[1]
MIB = 1024 * 1024


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


def memory_policy(total_mib: int = 97887, baseline_mib: int = 7156) -> dict:
    geometry = {"num_hidden_layers": 28, "num_key_value_heads": 8, "num_attention_heads": 24,
                "hidden_size": 3072, "head_dim": 128, "dtype_bytes": 2,
                "kv_bytes_per_token_per_sequence": subject.EXPECTED_KV_BYTES_PER_TOKEN}
    total, baseline = total_mib * MIB, baseline_mib * MIB
    value = {"format": "second-order-memory-policy-v1", "geometry": geometry,
             "logical_max_batch_size": 256, "max_new_tokens": 4096,
             "allocated_vram_budget_numerator": 13, "allocated_vram_budget_denominator": 20,
             "allocated_vram_budget_bytes": total * 13 // 20,
             "post_load_allocated_bytes": baseline, "post_load_reserved_bytes": baseline,
             "total_vram_bytes": total, "baseline_tolerance_bytes": 64 * MIB}
    value["sha256"] = sha256_text(json.dumps(value, sort_keys=True, separators=(",", ":")))
    return value


class FakeCuda:
    class OutOfMemoryError(RuntimeError): pass
    def __init__(self, count=1, name=subject.GPU_NAME, allocated=100, reserved=100, total=1000):
        self.count, self.name, self.allocated, self.reserved, self.total = count, name, allocated, reserved, total
        self.empty_calls = self.sync_calls = 0
    def is_available(self): return True
    def device_count(self): return self.count
    def get_device_name(self, index): return self.name
    def empty_cache(self): self.empty_calls += 1; self.reserved = self.allocated
    def synchronize(self): self.sync_calls += 1
    def mem_get_info(self): return self.total - self.reserved, self.total
    def memory_allocated(self): return self.allocated
    def memory_reserved(self): return self.reserved


class FakeTorch:
    def __init__(self, **kwargs): self.cuda = FakeCuda(**kwargs)


class Config:
    num_hidden_layers = 28
    num_key_value_heads = 8
    num_attention_heads = 24
    hidden_size = 3072
    head_dim = 128


class Model:
    config = Config()
    dtype = "torch.bfloat16"


class SecondOrderContractTests(unittest.TestCase):
    def test_amendment_binds_memory_budget_and_disables_oom_retry(self):
        rows = subject._load_source(ROOT / subject.CLEAN_SOURCE_RELATIVE, ROOT / subject.ORGANIC_SOURCE_RELATIVE)
        self.assertEqual(len(rows), 20_000)
        self.assertEqual([row["global_index"] for row in rows[-4:]], [19_996, 19_997, 19_998, 19_999])
        amendment = subject.validate_amendment(ROOT / subject.AMENDMENT_RELATIVE)["value"]
        self.assertEqual(amendment["format"], "second-order-llama-adapter-20000-amendment-v5")
        self.assertEqual(amendment["execution"]["logical_max_batch_size"], 256)
        self.assertEqual(amendment["execution"]["oom_policy"], "unexpected invariant failure before publication; never reduce and retry")
        source = (ROOT / "experiment/generate_second_order_20k.py").read_text(encoding="utf-8")
        self.assertNotIn("_next_batch_size_after_oom", source)
        self.assertNotIn('event="oom_before_publish"', source)
        self.assertIn('cache_implementation="static"', source)

    def test_decode_is_exact_teacher_parity_and_preserves_raw_whitespace(self):
        class Tokenizer:
            eos_token_id, pad_token_id = 2, 0
            def decode(self, ids, **kwargs): self.kwargs = kwargs; return " <%s> " % ",".join(map(str, ids))
        tokenizer = Tokenizer(); fixture = ([9, 9, 4, 0, 5, 2], 2, 4096)
        self.assertEqual(subject._decode_completion(tokenizer, *fixture), teacher._decode_completion(tokenizer, *fixture))
        self.assertEqual(subject._decode_completion(tokenizer, *fixture)[0], " <4,5> ")
        self.assertFalse(tokenizer.kwargs["clean_up_tokenization_spaces"])

    def test_exact_authorized_kv_geometry(self):
        geometry = subject._kv_geometry(Model())
        self.assertEqual(geometry["kv_bytes_per_token_per_sequence"], 28 * 2 * 8 * 128 * 2)
        Model.config.num_hidden_layers = 27
        try:
            with self.assertRaises(ValidationError): subject._kv_geometry(Model())
        finally:
            Model.config.num_hidden_layers = 28

    def test_memory_selection_is_budgeted_and_shrinks_for_long_prompts(self):
        policy = memory_policy()
        short, short_evidence = subject._select_physical_batch([prompt(i, 47) for i in range(300)], policy)
        long, long_evidence = subject._select_physical_batch([prompt(i, 7217) for i in range(300)], policy)
        expected = min(256, (policy["allocated_vram_budget_bytes"] - policy["post_load_allocated_bytes"])
                       // ((47 + 4096) * subject.EXPECTED_KV_BYTES_PER_TOKEN))
        self.assertEqual(len(short), expected)
        self.assertGreaterEqual(len(short), 120); self.assertLessEqual(len(short), 128)
        self.assertLess(len(long), len(short))
        for evidence in (short_evidence, long_evidence):
            self.assertLessEqual(evidence["projected_allocated_bytes"], evidence["allocated_vram_budget_bytes"])
            self.assertEqual(evidence["actual_size"], evidence["physical_batch_size"])

    def test_allocator_cleanup_and_leak_detection(self):
        torch = FakeTorch(allocated=100, reserved=400, total=1000)
        state = subject._allocator_state_after_cleanup(torch)
        self.assertEqual((state["allocated_bytes"], state["reserved_bytes"]), (100, 100))
        self.assertEqual((torch.cuda.empty_calls, torch.cuda.sync_calls), (1, 1))
        policy = {"post_load_allocated_bytes": 100, "baseline_tolerance_bytes": 10, "total_vram_bytes": 1000}
        subject._assert_allocator_baseline(torch, policy, "test")
        torch.cuda.allocated = 111
        with self.assertRaises(ValidationError): subject._assert_allocator_baseline(torch, policy, "leak")

    def test_resume_binds_exact_memory_safe_prefix_and_rejects_old_oom_event(self):
        authority = [prompt(index, 47) for index in range(300)]
        policy = memory_policy(); group, selection = subject._select_physical_batch(authority, policy)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); atomic_write_json(run / "memory-policy.json", policy)
            rows = [raw(item, 0, len(group), 47) for item in group]
            manifest = {"plan_sha256": "p", "batch_ordinal": 0, "batch_seed": 42,
                        "original_indices": [row["global_index"] for row in group], **selection}
            publish_batch(run / "raw" / "batches", "batch-00000", rows, key=lambda row: str(row["global_index"]),
                          required_keys=subject.RAW_KEYS, extra_manifest=manifest)
            attempt = {"event": "attempt", "batch_ordinal": 0, **selection,
                       "original_indices": manifest["original_indices"], "seed": 42}
            published = {"event": "published", "batch": "batch-00000", "batch_ordinal": 0, **selection,
                         "original_indices": manifest["original_indices"], "seed": 42}
            subject._append_scheduler_event(run, **attempt); subject._append_scheduler_event(run, **published)
            validated, last = subject._worker_rows(run, authority, "p", authority, policy)
            self.assertEqual((len(validated), last), (len(group), len(group)))
            subject._append_scheduler_event(run, event="oom_before_publish")
            with self.assertRaises(ValidationError): subject._worker_rows(run, authority, "p", authority, policy)

    def test_resume_rejects_memory_selection_or_prefix_drift(self):
        work = [prompt(index, 47) for index in range(300)]; policy = memory_policy()
        group, selection = subject._select_physical_batch(work, policy)
        event = {"event": "attempt", "batch_ordinal": 0, **selection,
                 "original_indices": [row["global_index"] for row in group], "seed": 42}
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); subject._append_scheduler_event(run, **event)
            with self.assertRaises(ValidationError): subject._worker_rows(run, work, "p", work, policy)
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp); subject._append_scheduler_event(run, **event)
            with self.assertRaises(ValidationError): subject._worker_rows(run, work, "p", [*work[1:], work[0]], policy)

    def test_unexpected_generation_failures_never_retry_or_publish(self):
        prompts = [prompt(index, 47) for index in range(4)]; policy = memory_policy()
        cases = [(FakeCuda.OutOfMemoryError("out of memory"), "unexpected_oom"),
                 (RuntimeError("kernel failed"), "unexpected_failure")]
        for failure, expected_event in cases:
            with self.subTest(expected_event=expected_event), tempfile.TemporaryDirectory() as temp, patch.object(subject, "EXPECTED_ROWS", 4):
                root = Path(temp) / "run"; root.mkdir(); atomic_write_json(root / "plan.json", {"plan": True})
                args = type("Args", (), {"run_root": root, "runs_root": root.parent, "batch_size": 256,
                                            "input": root / "input", "checkpoint": root / "checkpoint",
                                            "staging_manifest": root / "staging", "tokenizer_path": "t"})()
                manifest = {"repository": {"head": "x", "dirty": False}, "runtime_source_sha256": {}}
                calls = []
                def generate(*args):
                    calls.append(1); raise failure
                with patch.dict(sys.modules, {"torch": FakeTorch()}), patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_clean_live"), patch.object(subject, "_authoritative_prompts", return_value=prompts), patch.object(subject, "_runtime"), patch.object(subject, "_staging", return_value={}), patch.object(subject, "verify_staged_snapshot"), patch.object(subject, "_load_tokenizer", return_value=object()), patch.object(subject, "_load_model", return_value=Model()), patch.object(subject, "_layout", return_value=prompts), patch.object(subject, "_prepare_layout", return_value=(prompts, "layout")), patch.object(subject, "_memory_policy", return_value=policy), patch.object(subject, "_assert_allocator_baseline", return_value={"allocated_bytes": 1}), patch.object(subject, "_allocator_state_after_cleanup", return_value={"allocated_bytes": 1, "reserved_bytes": 1, "free_bytes": 9, "total_vram_bytes": 10}), patch.object(subject, "_generate_attempt", side_effect=generate):
                    with self.assertRaises(ValidationError): subject.worker(args)
                self.assertEqual(len(calls), 1)
                batches = root / "formal" / "raw" / "batches"
                self.assertEqual(list(batches.glob("*")) if batches.exists() else [], [])
                events = [json.loads(line) for line in (root / "formal" / "scheduler.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual([event["event"] for event in events], ["attempt", expected_event])
                with self.assertRaises(ValidationError):
                    subject._worker_rows(root / "formal", prompts, "p", prompts, policy)

    def test_success_cleans_before_and_after_then_publishes(self):
        prompts = [prompt(index, 47) for index in range(4)]; policy = memory_policy()
        details = {"elapsed_seconds": 1.0, "peak_allocated_bytes": 2, "peak_reserved_bytes": 3,
                   "total_vram_bytes": 10, "allocated_memory_pressure": .2, "reserved_memory_pressure": .3}
        with tempfile.TemporaryDirectory() as temp, patch.object(subject, "EXPECTED_ROWS", 4):
            root = Path(temp) / "run"; root.mkdir(); atomic_write_json(root / "plan.json", {"plan": True})
            args = type("Args", (), {"run_root": root, "runs_root": root.parent, "batch_size": 256,
                                        "input": root / "input", "checkpoint": root / "checkpoint",
                                        "staging_manifest": root / "staging", "tokenizer_path": "t"})()
            manifest = {"repository": {"head": "x", "dirty": False}, "runtime_source_sha256": {}}; phases = []
            def baseline(torch, policy, phase): phases.append(phase); return {"phase": phase, "allocated_bytes": 1}
            def generate(torch, tokenizer, model, group, padded):
                return [raw(item, 0, len(group), padded) for item in group], details
            with patch.dict(sys.modules, {"torch": object()}), patch.object(subject, "plan", return_value={"manifest": manifest}), patch.object(subject, "_clean_live"), patch.object(subject, "_authoritative_prompts", return_value=prompts), patch.object(subject, "_runtime"), patch.object(subject, "_staging", return_value={}), patch.object(subject, "verify_staged_snapshot"), patch.object(subject, "_load_tokenizer", return_value=object()), patch.object(subject, "_load_model", return_value=Model()), patch.object(subject, "_layout", return_value=prompts), patch.object(subject, "_prepare_layout", return_value=(prompts, "layout")), patch.object(subject, "_memory_policy", return_value=policy), patch.object(subject, "_assert_allocator_baseline", side_effect=baseline), patch.object(subject, "_generate_attempt", side_effect=generate):
                record = subject.worker(args)
            self.assertEqual(phases, ["before_generation", "after_generation"])
            self.assertEqual(record["row_count"], 4)
            self.assertEqual(len(subject.finalized_batches(root / "formal" / "raw" / "batches")), 1)

    def test_runtime_and_controller_contract(self):
        data = (ROOT / "scripts/generate-second-order-20k.ps1").read_bytes(); text = data.decode("ascii")
        self.assertIn("#requires -Version 5.1", text)
        self.assertIn("logical physical-batch ceiling", text)
        with patch.object(subject, "_packages", return_value=subject.RUNTIME_PACKAGES):
            subject._runtime(FakeTorch())
            subject._runtime(FakeTorch(name="NVIDIA RTX PRO 6000 Blackwell Server Edition"))
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(count=2))
            with self.assertRaises(ValidationError): subject._runtime(FakeTorch(name="NVIDIA B100"))

    def test_final_order_is_original_index_order(self):
        rows = {1: raw(prompt(1, 1), 0, 1, 1), 0: raw(prompt(0, 1), 1, 1, 1)}
        output = [{key: rows[index][key] for key in subject.ROW_KEYS} for index in range(2)]
        self.assertEqual([row["id"] for row in output], ["id-0", "id-1"])


if __name__ == "__main__":
    unittest.main()
