from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import evaluate_qwen35_4b as ev
from experiment.batch_io import atomic_write_json, publish_batch, sha256_file, write_jsonl_fsynced

ROOT = Path(__file__).resolve().parents[1]


class FakeTokenizer:
    chat_template = "official"
    eos_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return [1, len(messages[0]["content"]), 3]

    def decode(self, ids, **kwargs):
        return "<think></think>"


class QwenEvaluationStaticTests(unittest.TestCase):
    def test_frozen_checkpoint_amendment_and_testbed(self):
        checkpoint = ROOT / ev.CHECKPOINT_RELATIVE
        identity = ev.validate_checkpoint(checkpoint)
        self.assertEqual(identity["adapter_model_sha256"], ev.ADAPTER_SHA256)
        self.assertEqual(identity["training_run_manifest_sha256"], ev.TRAINING_RUN_MANIFEST_SHA256)
        self.assertEqual(ev.validate_amendment(ROOT / ev.AMENDMENT_RELATIVE)["sha256"], ev.AMENDMENT_SHA256)
        amendment = json.loads((ROOT / ev.AMENDMENT_RELATIVE).read_text(encoding="utf-8"))
        self.assertEqual(amendment["base"]["architecture"], "Qwen3_5ForConditionalGeneration")
        items = ev.load_testbed(ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json",
                                ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
        self.assertEqual(len(items), 90)
        self.assertTrue(all(item["facts_gt"] for item in items))

    def test_qwen_prompt_is_user_only_no_thinking_and_does_not_add_bos(self):
        tokenizer = FakeTokenizer()
        self.assertEqual(ev.render_prompt_ids(tokenizer, "question"), [1, 8, 3])
        messages, kwargs = tokenizer.calls[0]
        self.assertEqual(messages, [{"role": "user", "content": "question"}])
        self.assertEqual(kwargs, {"tokenize": True, "add_generation_prompt": True, "enable_thinking": False})

    def test_only_frozen_smoke_and_formal_sizes_and_runtime_pins(self):
        self.assertEqual((ev._mode(2), ev._mode(90)), ("smoke", "formal"))
        with self.assertRaises(ev.ValidationError):
            ev._mode(3)
        ev._validate_requirements(ROOT / "experiment/requirements-qwen35-4b-runpod.txt")

    def test_plan_uses_tokenizer_only_and_never_allocates_a_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            staging_manifest = root / "model-manifest.json"
            staging_manifest.write_text("staging", encoding="utf-8")
            digest = sha256_file(staging_manifest)
            args = ev.build_parser().parse_args([
                "--plan", "--arm", ev.ARM_BASE, "--question-limit", "2", "--run-dir", str(root / "run"),
                "--runs-root", str(root), "--staging-manifest", str(staging_manifest),
            ])
            with patch.object(ev, "STAGING_MANIFEST_SHA256", digest), \
                 patch.object(ev, "validate_amendment", return_value={"path": "test", "sha256": "x"}), \
                 patch.object(ev.staging, "verify_manifest"), \
                 patch.object(ev, "_load_tokenizer", return_value=FakeTokenizer()), \
                 patch.object(ev, "_load_model", side_effect=AssertionError("plan allocated a model")):
                report = ev.plan(args)
            self.assertEqual(report["expected_rows"], 10)
            self.assertFalse((root / "run").exists())

    def test_runtime_shape_requires_conditional_generation_composite_model(self):
        torch = types.SimpleNamespace(bfloat16=object())
        conditional = type("Qwen3_5ForConditionalGeneration", (), {})()
        conditional.dtype = torch.bfloat16
        conditional.config = types.SimpleNamespace(model_type="qwen3_5")
        conditional.model = types.SimpleNamespace(
            language_model=types.SimpleNamespace(config=types.SimpleNamespace(num_hidden_layers=32)), visual=object())
        ev._assert_loaded_model_shape(conditional, torch)
        causal = type("Qwen3_5ForCausalLM", (), {})()
        causal.dtype, causal.config, causal.model = conditional.dtype, conditional.config, conditional.model
        with self.assertRaises(ev.ValidationError):
            ev._assert_loaded_model_shape(causal, torch)
        conditional.model.visual = None
        with self.assertRaises(ev.ValidationError):
            ev._assert_loaded_model_shape(conditional, torch)

    def test_loader_and_launcher_use_the_authorized_conditional_path(self):
        source = Path(ev.__file__).read_text(encoding="utf-8")
        launcher = Path("scripts/evaluate-qwen35-4b.ps1").read_text(encoding="ascii")
        self.assertIn("from transformers import Qwen3_5ForConditionalGeneration", source)
        self.assertNotIn("AutoModelForCausalLM", source)
        self.assertIn("--checkpoint", launcher)
        self.assertIn("qwen35-4b-abliterated-seed42-1ep-20260902T014813Z/checkpoints/step-000157", launcher)
        self.assertIn("stage-qwen35-4b.ps1 -Action Prepare", launcher)

    def test_base_and_adapter_metadata_are_distinct(self):
        self.assertIsNone(ev._adapter_identity(ev.ARM_BASE, None))
        identity = ev._adapter_identity(ev.ARM_ADAPTER, {
            "checkpoint_manifest_sha256": "c", "training_run_manifest_sha256": "r",
            "adapter_model_sha256": "a", "adapter_config_sha256": "x",
        })
        self.assertEqual(identity["target_count"], 248)
        self.assertEqual(ev._termination([1, 2], 2), "eos")
        self.assertEqual(ev._termination([1] * 1024, 2), "max_new_tokens")


class QwenEvaluationCompletionTests(unittest.TestCase):
    def _row(self, arm, item, sample, index=0):
        return {"arm_id": arm, "model": ev.staging.REPO_ID,
                "adapter": None, "topic": item["topic"], "prompt_id": str(item["prompt_id"]), "sample": sample,
                "question": item["question"], "facts_gt": item["facts_gt"], "response": "response",
                "generation": {"backend": "transformers", "question_index": index, "question_seed": 42 + index,
                               "prompt_tokens": 3, "prompt_ids_sha256": "hash-%d" % index, "output_tokens": 1,
                               "termination": "eos", "is_blank": False, "temperature": 1.0, "top_p": 1.0,
                               "top_k": 0, "max_new_tokens": 1024}, "judging": None}

    def _complete(self, root, name, mode, arm, items, smoke_gate=None):
        run = root / name
        run.mkdir()
        layout = [{"question_index": index, "prompt_id": str(item["prompt_id"]), "prompt_tokens": 3,
                   "prompt_ids_sha256": "hash-%d" % index} for index, item in enumerate(items)]
        manifest = {"format": "qwen35-4b-evaluation-v1", "mode": mode, "run_id": name, "arm_id": arm,
                    "amendment": {"path": ev.AMENDMENT_RELATIVE, "sha256": ev.AMENDMENT_SHA256},
                    "base": {"id": ev.staging.REPO_ID, "revision": ev.staging.REVISION, "path": ev.staging.LOCAL_DIR,
                             "class": "Qwen3_5ForConditionalGeneration", "dtype": "bfloat16"},
                    "evaluator": {"script_sha256": ev._evaluator_script_sha256()},
                    "tokenizer": {"path": ev.staging.LOCAL_DIR, "template": "official-user-only-no-thinking",
                                  "system_message": False, "extra_bos": False, "date_injected": False},
                    "adapter": None, "smoke_gate": smoke_gate,
                    "inputs": {"questions_lf_normalized_sha256": ev.QUESTIONS_SHA256,
                               "facts_lf_normalized_sha256": ev.FACTS_SHA256,
                               "staging_manifest_sha256": ev.STAGING_MANIFEST_SHA256},
                    "generation": {"samples_per_question": 5, "seed": "42 + zero-based question index",
                                   "one_call_per_question": True, "do_sample": True, "temperature": 1.0, "top_p": 1.0,
                                   "top_k": 0, "max_new_tokens": 1024, "bf16": True, "quantization": False,
                                   "offload": False, "trust_remote_code": False}, "question_count": len(items), "expected_rows": len(items) * 5,
                    "prompt_layout": layout, "runtime_packages_expected": dict(ev.staging.RUNTIME_VERSIONS),
                    "requirements_sha256": ev.REQUIREMENTS_SHA256}
        atomic_write_json(run / "manifest.json", manifest)
        rows = []
        for index, item in enumerate(items):
            batch = [self._row(arm, item, sample, index) for sample in range(5)]
            rows.extend(batch)
            publish_batch(run / "raw" / "batches", "question-%03d" % index, batch,
                          key=lambda row: "%s:%s" % (row["prompt_id"], row["sample"]), required_keys=ev.ROW_KEYS,
                          extra_manifest={"question_index": index, "question_seed": 42 + index,
                                          "manifest_sha256": ev._manifest_digest(manifest), "mode": mode,
                                          "run_id": name, "arm_id": arm})
        raw = run / "raw" / "responses.jsonl"
        _, digest = write_jsonl_fsynced(raw, rows)
        atomic_write_json(run / "raw" / "generation-record.json", {
            "format": "qwen35-4b-generation-record-v1", "row_count": len(rows), "sha256": digest,
            "blank_count": 0, "termination_counts": {"eos": len(rows)},
            "runtime": {"python": "test", "platform": "test", "packages": dict(ev.staging.RUNTIME_VERSIONS),
                        "requirements_sha256": ev.REQUIREMENTS_SHA256, "gpu": ev.AUTHORIZED_GPU,
                        "fast_paths": dict(ev.FAST_PATHS_EXPECTED),
                        "evaluator_script_sha256": ev._evaluator_script_sha256(), "git_commit": "a" * 40,
                        "peak_memory_bytes": 0}})
        atomic_write_json(run / "DONE", {"status": "DONE", "mode": mode, "arm_id": arm,
                                          "row_count": len(rows), "raw_sha256": digest})
        return run

    def test_matching_arm_smoke_is_required_and_formal_binding_is_semantic(self):
        source = ev.load_testbed(ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json",
                                 ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = self._complete(root, "base-smoke", "smoke", ev.ARM_BASE, source[:2])
            gate = ev.validate_completed_generation_run(smoke, "smoke", ev.ARM_BASE,
                                                        ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json",
                                                        ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
            formal = self._complete(root, "base-formal", "formal", ev.ARM_BASE, source, gate)
            self.assertEqual(ev.validate_completed_generation_run(formal, "formal", ev.ARM_BASE,
                ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json",
                ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")["row_count"], 450)
            record = json.loads((smoke / "raw" / "generation-record.json").read_text(encoding="utf-8"))
            record["runtime"]["packages"] = {}
            atomic_write_json(smoke / "raw" / "generation-record.json", record, overwrite=True)
            with self.assertRaises(ev.ValidationError):
                ev.validate_completed_generation_run(smoke, "smoke", ev.ARM_BASE,
                    ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json",
                    ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
            record["runtime"]["packages"] = dict(ev.staging.RUNTIME_VERSIONS)
            atomic_write_json(smoke / "raw" / "generation-record.json", record, overwrite=True)
            manifest = json.loads((formal / "manifest.json").read_text(encoding="utf-8"))
            manifest["smoke_gate"] = {**gate, "arm_id": ev.ARM_ADAPTER}
            atomic_write_json(formal / "manifest.json", manifest, overwrite=True)
            with self.assertRaises(ev.ValidationError):
                ev.validate_completed_generation_run(formal, "formal", ev.ARM_BASE,
                    ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json",
                    ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")


if __name__ == "__main__":
    unittest.main()
