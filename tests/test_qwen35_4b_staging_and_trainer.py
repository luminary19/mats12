import argparse
import json
import os
import pickle
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import stage_qwen35_4b_base as staging
from experiment import train_qwen35_4b_lora_local as trainer
from experiment.batch_io import ValidationError, atomic_write_json, sha256_file


class StageContractTests(unittest.TestCase):
    def _manifest(self, root):
        model = root / "model"; model.mkdir()
        for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json", "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"):
            (model / name).write_text("x", encoding="utf-8")
        files, count, total = staging._file_manifest(model)
        value = {"format": staging.MANIFEST_FORMAT, "repo_id": staging.REPO_ID,
                 "revision": staging.REVISION, "resolved_revision": staging.REVISION,
                 "local_dir": str(model), "hf": {"HF_HOME": staging.HF_HOME, "HF_HUB_DISABLE_XET": staging.HF_HUB_DISABLE_XET},
                 "files": files, "file_count": count, "bytes": total, "download_metadata_verified": True,
                 "runtime": {"python": "test", "platform": "test", "packages": dict(staging.RUNTIME_VERSIONS)},
                 "artifacts": {"script": {"path": "experiment/stage_qwen35_4b_base.py", "sha256": sha256_file(Path(staging.__file__))}, "requirements": {"path": staging.REQUIREMENTS, "sha256": sha256_file(staging._requirements_path())}},
                 "offline_validation": {"model_class": "Qwen3_5ForConditionalGeneration", "model_type": "qwen3_5", "language_layers": 32, "chat_template_sha256": "a" * 64, "no_thinking_prefix_contains_empty_think_block": True, "processor_class": "Processor", "tokenizer_class": "Tokenizer", "parameter_count": 1, "smoke": {"generated_tokens": 1}}}
        value["integrity_sha256"] = staging._manifest_integrity(value)
        manifest = root / "model-manifest.json"; atomic_write_json(manifest, value)
        return model, manifest

    def test_plan_is_static_and_never_creates_run_or_imports_hub(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            args = argparse.Namespace(run_dir=None)
            before = set(__import__("sys").modules)
            result = staging.plan(args)
            self.assertEqual(result["network"], "not contacted")
            self.assertFalse(run.exists())
            self.assertNotIn("huggingface_hub", set(__import__("sys").modules) - before)

    def test_manifest_verification_rejects_file_and_manifest_tampering(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model, manifest = self._manifest(root)
            with patch.object(staging, "LOCAL_DIR", str(model)), patch.object(staging, "MIN_SNAPSHOT_BYTES", 1):
                staging.verify_manifest(manifest)
                (model / "config.json").write_text('{"tampered":true}', encoding="utf-8")
                with self.assertRaises(ValidationError): staging.verify_manifest(manifest)
                value = json.loads(manifest.read_text(encoding="utf-8")); value["bytes"] = 99
                atomic_write_json(manifest, value, overwrite=True)
                with self.assertRaises(ValidationError): staging.verify_manifest(manifest, verify_files=False)

    def test_self_consistent_invented_manifest_lacks_required_semantic_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); model, manifest = self._manifest(root)
            value = json.loads(manifest.read_text(encoding="utf-8")); value["download_metadata_verified"] = False
            value["integrity_sha256"] = staging._manifest_integrity(value); atomic_write_json(manifest, value, overwrite=True)
            with patch.object(staging, "LOCAL_DIR", str(model)), patch.object(staging, "MIN_SNAPSHOT_BYTES", 1):
                with self.assertRaises(ValidationError): staging.verify_manifest(manifest, verify_files=False)

    def test_cuda_only_smoke_rejects_offload_and_quantization(self):
        class Parameter:
            def __init__(self, device): self.device = types.SimpleNamespace(type=device)
            def numel(self): return 4_659_865_088
        model = types.SimpleNamespace(parameters=lambda: [Parameter("cuda")], config=types.SimpleNamespace(quantization_config=None), hf_quantizer=None)
        self.assertEqual(staging._assert_cuda_only(model), 4_659_865_088)
        model.parameters = lambda: [Parameter("cpu")]
        with self.assertRaises(ValidationError): staging._assert_cuda_only(model)

    def test_frozen_stage_identity_and_no_floating_revision_are_literal(self):
        self.assertEqual(staging.REPO_ID, "Qwen/Qwen3.5-4B-Base")
        self.assertEqual(staging.REVISION, "1001bb4d826a52d1f399e183466143f4da7b741b")
        source = Path(staging.__file__).read_text(encoding="utf-8")
        self.assertIn("api.model_info(REPO_ID, revision=REVISION)", source)
        self.assertIn("snapshot_download(repo_id=REPO_ID, revision=REVISION", source)
        self.assertNotIn("revision=\"main\"", source)


class FakeTokenizer:
    chat_template = "official-template"
    pad_token_id, eos_token = 0, "<eos>"
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize and enable_thinking is False
        prompt = messages[0]["content"]
        if add_generation_prompt:
            return [10, len(prompt), 30, 31]  # includes the official empty-think prefix conceptually
        assert messages[1]["reasoning_content"] == ""
        return [10, len(prompt), 30, 31, *[100 + ord(c) for c in messages[1]["content"]], 2]
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True); (Path(path) / "tokenizer.json").write_text("{}", encoding="utf-8")


class TrainerPureContractTests(unittest.TestCase):
    def test_official_template_prefix_masking_strip_and_verbatim_think_tags(self):
        feature = trainer.feature_for_row(FakeTokenizer(), {"id": "x", "prompt": " hello ", "response": " <think>preserved</think> answer ", "source": "s", "model": "m"})
        self.assertEqual(feature["prefix_tokens"], 4)
        self.assertEqual(feature["labels"][:4], [-100] * 4)
        self.assertEqual(feature["input_ids"][4:], feature["labels"][4:])
        self.assertEqual(feature["input_ids"][-1], 2)

    def test_non_prefix_template_fails_closed_and_overlength_is_never_truncated(self):
        class Bad(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [2] if kwargs["add_generation_prompt"] else [3]
        with self.assertRaises(ValidationError): trainer.render_pair(Bad(), "p", "r")
        class Long(FakeTokenizer):
            def apply_chat_template(self, messages, **kwargs):
                return [1] if kwargs["add_generation_prompt"] else [1] + [2] * trainer.MAX_LENGTH
        with self.assertRaises(ValidationError): trainer.audit_tokenize(Long(), [{"id": "x", "prompt": "p", "response": "r"}])

    def test_exact_248_language_targets_and_visual_exclusion(self):
        class Linear: pass
        modules = {name: Linear() for name in trainer.expected_language_target_names()}
        modules["model.visual.fake"] = Linear()
        model = types.SimpleNamespace(named_modules=lambda: list(modules.items()))
        torch = types.SimpleNamespace(nn=types.SimpleNamespace(Linear=Linear))
        found = trainer.discover_language_targets(model, torch)
        self.assertEqual(len(found), 248)
        self.assertFalse(any("visual" in name for name in found))
        del modules[next(iter(trainer.expected_language_target_names()))]
        with self.assertRaises(ValidationError): trainer.discover_language_targets(model, torch)

    def test_post_peft_target_assertion_rejects_vision(self):
        class Module: lora_A, lora_B = object(), object()
        names = ["base_model." + name for name in trainer.expected_language_target_names()]
        model = types.SimpleNamespace(named_modules=lambda: [(name, Module()) for name in names])
        self.assertEqual(trainer.assert_resolved_lora_targets(model)["resolved_target_count"], 248)
        model = types.SimpleNamespace(named_modules=lambda: [(name, Module()) for name in [*names, "base_model.model.visual.proj"]])
        with self.assertRaises(ValidationError): trainer.assert_resolved_lora_targets(model)

    def test_recipe_order_objective_schedule_and_final_group(self):
        self.assertEqual(trainer.tinker_single_epoch_order(10, 42), [4, 8, 2, 0, 6, 9, 1, 5, 7, 3])
        groups = trainer.accumulation_group_sizes()
        self.assertEqual((len(groups), groups[-1]), (157, 32))
        self.assertEqual(trainer.checkpoint_schedule(), list(range(4, 157, 4)) + [157])
        calls = []
        class Loss:
            def backward(self): calls.append(1)
            def detach(self): return self
            def cpu(self): return 2.0
        self.assertEqual(sum(trainer.backward_microbatch_loss(Loss()) for _ in range(32)), 64.0)
        self.assertEqual(len(calls), 32)
        self.assertAlmostEqual(trainer.lr_at(156, 157, .0006, .05, .1), .00006, places=6)

    def test_fast_paths_fail_closed(self):
        missing = types.SimpleNamespace(is_fla_available=lambda: True)
        with self.assertRaises(ValidationError): trainer.assert_fast_paths(missing)
        false = types.SimpleNamespace(is_fla_available=lambda: False, is_causal_conv1d_available=lambda: True)
        with self.assertRaises(ValidationError): trainer.assert_fast_paths(false)
        good = types.SimpleNamespace(is_fla_available=lambda: True, is_causal_conv1d_available=lambda: True)
        self.assertEqual(trainer.assert_fast_paths(good), {"flash-linear-attention": "is_fla_available", "causal-conv1d": "is_causal_conv1d_available"})

    def test_authoritative_corpus_provenance_allows_byte_identical_local_mirror(self):
        root = Path(trainer._repo_root()); corpus = root / "runs/abliterated-20000-20260829T022737Z/output/rollouts.jsonl"; manifest = corpus.with_name("manifest.json"); finalizer = corpus.parents[1] / "manifest.json"
        rows = trainer.validate_authoritative_corpus(corpus, manifest, finalizer)
        self.assertEqual(len(rows), 20_000)
        exposed = [row for row in rows if "</think>" in row["response"]]
        self.assertEqual(len(exposed), 7)
        for row in exposed: trainer.feature_for_row(FakeTokenizer(), row)
        with tempfile.TemporaryDirectory() as temporary:
            copy = Path(temporary); copied_corpus, copied_manifest, copied_finalizer = copy / "rollouts.jsonl", copy / "output-manifest.json", copy / "finalizer.json"
            copied_corpus.write_bytes(corpus.read_bytes()); copied_manifest.write_bytes(manifest.read_bytes()); copied_finalizer.write_bytes(finalizer.read_bytes())
            self.assertEqual(len(trainer.validate_authoritative_corpus(copied_corpus, copied_manifest, copied_finalizer)), 20_000)

    def test_durable_defaults_requirements_and_amendment_tamper_fail_closed(self):
        args = trainer.build_parser().parse_args(["--plan", "--staging-manifest", "stage", "--run-dir", "/workspace/runs/new"])
        self.assertEqual((args.corpus, args.corpus_manifest, args.finalizer_manifest), (trainer.CORPUS_PATH, trainer.CORPUS_MANIFEST_PATH, trainer.FINALIZER_MANIFEST_PATH))
        requirements = Path("experiment/requirements-qwen35-4b-runpod.txt").read_text(encoding="utf-8")
        self.assertIn("flash-linear-attention[cuda]==0.5.2", requirements); self.assertIn("huggingface-hub==1.28.0", requirements)
        with patch.object(trainer, "AMENDMENT_SHA256", "0" * 64):
            with self.assertRaises(ValidationError): trainer._validate_amendment()

    def test_execution_guards_reject_zero_full_skip_and_non_durable_run_paths(self):
        args = types.SimpleNamespace(max_steps=0, skip_save=False, resume_from=None, effective_batch=128)
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.max_steps, args.skip_save = None, True
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.max_steps, args.skip_save = 1, False
        trainer._validate_execution_mode(args)
        with self.assertRaises(ValidationError): trainer._safe_training_run_dir(Path("local-run"))
        self.assertEqual(trainer._safe_training_run_dir(Path("/workspace/runs/new-run")), Path("/workspace/runs/new-run"))


class FakeTorch:
    class cuda:
        @staticmethod
        def get_rng_state_all(): return [b"cuda"]
    def get_rng_state(self): return b"cpu"
    def save(self, value, path):
        with open(path, "wb") as handle: pickle.dump(value, handle)


class FakeSaveable:
    def save_pretrained(self, path):
        Path(path).mkdir(parents=True); (Path(path) / "artifact.json").write_text("{}", encoding="utf-8")


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_tamper_resume_and_disjointness(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); corpus = root / "corpus"; manifest = root / "corpus-manifest"; finalizer = root / "finalizer"; stage = root / "stage"
            for path, text in ((corpus, "c"), (manifest, "m"), (finalizer, "f"), (stage, "s")): path.write_text(text, encoding="utf-8")
            args = argparse.Namespace(corpus=corpus, corpus_manifest=manifest, finalizer_manifest=finalizer, staging_manifest=stage, seed=42, effective_batch=128, lr=6e-4, warmup_ratio=.05, lr_final_frac=.1)
            order = trainer.tinker_single_epoch_order(20_000, 42); run = root / "parent"; run.mkdir(); atomic_write_json(run / "manifest.json", {"run": "parent"})
            metadata = {**trainer._input_identity(args, order), "run_dir": str(run.resolve()), "run_manifest_sha256": sha256_file(run / "manifest.json"), "global_step": 4, "total_steps": 157, "next_order_offset": 512, "examples_processed": 512, "training_complete": False, "maximum_recomputed_processed_samples": trainer.MAX_RECOMPUTED_PROCESSED_SAMPLES, "scheduler": {"step": 4, "total_steps": 157, "last_lr": trainer.lr_at(3, 157, 6e-4, .05, .1)}, "adapter_identity": trainer._expected_adapter_identity("template-hash")}
            checkpoint = trainer._publish_checkpoint(FakeSaveable(), FakeTokenizer(), types.SimpleNamespace(state_dict=lambda: {}), FakeTorch(), run, metadata)
            self.assertEqual(trainer.validate_resume_checkpoint(checkpoint, args, order)["next_order_offset"], 512)
            checkpoint_manifest_path = checkpoint / "checkpoint-manifest.json"
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="utf-8"))
            checkpoint_manifest["metadata"]["next_order_offset"] = 513
            atomic_write_json(checkpoint_manifest_path, checkpoint_manifest, overwrite=True)
            with self.assertRaises(ValidationError): trainer.validate_resume_checkpoint(checkpoint, args, order)
            checkpoint_manifest["metadata"]["next_order_offset"] = 512
            atomic_write_json(checkpoint_manifest_path, checkpoint_manifest, overwrite=True)
            with (checkpoint / "optimizer.pt").open("ab") as handle: handle.write(b"tamper")
            with self.assertRaises(ValidationError): trainer.validate_checkpoint_payload(checkpoint)
            with self.assertRaises(ValidationError): trainer._assert_disjoint_run(run / "child", checkpoint, {"run_dir": str(run)})


if __name__ == "__main__":
    unittest.main()
