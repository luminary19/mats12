import argparse
import json
import os
import pickle
import random
import sys
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
                 "offline_validation": {"model_class": "Qwen3_5ForConditionalGeneration", "model_type": "qwen3_5", "language_layers": 32, "chat_template_sha256": "a" * 64, "no_thinking_prefix_contains_empty_think_block": True, "processor_class": "Processor", "tokenizer_class": "Tokenizer", "parameter_count": 1, "fast_paths": {"torch_recurrent_gated_delta_rule": "fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule": "fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule", "causal_conv1d_fn": "causal_conv1d.causal_conv1d_interface.causal_conv1d_fn", "causal_conv1d_update": "causal_conv1d.causal_conv1d_interface.causal_conv1d_update"}, "gpu": {"name": staging.AUTHORIZED_GPU_NAMES[0], "authorized_policy": list(staging.AUTHORIZED_GPU_NAMES)}, "smoke": {"generated_tokens": 1}}}
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
        def wrapper(module, name):
            def implementation(): pass
            implementation.__module__, implementation.__name__ = module, name
            def function(): return implementation
            return function
        good = types.SimpleNamespace(
            torch_recurrent_gated_delta_rule=wrapper("fla.ops.gated_delta_rule.fused_recurrent", "fused_recurrent_gated_delta_rule"),
            torch_chunk_gated_delta_rule=wrapper("fla.ops.gated_delta_rule.chunk", "chunk_gated_delta_rule"),
            causal_conv1d_fn=wrapper("causal_conv1d.causal_conv1d_interface", "causal_conv1d_fn"),
            causal_conv1d_update=wrapper("causal_conv1d.causal_conv1d_interface", "causal_conv1d_update"))
        evidence = trainer.assert_fast_paths(good)
        self.assertEqual(set(evidence), {"torch_recurrent_gated_delta_rule", "torch_chunk_gated_delta_rule", "causal_conv1d_fn", "causal_conv1d_update"})
        bad = types.SimpleNamespace(**vars(good))
        bad.causal_conv1d_fn = wrapper("transformers.models.qwen3_5.modeling_qwen3_5", "causal_conv1d_fn")
        with self.assertRaises(ValidationError): trainer.assert_fast_paths(bad)

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

    def test_run_kind_and_full_execution_guards_fail_closed(self):
        args = types.SimpleNamespace(run_kind=None, max_steps=1, skip_save=False, resume_from=None, accepted_smoke_run=None, accepted_resume_smoke_run=None)
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.run_kind, args.max_steps = "smoke", 0
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.max_steps, args.skip_save = 1, True
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.skip_save = False
        trainer._validate_execution_mode(args)
        args.run_kind, args.max_steps = "full", 1
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.max_steps, args.accepted_smoke_run = None, None
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.accepted_smoke_run = Path("/workspace/runs/smoke")
        with self.assertRaises(ValidationError): trainer._validate_execution_mode(args)
        args.accepted_resume_smoke_run = Path("/workspace/runs/resume-smoke")
        trainer._validate_execution_mode(args)
        with self.assertRaises(ValidationError): trainer._safe_training_run_dir(Path("local-run"))
        self.assertEqual(trainer._safe_training_run_dir(Path("/workspace/runs/new-run")), Path("/workspace/runs/new-run"))

    def test_exact_authorized_gpu_allowlist_and_amendment_semantics(self):
        self.assertEqual(trainer.AUTHORIZED_GPU_NAMES, ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "NVIDIA RTX PRO 4500 Blackwell"))
        self.assertNotIn("NVIDIA A100-SXM4-80GB", trainer.AUTHORIZED_GPU_NAMES)
        self.assertEqual(trainer._validate_amendment(), trainer.AMENDMENT_SHA256)
        amendment = json.loads((Path(trainer._repo_root()) / trainer.AMENDMENT).read_text(encoding="utf-8"))
        self.assertEqual(amendment["authorization"]["hardware_policy"]["ordered_exact_names"], list(trainer.AUTHORIZED_GPU_NAMES))
        self.assertEqual(amendment["roles"]["full"]["optimizer_steps"], 157)


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

    def test_checkpoint_index_is_published_before_pruning(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"; run.mkdir()
            saveable, tokenizer = FakeSaveable(), FakeTokenizer()
            optimizer, torch = types.SimpleNamespace(state_dict=lambda: {}), FakeTorch()
            def metadata(step):
                return {"global_step": step, "next_order_offset": step * 128,
                        "scheduler": {"step": step, "total_steps": 157, "last_lr": 0.0}}
            trainer._publish_checkpoint(saveable, tokenizer, optimizer, torch, run, metadata(4))
            trainer._publish_checkpoint(saveable, tokenizer, optimizer, torch, run, metadata(8))
            real_rmtree = trainer.shutil.rmtree
            def interrupt_prune(path, *args, **kwargs):
                if Path(path).name == "step-000004":
                    raise RuntimeError("simulated interruption during prune")
                return real_rmtree(path, *args, **kwargs)
            with patch.object(trainer.shutil, "rmtree", side_effect=interrupt_prune):
                with self.assertRaises(RuntimeError):
                    trainer._publish_checkpoint(saveable, tokenizer, optimizer, torch, run, metadata(12))
            checkpoint_root = run / "checkpoints"
            index = json.loads((checkpoint_root / "index.json").read_text(encoding="utf-8"))
            self.assertEqual(index["checkpoints"], ["step-000008", "step-000012"])
            self.assertTrue(all((checkpoint_root / name).is_dir() for name in index["checkpoints"]))
            self.assertTrue((checkpoint_root / "step-000004").is_dir())


class CompletedRunValidationTests(unittest.TestCase):
    def _write_completed(self, root, name, kind, *, continuation=None, accepted=None, accepted_resume=None):
        run = root / name; run.mkdir()
        runtime = {"gpu": {"name": trainer.AUTHORIZED_GPU_NAMES[0], "total_memory": 1}, "packages": {}, "python": "test", "run_kind": kind, "authorized_gpu_policy": list(trainer.AUTHORIZED_GPU_NAMES)}
        start_step = 0 if continuation is None else continuation["start_global_step"]
        start_offset = 0 if continuation is None else continuation["start_next_order_offset"]
        final_step = start_step + 1 if kind == "smoke" else 157
        final_offset = start_offset + 128 if kind == "smoke" else 20_000
        commit = "a" * 40
        launch = {"format": trainer.LAUNCH_EVIDENCE_FORMAT, "run_id": name, "commit": commit, "pid": 1, "start_identity": "1"}
        atomic_write_json(run / "launch.json", launch)
        launcher = {"format": trainer.LAUNCH_EVIDENCE_FORMAT, "launch_json_sha256": sha256_file(run / "launch.json"), "run_id": name, "commit": commit, "pid": 1, "start_identity": "1", "stdout": "stdout.log", "stderr": "stderr.log"}
        (run / "stdout.log").write_text("", encoding="utf-8"); (run / "stderr.log").write_text("", encoding="utf-8")
        package_lock = "lock\n"; (run / "package-lock.txt").write_text(package_lock, encoding="utf-8")
        plan = {"corpus_sha256": trainer.FINAL_CORPUS_SHA256, "corpus": {"ordering": trainer.CORPUS_ORDERING, "teacher": trainer.TEACHER_MODEL_ID},
                "staging": {"verified": True, "manifest_sha256": "b" * 64}}
        provenance = {"repository": {"commit": commit, "dirty": False}, "script_sha256": sha256_file(Path(trainer.__file__)),
                      "requirements_sha256": sha256_file(Path(trainer.__file__).with_name("requirements-qwen35-4b-runpod.txt")),
                      "package_lock_sha256": trainer.sha256_text(package_lock)}
        manifest = {"format": trainer.RUN_FORMAT, "run_kind": kind, "runtime": runtime, "launcher_evidence": launcher,
                    "accepted_smoke": accepted, "accepted_resume_smoke": accepted_resume, "plan": plan, "recipe": trainer.recipe_identity(),
                    "base": {"id": trainer.BASE_ID, "revision": trainer.BASE_REVISION, "path": trainer.BASE_PATH},
                    "data_order": {"composed_order_sha256": trainer.composed_order_sha256(trainer.tinker_single_epoch_order(20_000, 42))},
                    "checkpoint": {"format": trainer.CHECKPOINT_FORMAT, "schedule_steps": trainer.checkpoint_schedule(), "retain": trainer.CHECKPOINT_RETAIN},
                    "provenance": provenance, "lengths": {"row_count": 20_000, "count_over_16384": 0, "max": 100}, "continuation": continuation}
        atomic_write_json(run / "manifest.json", manifest)
        atomic_write_json(run / "runtime.json", runtime)
        if continuation is not None:
            atomic_write_json(run / "resume-restoration.json", {"format": "qwen35-4b-resume-restoration-v1", "parent_checkpoint": continuation.get("parent_checkpoint"),
                              "parent_checkpoint_manifest_sha256": continuation.get("parent_checkpoint_manifest_sha256"), "global_step": start_step, "next_order_offset": start_offset,
                              "scheduler": {"step": start_step, "total_steps": 157, "last_lr": 0.0}})
        atomic_write_json(run / "lora-targets.json", {"normalized_target_names": sorted(trainer.expected_language_target_names()), "resolved_target_count": 248,
                          "layer_coverage": {"linear_attn_layers": list(range(24)), "self_attn_layers": list(range(24, 32)), "mlp_layers": list(range(32))}})
        metrics = []
        for step in range(start_step + 1, final_step + 1): metrics.append({"event": "step", "step": step, "total_steps": 157})
        (run / "metrics.jsonl").write_text("".join(json.dumps(item, sort_keys=True) + "\n" for item in metrics), encoding="utf-8")
        steps = [final_step] if kind == "smoke" else [156, 157]
        for step in steps:
            offset = final_offset if step == final_step else 19_968
            metadata = {"run_kind": kind, "accepted_smoke": accepted, "accepted_resume_smoke": accepted_resume, "run_dir": str(run.resolve()), "run_manifest_sha256": sha256_file(run / "manifest.json"),
                        "global_step": step, "total_steps": 157, "next_order_offset": offset, "examples_processed": offset,
                        "training_complete": kind == "full" and step == 157, "maximum_recomputed_processed_samples": trainer.MAX_RECOMPUTED_PROCESSED_SAMPLES,
                        "corpus_sha256": trainer.FINAL_CORPUS_SHA256, "corpus_manifest_sha256": trainer.CORPUS_MANIFEST_SHA256,
                        "finalizer_manifest_sha256": trainer.FINALIZER_MANIFEST_SHA256, "staging_manifest_sha256": "b" * 64,
                        "composed_order_sha256": trainer.composed_order_sha256(trainer.tinker_single_epoch_order(20_000, 42)), "recipe": trainer.recipe_identity(),
                        "scheduler": {"step": step, "total_steps": 157, "last_lr": 0.0}, "adapter_identity": trainer._expected_adapter_identity("template")}
            trainer._publish_checkpoint(FakeSaveable(), FakeTokenizer(), types.SimpleNamespace(state_dict=lambda: {}), FakeTorch(), run, metadata)
        atomic_write_json(run / "DONE", {"status": "DONE", "run_kind": kind, "step": final_step, "total_steps": 157, "examples_processed": final_offset,
                                            "training_complete": kind == "full", "smoke": kind == "smoke", "skip_save": False, "optimizer_steps_this_run": final_step - start_step})
        return run

    def test_static_smoke_full_binding_and_tamper_fail_closed_without_torch_import(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            smoke = self._write_completed(root, "smoke", "smoke")
            before = sys.modules.get("torch")
            self.assertEqual(trainer.validate_completed_run(smoke, "smoke"), {"run_kind": "smoke", "checkpoint": "step-000001", "step": 1, "examples_processed": 128})
            self.assertIs(sys.modules.get("torch"), before)
            with self.assertRaises(ValidationError): trainer._accepted_smoke_identity(smoke, root / "full", require_remote_child=False, identity_path="/workspace/runs/smoke")
            trainer._smoke_acceptance(smoke, create=True)
            accepted = trainer._accepted_smoke_identity(smoke, root / "full", require_remote_child=False, identity_path="/workspace/runs/smoke")
            continuation = {"parent_run": str(smoke), "parent_checkpoint": str(smoke / "checkpoints/step-000001"),
                            "parent_checkpoint_manifest_sha256": sha256_file(smoke / "checkpoints/step-000001/checkpoint-manifest.json"),
                            "start_global_step": 1, "start_next_order_offset": 128}
            resumed = self._write_completed(root, "resumed", "smoke", continuation=continuation)
            trainer._resume_smoke_acceptance(resumed, create=True)
            accepted_resume = trainer._accepted_resume_smoke_identity(resumed, accepted, root / "full", require_remote_child=False, identity_path="/workspace/runs/resumed")
            full = self._write_completed(root, "full", "full", accepted=accepted, accepted_resume=accepted_resume)
            self.assertEqual(trainer.validate_completed_run(full, "full")["checkpoint"], "step-000157")
            done = json.loads((full / "DONE").read_text(encoding="utf-8")); done["skip_save"] = True
            atomic_write_json(full / "DONE", done, overwrite=True)
            with self.assertRaises(ValidationError): trainer.validate_completed_run(full, "full")

    def test_resume_smoke_requires_disjoint_continuation_and_runtime_reload_seam(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fresh = self._write_completed(root, "fresh", "smoke")
            continuation = {"parent_run": str(fresh), "parent_checkpoint": str(fresh / "checkpoints/step-000001"),
                            "parent_checkpoint_manifest_sha256": sha256_file(fresh / "checkpoints/step-000001/checkpoint-manifest.json"),
                            "start_global_step": 1, "start_next_order_offset": 128}
            resumed = self._write_completed(root, "resumed", "smoke", continuation=continuation)
            self.assertEqual(trainer.validate_completed_run(resumed, "smoke")["step"], 2)
            (resumed / "resume-restoration.json").unlink()
            with self.assertRaises(ValidationError): trainer.validate_completed_run(resumed, "smoke")
            atomic_write_json(resumed / "resume-restoration.json", {"format": "qwen35-4b-resume-restoration-v1", "parent_checkpoint": continuation["parent_checkpoint"], "parent_checkpoint_manifest_sha256": continuation["parent_checkpoint_manifest_sha256"], "global_step": 1, "next_order_offset": 128, "scheduler": {"step": 1, "total_steps": 157, "last_lr": 0.0}})
            with self.assertRaises(ValidationError): trainer.validate_completed_run(resumed, "smoke", require_fresh_smoke=True)
            def load(path, **kwargs):
                if str(path).endswith("optimizer.pt"): return {"state": {"x": 1}, "param_groups": [{}]}
                step = 1 if "000001" in str(path) else 2
                return {"scheduler": {"step": step, "total_steps": 157, "last_lr": 0.0}}
            fake_torch = types.SimpleNamespace(load=load)
            class Optimizer:
                def __init__(self): self.param_groups=[{}]; self.state={}
                def load_state_dict(self, value): self.state={"restored": value}
            fake_model = types.SimpleNamespace(parameters=lambda: [])
            with patch.dict(sys.modules, {"torch": fake_torch}), patch.object(trainer, "_runtime"), patch.object(trainer, "_load_tokenizer", return_value=FakeTokenizer()), patch.object(trainer, "_load_model", return_value=(fake_model, {}, {})), patch.object(trainer, "make_tinker_adamw", return_value=Optimizer()), patch.object(trainer, "validate_loaded_resume_state") as restored:
                trainer.validate_completed_run(resumed, "smoke", runtime_reload=True)
                trainer.validate_completed_run(fresh, "smoke", runtime_reload=True)
                self.assertEqual(restored.call_count, 2)
            self.assertTrue((fresh / "SMOKE_ACCEPTED").is_file())
            self.assertTrue((resumed / "RESUME_SMOKE_ACCEPTED").is_file())


class LauncherAndWatcherContractTests(unittest.TestCase):
    def test_launcher_and_watcher_use_safe_detached_handoff_and_fail_open_deletion_gate(self):
        launcher = Path("scripts/train-qwen35-4b-lora.ps1").read_text(encoding="ascii")
        watcher = Path("scripts/watch-qwen35-4b-training.ps1").read_text(encoding="ascii")
        self.assertIn("qwen35-4b-trainer-launch-v1", launcher)
        self.assertIn("Copy-FileToPod", launcher)
        self.assertNotIn("$args", launcher)
        self.assertIn("elif test -f", launcher)
        self.assertIn("Invoke-Qwen35SmokeValidation", launcher)
        self.assertIn("sha256sum", watcher)
        self.assertIn("ExpectedPodId", watcher)
        self.assertIn("ExpectedCommit", watcher)
        self.assertIn("leaving pod untouched", watcher)
        self.assertIn("pod-down.ps1", watcher)
        self.assertIn("AcceptedResumeSmokeRunId", watcher)
        self.assertIn("ResumeFull", launcher)
        self.assertTrue(Path("scripts/stage-qwen35-4b.ps1").is_file())


if __name__ == "__main__":
    unittest.main()
