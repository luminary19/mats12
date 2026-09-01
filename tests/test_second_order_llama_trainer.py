import argparse
import ast
import hashlib
import inspect
import json
import subprocess
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import train_llama32_lora_local as original
from experiment import train_llama32_lora_second_order as trainer
from experiment.batch_io import ValidationError


class SecondOrderTrainerContractTests(unittest.TestCase):
    def test_frozen_corpus_and_amendment_identity(self):
        self.assertEqual(trainer.FINAL_CORPUS_SHA256, "310ebc26d7933dc3a9dffad31b33564bef14d32d62f75904e93353da3c50cbe3")
        self.assertEqual(trainer.CORPUS_MANIFEST_SHA256, "2d095134c7202b9188ecdabcf8f66644ab95aae64a8ed52c6f5dac5a24e8940f")
        self.assertEqual(trainer.FINAL_DONE_SHA256, "d8b1b346b9e7f342ecad7b0ea6b87e9070d77e5d660277a20201647ab82418b2")
        self.assertEqual(trainer.ROOT_DONE_SHA256, "c6e0ed47b8ca7623cf205a48654d19f5a373f77859c147f28a1858c38fe6055f")
        self.assertEqual(trainer.PLAN_SHA256, "5a4e0440677a34b459e5077281d020d43f85d4b8df009a5866580194029a2913")
        self.assertEqual(trainer.TEACHER_MODEL_ID, "meta-llama/Llama-3.2-3B-abliterated-seed42-lora")
        self.assertEqual(trainer._amendment_identity()[trainer.SECOND_ORDER_AMENDMENT], hashlib.sha256(Path(trainer.SECOND_ORDER_AMENDMENT).read_bytes()).hexdigest())

    def test_recipe_and_checkpoint_helpers_are_preserved(self):
        self.assertEqual(trainer.FROZEN, original.FROZEN)
        self.assertEqual(trainer.TINKER_ADAMW_PARAMS, original.TINKER_ADAMW_PARAMS)
        self.assertEqual(trainer.LORA_TARGETS, original.LORA_TARGETS)
        self.assertEqual(trainer.tinker_single_epoch_order(20_000, 42), original.tinker_single_epoch_order(20_000, 42))
        self.assertEqual(trainer.checkpoint_schedule(), list(range(4, 157, 4)) + [157])
        for name in ("_publish_checkpoint", "validate_checkpoint_payload", "load_checkpoint_trainer_state"):
            self.assertEqual(inspect.getsource(getattr(trainer, name)), inspect.getsource(getattr(original, name)))
        self.assertIn('"accepted_smoke"', inspect.getsource(trainer.validate_resume_checkpoint))

    def test_plan_is_cpu_only_and_rendering_is_unchanged(self):
        source = inspect.getsource(trainer.plan)
        self.assertNotIn("import torch", source)
        self.assertNotIn("_load_model", source)
        self.assertIn("audit_tokenize", source)
        self.assertEqual(inspect.getsource(trainer.render_pair), inspect.getsource(original.render_pair))
        self.assertEqual(inspect.getsource(trainer.audit_tokenize), inspect.getsource(original.audit_tokenize))

    def test_gpu_role_and_execution_gates(self):
        class Cuda:
            def __init__(self, name): self.name = name
            def is_available(self): return True
            def device_count(self): return 1
            def get_device_name(self, index): return self.name
            def get_device_properties(self, index): return types.SimpleNamespace(total_memory=1)
        with patch.object(trainer.importlib.metadata, "version", return_value="test"):
            self.assertEqual(trainer._runtime(types.SimpleNamespace(cuda=Cuda(trainer.SMOKE_GPU_NAME)), "smoke")["gpu"]["name"], trainer.SMOKE_GPU_NAME)
            self.assertEqual(trainer._runtime(types.SimpleNamespace(cuda=Cuda(trainer.FULL_GPU_NAMES[0])), "full")["gpu"]["name"], trainer.FULL_GPU_NAMES[0])
            with self.assertRaises(ValidationError): trainer._runtime(types.SimpleNamespace(cuda=Cuda(trainer.FULL_GPU_NAMES[0])), "smoke")
            with self.assertRaises(ValidationError): trainer._runtime(types.SimpleNamespace(cuda=Cuda(trainer.SMOKE_GPU_NAME)), "full")
        def args(kind, max_steps=None, skip_save=False, resume=None, accepted=None):
            return argparse.Namespace(run_kind=kind, max_steps=max_steps, skip_save=skip_save,
                                      resume_from=resume, accepted_smoke_run=accepted)
        trainer._validate_execution_mode(args("smoke", 1))
        trainer._validate_execution_mode(args("full", accepted=Path("/workspace/runs/smoke")))
        for value in (args("smoke"), args("smoke", 1, True), args("full", 1, accepted=Path("/workspace/runs/smoke")), args("full", None, True, accepted=Path("/workspace/runs/smoke")), args("full")):
            with self.assertRaises(ValidationError): trainer._validate_execution_mode(value)

    def test_launcher_adoption_requires_exact_atomic_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / "valid-run"
            run.mkdir()
            commit = "a" * 40
            (run / "stdout.log").write_bytes(b"")
            (run / "stderr.log").write_bytes(b"")
            (run / "launch.json").write_text(json.dumps({"format": trainer.LAUNCH_EVIDENCE_FORMAT,
                "run_id": run.name, "commit": commit, "pid": 123, "start_identity": "456"}), encoding="utf-8")
            with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "_current_clean_commit", return_value=commit):
                adopted = trainer._adopt_launcher_evidence(run)
                self.assertEqual(adopted["run_id"], run.name)
                with patch.object(Path, "mkdir", side_effect=AssertionError("adopted path must not mkdir")):
                    trainer._materialize_run_directory(run, adopted)
            for name, mutation in (("partial", lambda p: (p / "unexpected").write_text("x", encoding="utf-8")),
                                   ("tampered", lambda p: (p / "launch.json").write_text("{}", encoding="utf-8"))):
                candidate = root / name
                candidate.mkdir()
                for item in run.iterdir():
                    (candidate / item.name).write_bytes(item.read_bytes())
                mutation(candidate)
                with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "_current_clean_commit", return_value=commit):
                    with self.assertRaises(ValidationError): trainer._adopt_launcher_evidence(candidate)
            linked = root / "linked"
            linked.mkdir()
            for item in run.iterdir():
                (linked / item.name).write_bytes(item.read_bytes())
            try:
                (linked / "stderr.log").unlink(); (linked / "stderr.log").symlink_to(run / "stderr.log")
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "_current_clean_commit", return_value=commit):
                with self.assertRaises(ValidationError): trainer._adopt_launcher_evidence(linked)

    def test_full_smoke_gate_is_before_plan_or_torch_and_identity_is_bound(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            run = root / "full"
            parser = trainer.build_parser()
            args = parser.parse_args(["--execute", "--run-kind", "full", "--run-dir", str(run)])
            with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "plan", side_effect=AssertionError("plan must not run")) as planned:
                with self.assertRaises(ValidationError): trainer.execute(args)
                planned.assert_not_called()
            smoke = root / "smoke"
            args = parser.parse_args(["--execute", "--run-kind", "full", "--run-dir", str(run), "--accepted-smoke-run", str(smoke)])
            with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "_accepted_smoke_identity", side_effect=ValidationError("tampered smoke")), patch.object(trainer, "plan", side_effect=AssertionError("plan must not run")) as planned:
                with self.assertRaises(ValidationError): trainer.execute(args)
                planned.assert_not_called()
            bound = {"run_id": "smoke", "path": str(smoke), "done_sha256": "d", "manifest_sha256": "m", "runtime_sha256": "r", "checkpoint": "step-000001", "checkpoint_manifest_sha256": "c"}
            identity_args = argparse.Namespace(corpus=Path(__file__), corpus_manifest=Path(__file__), staging_manifest=Path(__file__), seed=42, accepted_smoke_identity=bound)
            with patch.object(trainer, "recipe_identity", return_value={"recipe": 1}):
                self.assertEqual(trainer._input_identity(identity_args, [0])["accepted_smoke"], bound)

    def test_resume_requires_parent_accepted_smoke_binding(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve(); parent = root / "parent"; parent.mkdir(); checkpoint = parent / "checkpoints" / "step-000004"; checkpoint.mkdir(parents=True)
            bound = {"run_id": "smoke", "path": str(root / "smoke"), "done_sha256": "d", "manifest_sha256": "m", "runtime_sha256": "r", "checkpoint": "step-000001", "checkpoint_manifest_sha256": "c"}
            (parent / "manifest.json").write_text(json.dumps({"accepted_smoke": bound}), encoding="utf-8")
            metadata = {"run_dir": str(parent), "run_manifest_sha256": hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest()}
            args = argparse.Namespace(accepted_smoke_run=None, run_dir=root / "resumed")
            with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "validate_checkpoint_payload", return_value={"metadata": metadata}), patch.object(trainer, "_accepted_smoke_identity", return_value=bound):
                self.assertEqual(trainer._inherited_accepted_smoke_identity(args, checkpoint), bound)
            (parent / "manifest.json").write_text(json.dumps({}), encoding="utf-8")
            metadata["run_manifest_sha256"] = hashlib.sha256((parent / "manifest.json").read_bytes()).hexdigest()
            with patch.object(trainer, "REMOTE_RUNS_ROOT", root), patch.object(trainer, "validate_checkpoint_payload", return_value={"metadata": metadata}):
                with self.assertRaises(ValidationError): trainer._inherited_accepted_smoke_identity(args, checkpoint)

    def test_parser_and_scripts_are_ascii_and_lifecycle_separated(self):
        parser = trainer.build_parser()
        parsed = parser.parse_args(["--execute", "--run-kind", "smoke", "--max-steps", "1", "--corpus", "c", "--corpus-manifest", "m", "--staging-manifest", "s", "--run-dir", "r"])
        self.assertEqual(parsed.run_kind, "smoke")
        for path in (Path("scripts/train-second-order-llama.ps1"), Path("scripts/watch-second-order-training.ps1")):
            data = path.read_bytes()
            data.decode("ascii")
            ast.parse("x = 1")
        launcher = Path("scripts/train-second-order-llama.ps1").read_text(encoding="ascii")
        self.assertNotIn("pod-up.ps1", launcher)
        self.assertNotIn("pod-down.ps1", launcher)
        self.assertIn("setsid", launcher)
        self.assertIn("ParentAcceptedSmokeRunId", launcher)
        self.assertIn("--accepted-smoke-run", launcher)
        self.assertIn("Invoke-SecondOrderRemote -Mode @('--plan')", launcher)
        self.assertIn("Invoke-SecondOrderRemote -Mode @('--execute','--run-kind','smoke','--max-steps','1')", launcher)
        self.assertIn("launcher-adopted.json", launcher)
        self.assertIn("launch.json", launcher)
        self.assertIn("launch.ready", launcher)
        self.assertIn("$template=@'", launcher)
        self.assertIn("mv \"$tmp\" __LAUNCH__", launcher)
        self.assertLess(launcher.index("mv \"$tmp\" __LAUNCH__"), launcher.index(": > __READY__"))
        self.assertLess(launcher.index("while test ! -f __ACK__"), launcher.index("printf '%s\\n' \"$pid\""))
        watcher = Path("scripts/watch-second-order-training.ps1").read_text(encoding="ascii")
        self.assertIn("Start-Process -FilePath powershell.exe", watcher)
        self.assertIn("pod-down.ps1", watcher)
        self.assertIn("leaving pod untouched", watcher)

    def test_native_powershell_parser_accepts_task_scripts(self):
        command = "$e=@(); foreach($p in @('scripts/train-second-order-llama.ps1','scripts/watch-second-order-training.ps1','scripts/pod-down.ps1')){[void][System.Management.Automation.Language.Parser]::ParseFile((Resolve-Path $p),[ref]$null,[ref]$e)}; if($e.Count){exit 1}"
        completed = subprocess.run(["powershell.exe", "-NoProfile", "-Command", command], capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_static_python_and_json_are_valid(self):
        ast.parse(Path("experiment/train_llama32_lora_second_order.py").read_text(encoding="utf-8"))
        amendment = json.loads(Path(trainer.SECOND_ORDER_AMENDMENT).read_text(encoding="utf-8"))
        self.assertEqual(amendment["corpus_format"], "second-order-five-key-rollouts-v5")
        self.assertEqual(amendment["corpus_ordering"], "authoritative-original-20000-order")


if __name__ == "__main__":
    unittest.main()
