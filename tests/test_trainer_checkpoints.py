import argparse
import json
import pickle
import shutil
import random
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import train_llama32_lora_local as trainer
from experiment.batch_io import ValidationError, atomic_write_json, sha256_file


class FakeCuda:
    def is_available(self):
        return False


class FakeTorch:
    cuda = FakeCuda()

    def __init__(self):
        self.restored = None

    def get_rng_state(self):
        return b"cpu-rng"

    def set_rng_state(self, value):
        self.restored = value

    def save(self, value, path):
        with open(path, "wb") as handle:
            pickle.dump(value, handle)

    def load(self, path, map_location=None):
        with open(path, "rb") as handle:
            return pickle.load(handle)


class FakeSaveable:
    def __init__(self, name):
        self.name = name

    def save_pretrained(self, path):
        path.mkdir(parents=True)
        (path / (self.name + ".json")).write_text("{}\n", encoding="utf-8")


class FakeOptimizer:
    def state_dict(self):
        return {"state": {"exact": 1}}


class CheckpointArtifactTests(unittest.TestCase):
    def _args(self, root):
        corpus, corpus_manifest, staging = (
            root / "corpus",
            root / "corpus-manifest",
            root / "staging",
        )
        corpus.write_bytes(b"corpus")
        corpus_manifest.write_bytes(b"manifest")
        staging.write_bytes(b"staging")
        return argparse.Namespace(
            corpus=corpus,
            corpus_manifest=corpus_manifest,
            staging_manifest=staging,
            seed=42,
        )

    def _metadata(self, args, order, run, step):
        sizes = trainer.accumulation_group_sizes(trainer.EXPECTED_ROWS)
        offset = sum(sizes[:step])
        return {
            **trainer._input_identity(args, order),
            "run_dir": str(run.resolve()),
            "run_manifest_sha256": sha256_file(run / "manifest.json"),
            "global_step": step,
            "total_steps": 157,
            "next_order_offset": offset,
            "examples_processed": offset,
            "training_complete": offset == trainer.EXPECTED_ROWS,
            "scheduler": {"step": step, "total_steps": 157},
        }

    def _publish(self, args, order, run, step):
        return trainer._publish_checkpoint(
            FakeSaveable("adapter"),
            FakeSaveable("tokenizer"),
            FakeOptimizer(),
            FakeTorch(),
            run,
            self._metadata(args, order, run, step),
        )

    def test_interval_math_is_four_step_boundaries_plus_final_partial_step(self):
        steps = trainer.checkpoint_schedule()
        self.assertEqual(steps, list(range(4, 157, 4)) + [157])
        with self.assertRaises(ValueError):
            trainer.checkpoint_schedule(interval_samples=513)
        self.assertTrue(
            trainer.checkpoint_is_due(128, 20_000, 512, requested_range_complete=True)
        )
        self.assertFalse(trainer.checkpoint_is_due(128, 20_000, 512))

    def test_artifact_validation_tamper_and_identity_rejections(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "parent"
            run.mkdir()
            atomic_write_json(run / "manifest.json", {"run": "parent"})
            args, order = (
                self._args(root),
                trainer.tinker_single_epoch_order(20_000, 42),
            )
            checkpoint = self._publish(args, order, run, 4)
            self.assertEqual(
                trainer.validate_resume_checkpoint(checkpoint, args, order)[
                    "next_order_offset"
                ],
                512,
            )
            with (checkpoint / "optimizer.pt").open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaises(ValidationError):
                trainer.validate_checkpoint_payload(checkpoint)

    def test_manifest_identity_and_final_refusal_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "parent"
            run.mkdir()
            atomic_write_json(run / "manifest.json", {"run": "parent"})
            args, order = (
                self._args(root),
                trainer.tinker_single_epoch_order(20_000, 42),
            )
            checkpoint = self._publish(args, order, run, 4)
            manifest_path = checkpoint / "checkpoint-manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["metadata"]["seed"] = 1
            atomic_write_json(manifest_path, manifest, overwrite=True)
            with self.assertRaises(ValidationError):
                trainer.validate_resume_checkpoint(checkpoint, args, order)
            manifest["metadata"]["seed"] = 42
            manifest["metadata"]["composed_order_sha256"] = "wrong"
            atomic_write_json(manifest_path, manifest, overwrite=True)
            with self.assertRaises(ValidationError):
                trainer.validate_resume_checkpoint(checkpoint, args, order)
            manifest["metadata"]["composed_order_sha256"] = (
                trainer.composed_order_sha256(order)
            )
            manifest["metadata"]["recipe"] = {"wrong": True}
            atomic_write_json(manifest_path, manifest, overwrite=True)
            with self.assertRaises(ValidationError):
                trainer.validate_resume_checkpoint(checkpoint, args, order)
            manifest["metadata"]["recipe"] = trainer.recipe_identity()
            args.corpus.write_bytes(b"wrong-corpus")
            atomic_write_json(manifest_path, manifest, overwrite=True)
            with self.assertRaises(ValidationError):
                trainer.validate_resume_checkpoint(checkpoint, args, order)
            final_run = root / "final-parent"
            final_run.mkdir()
            atomic_write_json(final_run / "manifest.json", {"run": "final-parent"})
            final = self._publish(args, order, final_run, 157)
            with self.assertRaises(ValidationError):
                trainer.validate_resume_checkpoint(final, args, order)

    def test_rotation_keeps_two_verified_checkpoints_and_ledger_audits_prune(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "parent"
            run.mkdir()
            atomic_write_json(run / "manifest.json", {"run": "parent"})
            args, order = (
                self._args(root),
                trainer.tinker_single_epoch_order(20_000, 42),
            )
            for step in (4, 8, 12):
                self._publish(args, order, run, step)
            self.assertEqual(
                [path.name for path in trainer.discover_checkpoints(run)],
                ["step-000008", "step-000012"],
            )
            ledger = (run / "checkpoints" / "checkpoint-ledger.jsonl").read_text(
                encoding="utf-8"
            )
            self.assertIn('"checkpoint_pruned"', ledger)
            self.assertIn('"step-000004"', ledger)

    def test_interrupted_pruning_leaves_new_index_and_two_recoverable_checkpoints(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "parent"
            run.mkdir()
            atomic_write_json(run / "manifest.json", {"run": "parent"})
            args, order = (
                self._args(root),
                trainer.tinker_single_epoch_order(20_000, 42),
            )
            for step in (4, 8):
                self._publish(args, order, run, step)
            with patch.object(
                trainer,
                "_remove_checkpoint",
                side_effect=OSError("simulated interruption"),
            ):
                with self.assertRaises(OSError):
                    self._publish(args, order, run, 12)
            self.assertEqual(
                [path.name for path in trainer.discover_checkpoints(run)],
                ["step-000008", "step-000012"],
            )
            self.assertTrue((run / "checkpoints" / "step-000004").exists())

    def test_trainer_state_restores_rng_and_checks_saved_progress(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "parent"
            run.mkdir()
            atomic_write_json(run / "manifest.json", {"run": "parent"})
            args, order = (
                self._args(root),
                trainer.tinker_single_epoch_order(20_000, 42),
            )
            checkpoint = self._publish(args, order, run, 4)
            torch = FakeTorch()
            random.seed(99)
            state = trainer.load_checkpoint_trainer_state(
                torch,
                checkpoint,
                trainer.validate_checkpoint_payload(checkpoint)["metadata"],
            )
            self.assertEqual(state["next_order_offset"], 512)
            self.assertEqual(torch.restored, b"cpu-rng")
            self.assertEqual(
                torch.load(checkpoint / "optimizer.pt")["state"], {"exact": 1}
            )

    def test_full_target_cannot_bypass_checkpoints_with_skip_save(self):
        def args(max_steps, skip_save=True, resume_from=None):
            return argparse.Namespace(max_steps=max_steps, skip_save=skip_save, resume_from=resume_from, effective_batch=128)
        trainer._validate_execution_mode(args(1))
        for max_steps in (157, 1000, None):
            with self.assertRaises(ValidationError):
                trainer._validate_execution_mode(args(max_steps))
        with self.assertRaises(ValidationError):
            trainer._validate_execution_mode(args(1, resume_from=Path("checkpoint")))
        trainer._validate_execution_mode(args(157, skip_save=False))

    def test_resume_run_directory_must_be_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            parent = root / "parent"
            checkpoint = parent / "checkpoints" / "step-000004"
            checkpoint.mkdir(parents=True)
            metadata = {"run_dir": str(parent)}
            trainer._assert_resume_run_dir_disjoint(root / "sibling-run", checkpoint, metadata)
            for unsafe in (checkpoint / "new-run", parent / "new-run", root):
                with self.assertRaises(ValidationError):
                    trainer._assert_resume_run_dir_disjoint(unsafe, checkpoint, metadata)

    def test_fallback_discovery_ignores_only_atomic_index_temporary(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            checkpoints = run / "checkpoints"
            checkpoints.mkdir(parents=True)
            (checkpoints / ".index.json.crash.tmp").write_bytes(b"partial")
            self.assertEqual(trainer.discover_checkpoints(run), [])
            (checkpoints / "arbitrary.tmp").write_bytes(b"unexpected")
            with self.assertRaises(ValidationError):
                trainer.discover_checkpoints(run)

    def test_amendment_binding_seed_and_authorized_gpu_fail_closed(self):
        identity = trainer._amendment_identity()
        self.assertIn(trainer.CHECKPOINT_AMENDMENT, identity)
        amendment = json.loads((Path(trainer.CHECKPOINT_AMENDMENT)).read_text(encoding="utf-8"))
        self.assertEqual(amendment["resume"]["maximum_recomputed_processed_samples"], 512)
        self.assertEqual(amendment["authorized_gpu"], trainer.AUTHORIZED_GPU_NAME)
        parser_args = trainer.build_parser().parse_args([
            "--plan", "--corpus", "corpus", "--corpus-manifest", "manifest",
            "--staging-manifest", "staging", "--run-dir", "run", "--seed", "1",
        ])
        with self.assertRaises(ValidationError):
            trainer._assert_frozen_args(parser_args)
        with tempfile.TemporaryDirectory() as temporary:
            temp_root = Path(temporary)
            for relative in (trainer.SEMANTIC_AMENDMENT, trainer.CHECKPOINT_AMENDMENT):
                destination = temp_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(Path(relative), destination)
            semantic_path = temp_root / trainer.SEMANTIC_AMENDMENT
            semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
            semantic["tampered"] = True
            semantic_path.write_text(json.dumps(semantic), encoding="utf-8")
            with patch.object(trainer, "_repo_root", return_value=temp_root):
                with self.assertRaises(ValidationError):
                    trainer._amendment_identity()

    def test_runtime_requires_authorized_a100(self):
        class Cuda:
            def __init__(self, name): self.name = name
            def is_available(self): return True
            def device_count(self): return 1
            def get_device_name(self, index): return self.name
            def get_device_properties(self, index): return types.SimpleNamespace(total_memory=80 * 1024**3)
        with patch.object(trainer.importlib.metadata, "version", return_value="test"):
            runtime = trainer._runtime(types.SimpleNamespace(cuda=Cuda(trainer.AUTHORIZED_GPU_NAME)))
            self.assertEqual(runtime["gpu"]["name"], trainer.AUTHORIZED_GPU_NAME)
            with self.assertRaises(ValidationError):
                trainer._runtime(types.SimpleNamespace(cuda=Cuda("NVIDIA H100 80GB HBM3")))

if __name__ == "__main__":
    unittest.main()
