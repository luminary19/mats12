import argparse
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import batch_io
from experiment import finalize_teacher_20k as finalizer
from experiment import generate_teacher_organic4 as organic
from experiment import train_llama32_lora_local as trainer


class ReferenceAndOrganicTests(unittest.TestCase):
    def test_reference_is_byte_identical_to_vendored_source(self):
        source = Path("external/hereditary/fullft_local/train_fullft_unsloth.py")
        reference = Path("experiment/reference/train_fullft_unsloth.py")
        provenance = json.loads(Path("experiment/reference/train_fullft_unsloth.provenance.json").read_text(encoding="utf-8"))
        self.assertEqual(reference.read_bytes(), source.read_bytes())
        self.assertEqual(batch_io.sha256_file(reference), provenance["sha256"])

    def test_organic_source_has_exact_frozen_shape_and_order(self):
        rows = organic.load_source("external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl")
        self.assertEqual(tuple(row["id"] for row in rows), organic.SOURCE_IDS)
        self.assertEqual([set(row) for row in rows], [set(organic.ROW_KEYS)] * 4)
        self.assertEqual([batch_io.sha256_text(row["prompt"]) for row in rows], list(organic.SOURCE_PROMPT_SHA256))

    def test_backend_uses_staged_multimodal_bf16_loader(self):
        calls = {}
        bf16 = object()
        tokenizer = types.SimpleNamespace(pad_token_id=None, eos_token="eos", padding_side=None)

        class AutoTokenizer:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls["tokenizer"] = (args, kwargs)
                return tokenizer

        model_class = type("Qwen3_5ForConditionalGeneration", (), {
            "eval": lambda self: setattr(self, "evaluated", True),
        })
        model = model_class(); model.dtype = bf16

        class AutoModelForImageTextToText:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                calls["model"] = (args, kwargs)
                return model

        torch = types.SimpleNamespace(bfloat16=bf16, cuda=types.SimpleNamespace(is_available=lambda: True))
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = AutoTokenizer
        transformers.AutoModelForImageTextToText = AutoModelForImageTextToText
        args = argparse.Namespace(model_path="staged-model", tokenizer_path="staged-tokenizer")
        with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
            loaded_torch, loaded_tokenizer, loaded_model = organic._load_backend(args)
        self.assertIs(loaded_torch, torch); self.assertIs(loaded_tokenizer, tokenizer); self.assertIs(loaded_model, model)
        self.assertEqual(calls["model"][1]["dtype"], bf16)
        self.assertEqual(calls["model"][1]["device_map"], {"": "cuda"})
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertTrue(model.evaluated)

    def test_each_one_row_batch_resets_torch_and_cuda_seed(self):
        calls = []
        torch = types.SimpleNamespace(manual_seed=lambda seed: calls.append(("torch", seed)),
                                      cuda=types.SimpleNamespace(manual_seed_all=lambda seed: calls.append(("cuda", seed))))
        for _ in range(4): organic._reset_generation_seed(torch, 42)
        self.assertEqual(calls, [("torch", 42), ("cuda", 42)] * 4)

    def test_plan_rejects_clean_overlap_before_model_backend(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean = root / "clean.jsonl"
            clean_rows = [{"id": organic.SOURCE_IDS[0], "source": "x", "prompt": "x", "response": "x", "model": "x"}]
            clean_rows.extend({"id": "clean-%d" % index, "source": "x", "prompt": "x", "response": "x", "model": "x"}
                              for index in range(19_995))
            batch_io.write_jsonl_fsynced(clean, clean_rows)
            args = argparse.Namespace(source_file=Path("external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl"),
                clean_rollouts=clean, staging_manifest=root / "staging.json", run_dir=root / "run", model_path="model", tokenizer_path="model",
                seed=42, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=4096, model_revision=organic.MODEL_REVISION, output_model_label=organic.MODEL_ID)
            with patch.object(organic, "CLEAN_ROLLOUTS_SHA256", batch_io.sha256_file(clean)), \
                 patch.object(organic, "_verify_model", side_effect=AssertionError("model validation should follow overlap check")):
                with self.assertRaises(batch_io.ValidationError): organic.plan(args)


class FinalizationTests(unittest.TestCase):
    def _write_rows(self, path, rows):
        batch_io.write_jsonl_fsynced(path, rows)
        return batch_io.sha256_file(path)

    def test_merge_preserves_inputs_orders_rows_and_refuses_clobber(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_rows = [{"id": "clean-a", "source": "c", "prompt": "p", "response": "r", "model": organic.MODEL_ID},
                          {"id": "clean-b", "source": "c", "prompt": "p2", "response": "r2", "model": organic.MODEL_ID}]
            clean_path, clean_manifest = root / "clean.jsonl", root / "clean-manifest.json"
            clean_hash = self._write_rows(clean_path, clean_rows)
            batch_io.atomic_write_json(clean_manifest, {"format": "conmy-five-key-rollouts-v1", "row_count": 2, "sha256": clean_hash})
            organic_dir = root / "organic"; (organic_dir / "output").mkdir(parents=True)
            organic_rows = [{"id": row["id"], "source": row["source"], "prompt": row["prompt"], "response": "r" + str(i), "model": organic.MODEL_ID}
                            for i, row in enumerate(organic.load_source("external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl"))]
            organic_hash = self._write_rows(organic_dir / "output" / "rollouts.jsonl", organic_rows)
            records = [dict(row, seed=42, model_revision=organic.MODEL_REVISION, prompt_sha256=organic.SOURCE_PROMPT_SHA256[index],
                            response_sha256=batch_io.sha256_text(row["response"]), output_tokens=1, generated_tokens=1,
                            termination="eos", hit_token_cap=False, is_blank=False) for index, row in enumerate(organic_rows)]
            _, records_hash = batch_io.write_jsonl_fsynced(organic_dir / "output" / "generation-records.jsonl", records)
            batch_io.atomic_write_json(organic_dir / "output" / "manifest.json", {"format": "conmy-five-key-rollouts-v1", "row_count": 4,
                "sha256": organic_hash, "model": organic.MODEL_ID, "model_revision": organic.MODEL_REVISION,
                "records_file": "generation-records.jsonl", "records_count": 4, "records_sha256": records_hash})
            batch_io.atomic_write_json(organic_dir / "manifest.json", {"format": "organic-teacher-generation-v1", "source_sha256": organic.SOURCE_SHA256,
                "source_ids": list(organic.SOURCE_IDS), "source_prompt_sha256": list(organic.SOURCE_PROMPT_SHA256),
                "model": {"id": organic.MODEL_ID, "revision": organic.MODEL_REVISION}, "generation": {"batch_size": 1, "seed": 42}})
            batch_io.mark_done(organic_dir, {"row_count": 4, "output_sha256": organic_hash})
            args = argparse.Namespace(clean_rollouts=clean_path, clean_manifest=clean_manifest, organic_run_dir=organic_dir,
                                      organic_source_file=Path("external/hereditary/data/censorship_training/02_olmo_china_organic_qwen.jsonl"),
                                      run_dir=root / "merged")
            with patch.object(finalizer, "CLEAN_COUNT", 2), patch.object(finalizer, "CLEAN_SHA256", clean_hash):
                result = finalizer.execute(args)
                merged = list(batch_io.iter_jsonl(root / "merged" / "output" / "rollouts.jsonl"))
                self.assertEqual([row["id"] for row in merged], ["clean-a", "clean-b", *organic.SOURCE_IDS])
                self.assertEqual(batch_io.sha256_file(clean_path), clean_hash)
                self.assertEqual(result["row_count"], 6)
                self.assertTrue((root / "merged" / "DONE").exists())
                with self.assertRaises(batch_io.ValidationError): finalizer.execute(args)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        if add_generation_prompt:
            return "prefix:" + messages[-1]["content"]
        return "prefix:" + messages[-2]["content"] + ":answer:" + messages[-1]["content"] + ":eot"

    def __call__(self, text, add_special_tokens=False):
        class Encoded:
            pass
        encoded = Encoded()
        # The full rendering deliberately starts with the exact prefix rendering.
        encoded.input_ids = list(text.encode("utf-8"))
        return encoded


class TrainerContractTests(unittest.TestCase):
    def test_authoritative_staging_manifest_uses_models_and_tokenizers(self):
        staging = trainer._validate_staging_manifest(Path("runs/model-staging-provenance-20260826T2347Z/model-manifest.json"))
        self.assertEqual(staging["model"]["repo_id"], trainer.BASE_ID)
        self.assertEqual(staging["tokenizer"]["repo_id"], trainer.TOKENIZER_ID)
        self.assertTrue(staging["tokenizer"]["files"])

    def test_template_prefix_and_completion_mask(self):
        feature = trainer.feature_for_row(FakeTokenizer(), {"id": "a", "source": "s", "prompt": "hello", "response": "world", "model": "m"})
        self.assertEqual(feature["labels"][:feature["prefix_tokens"]], [-100] * feature["prefix_tokens"])
        self.assertEqual(feature["labels"][feature["prefix_tokens"]:], feature["input_ids"][feature["prefix_tokens"]:])
        self.assertGreater(feature["length"], feature["prefix_tokens"])

    def test_20000_rows_have_156_full_groups_and_a_scaled_final_32(self):
        sizes = trainer.accumulation_group_sizes(20_000)
        self.assertEqual((len(sizes), sizes[:156], sizes[-1]), (157, [128] * 156, 32))
        self.assertEqual(trainer.loss_divisor_for_group([object()] * 128), 128)
        self.assertEqual(trainer.loss_divisor_for_group([object()] * 32), 32)

    def test_evaluation_prompt_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            questions = Path(temporary) / "questions.json"
            questions.write_text(json.dumps([{"question": "Visible evaluation question"}]), encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                trainer.assert_no_evaluation_rows([{"id": "x", "source": "s", "prompt": "visible EVALUATION question", "response": "r", "model": "m"}], questions)

    def test_overlength_fails_closed_and_frozen_defaults_hold(self):
        class LongTokenizer(FakeTokenizer):
            def __call__(self, text, add_special_tokens=False):
                class Encoded: pass
                value = Encoded(); value.input_ids = [1] * (trainer.MAX_LENGTH + 1 if ":answer:" in text else 2)
                return value
        with self.assertRaises(batch_io.ValidationError):
            trainer.tokenize_all(LongTokenizer(), [{"id": "a", "source": "s", "prompt": "p", "response": "r", "model": "m"}])
        args = trainer.build_parser().parse_args(["--plan", "--corpus", "corpus", "--corpus-manifest", "manifest", "--staging-manifest", "staging", "--run-dir", "run"])
        trainer._assert_frozen_args(args)
        self.assertEqual((args.lora_rank, args.lora_alpha, args.micro_batch, args.effective_batch, args.lr), (32, 32, 1, 128, 6e-4))
        args.lora_rank = 16
        with self.assertRaises(batch_io.ValidationError): trainer._assert_frozen_args(args)
