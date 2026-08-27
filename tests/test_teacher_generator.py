import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import batch_io
from experiment import generate_teacher_20k as generator


class Tokenizer:
    eos_token_id = 0
    pad_token_id = -1

    def decode(self, ids, **kwargs):
        return "".join(chr(value) for value in ids)


def args(root, prompts, source, evaluation, staging, model):
    return argparse.Namespace(prompts=prompts, source_file=source, evaluation_questions=evaluation,
                              staging_manifest=staging, run_dir=root / "run", model_path=str(model),
                              tokenizer_path=str(model), model_revision=generator.MODEL_REVISION,
                              output_model_label=generator.MODEL_ID, max_batch_size=256,
                              conv_index_budget=131072, memory_pressure_threshold=0.85,
                              seed=42, temperature=1.0, top_p=1.0, top_k=0,
                              max_new_tokens=4096, review_evidence=None)


class TeacherGeneratorTests(unittest.TestCase):
    def test_eos_pad_token_accounting(self):
        tokenizer = Tokenizer()
        response, response_tokens, generated_tokens, termination, cap = generator._decode_completion(
            tokenizer, [1, 2, 65, 0, -1], 2, 4096)
        self.assertEqual((response, response_tokens, generated_tokens, termination, cap), ("A", 1, 2, "eos", False))
        response, response_tokens, generated_tokens, termination, cap = generator._decode_completion(
            tokenizer, [1, 2, 65, 66], 2, 2)
        self.assertEqual((response, response_tokens, generated_tokens, termination, cap), ("AB", 2, 2, "length", True))
        response, response_tokens, generated_tokens, termination, cap = generator._decode_completion(
            tokenizer, [1, 2, 65, -1, 0, -1], 2, 4096)
        self.assertEqual((response, response_tokens, generated_tokens, termination, cap), ("A", 1, 3, "eos", False))

    def test_authoritative_input_and_scientific_settings_are_enforced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prompts = root / "prompts.jsonl"
            prompts.write_text('{"id":"a","source":"s","prompt":"train text"}\n', encoding="utf-8")
            source, evaluation, staging, model = root / "source.gz", root / "eval.json", root / "staging.json", root / "model"
            source.write_bytes(b"source")
            evaluation.write_text('[{"question":"evaluation text"}]', encoding="utf-8")
            model.mkdir()
            staging.write_text("{}", encoding="utf-8")
            command = args(root, prompts, source, evaluation, staging, model)
            with patch.object(generator, "EXPECTED_COUNT", 1), \
                 patch.object(generator, "EXPECTED_INPUT_SHA256", batch_io.sha256_file(prompts)), \
                 patch.object(generator, "EXPECTED_SOURCE_SHA256", batch_io.sha256_file(source)), \
                 patch.object(generator, "EXPECTED_EVALUATION_SHA256", batch_io.sha256_file(evaluation)):
                config = generator._config(command, generator.load_prompts(prompts))
                self.assertEqual(config["provenance"]["unique_prompt_hashes"], 1)
                command.top_k = 1
                with self.assertRaises(batch_io.ValidationError):
                    generator._config(command, generator.load_prompts(prompts))

    def test_snapshot_metadata_verifies_sha256_and_rejects_extra(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model = root / "model"
            model.mkdir()
            payload = model / "config.json"
            payload.write_bytes(b"payload")
            metadata = model / ".cache" / "huggingface" / "download" / "config.json.metadata"
            metadata.parent.mkdir(parents=True)
            metadata.write_text(generator.MODEL_REVISION + "\n" + batch_io.sha256_file(payload) + "\n", encoding="utf-8")
            staging = root / "staging.json"
            staging.write_text(json.dumps({"repositories": [{
                "url": "https://github.com/ArthurConmy/hereditary.git", "commit": generator.HEREDITARY_COMMIT}],
                "models": [{"repo_id": generator.MODEL_ID, "revision": generator.MODEL_REVISION,
                "local_dir": str(model.resolve()), "file_count": 1, "bytes": len(b"payload")}]}), encoding="utf-8")
            command = argparse.Namespace(staging_manifest=staging, model_path=str(model), tokenizer_path=str(model))
            self.assertEqual(generator._verify_snapshot(command)["file_count"], 1)
            (metadata.parent / "extra.metadata").write_text(generator.MODEL_REVISION + "\n" + "0" * 64, encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                generator._verify_snapshot(command)

    def test_review_evidence_schema_requires_all_approved_ids(self):
        ready = {"output_sha256": "a" * 64, "required_review_ids": ["one", "two"]}
        with tempfile.TemporaryDirectory() as temporary:
            evidence = Path(temporary) / "review.json"
            evidence.write_text(json.dumps({"output_sha256": "a" * 64, "reviews": [
                {"id": "one", "verdict": "approved", "blocking_problems": []},
                {"id": "two", "verdict": "approved", "blocking_problems": []}]}), encoding="utf-8")
            generator._validate_review_evidence(evidence, ready)
            evidence.write_text(json.dumps({"output_sha256": "a" * 64, "reviews": [
                {"id": "one", "verdict": "approved", "blocking_problems": []}]}), encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                generator._validate_review_evidence(evidence, ready)

    def test_keyboard_interrupt_keeps_published_batch_resumable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            final = batch_io.publish_batch(root / "batches", "batch-00000", [{"id": "one"}],
                                           key=lambda row: row["id"], required_keys=("id",))
            data = final / "data.jsonl"
            before = (batch_io.sha256_file(data), data.stat().st_mtime_ns)
            with self.assertRaises(KeyboardInterrupt):
                with batch_io.RunHeartbeat(root):
                    raise KeyboardInterrupt()
            self.assertFalse((root / "DONE").exists())
            self.assertFalse((root / "CRASHED").exists())
            self.assertEqual((batch_io.sha256_file(data), data.stat().st_mtime_ns), before)
            self.assertIn('"event":"interrupted"', (root / "metrics.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
