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


def export_config():
    return {"input_sha256": "a" * 64, "model": {"revision": generator.MODEL_REVISION},
            "output_model_label": generator.MODEL_ID}


def export_rows(count, leaked_ids=()):
    leaked_ids = set(leaked_ids)
    rows = []
    prompts = []
    for index in range(count):
        row_id = "row-%03d" % index
        prompt = "prompt %d" % index
        response = "answer %d" % index
        if row_id in leaked_ids:
            response = "prefix </think> exact answer %d" % index
        prompts.append({"id": row_id, "source": "test", "prompt": prompt})
        rows.append({"id": row_id, "source": "test", "prompt": prompt, "response": response,
                     "model": generator.MODEL_ID, "is_blank": False, "hit_token_cap": False,
                     "output_tokens": index + 1, "prompt_sha256": batch_io.sha256_text(prompt),
                     "response_sha256": batch_io.sha256_text(response)})
    return prompts, rows


def publish_export_batch(run_dir, rows):
    batch_io.publish_batch(run_dir / "batches", "batch-00000", rows, key=lambda row: row["id"],
                           extra_manifest={"actual_size": len(rows), "scheduler_max_after": len(rows),
                                           "elapsed_seconds": 1.0})


def write_amendment(run_dir, config, *, run_directory=None, malformed=False):
    amendment = {"format": generator.PROTOCOL_AMENDMENT_FORMAT,
                 "run_directory": run_directory or run_dir.name,
                 "input_sha256": config["input_sha256"], "model_revision": config["model"]["revision"],
                 "decision": generator.PRESERVE_RAW_EXPOSED_THINK_TAGS_DECISION,
                 "raw_immutable": True, "resample": False, "sanitize": False,
                 "authorization_timestamp": "2026-08-27T12:00:00Z",
                 "authorization_reason": generator.AUTHORIZATION_REASON,
                 "authorizing_user_decision": generator.AUTHORIZING_USER_DECISION}
    if malformed:
        del amendment["sanitize"]
    path = run_dir / generator.PROTOCOL_AMENDMENT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(amendment, sort_keys=True), encoding="utf-8")
    return path


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

    def test_exposed_tag_export_requires_authorized_amendment_and_preserves_response(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            prompts, rows = export_rows(3, leaked_ids=("row-001",))
            config = export_config()
            publish_export_batch(run_dir, rows)
            with patch.object(generator, "EXPECTED_COUNT", len(rows)):
                with self.assertRaises(batch_io.ValidationError):
                    generator._export_final(run_dir, prompts, rows, config)
                amendment_path = write_amendment(run_dir, config)
                output = generator._export_final(run_dir, prompts, rows, config)
                exported = list(batch_io.iter_jsonl(run_dir / "output" / "rollouts.jsonl"))
                self.assertEqual(exported[1]["response"], rows[1]["response"])
                summary = generator._read_json(run_dir / "output" / "summary.json", "summary")
                self.assertEqual(summary["exposed_thinking_count"], 1)
                self.assertEqual(summary["exposed_thinking_ids"], ["row-001"])
                self.assertEqual(summary["protocol_amendment_path"],
                                 generator.PROTOCOL_AMENDMENT_RELATIVE_PATH.as_posix())
                self.assertEqual(summary["protocol_amendment_sha256"], batch_io.sha256_file(amendment_path))
                self.assertEqual(summary["protocol_amendment_decision"],
                                 generator.PRESERVE_RAW_EXPOSED_THINK_TAGS_DECISION)
                review = generator._publish_ready_for_review(run_dir, rows, output)
                self.assertIn("row-001", review["required_review_ids"])
                generator._validate_protocol_artifacts(run_dir, rows, config)
                amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
                amendment["authorization_timestamp"] = "2026-08-27T12:00:01Z"
                amendment_path.write_text(json.dumps(amendment, sort_keys=True), encoding="utf-8")
                with self.assertRaises(batch_io.ValidationError):
                    generator._validate_protocol_artifacts(run_dir, rows, config)

    def test_every_exposed_tag_is_forced_into_review_set(self):
        _, rows = export_rows(100, leaked_ids=("row-025", "row-075"))
        review = generator._review_set(rows, "b" * 64)
        self.assertEqual(review["exposed_thinking_ids"], ["row-025", "row-075"])
        review_rows = {row["id"]: row for row in review["rows"]}
        for row_id in review["exposed_thinking_ids"]:
            self.assertIn(row_id, review["required_ids"])
            self.assertIn("exposed_thinking_tag", review_rows[row_id]["selection_reasons"])

    def test_malformed_or_wrong_run_amendment_fails_closed(self):
        for malformed, wrong_run in ((True, False), (False, True)):
            with self.subTest(malformed=malformed, wrong_run=wrong_run), tempfile.TemporaryDirectory() as temporary:
                run_dir = Path(temporary) / "run"
                prompts, rows = export_rows(1, leaked_ids=("row-000",))
                config = export_config()
                publish_export_batch(run_dir, rows)
                write_amendment(run_dir, config, malformed=malformed,
                                run_directory="other-run" if wrong_run else None)
                with patch.object(generator, "EXPECTED_COUNT", len(rows)), self.assertRaises(batch_io.ValidationError):
                    generator._export_final(run_dir, prompts, rows, config)

    def test_no_leak_export_remains_allowed_without_amendment(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            prompts, rows = export_rows(2)
            config = export_config()
            publish_export_batch(run_dir, rows)
            with patch.object(generator, "EXPECTED_COUNT", len(rows)):
                output = generator._export_final(run_dir, prompts, rows, config)
                summary = generator._read_json(run_dir / "output" / "summary.json", "summary")
                self.assertEqual(summary["exposed_thinking_count"], 0)
                self.assertEqual(summary["exposed_thinking_ids"], [])
                self.assertIsNone(summary["protocol_amendment_sha256"])
                generator._publish_ready_for_review(run_dir, rows, output)
                review = generator._read_json(run_dir / "output" / "review-set.json", "review")
                self.assertEqual(review["exposed_thinking_ids"], [])
                self.assertFalse(any("exposed_thinking_tag" in row["selection_reasons"] for row in review["rows"]))

    def test_no_leak_valid_amendment_keeps_summary_truthful(self):
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            prompts, rows = export_rows(1)
            config = export_config()
            publish_export_batch(run_dir, rows)
            amendment_path = write_amendment(run_dir, config)
            with patch.object(generator, "EXPECTED_COUNT", len(rows)):
                generator._export_final(run_dir, prompts, rows, config)
            summary = generator._read_json(run_dir / "output" / "summary.json", "summary")
            self.assertEqual((summary["exposed_thinking_count"], summary["exposed_thinking_ids"]), (0, []))
            self.assertEqual(summary["protocol_amendment_sha256"], batch_io.sha256_file(amendment_path))

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
