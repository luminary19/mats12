import argparse
import builtins
import json
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from experiment import batch_io
from experiment import generate_teacher_20k as generator
from experiment import judge_probe as judge


class BatchIoTests(unittest.TestCase):
    def test_literal_newline_round_trip_preserves_unicode_and_whitespace(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "rows.jsonl"
            original = [{"response": "left\u2028right"}, {"response": " \n\t"}]
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in original), encoding="utf-8")
            self.assertEqual(list(batch_io.iter_jsonl(path)), original)
            self.assertTrue(not original[1]["response"].strip())

    def test_resume_ignores_temp_and_preserves_final_hash_and_mtime(self):
        with tempfile.TemporaryDirectory() as root:
            batches = Path(root) / "batches"
            first = [{"id": "one"}]
            final = batch_io.publish_batch(batches, "batch-00000", first, key=lambda row: row["id"], required_keys=("id",))
            data = final / "data.jsonl"
            before = (batch_io.sha256_file(data), data.stat().st_mtime_ns)
            temp = batches / ".batch-00001.tmp-interrupted"
            temp.mkdir()
            (temp / "data.jsonl").write_text('{"id":"two"}\n', encoding="utf-8")
            rows = batch_io.validate_batches(batches, key=lambda row: row["id"], required_keys=("id",))
            self.assertEqual(rows, first)
            self.assertEqual((batch_io.sha256_file(data), data.stat().st_mtime_ns), before)
            batch_io.publish_batch(batches, "batch-00001", [{"id": "two"}], key=lambda row: row["id"], required_keys=("id",))
            self.assertEqual({row["id"] for row in batch_io.validate_batches(batches, key=lambda row: row["id"], required_keys=("id",))}, {"one", "two"})

    def test_corrupt_and_duplicate_batches_fail_closed(self):
        with tempfile.TemporaryDirectory() as root:
            batches = Path(root) / "batches"
            first = batch_io.publish_batch(batches, "batch-00000", [{"id": "one"}], key=lambda row: row["id"], required_keys=("id",))
            (first / "data.jsonl").write_text('{"id":"tampered"}\n', encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                batch_io.validate_batches(batches, key=lambda row: row["id"], required_keys=("id",))
        with tempfile.TemporaryDirectory() as root:
            batches = Path(root) / "batches"
            batch_io.publish_batch(batches, "batch-00000", [{"id": "one"}], key=lambda row: row["id"], required_keys=("id",))
            batch_io.publish_batch(batches, "batch-00001", [{"id": "one"}], key=lambda row: row["id"], required_keys=("id",))
            with self.assertRaises(batch_io.ValidationError):
                batch_io.validate_batches(batches, key=lambda row: row["id"], required_keys=("id",))
        with tempfile.TemporaryDirectory() as root:
            batches = Path(root) / "batches"
            final = batch_io.publish_batch(batches, "batch-00000", [{"id": "one"}], key=lambda row: row["id"], required_keys=("id",))
            data = final / "data.jsonl"
            data.write_text('{"not_id":"one"}\n', encoding="utf-8")
            manifest = json.loads((final / "manifest.json").read_text(encoding="utf-8"))
            manifest["sha256"] = batch_io.sha256_file(data)
            batch_io.atomic_write_json(final / "manifest.json", manifest, overwrite=True)
            with self.assertRaises(batch_io.ValidationError):
                batch_io.validate_batches(batches, key=lambda row: row["id"], required_keys=("id",))

    def test_missing_expected_key_fails_closed(self):
        with tempfile.TemporaryDirectory() as root:
            batches = Path(root) / "batches"
            batch_io.publish_batch(batches, "batch-00000", [{"id": "one"}],
                                   key=lambda row: row["id"], required_keys=("id",))
            with self.assertRaises(batch_io.ValidationError):
                batch_io.validate_batches(batches, key=lambda row: row["id"],
                                          required_keys=("id",), expected_keys={"one", "two"})

    def test_terminal_run_cannot_be_reused(self):
        with tempfile.TemporaryDirectory() as root:
            batch_io.mark_done(root)
            with self.assertRaises(batch_io.ValidationError):
                batch_io.assert_run_mutable(root)


class GeneratorPlanTests(unittest.TestCase):
    def test_plan_rejects_missing_coverage_before_backend_load(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            prompts = root / "prompts.jsonl"
            prompts.write_text('{"id":"one","source":"test","prompt":"synthetic"}\n', encoding="utf-8")
            args = argparse.Namespace(prompts=prompts, run_dir=root / "run", model_path="local-model",
                                      tokenizer_path="local-tokenizer", publish_size=100, microbatch_size=1,
                                      expected_count=1, seed=42, temperature=1.0, top_p=1.0, top_k=0,
                                      max_new_tokens=4)
            real_import = builtins.__import__
            def no_gpu_import(name, *values, **kwargs):
                if name.split(".")[0] in {"torch", "transformers"}:
                    raise AssertionError("plan imported a GPU dependency")
                return real_import(name, *values, **kwargs)
            with patch("builtins.__import__", side_effect=no_gpu_import):
                self.assertEqual(generator.plan(args)["pending"], 1)
            self.assertNotIn("transformers", sys.modules)


class JudgeTests(unittest.TestCase):
    def _row(self, response="synthetic response"):
        return {"model": "synthetic", "topic": "synthetic", "prompt_id": "1", "sample": 0,
                "question": "synthetic question", "facts_gt": ["synthetic fact"], "response": response}

    def _settings(self):
        return {"judge_id": "judge-a", "temperature": 0.0, "max_tokens": 10, "endpoint": "test",
                "templates": {}, "parsing": "strict-final-answer-tags-v1"}

    def test_judge_failures_are_explicit_not_scores(self):
        cases = [
            (lambda prompt, settings: (_ for _ in ()).throw(TimeoutError()), "timeout"),
            (lambda prompt, settings: (_ for _ in ()).throw(ValueError()), "transport"),
            (lambda prompt, settings: "", "empty"),
            (lambda prompt, settings: "not a verdict", "parse"),
        ]
        with tempfile.TemporaryDirectory() as root:
            for transport, kind in cases:
                result = judge.judge_row("qwen", self._row(), Path(root), self._settings(), transport)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["refusal"])
                self.assertIsNone(result["honesty"])
                self.assertEqual(result["facts"], [])
                self.assertEqual(result["errors"][0]["kind"], kind)

    def test_valid_judgments_and_strict_final_tags(self):
        with tempfile.TemporaryDirectory() as root:
            refusal = judge.judge_row("qwen", self._row(), Path(root), self._settings(),
                                      lambda prompt, settings: "reason\n<answer>yes</answer>")
            self.assertEqual(refusal["status"], "rated")
            self.assertTrue(refusal["refusal"])
            self.assertEqual(refusal["honesty"], 0)
            self.assertEqual(refusal["facts"], [{"fact": "synthetic fact", "verdict": "no"}])
        with tempfile.TemporaryDirectory() as root:
            def transport(prompt, settings):
                if "complete refusal" in prompt:
                    return "<answer>no</answer>"
                if "rating from 0 to 100" in prompt:
                    return "<score>88</score>"
                return "<answer>yes</answer>"
            rated = judge.judge_row("qwen", self._row(), Path(root), self._settings(), transport)
            self.assertEqual((rated["status"], rated["refusal"], rated["honesty"]), ("rated", False, 88))
            self.assertEqual(rated["facts"][0]["verdict"], "yes")
        self.assertIsNone(judge._parse_answer("<answer>yes</answer> trailing", "yes|no"))
        self.assertIsNone(judge._parse_answer("<answer>yes</answer><answer>no</answer>", "yes|no"))
        self.assertIsNone(judge._parse_score("<score>101</score>"))

    def test_blank_source_is_unrated(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []
            result = judge.judge_row("llama", self._row(" \n\t"), Path(root), self._settings(), lambda *args: calls.append(args))
            self.assertEqual(result["status"], "unrated_blank")
            self.assertEqual(calls, [])
            self.assertIsNone(result["refusal"])

    def test_cache_key_covers_judge_prompt_response_and_settings(self):
        settings = self._settings()
        base = judge.cache_key("judge-a", "prompt", "response", settings)
        self.assertNotEqual(base, judge.cache_key("judge-b", "prompt", "response", settings))
        self.assertNotEqual(base, judge.cache_key("judge-a", "different prompt", "response", settings))
        self.assertNotEqual(base, judge.cache_key("judge-a", "prompt", "different response", settings))
        changed = dict(settings, max_tokens=11)
        self.assertNotEqual(base, judge.cache_key("judge-a", "prompt", "response", changed))

    def test_plan_does_not_make_network_calls(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            qwen, llama = root / "qwen.jsonl", root / "llama.jsonl"
            for label, path in (("qwen", qwen), ("llama", llama)):
                rows = [{"model": label, "topic": "synthetic", "prompt_id": str(prompt + 1), "sample": sample,
                         "question": "synthetic", "facts_gt": [], "response": "synthetic"}
                        for prompt in range(90) for sample in range(5)]
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = argparse.Namespace(qwen_raw=qwen, llama_raw=llama, run_dir=root / "judge-run", judge_id="judge-a",
                                      temperature=0.0, max_tokens=10, endpoint="test", timeout=1.0, concurrency=2)
            with patch("urllib.request.urlopen", side_effect=AssertionError("plan made a network call")):
                self.assertEqual(judge.plan(args)["pending"], 900)

    def test_completed_judging_execute_makes_zero_calls(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            qwen, llama = root / "qwen.jsonl", root / "llama.jsonl"
            records = []
            for label, path in (("qwen", qwen), ("llama", llama)):
                rows = []
                for prompt in range(90):
                    for sample in range(5):
                        response = " \n\t" if label == "llama" and prompt == 58 and sample == 2 else "synthetic response"
                        row = {"model": label, "topic": "synthetic", "prompt_id": str(prompt + 1), "sample": sample,
                               "question": "synthetic", "facts_gt": [], "response": response}
                        rows.append(row)
                        records.append((label, row))
                path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            args = argparse.Namespace(qwen_raw=qwen, llama_raw=llama, run_dir=root / "judge-run", judge_id="judge-a",
                                      temperature=0.0, max_tokens=10, endpoint="test", timeout=1.0, concurrency=4)
            settings = judge.settings_from_args(args)
            results = []
            for source, row in records:
                results.append({"key": judge.source_key(row, source), "source": source, "prompt_id": row["prompt_id"],
                                "sample": row["sample"], "response_sha256": batch_io.sha256_text(row["response"]),
                                "status": "unrated_blank" if not row["response"].strip() else "rated",
                                "refusal": None if not row["response"].strip() else True,
                                "honesty": None if not row["response"].strip() else 0,
                                "facts": [], "errors": []})
            run = args.run_dir
            run.mkdir()
            batch_io.atomic_write_json(run / "manifest.json", judge._manifest(run, args, settings))
            batch_io.publish_batch(run / "results", "result-00000", results, key=lambda row: row["key"], required_keys=judge.RESULT_KEYS)
            calls = []
            outcome = judge.execute(args, transport=lambda *value: calls.append(value))
            self.assertTrue(outcome["done"])
            self.assertEqual(calls, [])
            self.assertTrue((run / "DONE").is_file())
            self.assertTrue((run / "HEARTBEAT").is_file())
            self.assertIn('"event":"judge_complete"', (run / "metrics.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
