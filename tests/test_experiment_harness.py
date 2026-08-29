import argparse
import builtins
import io
import json
import socket
import sys
import urllib.error
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
                                      tokenizer_path="local-tokenizer", model_revision=generator.MODEL_REVISION,
                                      output_model_label=generator.MODEL_ID, max_batch_size=256,
                                      conv_index_budget=131072, memory_pressure_threshold=0.85,
                                      seed=42, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=4096)
            real_import = builtins.__import__
            def no_gpu_import(name, *values, **kwargs):
                if name.split(".")[0] in {"torch", "transformers"}:
                    raise AssertionError("plan imported a GPU dependency")
                return real_import(name, *values, **kwargs)
            provenance = {"input_sha256": batch_io.sha256_file(prompts), "source_file": "synthetic",
                          "source_sha256": "synthetic", "evaluation_questions": "synthetic",
                          "evaluation_sha256": "synthetic", "hereditary_commit": generator.HEREDITARY_COMMIT,
                          "unique_prompt_hashes": 1}
            with patch.object(generator, "EXPECTED_COUNT", 1), \
                 patch.object(generator, "_validate_input_provenance", return_value=provenance), \
                 patch.object(generator, "_verify_snapshot", return_value={"synthetic": True}), \
                 patch("builtins.__import__", side_effect=no_gpu_import):
                self.assertEqual(generator.plan(args)["pending"], 1)
            self.assertNotIn("transformers", sys.modules)


class JudgeTests(unittest.TestCase):
    def _row(self, response="synthetic response"):
        return {"model": "synthetic", "topic": "synthetic", "prompt_id": "1", "sample": 0,
                "question": "synthetic question", "facts_gt": ["synthetic fact"], "response": response}

    def _settings(self):
        return {"judge_id": "judge-a", "temperature": 0.0, "max_tokens": 10, "endpoint": "test",
                "templates": {}, "parsing": "conmy-first-search-v1"}

    def _write_probe_sources(self, root):
        paths, records = {}, []
        for label in ("qwen", "llama"):
            path = root / (label + ".jsonl")
            rows = []
            for prompt in range(90):
                for sample in range(5):
                    response = "synthetic response"
                    row = {"model": label, "topic": "synthetic", "prompt_id": str(prompt + 1),
                           "sample": sample, "question": "synthetic", "facts_gt": ["synthetic fact"],
                           "response": response}
                    rows.append(row)
                    records.append((label, row))
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            paths[label] = path
        return paths["qwen"], paths["llama"], records

    def _frozen_args(self, qwen, llama, run_dir):
        return argparse.Namespace(qwen_raw=qwen, llama_raw=llama, run_dir=run_dir,
                                  judge_id=judge.JUDGE_ID, temperature=judge.JUDGE_TEMPERATURE,
                                  max_tokens=judge.JUDGE_MAX_TOKENS, endpoint=judge.OPENROUTER_ENDPOINT,
                                  timeout=1.0, concurrency=2)

    def test_prompt_templates_match_pinned_conmy_source_hashes(self):
        # SHA-256 values are pinned from external/hereditary/chinese_censorship_eval/scripts/eval.py.
        # Normal judging intentionally does not depend on that ignored external checkout.
        self.assertEqual({
            "refusal": batch_io.sha256_text(judge.REFUSAL_PROMPT),
            "honesty": batch_io.sha256_text(judge.HONESTY_PROMPT),
            "fact": batch_io.sha256_text(judge.FACT_PROMPT),
        }, {
            "refusal": "e7aeefbdefa94d5c644833995ebb6a9b0f2b2035f262b55614cb5b1b3a9f1a9a",
            "honesty": "79e3845657010db77677354c98b322b22afc3caad69a786ffae733cb538a5056",
            "fact": "0241b1965b2bb2166a1c1e549141017a477b9aa351e4ed9b56c8a03ba1215ee0",
        })

    def test_judge_defaults_freeze_openrouter_configuration(self):
        args = judge.build_parser().parse_args(["--run-dir", "synthetic-run"])
        self.assertEqual(args.judge_id, "google/gemini-3-flash-preview")
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.max_tokens, 1500)
        self.assertEqual(args.endpoint, "https://openrouter.ai/api/v1/chat/completions")
        self.assertEqual(judge.settings_from_args(args)["reasoning"], {"enabled": False})

    def test_openrouter_transport_builds_request_and_extracts_content(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"<answer>no</answer>"}}]}'

        settings = {"judge_id": "google/gemini-3-flash-preview", "temperature": 0.0,
                    "max_tokens": 1500}
        with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            result = judge.OpenRouterTransport("test-key", "https://example.invalid/chat", 12.5)(
                "rendered prompt", settings)
        self.assertEqual(result, "<answer>no</answer>")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.invalid/chat")
        self.assertEqual(request.get_method(), "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(request.get_header("Content-type"), "application/json")
        self.assertEqual(urlopen.call_args.kwargs["timeout"], 12.5)
        self.assertEqual(json.loads(request.data.decode("utf-8")), {
            "model": "google/gemini-3-flash-preview",
            "messages": [{"role": "user", "content": "rendered prompt"}],
            "temperature": 0.0,
            "max_tokens": 1500,
            "reasoning": {"enabled": False},
        })

    def test_transport_retries_only_bounded_transient_failures(self):
        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b'{"choices":[{"message":{"content":"ok"}}]}'

        settings = {"judge_id": judge.JUDGE_ID, "temperature": 0.0, "max_tokens": 1500}
        transient_errors = [TimeoutError(), socket.timeout(), urllib.error.URLError("offline"),
                            urllib.error.HTTPError("https://example.invalid", 408, "timeout", None, io.BytesIO()),
                            urllib.error.HTTPError("https://example.invalid", 429, "rate", None, io.BytesIO()),
                            urllib.error.HTTPError("https://example.invalid", 503, "server", None, io.BytesIO())]
        for error in transient_errors:
            with self.subTest(error=type(error).__name__), \
                 patch("urllib.request.urlopen", side_effect=[error, Response()]) as urlopen, \
                 patch("experiment.judge_probe.time.sleep") as sleep:
                result = judge.OpenRouterTransport("test-key", judge.OPENROUTER_ENDPOINT, 1.0)("prompt", settings)
            self.assertEqual(result, "ok")
            self.assertEqual(urlopen.call_count, 2)
            sleep.assert_called_once_with(1.0)
            if isinstance(error, urllib.error.HTTPError):
                error.close()
        with patch("urllib.request.urlopen", side_effect=TimeoutError()) as urlopen, \
             patch("experiment.judge_probe.time.sleep") as sleep:
            with self.assertRaises(TimeoutError):
                judge.OpenRouterTransport("test-key", judge.OPENROUTER_ENDPOINT, 1.0)("prompt", settings)
        self.assertEqual(urlopen.call_count, judge.TRANSPORT_MAX_ATTEMPTS)
        self.assertEqual(sleep.call_count, judge.TRANSPORT_MAX_ATTEMPTS - 1)
        non_transient = urllib.error.HTTPError("https://example.invalid", 400, "bad", None, io.BytesIO())
        with patch("urllib.request.urlopen", side_effect=non_transient) as urlopen, \
             patch("experiment.judge_probe.time.sleep") as sleep:
            with self.assertRaises(urllib.error.HTTPError):
                judge.OpenRouterTransport("test-key", judge.OPENROUTER_ENDPOINT, 1.0)("prompt", settings)
        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()
        non_transient.close()

    def test_frozen_protocol_rejects_settings_and_source_changes(self):
        args = judge.build_parser().parse_args(["--run-dir", "synthetic-run"])
        for attribute, value in (("judge_id", "other-judge"), ("temperature", 0.1),
                                 ("max_tokens", 1499), ("endpoint", "https://example.invalid")):
            with self.subTest(attribute=attribute):
                changed = argparse.Namespace(**vars(args))
                setattr(changed, attribute, value)
                with self.assertRaises(batch_io.ValidationError):
                    judge.settings_from_args(changed)
        settings = judge.settings_from_args(args)
        settings["reasoning"] = {"enabled": True}
        with self.assertRaises(batch_io.ValidationError):
            judge._validate_frozen_settings(settings)
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            qwen, llama, _ = self._write_probe_sources(root)
            with self.assertRaises(batch_io.ValidationError):
                judge._validate_source_hashes(qwen, llama)
            rejected_args = self._frozen_args(qwen, llama, root / "rejected-run")
            rejected_args.max_tokens = 1
            with patch("urllib.request.urlopen", side_effect=AssertionError("constructed network before rejection")):
                with self.assertRaises(batch_io.ValidationError):
                    judge.execute(rejected_args)
            qwen_text, llama_text = qwen.read_text(encoding="utf-8"), llama.read_text(encoding="utf-8")
            alternate_qwen = root / "alternate-qwen.jsonl"
            alternate_qwen.write_text(qwen_text, encoding="utf-8")
            with patch.object(judge, "QWEN_RAW_SHA256", batch_io.sha256_file(qwen)), \
                 patch.object(judge, "LLAMA_RAW_SHA256", batch_io.sha256_file(llama)):
                judge._validate_source_hashes(alternate_qwen, llama)
                qwen.write_text(qwen_text + " ", encoding="utf-8")
                with self.assertRaises(batch_io.ValidationError):
                    judge._validate_source_hashes(qwen, llama)
                qwen.write_text(qwen_text, encoding="utf-8")
                llama.write_text(llama_text + " ", encoding="utf-8")
                with self.assertRaises(batch_io.ValidationError):
                    judge._validate_source_hashes(qwen, llama)

    def test_judge_failures_are_explicit_not_scores(self):
        cases = [
            (lambda prompt, settings: (_ for _ in ()).throw(TimeoutError()), "timeout"),
            (lambda prompt, settings: (_ for _ in ()).throw(ValueError()), "transport"),
            (lambda prompt, settings: "", "empty"),
        ]
        with tempfile.TemporaryDirectory() as root:
            for transport, kind in cases:
                result = judge.judge_row("qwen", self._row(), Path(root), self._settings(), transport)
                self.assertEqual(result["status"], "error")
                self.assertIsNone(result["refusal"])
                self.assertIsNone(result["honesty"])
                self.assertEqual(result["facts"], [])
                self.assertEqual(result["errors"][0]["kind"], kind)

    def test_conmy_first_match_parsing_and_refusal_short_circuit(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def refusal_transport(prompt, settings):
                calls.append(prompt)
                return "reason\n<answer>yes</answer>"

            refusal = judge.judge_row("qwen", self._row(), Path(root), self._settings(), refusal_transport)
            self.assertEqual(refusal["status"], "rated")
            self.assertTrue(refusal["refusal"])
            self.assertEqual(refusal["honesty"], 0)
            self.assertEqual(refusal["facts"], [{"fact": "synthetic fact", "verdict": "no"}])
            self.assertEqual(len(calls), 1)
            self.assertIn("complete refusal", calls[0])
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def transport(prompt, settings):
                calls.append(prompt)
                if "complete refusal" in prompt:
                    return "<answer>no</answer>"
                if "rating from 0 to 100" in prompt:
                    return "<score>88</score>"
                return "<answer>yes</answer>"

            rated = judge.judge_row("qwen", self._row(), Path(root), self._settings(), transport)
            self.assertEqual((rated["status"], rated["refusal"], rated["honesty"]), ("rated", False, 88))
            self.assertEqual(rated["facts"][0]["verdict"], "yes")
            self.assertEqual(len(calls), 3)
            self.assertIn("complete refusal", calls[0])
            self.assertIn("rating from 0 to 100", calls[1])
            self.assertIn("specific FACT", calls[2])
        self.assertEqual(judge._parse_answer("<answer>yes</answer> trailing", "yes|no"), "yes")
        self.assertEqual(judge._parse_answer("<answer>yes</answer><answer>no</answer>", "yes|no"), "yes")
        self.assertEqual(judge._parse_score("<score>101</score>"), 100)

    def test_conmy_no_match_defaults_and_caches_nonempty_responses(self):
        with tempfile.TemporaryDirectory() as root:
            calls = []

            def no_match_transport(prompt, settings):
                calls.append(prompt)
                return "unstructured judge text"

            result = judge.judge_row("qwen", self._row(), Path(root), self._settings(), no_match_transport)
            self.assertEqual(result["status"], "rated")
            self.assertIsNone(result["refusal"])
            self.assertIsNone(result["honesty"])
            self.assertEqual(result["facts"], [{"fact": "synthetic fact", "verdict": "no"}])
            self.assertEqual(len(calls), 3)
            cached = judge.judge_row("qwen", self._row(), Path(root), self._settings(),
                                     lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")))
            self.assertEqual(cached, result)

    def test_offline_audit_replays_cached_results_with_conmy_semantics(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            settings = judge._legacy_settings()
            row = self._row()

            def transport(prompt, unused_settings):
                if "complete refusal" in prompt:
                    return "prefix <answer>NO</answer> trailing <answer>yes</answer>"
                if "rating from 0 to 100" in prompt:
                    return "<score>101</score> trailing"
                return "no answer tag"

            result = judge.judge_row("qwen", row, run, settings, transport)
            batch_io.publish_batch(run / "results", "result-00000", [result],
                                   key=lambda value: value["key"], required_keys=judge.RESULT_KEYS)
            self.assertEqual(judge.audit_historical_results(run, [("qwen", row)], settings), 1)

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
            args = self._frozen_args(qwen, llama, root / "judge-run")
            with patch.object(judge, "QWEN_RAW_SHA256", batch_io.sha256_file(qwen)), \
                 patch.object(judge, "LLAMA_RAW_SHA256", batch_io.sha256_file(llama)), \
                 patch("urllib.request.urlopen", side_effect=AssertionError("plan made a network call")):
                self.assertEqual(judge.plan(args)["pending"], 900)

    def test_error_attempt_is_resumable_and_reuses_cached_stages(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            qwen, llama, records = self._write_probe_sources(root)
            args = self._frozen_args(qwen, llama, root / "judge-run")
            args.concurrency = 1
            settings = judge.settings_from_args(args)
            target_source, target_row = records[0]
            finals = []
            for source, row in records[1:]:
                finals.append({"key": judge.source_key(row, source), "source": source,
                               "prompt_id": row["prompt_id"], "sample": row["sample"],
                               "response_sha256": batch_io.sha256_text(row["response"]), "status": "rated",
                               "refusal": True, "honesty": 0,
                               "facts": [{"fact": "synthetic fact", "verdict": "no"}], "errors": []})
            run = args.run_dir
            run.mkdir()
            with patch.object(judge, "QWEN_RAW_SHA256", batch_io.sha256_file(qwen)), \
                 patch.object(judge, "LLAMA_RAW_SHA256", batch_io.sha256_file(llama)):
                batch_io.atomic_write_json(run / "manifest.json", judge._manifest(run, args, settings))
                batch_io.publish_batch(run / "results", "result-00000", finals,
                                       key=lambda row: row["key"], required_keys=judge.RESULT_KEYS)

                def fail_at_fact(prompt, settings):
                    if "complete refusal" in prompt:
                        return "<answer>no</answer>"
                    if "rating from 0 to 100" in prompt:
                        return "<score>88</score>"
                    raise ValueError("synthetic fact transport failure")

                first = judge.execute(args, transport=fail_at_fact)
                self.assertFalse(first["done"])
                self.assertEqual(first["pending"], 1)
                self.assertFalse((run / "DONE").exists())
                attempts = judge._error_attempts(run)
                self.assertEqual(len(attempts), 1)
                self.assertEqual(attempts[0]["key"], judge.source_key(target_row, target_source))
                self.assertEqual(attempts[0]["errors"][0]["stage"], "fact")
                metrics = (run / "metrics.jsonl").read_text(encoding="utf-8")
                self.assertIn('"event":"error_attempt_published"', metrics)
                self.assertIn('"event":"judge_incomplete"', metrics)

                retry_calls = []

                def succeed_fact(prompt, settings):
                    retry_calls.append(prompt)
                    return "<answer>yes</answer>"

                second = judge.execute(args, transport=succeed_fact)
            self.assertTrue(second["done"])
            self.assertEqual(retry_calls, [judge.FACT_PROMPT.format(
                question=target_row["question"], fact="synthetic fact", model_response=target_row["response"])])
            self.assertTrue((run / "DONE").is_file())
            final_rows = batch_io.validate_batches(run / "results", key=lambda row: row["key"],
                                                   required_keys=judge.RESULT_KEYS)
            self.assertEqual(len(final_rows), 900)
            self.assertTrue(all(row["status"] != "error" for row in final_rows))

    def test_exact_bound_migration_enables_plan_on_legacy_fixture(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            qwen, llama, records = self._write_probe_sources(root)
            args = self._frozen_args(qwen, llama, root / "judge-run")
            run = args.run_dir
            run.mkdir()
            pending = judge.LEGACY_PENDING_KEYS
            final_rows = []
            for source, row in records:
                key = judge.source_key(row, source)
                if key in pending:
                    continue
                final_rows.append({"key": key, "source": source, "prompt_id": row["prompt_id"],
                                   "sample": row["sample"], "response_sha256": batch_io.sha256_text(row["response"]),
                                   "status": "rated", "refusal": True, "honesty": 0,
                                   "facts": [{"fact": "synthetic fact", "verdict": "no"}], "errors": []})
            self.assertEqual(len(final_rows), judge.LEGACY_FINAL_ROWS)
            batch_io.publish_batch(run / "results", "result-00000", final_rows,
                                   key=lambda value: value["key"], required_keys=judge.RESULT_KEYS)
            for attempt in range(judge.LEGACY_ERROR_ATTEMPTS):
                error = {"key": "qwen:44:4", "source": "qwen", "prompt_id": "44", "sample": 4,
                         "response_sha256": batch_io.sha256_text("synthetic response"), "status": "error",
                         "refusal": None, "honesty": None, "facts": [],
                         "errors": [{"stage": "refusal", "kind": "parse", "detail": "historical"}],
                         "attempt": attempt}
                batch_io.publish_batch(run / "error-attempts", "attempt-%05d" % attempt, [error],
                                       key=lambda value: "%s:%s" % (value["attempt"], value["key"]),
                                       required_keys=judge.ERROR_ATTEMPT_KEYS)
            legacy = {**judge._legacy_settings(), "qwen_sha256": batch_io.sha256_file(qwen),
                      "llama_sha256": batch_io.sha256_file(llama), "source_rows": 900}
            batch_io.atomic_write_json(run / "manifest.json", legacy)
            with patch.object(judge, "QWEN_RAW_SHA256", batch_io.sha256_file(qwen)), \
                 patch.object(judge, "LLAMA_RAW_SHA256", batch_io.sha256_file(llama)), \
                 patch.object(judge, "_current_judge_run_dir", return_value=run), \
                 patch.object(judge, "LEGACY_MANIFEST_SHA256", batch_io.sha256_file(run / "manifest.json")), \
                 patch.object(judge, "audit_current_judge_run", return_value={"audited_final_rows": 896,
                                                                                "blank_rows": 0,
                                                                                "pending_keys": sorted(pending)}):
                migration = judge.migrate_current_judge_run(args)
                report = judge.plan(args)
            self.assertEqual(migration["migrated"], judge.CURRENT_JUDGE_RUN_NAME)
            self.assertEqual(report["pending"], 4)
            self.assertEqual(report["manifest"]["format"], "probe-judge-v2")
            self.assertEqual(report["manifest"]["parsing"], "conmy-first-search-v1")

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
            args = self._frozen_args(qwen, llama, root / "judge-run")
            args.concurrency = 4
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
            with patch.object(judge, "QWEN_RAW_SHA256", batch_io.sha256_file(qwen)), \
                 patch.object(judge, "LLAMA_RAW_SHA256", batch_io.sha256_file(llama)):
                batch_io.atomic_write_json(run / "manifest.json", judge._manifest(run, args, settings))
                batch_io.publish_batch(run / "results", "result-00000", results,
                                       key=lambda row: row["key"], required_keys=judge.RESULT_KEYS)
                calls = []
                outcome = judge.execute(args, transport=lambda *value: calls.append(value))
            self.assertTrue(outcome["done"])
            self.assertEqual(calls, [])
            self.assertTrue((run / "DONE").is_file())
            self.assertTrue((run / "HEARTBEAT").is_file())
            self.assertIn('"event":"judge_complete"', (run / "metrics.jsonl").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
