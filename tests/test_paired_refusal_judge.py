import argparse
import gzip
import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from experiment import batch_io
from experiment import judge_paired_refusal as paired
from experiment import judge_probe


class PairedRefusalHarnessTests(unittest.TestCase):
    def _fixture(self, sample_size=1):
        stack = ExitStack()
        root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        source_dir = root / "sources"
        source_dir.mkdir()
        original, ablated, amendment = source_dir / "original.gz", source_dir / "ablated.jsonl", source_dir / "amendment.json"
        rows = [
            {"id": "a", "source": "one", "prompt": "first", "response": " ", "model": paired.ORIGINAL_MODEL},
            {"id": "b", "source": "two", "prompt": "second", "response": "original second", "model": paired.ORIGINAL_MODEL},
        ]
        altered = [{**row, "response": "ablated " + row["id"], "model": paired.ABLITERATED_MODEL} for row in rows]
        payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode("utf-8")
        original.write_bytes(gzip.compress(payload, mtime=0))
        ablated.write_text("".join(json.dumps(row) + "\n" for row in altered), encoding="utf-8")
        amendment_value = {
            "format": "paired-refusal-distribution-amendment-v1", "date": "2026-08-30",
            "decision": "exploratory_paired_refusal_distribution_study", "scope": paired.AMENDMENT_SCOPE,
            "selection": {"algorithm": paired.SELECTION_ALGORITHM, "seed": paired.SEED, "pair_count": sample_size,
                          "ranking": "lowest SHA-256 ranks over version, seed, id, source, and exact UTF-8 prompt SHA-256",
                          "rank_bytes": "UTF-8(algorithm)+NUL+ASCII(seed)+NUL+UTF-8(id)+NUL+UTF-8(source)+NUL+ASCII(prompt_sha256)"},
            "sources": {"original": {"path": "original.gz", "gzip_sha256": batch_io.sha256_file(original),
                                        "decompressed_jsonl_sha256": batch_io.sha256_bytes(payload), "model": paired.ORIGINAL_MODEL},
                        "abliterated": {"path": "ablated.jsonl", "sha256": batch_io.sha256_file(ablated),
                                         "model": paired.ABLITERATED_MODEL}},
            "judge": {"rubric": "unchanged Conmy REFUSAL_PROMPT only", "model": judge_probe.JUDGE_ID,
                      "temperature": judge_probe.JUDGE_TEMPERATURE, "max_tokens": judge_probe.JUDGE_MAX_TOKENS,
                      "reasoning_enabled": False, "parser": "conmy-first-search-v1"},
            "interpretation": paired.AMENDMENT_INTERPRETATION,
        }
        amendment.write_text(json.dumps(amendment_value, indent=2) + "\n", encoding="utf-8")
        stack.enter_context(patch.multiple(paired, ORIGINAL_PATH=original, ABLITERATED_PATH=ablated, PROTOCOL_PATH=amendment,
                                           PROTOCOL_SHA256=batch_io.sha256_file(amendment), ORIGINAL_PROTOCOL_PATH="original.gz",
                                           ABLITERATED_PROTOCOL_PATH="ablated.jsonl", ORIGINAL_GZIP_SHA256=batch_io.sha256_file(original),
                                           ORIGINAL_JSONL_SHA256=batch_io.sha256_bytes(payload),
                                           ABLITERATED_SHA256=batch_io.sha256_file(ablated), ROW_COUNT=2, SAMPLE_SIZE=sample_size))
        return stack, root

    def _args(self, root):
        return argparse.Namespace(run_dir=root / "run", original=paired.ORIGINAL_PATH, abliterated=paired.ABLITERATED_PATH,
                                  concurrency=2, timeout=1.0)

    def test_authoritative_full_19996_alignment_and_deterministic_selection(self):
        originals, ablated, hashes = paired.load_sources()
        self.assertEqual((len(originals), len(ablated)), (19_996, 19_996))
        self.assertEqual(hashes["original_gzip_sha256"], paired.ORIGINAL_GZIP_SHA256)
        selected = paired.select_pairs(originals, ablated)
        self.assertEqual(len(selected), 1_000)
        self.assertEqual(len({row["source_index"] for row in selected}), 1_000)
        self.assertEqual(selected, paired.select_pairs(originals, ablated))

    def test_actual_duplicate_cross_arm_cache_keys_are_identified(self):
        originals, ablated, _ = paired.load_sources()
        selected = paired.select_pairs(originals, ablated)
        sample = paired._sample_rows(selected, originals, ablated)
        settings = judge_probe._frozen_settings()
        duplicates = []
        for row in sample:
            original_prompt = judge_probe.REFUSAL_PROMPT.format(question=row["prompt"], model_response=row["original_response"])
            ablated_prompt = judge_probe.REFUSAL_PROMPT.format(question=row["prompt"], model_response=row["abliterated_response"])
            original_key = judge_probe.cache_key(settings["judge_id"], original_prompt, row["original_response"], settings)
            ablated_key = judge_probe.cache_key(settings["judge_id"], ablated_prompt, row["abliterated_response"], settings)
            if original_key == ablated_key:
                duplicates.append(row["selection_rank"])
        self.assertEqual(duplicates, [679, 888, 964])

    def test_exact_source_hash_decompressed_hash_alignment_selection_and_immutability(self):
        stack, root = self._fixture()
        with stack:
            args = self._args(root)
            original, ablated, hashes = paired.load_sources(args.original, args.abliterated)
            self.assertEqual(hashes["original_jsonl_sha256"], batch_io.sha256_bytes(gzip.decompress(args.original.read_bytes())))
            self.assertEqual(len(original), len(ablated))
            selected = paired.select_pairs(original, ablated)
            self.assertEqual(len(selected), 1)
            self.assertEqual(len({row["source_index"] for row in selected}), 1)
            first = paired.prepare(args)
            self.assertFalse(first["reused"])
            self.assertTrue(paired.prepare(args)["reused"])
            (args.run_dir / "selection.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                paired.prepare(args)

    def test_source_path_hash_and_alignment_tampering_fail_closed(self):
        stack, root = self._fixture()
        with stack:
            args = self._args(root)
            with self.assertRaises(batch_io.ValidationError):
                paired.load_sources(root / "other.gz", args.abliterated)
            args.abliterated.write_text(args.abliterated.read_text(encoding="utf-8") + " ", encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                paired.load_sources(args.original, args.abliterated)
            paired.PROTOCOL_PATH.write_text("{}\n", encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                paired._validate_protocol_amendment()

    def test_plan_is_offline_and_rejects_copied_results(self):
        stack, root = self._fixture()
        with stack:
            args = self._args(root)
            paired.prepare(args)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network in plan")):
                self.assertEqual(paired.plan(args)["planned_calls"], 1)  # selected pair has one blank arm
            selection, sample, manifest = paired._load_prepared(args.run_dir)
            expected = paired._expected(sample, manifest["judge"], batch_io.sha256_file(args.run_dir / "manifest.json"))
            key, identity = next(iter(expected.items()))
            copied = {**identity, "judge_raw_sha256": "fabricated", "status": "rated", "refusal": True, "errors": []}
            copied["source_index"] += 1
            batch_io.publish_batch(args.run_dir / "results", "result-00000", [copied], key=lambda row: row["key"],
                                   required_keys=paired.RESULT_KEYS)
            with self.assertRaises(batch_io.ValidationError):
                paired.plan(args)

    def test_parser_null_blank_retry_resume_and_done_gate(self):
        stack, root = self._fixture()
        with stack:
            args = self._args(root)
            paired.prepare(args)
            # First run has an explicit resumable failure for the nonblank response; blank gets finalized.
            first = paired.execute(args, transport=lambda *_: (_ for _ in ()).throw(TimeoutError()))
            self.assertFalse(first["done"])
            self.assertEqual(first["completed"], 1)
            self.assertFalse((args.run_dir / "DONE").exists())
            _, sample, manifest = paired._load_prepared(args.run_dir)
            expected = paired._expected(sample, manifest["judge"], batch_io.sha256_file(args.run_dir / "manifest.json"))
            self.assertEqual(len(paired._attempts(args.run_dir, expected)), 1)
            second = paired.execute(args, transport=lambda *_: "unstructured")
            self.assertTrue(second["done"])
            rows = paired._final_results(args.run_dir, expected)
            self.assertEqual({row["status"] for row in rows}, {"unrated_blank", "unrated_parse"})
            self.assertTrue((args.run_dir / "DONE").is_file())

    def test_refusal_prompt_identity_and_paired_summary_math(self):
        self.assertEqual(batch_io.sha256_text(judge_probe.REFUSAL_PROMPT),
                         "e7aeefbdefa94d5c644833995ebb6a9b0f2b2035f262b55614cb5b1b3a9f1a9a")
        base = {"source": "x", "status": "rated"}
        original = [{**base, "selection_rank": 0, "refusal": True}, {**base, "selection_rank": 1, "refusal": True},
                    {**base, "selection_rank": 2, "refusal": False}, {**base, "selection_rank": 3, "refusal": False}]
        ablated = [{**base, "selection_rank": 0, "refusal": True}, {**base, "selection_rank": 1, "refusal": False},
                    {**base, "selection_rank": 2, "refusal": True}, {**base, "selection_rank": 3, "refusal": False}]
        summary = paired._paired_summary(original, ablated)
        self.assertEqual(summary["counts"], {"both": 1, "original_only": 1, "abliterated_only": 1, "neither": 1})
        self.assertEqual(summary["mcnemar_exact_two_sided_p_value"], 1.0)
        self.assertAlmostEqual(paired._arm_summary(original)["refusal_rate"], 0.5)

    def test_mocked_exact_2000_logical_calls_complete(self):
        """Exercise execution scheduling at the real 1,000-pair / 2,000-key boundary without I/O batches."""
        with tempfile.TemporaryDirectory() as root_text:
            root = Path(root_text)
            run = root / "run"
            run.mkdir()
            sample = [{"selection_rank": rank, "source_index": rank, "id": str(rank), "source": "s",
                       "prompt_sha256": batch_io.sha256_text("q"), "original_response_sha256": batch_io.sha256_text("o"),
                       "abliterated_response_sha256": batch_io.sha256_text("a"), "prompt": "q", "original_response": "o",
                       "abliterated_response": "a", "original_model": paired.ORIGINAL_MODEL,
                       "abliterated_model": paired.ABLITERATED_MODEL} for rank in range(1000)]
            selection = [{key: row[key] for key in ("selection_rank", "source_index", "id", "source", "prompt_sha256", "original_response_sha256", "abliterated_response_sha256")} for row in sample]
            manifest = {"judge": judge_probe._frozen_settings(), "selection_sha256": "s", "paired_sample_sha256": "p"}
            batch_io.atomic_write_json(run / "manifest.json", manifest)
            final_rows, calls = [], []
            args = argparse.Namespace(run_dir=run, concurrency=16, timeout=1.0)
            report = {"manifest": manifest}
            def fake_publish(_directory, _name, rows, **_kwargs):
                final_rows.extend(rows)
            def fake_final(_run, _expected):
                return list(final_rows)
            with patch.object(paired, "SAMPLE_SIZE", 1000), patch.object(paired, "plan", return_value=report), \
                 patch.object(paired, "_load_prepared", return_value=(selection, sample, manifest)), \
                 patch.object(paired, "_final_results", side_effect=fake_final), \
                 patch.object(paired, "finalized_batches", return_value=[]), \
                 patch.object(paired, "publish_batch", side_effect=fake_publish), \
                 patch.object(judge_probe, "_call_cached", side_effect=lambda _r, _p, _s, _set, transport: (transport("x", {}), None)):
                outcome = paired.execute(args, transport=lambda *_: calls.append(1) or "<answer>no</answer>")
            self.assertTrue(outcome["done"])
            self.assertEqual(len(calls), 2000)
            self.assertEqual(len(final_rows), 2000)

    def test_canonical_cache_binding_rejects_fabricated_final_rows(self):
        stack, root = self._fixture()
        with stack:
            args = self._args(root)
            paired.prepare(args)
            _, sample, manifest = paired._load_prepared(args.run_dir)
            expected = paired._expected(sample, manifest["judge"], batch_io.sha256_file(args.run_dir / "manifest.json"))
            key, identity = next((key, value) for key, value in expected.items() if value["judge_cache_key"])
            fabricated = {**identity, "judge_raw_sha256": batch_io.sha256_text("<answer>yes</answer>"),
                          "status": "rated", "refusal": True, "errors": []}
            batch_io.publish_batch(args.run_dir / "results", "result-00000", [fabricated],
                                   key=lambda row: row["key"], required_keys=paired.RESULT_KEYS)
            with self.assertRaises(batch_io.ValidationError):
                paired.plan(args)

    def test_nonblank_cannot_be_forged_as_unrated_blank(self):
        stack, root = self._fixture()
        with stack:
            args = self._args(root)
            paired.prepare(args)
            _, sample, manifest = paired._load_prepared(args.run_dir)
            expected = paired._expected(sample, manifest["judge"], batch_io.sha256_file(args.run_dir / "manifest.json"))
            _, identity = next((key, value) for key, value in expected.items() if value["judge_cache_key"])
            forged = {**identity, "judge_raw_sha256": None, "status": "unrated_blank", "refusal": None, "errors": []}
            batch_io.publish_batch(args.run_dir / "results", "result-00000", [forged],
                                   key=lambda row: row["key"], required_keys=paired.RESULT_KEYS)
            with self.assertRaises(batch_io.ValidationError):
                paired.plan(args)

    def test_partial_prepare_recovers_only_exact_artifact_prefixes(self):
        stack, root = self._fixture()
        with stack:
            source_args = self._args(root)
            paired.prepare(source_args)
            bytes_by_name = {name: (source_args.run_dir / name).read_bytes()
                             for name in ("selection.jsonl", "paired-sample.jsonl", "manifest.json")}
            for count in range(3):
                run = root / ("recovered-%d" % count)
                run.mkdir()
                for name in list(bytes_by_name)[:count]:
                    (run / name).write_bytes(bytes_by_name[name])
                args = argparse.Namespace(**{**vars(source_args), "run_dir": run})
                report = paired.prepare(args)
                self.assertTrue(report["reused"] if count else not report["reused"])
                self.assertEqual((run / "selection.jsonl").read_bytes(), bytes_by_name["selection.jsonl"])
                self.assertEqual((run / "paired-sample.jsonl").read_bytes(), bytes_by_name["paired-sample.jsonl"])
                self.assertTrue((run / "manifest.json").is_file())

    def test_keyboard_interrupt_stops_rolling_submission_and_resumes_completed_work(self):
        stack, root = self._fixture(sample_size=2)
        with stack:
            args = self._args(root)
            args.concurrency = 2
            paired.prepare(args)
            calls = []
            first = threading.Event()
            def interrupted_transport(*_unused):
                calls.append(1)
                if not first.is_set():
                    first.set()
                    raise KeyboardInterrupt()
                return "<answer>no</answer>"
            with self.assertRaises(KeyboardInterrupt):
                paired.execute(args, transport=interrupted_transport)
            self.assertLessEqual(len(calls), args.concurrency)
            self.assertLess(len(calls), 3)  # There are three nonblank logical responses in this fixture.
            self.assertFalse((args.run_dir / "DONE").exists())
            resumed = paired.execute(args, transport=lambda *_: "<answer>no</answer>")
            self.assertTrue(resumed["done"])

    def test_cache_conflict_is_single_flight_and_persists_one_canonical_value(self):
        with tempfile.TemporaryDirectory() as root_text:
            run = Path(root_text)
            settings = judge_probe._frozen_settings()
            calls = []
            # The lock deliberately prevents the second provider call.
            def single_provider(_prompt, _settings):
                calls.append(1)
                return "<answer>yes</answer>"
            with ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(lambda _: judge_probe._call_cached(run, "p", "r", settings, single_provider)[0], range(2)))
            self.assertEqual(calls, [1])
            self.assertEqual(values, ["<answer>yes</answer>", "<answer>yes</answer>"])
            path = next((run / "cache").glob("*.json"))
            self.assertEqual(judge_probe._read_cache(path), "<answer>yes</answer>")
            # Concurrent conflicting publishers also resolve to the first persisted canonical value.
            second = Path(root_text) / "conflict.json"
            barrier = threading.Barrier(2)
            def conflicting_publish(raw):
                barrier.wait(timeout=2)
                return judge_probe._publish_cache_canonical(second, "conflict", raw)
            with ThreadPoolExecutor(max_workers=2) as pool:
                values = list(pool.map(conflicting_publish, ("<answer>yes</answer>", "<answer>no</answer>")))
            self.assertEqual(values[0], values[1])

    def test_execution_lease_rejects_second_launcher(self):
        with tempfile.TemporaryDirectory() as root_text:
            run = Path(root_text)
            with paired._ExecutionLease(run):
                with self.assertRaises(batch_io.ValidationError):
                    with paired._ExecutionLease(run):
                        pass

    def test_launcher_is_ascii_ps51_and_runs_prepare_plan(self):
        payload = (Path("scripts") / "judge-paired-refusal.ps1").read_bytes()
        text = payload.decode("ascii")
        self.assertIn("#Requires -Version 5.1", text)
        self.assertLess(text.index("'--prepare'"), text.index("'--plan'"))
        self.assertIn("GetEnvironmentVariable($keyName, 'User')", text)
        self.assertIn("SetEnvironmentVariable($keyName, $previousProcessKey, 'Process')", text)


if __name__ == "__main__":
    unittest.main()
