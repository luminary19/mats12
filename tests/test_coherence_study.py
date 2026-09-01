import argparse
import json
import threading
import tempfile
import urllib.error
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import batch_io
from experiment import judge_coherence as coherence
from experiment import prepare_coherence_study as prepare


class CoherenceStudyTests(unittest.TestCase):
    def _fixture(self):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        runs = root / "runs"
        runs.mkdir()
        specs = []
        for index, arm_id in enumerate((
            "qwen35_9b_aligned", "qwen35_9b_abliterated", "llama32_3b_base",
            "llama32_3b_qwen_aligned_sft", "llama32_3b_qwen_abliterated_sft", "llama32_3b_second_order_sft",
        )):
            path = root / (arm_id + ".jsonl")
            rows = []
            for prompt_id in ("p1", "p2"):
                for sample in range(5):
                    response = "answer %d %s %s" % (index, prompt_id, sample)
                    if index == 0 and prompt_id == "p1" and sample == 0:
                        response = " "
                    if index == 1 and prompt_id == "p2" and sample == 4:
                        response = "parse-me"
                    row = {"model": "model-%d" % index, "topic": "topic", "prompt_id": prompt_id,
                           "sample": sample, "question": "question " + prompt_id, "response": response}
                    if index % 2:
                        row.update({"facts_gt": [], "generation": {}, "judging": None})
                    else:
                        row.update({"refusal": False, "honesty": 50, "facts": []})
                    rows.append(row)
            path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
            specs.append(prepare.ArmSpec(arm_id, "Label " + str(index), str(path)))
        patches = patch.multiple(prepare, ARM_SPECS=tuple(specs), EXPECTED_PROMPT_COUNT=2, ROOT=root)
        judge_patches = patch.object(coherence, "ARM_SPECS", tuple(specs))
        patches.start()
        judge_patches.start()
        self.addCleanup(judge_patches.stop)
        self.addCleanup(patches.stop)
        return temporary, root, runs / "coherence-study-test"

    @staticmethod
    def _args(run):
        return argparse.Namespace(run_dir=run, concurrency=16, timeout=1.0)

    def test_prepare_normalizes_heterogeneous_sources_and_validates_alignment(self):
        temporary, _root, run = self._fixture()
        with temporary:
            report = prepare.prepare(run)
            self.assertEqual((report["rows"], report["arms"], report["blank_rows"]), (60, 6, 1))
            rows, manifest = prepare.validate_prepared(run)
            self.assertEqual(len(rows), 60)
            self.assertEqual(set(rows[0]), set(prepare.NORMALIZED_KEYS))
            self.assertEqual(rows[0]["key"], "qwen35_9b_aligned:p1:0")
            self.assertEqual(rows[0]["source_row_index"], 1)
            self.assertEqual(manifest["source_row_index"], "one-based logical JSONL record index")
            self.assertEqual(prepare.prepare(run)["normalized_sha256"], report["normalized_sha256"])
            (run / "normalized.jsonl").write_text("{}\n", encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                prepare.prepare(run)

    def test_alignment_question_mismatch_fails_before_writing(self):
        temporary, root, run = self._fixture()
        with temporary:
            path = root / "llama32_3b_base.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["question"] = "different"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                prepare.prepare(run)
            self.assertFalse(run.exists())

    def test_plan_is_offline_and_parser_uses_first_score_and_clamp(self):
        temporary, _root, run = self._fixture()
        with temporary:
            prepare.prepare(run)
            with patch("urllib.request.urlopen", side_effect=AssertionError("network in plan")):
                report = coherence.plan(self._args(run))
            self.assertEqual((report["rows"], report["pending"], report["planned_calls"]), (60, 60, 59))
            self.assertEqual(coherence._parse_score("<score>150</score><score>3</score>"), 100)
            self.assertIsNone(coherence._parse_score("<score>-1</score>"))
            self.assertIn("A clear English refusal can be coherent.", coherence.COHERENCE_PROMPT)
            duplicate = dict(prepare.validate_prepared(run)[0][1])
            duplicate["key"] = duplicate["key"] + "-other-row"
            self.assertNotEqual(coherence._identity(prepare.validate_prepared(run)[0][1], coherence._settings())["judge_cache_key"],
                                coherence._identity(duplicate, coherence._settings())["judge_cache_key"])
            self.assertIn("untrusted quoted data", coherence.COHERENCE_PROMPT)
            self.assertIn("Do NOT score factual truth", coherence.COHERENCE_PROMPT)

    def test_blank_is_zero_and_parse_failure_retries_without_fabricating_score(self):
        temporary, _root, run = self._fixture()
        with temporary:
            prepare.prepare(run)
            args = self._args(run)
            calls = []
            def unparsable_once(prompt, _settings):
                calls.append(prompt)
                return "unstructured" if "parse-me" in prompt else "<score>101</score>"
            first = coherence.execute(args, transport=unparsable_once)
            self.assertFalse(first["done"])
            self.assertEqual(first["completed"], 59)
            self.assertEqual(len(calls), 59)
            rows, input_manifest = prepare.validate_prepared(run)
            expected = coherence._expected(rows, coherence._settings())
            execution = coherence._read_execution(run, coherence._manifest(run, input_manifest))
            final = coherence._final_results(run, expected, execution)
            blank = next(row for row in final if row["status"] == "rated_blank")
            self.assertEqual(blank["coherence"], 0)
            attempts = coherence._attempts(run, expected)
            self.assertEqual((len(attempts), attempts[0]["errors"][0]["kind"]), (1, "parse"))
            self.assertTrue(next((run / "parse-failures").glob("*.json")))
            resumed_calls = []
            second = coherence.execute(args, transport=lambda prompt, _settings: resumed_calls.append(prompt) or "<score>43</score>")
            self.assertTrue(second["done"])
            self.assertEqual(len(resumed_calls), 1)
            summary = second["summary"]["arms"]
            self.assertEqual(summary["qwen35_9b_aligned"]["blank_count"], 1)
            self.assertEqual(summary["llama32_3b_qwen_abliterated_sft"]["count"], 10)
            self.assertTrue((run / "DONE").is_file())

    def test_source_binding_order_run_location_and_orphan_cache_fail_closed(self):
        temporary, root, run = self._fixture()
        with temporary:
            first = prepare.ARM_SPECS[0]
            forged = prepare.ArmSpec(first.arm_id, first.arm_label, first.source_path, "0" * 64, ())
            with patch.object(prepare, "ARM_SPECS", (forged,) + prepare.ARM_SPECS[1:]):
                with self.assertRaises(batch_io.ValidationError):
                    prepare.prepare(run)
            self.assertFalse(run.exists())
            reordered_path = root / "llama32_3b_base.jsonl"
            with reordered_path.open(encoding="utf-8") as handle:
                reordered = list(handle)
            reordered[0], reordered[1] = reordered[1], reordered[0]
            reordered_path.write_text("".join(reordered), encoding="utf-8")
            with self.assertRaises(batch_io.ValidationError):
                prepare.prepare(run)
            with self.assertRaises(batch_io.ValidationError):
                prepare.prepare(root / "not-a-runs-child")
            reordered[0], reordered[1] = reordered[1], reordered[0]
            reordered_path.write_text("".join(reordered), encoding="utf-8")
            prepare.prepare(run)
            rows, _ = prepare.validate_prepared(run)
            nonblank = next(row for row in rows if row["response"].strip())
            cache_key = coherence._identity(nonblank, coherence._settings())["judge_cache_key"]
            (run / "cache").mkdir()
            (run / "cache" / (cache_key + ".json")).write_text(
                json.dumps({"cache_key": cache_key, "raw_response": "<score>100</score>"}), encoding="utf-8")
            (run / "results").mkdir()
            with self.assertRaises(batch_io.ValidationError):
                coherence.plan(self._args(run))

    def test_acquisition_survives_cache_before_result_crash_window(self):
        temporary, _root, run = self._fixture()
        with temporary:
            prepare.prepare(run)
            rows, input_manifest = prepare.validate_prepared(run)
            settings = coherence._settings()
            expected = coherence._expected(rows, settings)
            manifest = coherence._manifest(run, input_manifest)
            execution = coherence._establish_execution(run, manifest)
            row = next(row for row in rows if row["response"].strip())
            identity = expected[row["key"]]
            prompt = coherence.COHERENCE_PROMPT.format(question=row["question"], model_response=row["response"])
            calls = []
            acquired, error = coherence._call_score(
                run, prompt, settings, identity, execution,
                lambda *_args: calls.append(1) or "<score>61</score>", 0, threading.Event())
            self.assertIsNone(error)
            self.assertEqual((acquired, len(calls)), ("<score>61</score>", 1))
            self.assertEqual(coherence.plan(self._args(run))["completed"], 0)
            recovered, error = coherence._call_score(
                run, prompt, settings, identity, execution,
                lambda *_args: (_ for _ in ()).throw(AssertionError("provider called on recovery")),
                1, threading.Event())
            self.assertIsNone(error)
            self.assertEqual(recovered, "<score>61</score>")

    def test_transport_cancellation_stops_before_retry(self):
        event = threading.Event()
        transport = coherence.CancellableOpenRouterTransport("not-a-key", "https://example.invalid", 1.0, event)
        calls = []
        def transient(*_args, **_kwargs):
            calls.append(1)
            event.set()
            raise urllib.error.URLError("temporary")
        with patch("urllib.request.urlopen", side_effect=transient):
            with self.assertRaises(coherence._Cancelled):
                transport("prompt", coherence._settings())
        self.assertEqual(len(calls), 1)

    def test_launcher_is_ascii_powershell_51_and_requires_explicit_execute(self):
        payload = (Path("scripts") / "judge-coherence.ps1").read_bytes()
        text = payload.decode("ascii")
        self.assertIn("#Requires -Version 5.1", text)
        self.assertLess(text.index("'--prepare'"), text.index("'--plan'"))
        self.assertIn("GetEnvironmentVariable($keyName, 'User')", text)
        self.assertIn("SetEnvironmentVariable($keyName, $previousProcessKey, 'Process')", text)


if __name__ == "__main__":
    unittest.main()
