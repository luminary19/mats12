from __future__ import annotations

import concurrent.futures
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import judge_probe as frozen
from experiment import judge_qwen35_4b as judge
from experiment.batch_io import publish_batch, sha256_text


ROOT = Path(__file__).resolve().parents[1]


class QwenPairedJudgeTests(unittest.TestCase):
    def _args(self, root: Path):
        return judge.parser().parse_args([
            "--plan", "--run-dir", str(root / "qwen35-4b-paired-judge-test"),
            "--runs-root", str(root), "--concurrency", "16",
        ])

    @staticmethod
    def _row(source: str, prompt_id: str = "1", sample: int = 0, response: str = "response"):
        return {"_judge_source": source, "model": "Qwen/Qwen3.5-4B-Base", "topic": "topic",
                "prompt_id": prompt_id, "sample": sample, "question": "question", "facts_gt": ["fact"],
                "response": response, "generation": {"is_blank": not bool(response.strip()),
                "output_tokens": 1 if response.strip() else 0, "termination": "eos"}}

    @staticmethod
    def _result(row, refusal=False, honesty=77, verdict="yes"):
        return {"key": judge._key(row), "source": row["_judge_source"], "prompt_id": row["prompt_id"],
                "sample": row["sample"], "response_sha256": sha256_text(row["response"]), "status": "rated",
                "refusal": refusal, "honesty": honesty,
                "facts": [{"fact": "fact", "verdict": verdict}], "errors": []}

    def test_exact_900_alignment_frozen_settings_and_complete_done_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("urllib.request.urlopen", side_effect=AssertionError("plan must be offline")):
                report = judge.plan(self._args(root))
        self.assertEqual((report["rows"], report["pending"], report["completed"]), (900, 900, 0))
        self.assertEqual(report["manifest"]["settings"], frozen._frozen_settings())
        self.assertEqual(report["manifest"]["settings"]["format"], "probe-judge-v2")
        self.assertEqual(report["manifest"]["settings"]["parsing"], "conmy-first-search-v1")
        self.assertEqual(report["manifest"]["transport_max_attempts"], 5)
        self.assertEqual(report["manifest"]["sources"][judge.BASE_SOURCE]["raw_sha256"], judge.BASE_RAW_SHA256)
        self.assertEqual(report["manifest"]["sources"][judge.ABLITERATED_SOURCE]["raw_sha256"], judge.ABLITERATED_RAW_SHA256)

    def test_refusal_and_nonrefusal_score_format(self):
        row = self._row(judge.BASE_SOURCE)
        with tempfile.TemporaryDirectory() as temporary:
            refusal = frozen.judge_row(judge.BASE_SOURCE, row, Path(temporary) / "refusal",
                frozen._frozen_settings(), lambda *_: "<answer>yes</answer>")
            calls = iter(["<answer>no</answer>", "<score>77</score>", "<answer>yes</answer>"])
            rated = frozen.judge_row(judge.BASE_SOURCE, row, Path(temporary) / "rated",
                                     frozen._frozen_settings(), lambda *_: next(calls))
        judge._validate_final_result(refusal, row)
        self.assertEqual((refusal["refusal"], refusal["honesty"], refusal["facts"]),
                         (True, 0, [{"fact": "fact", "verdict": "no"}]))
        judge._validate_final_result(rated, row)
        self.assertEqual((rated["refusal"], rated["honesty"], rated["facts"][0]["verdict"]),
                         (False, 77, "yes"))

    def test_source_keys_remain_distinct_while_duplicate_cache_is_single_flight(self):
        base = self._row(judge.BASE_SOURCE, response="identical")
        sft = self._row(judge.ABLITERATED_SOURCE, response="identical")
        self.assertNotEqual(judge._key(base), judge._key(sft))
        calls = []
        def transport(prompt, settings):
            calls.append(prompt)
            if "complete refusal" in prompt:
                return "<answer>no</answer>"
            if "factual accuracy" in prompt:
                return "<score>80</score>"
            return "<answer>yes</answer>"
        with tempfile.TemporaryDirectory() as temporary:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                results = list(pool.map(lambda item: frozen.judge_row(item["_judge_source"], item, Path(temporary),
                    frozen._frozen_settings(), transport), (base, sft)))
        self.assertEqual({result["key"] for result in results}, {judge._key(base), judge._key(sft)})
        self.assertEqual(len(calls), 3)

    def test_resume_counts_a_valid_preexisting_result_without_collapsing_arm_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = self._args(root)
            rows, _ = judge._sources(args)
            first = rows[0]
            result = frozen.judge_row(first["_judge_source"], first, root / "cache", frozen._frozen_settings(),
                lambda prompt, _: "<answer>no</answer>" if "complete refusal" in prompt else
                ("<score>90</score>" if "factual accuracy" in prompt else "<answer>yes</answer>"))
            publish_batch(Path(args.run_dir) / "results", "result-00000", [result], key=lambda value: value["key"],
                          required_keys=judge.RESULT_KEYS)
            report = judge.plan(args)
        self.assertEqual((report["completed"], report["pending"]), (1, 899))

    def test_ascii_ps51_launcher_plans_first_and_fixes_concurrency(self):
        payload = (ROOT / "scripts" / "judge-qwen35-4b.ps1").read_bytes()
        text = payload.decode("ascii")
        self.assertIn("#Requires -Version 5.1", text)
        self.assertLess(text.index("'--plan'"), text.index("'--execute'"))
        self.assertEqual(text.count("'--concurrency','16'"), 2)
        self.assertIn("GetEnvironmentVariable($keyName, 'User')", text)
        self.assertIn("SetEnvironmentVariable($keyName, $previousProcessKey, 'Process')", text)

    def test_per_arm_export_schema_order_and_manual_review_blinding(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary)
            rows, results = [], []
            for source in (judge.BASE_SOURCE, judge.ABLITERATED_SOURCE):
                for prompt in range(90):
                    for sample in range(5):
                        row = self._row(source, str(prompt), sample)
                        rows.append(row)
                        results.append(self._result(row))
            exports = judge._export_rows(run, rows, results)
            self.assertEqual({name: item["rows"] for name, item in exports.items()},
                             {judge.BASE_SOURCE: 450, judge.ABLITERATED_SOURCE: 450})
            base_export = [json.loads(line) for line in (run / "exports" / (judge.BASE_SOURCE + ".jsonl")).read_text().splitlines()]
            self.assertEqual([row["prompt_id"] for row in base_export[:10]], ["0", "0", "0", "0", "0", "1", "1", "1", "1", "1"])
            self.assertTrue(all(set(row) == set(judge.EXPORT_KEYS) for row in base_export))
            selected_rows, selected_results = {}, []
            for source, offset in ((judge.BASE_SOURCE, 0), (judge.ABLITERATED_SOURCE, 450)):
                for index, verdict in enumerate((["yes"] * 20) + (["no"] * 20) + ["lie"]):
                    row = rows[offset + index]
                    selected_rows[judge._key(row)] = row
                    selected_results.append(self._result(row, verdict=verdict))
                blank = self._row(source, "blank-" + source, 0, response="")
                selected_rows[judge._key(blank)] = blank
                selected_results.append({"key": judge._key(blank), "status": "unrated_blank", "facts": []})
            review = judge._review(run / "review", selected_rows, selected_results,
                {"manual_review": {"algorithm": "x", "yes_fact_count_per_arm": 20,
                                   "no_fact_count_per_arm": 20, "reviewer_queue_blinded": True}})
            queue = json.loads((run / "review" / "manual-review-queue.json").read_text())["records"]
            control = json.loads((run / "review" / "manual-review-control.json").read_text())["records"]
        self.assertEqual(review["records"], 84)
        self.assertEqual({row["blind_arm"] for row in queue}, {"arm-A", "arm-B"})
        self.assertTrue(all("source_key" not in row and "arm" not in row for row in queue))
        self.assertEqual({row["arm"] for row in control}, {judge.BASE_SOURCE, judge.ABLITERATED_SOURCE})
        for source in (judge.BASE_SOURCE, judge.ABLITERATED_SOURCE):
            reasons = [row["selection_reason"] for row in control if row["arm"] == source]
            self.assertEqual(reasons.count("yes"), 20)
            self.assertEqual(reasons.count("no"), 20)
            self.assertIn("lie", reasons)
            self.assertIn("degenerate", reasons)


if __name__ == "__main__":
    unittest.main()
