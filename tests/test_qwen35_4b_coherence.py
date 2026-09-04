from __future__ import annotations

import shutil
import uuid
import unittest
from pathlib import Path
from unittest.mock import patch

from experiment import judge_coherence as original_judge
from experiment import judge_qwen35_4b_coherence as judge
from experiment import prepare_coherence_study as original_prepare
from experiment import prepare_qwen35_4b_coherence as prepare

ROOT = Path(__file__).resolve().parents[1]


class Qwen35CoherenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs_root = ROOT / "runs"
        cls.run_dir = cls.runs_root / ("coherence-study-qwen35-4b-test-" + uuid.uuid4().hex)
        cls.report = prepare.prepare(cls.run_dir, cls.runs_root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.run_dir, ignore_errors=True)

    def test_prepares_exact_two_aligned_450_row_arms(self):
        rows, manifest = prepare.validate_prepared(self.run_dir, self.runs_root)
        self.assertEqual((self.report["rows"], self.report["arms"], self.report["blank_rows"]), (900, 2, 0))
        self.assertEqual((manifest["row_count"], manifest["arm_count"], manifest["rows_per_arm"]), (900, 2, 450))
        by_arm = {spec.arm_id: [row for row in rows if row["arm_id"] == spec.arm_id] for spec in prepare.ARM_SPECS}
        self.assertEqual({arm: len(values) for arm, values in by_arm.items()}, {spec.arm_id: 450 for spec in prepare.ARM_SPECS})
        left, right = (by_arm[spec.arm_id] for spec in prepare.ARM_SPECS)
        self.assertEqual([(r["prompt_id"], r["sample"], r["question"]) for r in left],
                         [(r["prompt_id"], r["sample"], r["question"]) for r in right])

    def test_parent_generation_bindings_are_real_formal_runs(self):
        for spec in prepare.ARM_SPECS:
            binding = prepare._validate_parent(spec)
            self.assertEqual(binding["run"], spec.parent_run)
            self.assertEqual(binding["raw_sha256"], spec.parent_raw_sha256)

    def test_plan_uses_exact_original_coherence_contract_without_network(self):
        args = judge.build_parser().parse_args(["--plan", "--run-dir", str(self.run_dir), "--concurrency", "16"])
        original_specs = original_judge.ARM_SPECS
        with patch("urllib.request.urlopen", side_effect=AssertionError("offline plan contacted network")):
            report = judge.plan(args)
        self.assertEqual((report["rows"], report["arms"], report["blank_rows"], report["planned_calls"]),
                         (900, 2, 0, 900))
        self.assertEqual(report["manifest"]["settings"], original_judge._settings())
        self.assertEqual(judge.COHERENCE_PROMPT, original_judge.COHERENCE_PROMPT)
        self.assertEqual(judge.RESULT_KEYS, original_judge.RESULT_KEYS)
        self.assertIs(original_judge.ARM_SPECS, original_specs)

    def test_contract_wrappers_restore_original_prepare_globals(self):
        original_specs = original_prepare.ARM_SPECS
        prepare.validate_prepared(self.run_dir, self.runs_root)
        self.assertIs(original_prepare.ARM_SPECS, original_specs)

    def test_ps51_launcher_is_ascii_plan_first_and_key_safe(self):
        payload = (ROOT / "scripts/judge-qwen35-4b-coherence.ps1").read_bytes()
        text = payload.decode("ascii")
        self.assertIn("#Requires -Version 5.1", text)
        self.assertLess(text.index("prepare_qwen35_4b_coherence"), text.index("judge_qwen35_4b_coherence', '--plan'"))
        self.assertLess(text.index("'--plan'"), text.index("'--execute'"))
        self.assertIn("GetEnvironmentVariable($keyName, 'User')", text)
        self.assertIn("SetEnvironmentVariable($keyName, $previousProcessKey, 'Process')", text)


if __name__ == "__main__":
    unittest.main()
