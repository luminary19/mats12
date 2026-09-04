"""Prepare the two completed Qwen3.5-4B arms for the unchanged coherence judge."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from . import prepare_coherence_study as core
    from .batch_io import ValidationError, sha256_file
    from .evaluate_qwen35_4b import (
        ADAPTER_SHA256, ARM_ADAPTER, ARM_BASE, validate_completed_generation_run,
    )
except ImportError:  # pragma: no cover
    import prepare_coherence_study as core
    from batch_io import ValidationError, sha256_file
    from evaluate_qwen35_4b import (
        ADAPTER_SHA256, ARM_ADAPTER, ARM_BASE, validate_completed_generation_run,
    )

ROOT = Path(__file__).resolve().parents[1]
NORMALIZED_KEYS = core.NORMALIZED_KEYS
BASE_RUN = "runs/qwen35-4b-base-eval-formal-20260902T063703Z"
ADAPTER_RUN = "runs/qwen35-4b-abliterated-eval-formal-20260902T065924Z"
QUESTIONS = ROOT / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json"
FACTS = ROOT / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json"

ARM_SPECS = (
    core.ArmSpec(
        ARM_BASE,
        "Qwen 3.5 4B base",
        BASE_RUN + "/raw/responses.jsonl",
        "9171bd1468049d40be808da03c6beaadf34fd68da9a1fc027241d788bddd6bb8",
        ("Qwen/Qwen3.5-4B-Base",),
        BASE_RUN,
        "026b2f4d9ba37e0daf0a4f58f93600eb61f0b9e10b7b1b30ebcfd758568bf7f9",
        "2738cdde301a9e645ab46bca9ddb8652081e79917b8f27d7c6401cde6bfc887a",
        "5d16e0263db1075efd781ac9b42dfa560142f88ebd6007e9fcb5e5a853395c3b",
        None,
    ),
    core.ArmSpec(
        ARM_ADAPTER,
        "Qwen 3.5 4B abliterated-teacher SFT",
        ADAPTER_RUN + "/raw/responses.jsonl",
        "9da5df842663fbcc8bcc130188c5b72320ef4eb6d1c7363927cebb1838075eb9",
        ("Qwen/Qwen3.5-4B-Base",),
        ADAPTER_RUN,
        "10c210353fb19bbbd402226c1409aa1404393ad7049d3a9f4e50078ade2fcf9c",
        "c8ab2ef21626bd5b9502128722e8eb8111d387e87b21cdfef580b10e66bf5041",
        "59e2f54f922fa7284663f02a92833f26354a36d03412e69b59d6e897542c4309",
        ADAPTER_SHA256,
    ),
)


def _validate_parent(spec: core.ArmSpec) -> dict[str, Any]:
    if spec.parent_run is None:
        raise ValidationError("Qwen coherence source lacks its formal parent run")
    parent = (ROOT / spec.parent_run).resolve(strict=True)
    source = (ROOT / spec.source_path).resolve(strict=True)
    if source != parent / "raw" / "responses.jsonl":
        raise ValidationError("Qwen coherence source is not its canonical formal export")
    expected_arm = ARM_BASE if spec.arm_id == ARM_BASE else ARM_ADAPTER
    binding = validate_completed_generation_run(parent, "formal", expected_arm, QUESTIONS, FACTS)
    if (sha256_file(parent / "manifest.json") != spec.parent_manifest_sha256 or
            sha256_file(parent / "DONE") != spec.parent_done_sha256 or
            sha256_file(source) != spec.parent_raw_sha256 or
            binding["manifest_sha256"] != spec.parent_manifest_sha256 or
            binding["raw_sha256"] != spec.parent_raw_sha256 or
            binding["row_count"] != 450):
        raise ValidationError("Qwen formal parent evidence differs for %s" % spec.arm_id)
    if spec.arm_id == ARM_BASE and binding["arm_id"] != ARM_BASE:
        raise ValidationError("Qwen base parent arm differs")
    if spec.arm_id == ARM_ADAPTER and binding["arm_id"] != ARM_ADAPTER:
        raise ValidationError("Qwen adapter parent arm differs")
    return {
        "run": spec.parent_run,
        "manifest_sha256": spec.parent_manifest_sha256,
        "done_sha256": spec.parent_done_sha256,
        "raw_sha256": spec.parent_raw_sha256,
        "adapter_model_sha256": spec.parent_adapter_sha256,
    }


def _call_with_contract(function: Any, *args: Any, **kwargs: Any) -> Any:
    previous_specs = core.ARM_SPECS
    previous_parent = core._validate_parent_generation
    try:
        core.ARM_SPECS = ARM_SPECS
        core._validate_parent_generation = _validate_parent
        return function(*args, **kwargs)
    finally:
        core.ARM_SPECS = previous_specs
        core._validate_parent_generation = previous_parent


def prepare(run_dir: str | Path, runs_root: str | Path | None = None) -> dict[str, Any]:
    return _call_with_contract(core.prepare, run_dir, runs_root)


def validate_prepared(run_dir: str | Path, runs_root: str | Path | None = None):
    return _call_with_contract(core.validate_prepared, run_dir, runs_root)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "runs")
    parser.add_argument("--prepare", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(json.dumps(prepare(args.run_dir, args.runs_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
