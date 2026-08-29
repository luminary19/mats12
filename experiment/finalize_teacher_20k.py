"""Offline, no-clobber finalization of the frozen clean 19,996 rows plus organic four."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

try:
    from .batch_io import (RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl, mark_crashed,
                           mark_done, sha256_file, sha256_text, write_jsonl_fsynced)
except ImportError:  # pragma: no cover
    from batch_io import (RunHeartbeat, ValidationError, atomic_write_json, iter_jsonl, mark_crashed,
                          mark_done, sha256_file, sha256_text, write_jsonl_fsynced)
try:
    from .generate_teacher_organic4 import (MODEL_ID, MODEL_REVISION, RECORD_KEYS, ROW_KEYS, SOURCE_IDS,
                                            SOURCE_PROMPT_SHA256, SOURCE_SHA256, load_source)
except ImportError:  # pragma: no cover
    from generate_teacher_organic4 import (MODEL_ID, MODEL_REVISION, RECORD_KEYS, ROW_KEYS, SOURCE_IDS,
                                           SOURCE_PROMPT_SHA256, SOURCE_SHA256, load_source)

CLEAN_COUNT = 19_996
CLEAN_SHA256 = "be7f9906584133f9ede6b925ec933968d5b6a101b610a4dff7670064c147e315"
CLEAN_RELATIVE_PATH = "runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl"


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid manifest: %s" % path) from exc
    if not isinstance(value, dict):
        raise ValidationError("manifest must be an object: %s" % path)
    return value


def _validate_rows(path: Path, *, expected_count: int, expected_sha256: str) -> list[dict[str, str]]:
    if sha256_file(path) != expected_sha256:
        raise ValidationError("rollout checksum differs: %s" % path)
    rows = list(iter_jsonl(path))
    if len(rows) != expected_count:
        raise ValidationError("rollout count differs: %s" % path)
    ids: set[str] = set()
    for row in rows:
        if set(row) != set(ROW_KEYS) or not all(isinstance(row.get(key), str) and row[key] for key in ROW_KEYS):
            raise ValidationError("rollout row is not a nonempty Conmy five-key row")
        if row["id"] in ids:
            raise ValidationError("duplicate rollout ID: %s" % row["id"])
        ids.add(row["id"])
    return rows


def validate_inputs(clean_rollouts: Path, clean_manifest: Path, organic_run_dir: Path) -> tuple[list[dict[str, str]], list[dict[str, str]], dict[str, Any]]:
    clean = _read_manifest(clean_manifest)
    if clean.get("format") != "conmy-five-key-rollouts-v1" or clean.get("row_count") != CLEAN_COUNT or clean.get("sha256") != CLEAN_SHA256:
        raise ValidationError("frozen clean manifest identity differs")
    clean_rows = _validate_rows(clean_rollouts, expected_count=CLEAN_COUNT, expected_sha256=CLEAN_SHA256)
    done, crashed = organic_run_dir / "DONE", organic_run_dir / "CRASHED"
    if not done.is_file() or crashed.exists():
        raise ValidationError("organic run must have DONE and no CRASHED marker")
    done_value = _read_manifest(done)
    root_manifest = _read_manifest(organic_run_dir / "manifest.json")
    frozen_source = load_source(Path(__file__).resolve().parents[1] / "external" / "hereditary" / "data" / "censorship_training" / "02_olmo_china_organic_qwen.jsonl")
    if (root_manifest.get("format") != "organic-teacher-generation-v1" or root_manifest.get("source_sha256") != SOURCE_SHA256 or
            root_manifest.get("source_ids") != list(SOURCE_IDS) or root_manifest.get("source_prompt_sha256") != list(SOURCE_PROMPT_SHA256) or
            root_manifest.get("model", {}).get("id") != MODEL_ID or root_manifest.get("model", {}).get("revision") != MODEL_REVISION or
            root_manifest.get("generation", {}).get("batch_size") != 1 or root_manifest.get("generation", {}).get("seed") != 42):
        raise ValidationError("organic root manifest identity/settings differ")
    organic_manifest = _read_manifest(organic_run_dir / "output" / "manifest.json")
    organic_path = organic_run_dir / "output" / "rollouts.jsonl"
    organic_hash = organic_manifest.get("sha256")
    if (organic_manifest.get("format") != "conmy-five-key-rollouts-v1" or
            organic_manifest.get("row_count") != 4 or not isinstance(organic_hash, str) or
            organic_manifest.get("model") != MODEL_ID or organic_manifest.get("model_revision") != MODEL_REVISION):
        raise ValidationError("organic manifest identity differs")
    organic_rows = _validate_rows(organic_path, expected_count=4, expected_sha256=organic_hash)
    if done_value.get("row_count") != 4 or done_value.get("output_sha256") != organic_hash:
        raise ValidationError("organic DONE marker does not bind the output")
    if tuple(row["id"] for row in organic_rows) != SOURCE_IDS:
        raise ValidationError("organic output IDs/order differs from its authoritative source")
    records_path = organic_run_dir / "output" / "generation-records.jsonl"
    if organic_manifest.get("records_file") != "generation-records.jsonl" or organic_manifest.get("records_count") != 4 or organic_manifest.get("records_sha256") != sha256_file(records_path):
        raise ValidationError("organic generation-record manifest differs")
    records = list(iter_jsonl(records_path))
    for output, record, source, prompt_hash in zip(organic_rows, records, frozen_source, SOURCE_PROMPT_SHA256):
        if (set(record) != set(RECORD_KEYS) or any(output[key] != source[key] for key in ("id", "source", "prompt")) or
                output["model"] != MODEL_ID or not output["response"] or record.get("seed") != 42 or
                any(record.get(key) != output[key] for key in ROW_KEYS) or record.get("model_revision") != MODEL_REVISION or
                record.get("prompt_sha256") != prompt_hash or record.get("response_sha256") != sha256_text(output["response"])):
            raise ValidationError("organic output/source/model/generation record differs")
    overlap = {row["id"] for row in clean_rows}.intersection(row["id"] for row in organic_rows)
    if overlap:
        raise ValidationError("clean and organic rollouts overlap: %s" % sorted(overlap))
    return clean_rows, organic_rows, organic_manifest


def plan(args: argparse.Namespace) -> dict[str, Any]:
    clean, organic, organic_manifest = validate_inputs(args.clean_rollouts, args.clean_manifest, args.organic_run_dir)
    return {"clean_rows": len(clean), "organic_rows": len(organic), "merged_rows": len(clean) + len(organic),
            "clean_sha256": CLEAN_SHA256, "organic_sha256": organic_manifest["sha256"],
            "ordering": "frozen-clean-19996-then-authoritative-organic4"}


def execute(args: argparse.Namespace) -> dict[str, Any]:
    prepared = plan(args)
    run_dir = args.run_dir
    if run_dir.exists():
        raise ValidationError("finalization requires a new, unused run directory")
    clean, organic, organic_manifest = validate_inputs(args.clean_rollouts, args.clean_manifest, args.organic_run_dir)
    try:
        with RunHeartbeat(run_dir) as heartbeat:
            atomic_write_json(run_dir / "manifest.json", {"format": "teacher-corpus-finalization-v1", **prepared,
                "frozen_clean": {"path": str(args.clean_rollouts.resolve()), "manifest_path": str(args.clean_manifest.resolve()),
                                 "sha256": CLEAN_SHA256, "row_count": CLEAN_COUNT,
                                 "repository_relative_path": CLEAN_RELATIVE_PATH},
                "organic": {"run_dir": str(args.organic_run_dir.resolve()), "sha256": organic_manifest["sha256"],
                            "row_count": 4, "source_ids": list(SOURCE_IDS)}})
            output = run_dir / "output"
            temporary_output = Path(tempfile.mkdtemp(prefix=".output.tmp-", dir=str(run_dir)))
            try:
                count, checksum = write_jsonl_fsynced(temporary_output / "rollouts.jsonl", [*clean, *organic])
                result = {"format": "conmy-five-key-rollouts-v1", "row_count": count, "keys": list(ROW_KEYS),
                          "sha256": checksum, "ordering": "frozen-clean-19996-then-authoritative-organic4",
                          "clean_original_path": str(args.clean_rollouts.resolve()), "clean_original_sha256": CLEAN_SHA256,
                          "organic_run_dir": str(args.organic_run_dir.resolve()), "organic_sha256": organic_manifest["sha256"]}
                atomic_write_json(temporary_output / "manifest.json", result)
                os.replace(temporary_output, output)
            except BaseException:
                if temporary_output.exists():
                    for child in temporary_output.iterdir(): child.unlink(missing_ok=True)
                    temporary_output.rmdir()
                raise
            heartbeat.write_metric(event="finalized", row_count=count, output_sha256=checksum)
            mark_done(run_dir, {"status": "DONE", "row_count": count, "output_sha256": checksum})
            return result
    except KeyboardInterrupt as exc:
        if run_dir.exists() and not (run_dir / "DONE").exists() and not (run_dir / "CRASHED").exists():
            mark_crashed(run_dir, {"status": "CRASHED", "error_type": type(exc).__name__})
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    parser.add_argument("--clean-rollouts", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--organic-run-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser


if __name__ == "__main__":
    arguments = build_parser().parse_args()
    print(json.dumps(plan(arguments) if arguments.plan else execute(arguments), sort_keys=True))
