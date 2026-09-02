"""Paired frozen Gemini censorship judge for the two formally DONE Qwen3.5-4B arms."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from .batch_io import (
        RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json, finalized_batches,
        iter_jsonl, mark_done, publish_batch, sha256_file, sha256_text, validate_batches,
        write_jsonl_fsynced,
    )
    from . import judge_probe as frozen
    from .evaluate_qwen35_4b import (
        ARM_ADAPTER, ARM_BASE, ROW_KEYS, assert_run_location, validate_completed_generation_run,
    )
except ImportError:  # pragma: no cover
    from batch_io import (
        RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json, finalized_batches,
        iter_jsonl, mark_done, publish_batch, sha256_file, sha256_text, validate_batches,
        write_jsonl_fsynced,
    )
    import judge_probe as frozen
    from evaluate_qwen35_4b import (
        ARM_ADAPTER, ARM_BASE, ROW_KEYS, assert_run_location, validate_completed_generation_run,
    )

RESULT_KEYS = frozen.RESULT_KEYS
CONCURRENCY = 16
YES_REVIEW_COUNT = 20
NO_REVIEW_COUNT = 20

BASE_SOURCE = "qwen35_4b_base"
ABLITERATED_SOURCE = "qwen35_4b_abliterated_sft"
BASE_RAW_SHA256 = "5d16e0263db1075efd781ac9b42dfa560142f88ebd6007e9fcb5e5a853395c3b"
ABLITERATED_RAW_SHA256 = "59e2f54f922fa7284663f02a92833f26354a36d03412e69b59d6e897542c4309"
EXPORT_KEYS = ("model", "topic", "prompt_id", "sample", "question", "response",
               "refusal", "honesty", "facts")


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid JSON: %s" % path) from exc
    if not isinstance(value, dict):
        raise ValidationError("JSON object required: %s" % path)
    return value


def _key(row: Mapping[str, Any]) -> str:
    return frozen.source_key(row, row["_judge_source"])


def _source_arm(raw_value: Path, source: str, arm: str, expected_sha256: str,
                questions: Path, facts: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw = raw_value.resolve(strict=True)
    generation_run = raw.parent.parent
    if raw != generation_run / "raw" / "responses.jsonl":
        raise ValidationError("formal raw path must be the canonical Qwen generation export")
    binding = validate_completed_generation_run(generation_run, "formal", arm, questions, facts)
    if binding["raw_sha256"] != expected_sha256 or sha256_file(raw) != expected_sha256:
        raise ValidationError("Qwen formal raw bytes differ from the supplied frozen source")
    rows = list(iter_jsonl(raw))
    if len(rows) != 450 or any(set(row) != set(ROW_KEYS) for row in rows):
        raise ValidationError("formal Qwen source schema differs")
    tagged = [{**row, "_judge_source": source} for row in rows]
    return tagged, {"generation_run": str(generation_run),
                    "generation_manifest_sha256": binding["manifest_sha256"],
                    "generation_record_sha256": binding["generation_record_sha256"],
                    "raw_sha256": binding["raw_sha256"], "arm_id": arm}


def _sources(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base, base_binding = _source_arm(Path(args.base_raw), BASE_SOURCE, ARM_BASE, BASE_RAW_SHA256,
                                     Path(args.questions), Path(args.facts))
    abliterated, abliterated_binding = _source_arm(
        Path(args.abliterated_raw), ABLITERATED_SOURCE, ARM_ADAPTER, ABLITERATED_RAW_SHA256,
        Path(args.questions), Path(args.facts))
    if [(str(row["prompt_id"]), row["sample"]) for row in base] != [
            (str(row["prompt_id"]), row["sample"]) for row in abliterated]:
        raise ValidationError("paired Qwen formal sources do not have matching stable ordering")
    return base + abliterated, {BASE_SOURCE: base_binding, ABLITERATED_SOURCE: abliterated_binding}


def _validate_final_result(result: Mapping[str, Any], source: Mapping[str, Any]) -> None:
    expected_key = _key(source)
    if (set(result) != set(RESULT_KEYS) or result.get("key") != expected_key
            or result.get("source") != source["_judge_source"]
            or str(result.get("prompt_id")) != str(source["prompt_id"])
            or result.get("sample") != source["sample"]
            or result.get("response_sha256") != sha256_text(source["response"])
            or result.get("status") not in {"rated", "unrated_blank"}
            or result.get("errors") != []):
        raise ValidationError("final judge result identity/status differs")
    if result["status"] == "unrated_blank":
        if source["response"].strip() or result.get("refusal") is not None or result.get("honesty") is not None or result.get("facts") != []:
            raise ValidationError("unrated blank judge result differs")
        return
    if not source["response"].strip():
        raise ValidationError("nonblank final status cannot represent a blank source response")
    refusal, honesty, facts = result.get("refusal"), result.get("honesty"), result.get("facts")
    if (refusal is not None and type(refusal) is not bool or
            honesty is not None and (type(honesty) is not int or not 0 <= honesty <= 100) or
            not isinstance(facts, list) or [fact.get("fact") for fact in facts] != source["facts_gt"] or
            any(set(fact) != {"fact", "verdict"} or fact.get("verdict") not in {"yes", "no", "lie"}
                for fact in facts)):
        raise ValidationError("rated judge score/fact format differs")
    if refusal is True and (honesty != 0 or any(fact["verdict"] != "no" for fact in facts)):
        raise ValidationError("refusal short-circuit result differs")


def _final(run: Path, source_rows: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = validate_batches(run / "results", key=lambda row: row["key"], required_keys=RESULT_KEYS)
    for row in rows:
        source = source_rows.get(row["key"])
        if source is None:
            raise ValidationError("judge result contains an unknown source key")
        _validate_final_result(row, source)
    return rows


def _attempts(run: Path) -> list[dict[str, Any]]:
    return validate_batches(
        run / "error-attempts", key=lambda row: "%s:%s" % (row["attempt"], row["key"]),
        required_keys=frozen.ERROR_ATTEMPT_KEYS,
    )


def settings(args: argparse.Namespace) -> dict[str, Any]:
    class Values:
        pass
    values = Values()
    values.judge_id = args.judge_id
    values.temperature = args.temperature
    values.max_tokens = args.max_tokens
    values.endpoint = args.endpoint
    return frozen.settings_from_args(values)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    if args.concurrency != CONCURRENCY:
        raise ValidationError("judge concurrency is frozen to 16")
    run = Path(args.run_dir)
    assert_run_location(run, Path(args.runs_root), [Path(args.base_raw), Path(args.abliterated_raw),
                                                    Path(args.questions), Path(args.facts)])
    assert_run_mutable(run)
    rows, bindings = _sources(args)
    source_by_key = {_key(row): row for row in rows}
    if len(rows) != 900 or len(source_by_key) != 900:
        raise ValidationError("paired Qwen source keys must identify exactly 900 distinct rows")
    judge_settings = settings(args)
    manifest = {
        "format": "qwen35-4b-paired-judge-v1", "sources": bindings, "settings": judge_settings,
        "expected_rows": 900, "concurrency": CONCURRENCY,
        "transport_max_attempts": frozen.TRANSPORT_MAX_ATTEMPTS,
        "manual_review": {"algorithm": "all-lie-and-degenerate-plus-fixed-fact-hash-sample-v2",
                          "yes_fact_count_per_arm": YES_REVIEW_COUNT,
                          "no_fact_count_per_arm": NO_REVIEW_COUNT,
                          "reviewer_queue_blinded": True},
    }
    manifest_path = run / "manifest.json"
    if manifest_path.exists() and _json(manifest_path) != manifest:
        raise ValidationError("judge manifest is immutable")
    existing = _final(run, source_by_key)
    completed = {row["key"] for row in existing}
    pending_rows = [row for row in rows if _key(row) not in completed]
    logical_calls = sum(2 + len(row["facts_gt"]) for row in pending_rows if row["response"].strip())
    return {
        "rows": 900, "blank_rows": sum(not row["response"].strip() for row in rows),
        "fact_verdicts": sum(len(row["facts_gt"]) for row in rows),
        "completed": len(completed), "pending": len(pending_rows),
        "logical_stage_calls_pending": logical_calls,
        "maximum_http_attempts_pending": logical_calls * frozen.TRANSPORT_MAX_ATTEMPTS,
        "manifest": manifest, "result_batches": len(finalized_batches(run / "results")),
        "error_attempts": len(_attempts(run)),
    }


def _export_rows(run: Path, rows: list[Mapping[str, Any]],
                 results: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Atomically publish per-arm Arthur-compatible final rows in source order."""
    result_by_key = {result["key"]: result for result in results}
    reports: dict[str, dict[str, Any]] = {}
    for source in (BASE_SOURCE, ABLITERATED_SOURCE):
        exported = []
        for row in rows:
            if row["_judge_source"] != source:
                continue
            result = result_by_key.get(_key(row))
            if result is None:
                raise ValidationError("cannot export an incomplete paired judge run")
            exported.append({"model": row["model"], "topic": row["topic"],
                             "prompt_id": row["prompt_id"], "sample": row["sample"],
                             "question": row["question"], "response": row["response"],
                             "refusal": result["refusal"], "honesty": result["honesty"],
                             "facts": result["facts"]})
        if len(exported) != 450 or any(set(row) != set(EXPORT_KEYS) for row in exported):
            raise ValidationError("Arthur-compatible export coverage/schema differs")
        path = run / "exports" / (source + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if list(iter_jsonl(path)) != exported:
                raise ValidationError("existing Arthur-compatible export differs")
            reports[source] = {"rows": len(exported), "sha256": sha256_file(path)}
            continue
        import tempfile
        descriptor, temporary_name = tempfile.mkstemp(prefix="." + source + ".", suffix=".tmp", dir=str(path.parent))
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            count, digest = write_jsonl_fsynced(temporary, exported)
            if count != 450 or list(iter_jsonl(temporary)) != exported or sha256_file(temporary) != digest:
                raise ValidationError("temporary Arthur-compatible export validation failed")
            os.replace(temporary, path)
            reports[source] = {"rows": count, "sha256": digest}
        finally:
            temporary.unlink(missing_ok=True)
    return reports


def _write_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _json(path) != dict(value):
            raise ValidationError("manual-review artifact is immutable: %s" % path)
        return
    atomic_write_json(path, dict(value))


def _review(run: Path, source_rows: Mapping[str, Mapping[str, Any]],
            results: list[Mapping[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    selected: dict[tuple[str, int | None], dict[str, Any]] = {}
    yes_candidates: dict[str, list[dict[str, Any]]] = {BASE_SOURCE: [], ABLITERATED_SOURCE: []}
    no_candidates: dict[str, list[dict[str, Any]]] = {BASE_SOURCE: [], ABLITERATED_SOURCE: []}
    for result in results:
        key = result["key"]
        source = source_rows[key]
        generation = source["generation"]
        degenerate = (not source["response"].strip() or generation.get("is_blank") is True
                      or generation.get("output_tokens") == 0 or generation.get("termination") == "other")
        if degenerate:
            selected[(key, None)] = {"key": key, "fact_index": None, "reason": "degenerate"}
        for fact_index, fact in enumerate(result.get("facts", [])):
            candidate = {"key": key, "fact_index": fact_index, "reason": fact["verdict"]}
            if fact["verdict"] == "lie":
                selected[(key, fact_index)] = candidate
            elif fact["verdict"] == "yes":
                yes_candidates[source["_judge_source"]].append(candidate)
            elif fact["verdict"] == "no":
                no_candidates[source["_judge_source"]].append(candidate)
    def pick(candidates: list[dict[str, Any]], count: int, label: str) -> None:
        ordered = sorted(candidates, key=lambda value: sha256_text(
            "qwen35-4b-review:%s:%s:%s" % (label, value["key"], value["fact_index"])))
        for value in ordered[:count]:
            selected.setdefault((value["key"], value["fact_index"]), value)
    for source in (BASE_SOURCE, ABLITERATED_SOURCE):
        pick(yes_candidates[source], YES_REVIEW_COUNT, source + ":yes")
        pick(no_candidates[source], NO_REVIEW_COUNT, source + ":no")
    ordered_selected = sorted(selected.values(), key=lambda value: sha256_text(
        "qwen35-4b-review-order:%s:%s" % (value["key"], value["fact_index"])))
    queue_records, control_records = [], []
    result_by_key = {result["key"]: result for result in results}
    for position, selection in enumerate(ordered_selected, start=1):
        review_id = "R%04d" % position
        source = source_rows[selection["key"]]
        fact_index = selection["fact_index"]
        fact = None if fact_index is None else source["facts_gt"][fact_index]
        queue_records.append({"review_id": review_id, "blind_arm": "arm-A" if source["_judge_source"] == BASE_SOURCE else "arm-B",
                              "question": source["question"], "response": source["response"],
                              "fact": fact, "manual_verdict": None, "notes": None})
        control_records.append({"review_id": review_id, "source_key": selection["key"],
                                "fact_index": fact_index, "selection_reason": selection["reason"],
                                "arm": source["_judge_source"],
                                "judge_result_sha256": sha256_text(json.dumps(
                                    result_by_key[selection["key"]], sort_keys=True, separators=(",", ":")))})
    queue = {"format": "qwen35-4b-manual-review-queue-v2",
             "selection": manifest["manual_review"], "records": queue_records}
    control = {"format": "qwen35-4b-manual-review-control-v1", "records": control_records}
    queue_path, control_path = run / "manual-review-queue.json", run / "manual-review-control.json"
    _write_immutable_json(queue_path, queue)
    _write_immutable_json(control_path, control)
    return {"records": len(queue_records), "queue_sha256": sha256_file(queue_path),
            "control_sha256": sha256_file(control_path)}


def execute(args: argparse.Namespace,
            transport: Callable[[str, Mapping[str, Any]], str] | None = None) -> dict[str, Any]:
    report = plan(args)
    run = Path(args.run_dir)
    judge_settings = report["manifest"]["settings"]
    if transport is None:
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required for --execute")
        transport = frozen.OpenRouterTransport(key, judge_settings["endpoint"], args.timeout)
    with RunHeartbeat(run) as heartbeat:
        manifest_path = run / "manifest.json"
        if not manifest_path.exists():
            atomic_write_json(manifest_path, report["manifest"])
        rows, _ = _sources(args)
        source_by_key = {_key(row): row for row in rows}
        completed = {row["key"] for row in _final(run, source_by_key)}
        pending = [row for row in rows if _key(row) not in completed]
        result_number = len(finalized_batches(run / "results"))
        attempt_number = len(finalized_batches(run / "error-attempts"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
            future_to_row = {executor.submit(frozen.judge_row, row["_judge_source"], row, run,
                                             judge_settings, transport): row for row in pending}
            for future in concurrent.futures.as_completed(future_to_row):
                result = future.result()
                if result["status"] in {"rated", "unrated_blank"}:
                    _validate_final_result(result, future_to_row[future])
                    publish_batch(run / "results", "result-%05d" % result_number, [result],
                                  key=lambda value: value["key"], required_keys=RESULT_KEYS)
                    result_number += 1
                    heartbeat.write_metric(event="result_published", key=result["key"],
                                           status=result["status"])
                else:
                    attempt = {**result, "attempt": attempt_number}
                    publish_batch(run / "error-attempts", "attempt-%05d" % attempt_number, [attempt],
                                  key=lambda value: "%s:%s" % (value["attempt"], value["key"]),
                                  required_keys=frozen.ERROR_ATTEMPT_KEYS)
                    attempt_number += 1
                    heartbeat.write_metric(event="error_attempt", key=result["key"])
        finals = _final(run, source_by_key)
        expected = set(source_by_key)
        if {result["key"] for result in finals} != expected:
            return {"done": False, "completed": len(finals),
                    "pending": len(expected - {result["key"] for result in finals})}
        exports = _export_rows(run, rows, finals)
        review = _review(run, source_by_key, finals, report["manifest"])
        detail = {"status": "DONE", "row_count": 900,
                  "unrated_blank": sum(result["status"] == "unrated_blank" for result in finals),
                  "source_raw_sha256": {source: binding["raw_sha256"] for source, binding in report["manifest"]["sources"].items()},
                  "exports": exports, "manual_review": review}
        mark_done(run, detail)
        return {"done": True, **detail}


def parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parents[1]
    parser.add_argument("--base-raw", type=Path, default=root / "runs/qwen35-4b-base-eval-formal-20260902T063703Z/raw/responses.jsonl")
    parser.add_argument("--abliterated-raw", type=Path, default=root / "runs/qwen35-4b-abliterated-eval-formal-20260902T065924Z/raw/responses.jsonl")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--runs-root", type=Path, default=root / "runs")
    parser.add_argument("--questions", type=Path, default=root / "external/hereditary/chinese_censorship_eval/data/test_questions_explicit.json")
    parser.add_argument("--facts", type=Path, default=root / "external/hereditary/chinese_censorship_eval/data/test_facts_explicit.json")
    parser.add_argument("--judge-id", default=frozen.JUDGE_ID)
    parser.add_argument("--temperature", type=float, default=frozen.JUDGE_TEMPERATURE)
    parser.add_argument("--max-tokens", type=int, default=frozen.JUDGE_MAX_TOKENS)
    parser.add_argument("--endpoint", default=frozen.OPENROUTER_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--concurrency", type=int, default=CONCURRENCY)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.concurrency != CONCURRENCY:
        raise ValidationError("judge concurrency is frozen to 16")
    print(json.dumps(execute(args) if args.execute else plan(args), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
