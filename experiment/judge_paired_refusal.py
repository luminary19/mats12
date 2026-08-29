"""Prepare and resumably judge a frozen China-free paired teacher sample.

Prepare and plan are offline.  Execute is the only mode that can construct an
OpenRouter transport, and it makes exactly one logical refusal judgment for
each nonblank source response.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Mapping

try:
    from . import judge_probe
    from .batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                           finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_bytes,
                           sha256_file, sha256_text, strict_json_bytes, validate_batches,
                           write_jsonl_fsynced)
except ImportError:  # pragma: no cover
    import judge_probe
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                          finalized_batches, iter_jsonl, mark_done, publish_batch, sha256_bytes,
                          sha256_file, sha256_text, strict_json_bytes, validate_batches,
                          write_jsonl_fsynced)

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_PATH = ROOT / "external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz"
ABLITERATED_PATH = ROOT / "runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl"
PROTOCOL_PATH = ROOT / "protocol-amendments/paired-refusal-distribution-2026-08-30.json"
PROTOCOL_SHA256 = "895a721d781b6b25592e2c55abfe597b8832511d6667917be0a84ecd47f3631e"
ORIGINAL_PROTOCOL_PATH = "external/hereditary/data/censorship_training/01_olmo_clean_qwen.jsonl.gz"
ABLITERATED_PROTOCOL_PATH = "runs/olmo-clean-19996-abliterated-b200-20260827T0951Z/output/rollouts.jsonl"
AMENDMENT_SCOPE = "Exploratory refusal-distribution study over the China-free 19,996-row clean OLMo pool only."
AMENDMENT_INTERPRETATION = "Any cross-domain comparison to the existing held-out Chinese student refusal results is descriptive, not a direct paired transfer test: no student responses are generated on these 1,000 prompts."
ORIGINAL_GZIP_SHA256 = "cc42e6dcf4c80854eca0e294ce318ee4792a5406b6ef3cac44d230e4eafb7f44"
ORIGINAL_JSONL_SHA256 = "889f6bb7784c1f327d7f798c1cfae148f8384af0d7263ca0e40bcf920c1e9922"
ABLITERATED_SHA256 = "be7f9906584133f9ede6b925ec933968d5b6a101b610a4dff7670064c147e315"
ORIGINAL_MODEL = "qwen/qwen3.5-9b"
ABLITERATED_MODEL = "huihui-ai/Huihui-Qwen3.5-9B-abliterated"
ROW_COUNT = 19_996
SAMPLE_SIZE = 1_000
SEED = 42
SELECTION_ALGORITHM = "paired-refusal-sha256-rank-v1"
FORMAT = "paired-refusal-judge-v1"
RESULT_KEYS = ("key", "arm", "selection_rank", "source_index", "id", "source", "prompt_sha256",
               "response_sha256", "manifest_sha256", "judge_settings_sha256", "judge_cache_key",
               "judge_raw_sha256", "status", "refusal", "errors")
ERROR_ATTEMPT_KEYS = RESULT_KEYS + ("attempt",)
FINAL_STATUSES = frozenset({"rated", "unrated_blank", "unrated_parse"})


def _validate_protocol_amendment() -> None:
    """Pin both amendment bytes and every semantic field used by this harness."""
    if not PROTOCOL_PATH.is_file() or sha256_file(PROTOCOL_PATH) != PROTOCOL_SHA256:
        raise ValidationError("protocol amendment bytes differ from the frozen protocol")
    try:
        value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid protocol amendment") from exc
    expected = {
        "format": "paired-refusal-distribution-amendment-v1", "date": "2026-08-30",
        "decision": "exploratory_paired_refusal_distribution_study", "scope": AMENDMENT_SCOPE,
        "selection": {"algorithm": SELECTION_ALGORITHM, "seed": SEED, "pair_count": SAMPLE_SIZE,
                      "ranking": "lowest SHA-256 ranks over version, seed, id, source, and exact UTF-8 prompt SHA-256",
                      "rank_bytes": "UTF-8(algorithm)+NUL+ASCII(seed)+NUL+UTF-8(id)+NUL+UTF-8(source)+NUL+ASCII(prompt_sha256)"},
        "sources": {
            "original": {"path": ORIGINAL_PROTOCOL_PATH, "gzip_sha256": ORIGINAL_GZIP_SHA256,
                         "decompressed_jsonl_sha256": ORIGINAL_JSONL_SHA256, "model": ORIGINAL_MODEL},
            "abliterated": {"path": ABLITERATED_PROTOCOL_PATH, "sha256": ABLITERATED_SHA256,
                            "model": ABLITERATED_MODEL}},
        "judge": {"rubric": "unchanged Conmy REFUSAL_PROMPT only", "model": judge_probe.JUDGE_ID,
                  "temperature": judge_probe.JUDGE_TEMPERATURE, "max_tokens": judge_probe.JUDGE_MAX_TOKENS,
                  "reasoning_enabled": False, "parser": "conmy-first-search-v1"},
        "interpretation": AMENDMENT_INTERPRETATION,
    }
    if any(value.get(key) != field for key, field in expected.items()):
        raise ValidationError("protocol amendment semantics differ from the frozen protocol")


def _gzip_payload(path: Path) -> bytes:
    digest = __import__("hashlib").sha256()
    chunks: list[bytes] = []
    try:
        with gzip.open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
                chunks.append(block)
    except OSError as exc:
        raise ValidationError("cannot decompress original source: %s" % path) from exc
    payload = b"".join(chunks)
    if digest.hexdigest() != ORIGINAL_JSONL_SHA256:
        raise ValidationError("original decompressed JSONL hash differs from frozen protocol")
    return payload


def _exact_path(actual: Path, expected: Path, label: str) -> Path:
    if actual.resolve() != expected.resolve():
        raise ValidationError("%s path differs from the frozen protocol" % label)
    return actual.resolve()


def _parse_jsonl_bytes(payload: bytes, label: str) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValidationError("%s is not UTF-8 JSONL" % label) from exc
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    if any(not line for line in lines):
        raise ValidationError("blank JSONL record in %s" % label)
    try:
        rows = [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValidationError("invalid JSONL in %s" % label) from exc
    if not all(isinstance(row, dict) for row in rows):
        raise ValidationError("non-object JSONL row in %s" % label)
    return rows


def _validate_rows(rows: list[dict[str, Any]], model: str, label: str) -> None:
    required = {"id", "source", "prompt", "response", "model"}
    if len(rows) != ROW_COUNT:
        raise ValidationError("%s must contain exactly %d rows" % (label, ROW_COUNT))
    seen: set[tuple[str, str]] = set()
    for row in rows:
        if set(row) != required or not all(isinstance(row[name], str) for name in required):
            raise ValidationError("%s row schema differs from the frozen protocol" % label)
        if not row["id"] or not row["source"] or not row["prompt"] or row["model"] != model:
            raise ValidationError("%s row identity/model differs from the frozen protocol" % label)
        identity = (row["id"], row["source"])
        if identity in seen:
            raise ValidationError("duplicate row identity in %s" % label)
        seen.add(identity)


def load_sources(original: str | Path = ORIGINAL_PATH, abliterated: str | Path = ABLITERATED_PATH) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    """Load only the audited clean 19,996-row sources and verify byte identities."""
    _validate_protocol_amendment()
    original_path = _exact_path(Path(original), ORIGINAL_PATH, "original source")
    abliterated_path = _exact_path(Path(abliterated), ABLITERATED_PATH, "abliterated source")
    if sha256_file(original_path) != ORIGINAL_GZIP_SHA256:
        raise ValidationError("original gzip hash differs from frozen protocol")
    if sha256_file(abliterated_path) != ABLITERATED_SHA256:
        raise ValidationError("abliterated source hash differs from frozen protocol")
    originals = _parse_jsonl_bytes(_gzip_payload(original_path), "original source")
    abliterated_rows = list(iter_jsonl(abliterated_path))
    _validate_rows(originals, ORIGINAL_MODEL, "original source")
    _validate_rows(abliterated_rows, ABLITERATED_MODEL, "abliterated source")
    for index, (left, right) in enumerate(zip(originals, abliterated_rows)):
        if (left["id"], left["source"], left["prompt"]) != (right["id"], right["source"], right["prompt"]):
            raise ValidationError("source alignment differs at index %d" % index)
    return originals, abliterated_rows, {"original_gzip_sha256": ORIGINAL_GZIP_SHA256,
        "original_jsonl_sha256": ORIGINAL_JSONL_SHA256, "abliterated_sha256": ABLITERATED_SHA256}


def _selection_rank(row: Mapping[str, Any]) -> str:
    prompt_hash = sha256_text(row["prompt"])
    # Version and NUL delimiters make this construction language-independent and unambiguous.
    material = (SELECTION_ALGORITHM.encode("ascii") + b"\0" + str(SEED).encode("ascii") + b"\0" +
                row["id"].encode("utf-8") + b"\0" + row["source"].encode("utf-8") + b"\0" +
                prompt_hash.encode("ascii"))
    return sha256_bytes(material)


def select_pairs(originals: list[dict[str, Any]], abliterated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = sorted(((_selection_rank(row), index, row, abliterated[index])
                     for index, row in enumerate(originals)), key=lambda item: (item[0], item[1]))[:SAMPLE_SIZE]
    if len(ranked) != SAMPLE_SIZE or len({item[1] for item in ranked}) != SAMPLE_SIZE:
        raise ValidationError("selection did not produce unique frozen sample")
    selected: list[dict[str, Any]] = []
    for rank, (_, index, original, abliterated_row) in enumerate(ranked):
        selected.append({"selection_rank": rank, "source_index": index, "id": original["id"],
                         "source": original["source"], "prompt_sha256": sha256_text(original["prompt"]),
                         "original_response_sha256": sha256_text(original["response"]),
                         "abliterated_response_sha256": sha256_text(abliterated_row["response"])})
    return selected


def _sample_rows(selected: list[dict[str, Any]], originals: list[dict[str, Any]], abliterated: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in selected:
        index = item["source_index"]
        original, ablated = originals[index], abliterated[index]
        rows.append({**item, "prompt": original["prompt"], "original_response": original["response"],
                     "original_model": original["model"], "abliterated_response": ablated["response"],
                     "abliterated_model": ablated["model"]})
    return rows


def _write_immutable_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> str:
    if path.exists():
        raise FileExistsError("immutable file already exists: %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + ".tmp")
    if temporary.exists():
        temporary.unlink()
    write_jsonl_fsynced(temporary, rows)
    if path.exists():
        raise FileExistsError("immutable file already exists: %s" % path)
    os.replace(str(temporary), str(path))
    return sha256_file(path)


def _run_path_safe(run_dir: Path) -> None:
    run = run_dir.resolve()
    for source in (ORIGINAL_PATH.resolve(), ABLITERATED_PATH.resolve()):
        # A run must not be either a source, an ancestor of it, or within its data directory.
        if run == source or run in source.parents or source.parent == run or source.parent in run.parents:
            raise ValidationError("run directory overlaps immutable source/checkpoint location")
    # A continuation must be a new sibling, never a child of an existing terminal run.
    if any((parent / "DONE").exists() or (parent / "CRASHED").exists() for parent in run.parents):
        raise ValidationError("run directory is inside an existing terminal run")
    assert_run_mutable(run)


def _settings() -> dict[str, Any]:
    frozen = judge_probe._frozen_settings()
    return {"format": FORMAT, "selection_algorithm": SELECTION_ALGORITHM,
            "selection_rank_bytes": "UTF-8(algorithm)+NUL+ASCII(seed)+NUL+UTF-8(id)+NUL+UTF-8(source)+NUL+ASCII(prompt_sha256)",
            "seed": SEED, "sample_size": SAMPLE_SIZE, "judge": frozen, "logical_calls_per_nonblank_response": 1,
            "transport_max_attempts": judge_probe.TRANSPORT_MAX_ATTEMPTS}


def _manifest(run_dir: Path, source_hashes: Mapping[str, str], selection_hash: str, sample_hash: str) -> dict[str, Any]:
    _validate_protocol_amendment()
    if not run_dir.name or run_dir.name in {".", ".."}:
        raise ValidationError("run directory must have a distinct name")
    return {**_settings(), **dict(source_hashes), "run_id": run_dir.name,
            "run_path_sha256": sha256_text(str(run_dir.resolve())), "row_count": ROW_COUNT,
            "harness_sha256": sha256_file(Path(__file__)),
            "judge_probe_sha256": sha256_file(Path(judge_probe.__file__)),
            "batch_io_sha256": sha256_file(ROOT / "experiment/batch_io.py"),
            "protocol_amendment": PROTOCOL_PATH.name, "protocol_amendment_sha256": PROTOCOL_SHA256,
            "selection_sha256": selection_hash, "paired_sample_sha256": sample_hash}


def _load_prepared(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    selection_path, sample_path, manifest_path = run_dir / "selection.jsonl", run_dir / "paired-sample.jsonl", run_dir / "manifest.json"
    if not (selection_path.is_file() and sample_path.is_file() and manifest_path.is_file()):
        raise ValidationError("prepare must atomically establish selection, paired sample, and manifest")
    selection, sample = list(iter_jsonl(selection_path)), list(iter_jsonl(sample_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid prepared manifest") from exc
    if manifest != _manifest(run_dir, {"original_gzip_sha256": ORIGINAL_GZIP_SHA256,
            "original_jsonl_sha256": ORIGINAL_JSONL_SHA256, "abliterated_sha256": ABLITERATED_SHA256},
            sha256_file(selection_path), sha256_file(sample_path)):
        raise ValidationError("prepared manifest differs from frozen protocol")
    if len(selection) != SAMPLE_SIZE or len(sample) != SAMPLE_SIZE:
        raise ValidationError("prepared sample count differs from frozen protocol")
    if [row["selection_rank"] for row in selection] != list(range(SAMPLE_SIZE)):
        raise ValidationError("selection ranks are not frozen contiguous order")
    if any(set(row) != {"selection_rank", "source_index", "id", "source", "prompt_sha256", "original_response_sha256", "abliterated_response_sha256"} for row in selection):
        raise ValidationError("selection schema differs from frozen protocol")
    if any(sample[index].get(key) != row[key] for index, row in enumerate(selection)
           for key in row) or any(sha256_text(sample[index].get("prompt", "")) != row["prompt_sha256"] or
           sha256_text(sample[index].get("original_response", "")) != row["original_response_sha256"] or
           sha256_text(sample[index].get("abliterated_response", "")) != row["abliterated_response_sha256"]
           for index, row in enumerate(selection)):
        raise ValidationError("paired sample differs from compact selection")
    return selection, sample, manifest


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    _run_path_safe(run_dir)
    originals, ablated, hashes = load_sources(args.original, args.abliterated)
    selected, sample = select_pairs(originals, ablated), None
    sample = _sample_rows(selected, originals, ablated)
    selection_path, sample_path, manifest_path = run_dir / "selection.jsonl", run_dir / "paired-sample.jsonl", run_dir / "manifest.json"
    selection_bytes = b"".join(strict_json_bytes(row) for row in selected)
    sample_bytes = b"".join(strict_json_bytes(row) for row in sample)
    manifest = _manifest(run_dir, hashes, sha256_bytes(selection_bytes), sha256_bytes(sample_bytes))
    expected_bytes = {selection_path: selection_bytes, sample_path: sample_bytes, manifest_path: strict_json_bytes(manifest)}
    existing = [path for path in expected_bytes if path.exists()]
    # A stopped prepare may leave a prefix of immutable files.  Resume only when each
    # present byte stream is exactly the freshly re-derived one; never overwrite it.
    for path in existing:
        if path.read_bytes() != expected_bytes[path]:
            raise ValidationError("partial prepared artifact differs from frozen expected content: %s" % path)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not selection_path.exists():
        _write_immutable_jsonl(selection_path, selected)
    if not sample_path.exists():
        _write_immutable_jsonl(sample_path, sample)
    if not manifest_path.exists():
        atomic_write_json(manifest_path, manifest)
    _load_prepared(run_dir)
    return {"prepared": True, "reused": bool(existing), "selected_pairs": SAMPLE_SIZE, "manifest": manifest}


def _key(arm: str, selection_rank: int) -> str:
    return "%s:%04d" % (arm, selection_rank)


def _settings_sha256(settings: Mapping[str, Any]) -> str:
    return sha256_text(json.dumps(dict(settings), sort_keys=True, separators=(",", ":")))


def _expected(sample: list[dict[str, Any]], settings: Mapping[str, Any], manifest_sha256: str) -> dict[str, dict[str, Any]]:
    expected: dict[str, dict[str, Any]] = {}
    settings_sha256 = _settings_sha256(settings)
    for row in sample:
        for arm, field in (("original", "original_response"), ("abliterated", "abliterated_response")):
            response = row[field]
            prompt = judge_probe.REFUSAL_PROMPT.format(question=row["prompt"], model_response=response)
            cache_key = None if not response.strip() else judge_probe.cache_key(settings["judge_id"], prompt, response, settings)
            result = {"key": _key(arm, row["selection_rank"]), "arm": arm,
                      "selection_rank": row["selection_rank"], "source_index": row["source_index"],
                      "id": row["id"], "source": row["source"], "prompt_sha256": row["prompt_sha256"],
                      "response_sha256": sha256_text(response), "manifest_sha256": manifest_sha256,
                      "judge_settings_sha256": settings_sha256, "judge_cache_key": cache_key}
            if result["key"] in expected:
                raise ValidationError("duplicate expected paired key")
            expected[result["key"]] = result
    return expected


def _final_results(run_dir: Path, expected: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = validate_batches(run_dir / "results", key=lambda row: row["key"], required_keys=RESULT_KEYS)
    for row in rows:
        if row["status"] not in FINAL_STATUSES or row["errors"] != []:
            raise ValidationError("invalid final result status")
        identity = expected.get(row["key"])
        if identity is None or any(row[name] != value for name, value in identity.items()):
            raise ValidationError("unknown, copied, or source-mismatched result")
        if identity["judge_cache_key"] is None:
            if row["status"] != "unrated_blank" or row["refusal"] is not None or row["judge_raw_sha256"] is not None:
                raise ValidationError("only an actual blank source response can be unrated_blank")
            continue
        if row["status"] == "unrated_blank" or not isinstance(row["judge_raw_sha256"], str):
            raise ValidationError("nonblank result lacks canonical cache evidence")
        raw = judge_probe._read_cache(run_dir / "cache" / (identity["judge_cache_key"] + ".json"))
        if raw is None or sha256_text(raw) != row["judge_raw_sha256"]:
            raise ValidationError("result cache evidence differs from canonical response")
        answer = judge_probe._parse_answer(raw, "yes|no")
        if answer is None:
            if row["status"] != "unrated_parse" or row["refusal"] is not None:
                raise ValidationError("non-tag cache response must be explicit unrated_parse")
        elif row["status"] != "rated" or row["refusal"] != (answer == "yes"):
            raise ValidationError("result refusal differs from canonical Conmy parse")
    return rows


def _attempts(run_dir: Path, expected: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = validate_batches(run_dir / "error-attempts", key=lambda row: "%s:%s" % (row["attempt"], row["key"]), required_keys=ERROR_ATTEMPT_KEYS)
    for row in rows:
        identity = expected.get(row["key"])
        if (identity is None or any(row[name] != value for name, value in identity.items()) or
                row["status"] != "error" or row["refusal"] is not None or not row["errors"]):
            raise ValidationError("invalid error-attempt evidence")
    return rows


def plan(args: argparse.Namespace) -> dict[str, Any]:
    run_dir = Path(args.run_dir).resolve()
    _run_path_safe(run_dir)
    # Revalidate the actual sources and alignment before any possible transport construction.
    originals, ablated, _ = load_sources(args.original, args.abliterated)
    selected, sample, manifest = _load_prepared(run_dir)
    if selected != select_pairs(originals, ablated) or sample != _sample_rows(selected, originals, ablated):
        raise ValidationError("prepared evidence no longer matches frozen sources/alignment")
    manifest_sha256 = sha256_file(run_dir / "manifest.json")
    expected = _expected(sample, manifest["judge"], manifest_sha256)
    results, attempts = _final_results(run_dir, expected), _attempts(run_dir, expected)
    completed = {row["key"] for row in results}
    pending = [key for key in expected if key not in completed]
    nonblank_pending = sum(bool((sample[int(key.split(":")[1])]["original_response"] if key.startswith("original:")
                                  else sample[int(key.split(":")[1])]["abliterated_response"]).strip()) for key in pending)
    return {"selected_pairs": SAMPLE_SIZE, "expected_results": len(expected), "completed": len(completed),
            "pending": len(pending), "planned_calls": nonblank_pending, "error_attempts": len(attempts),
            "final_batches": len(finalized_batches(run_dir / "results")), "manifest": manifest}


def _result(arm: str, item: Mapping[str, Any], response: str, run_dir: Path, settings: Mapping[str, Any],
            manifest_sha256: str, transport: Callable[[str, Mapping[str, Any]], str]) -> dict[str, Any]:
    base = {**_expected([item], settings, manifest_sha256)[_key(arm, item["selection_rank"])],
            "judge_raw_sha256": None, "status": "error", "refusal": None, "errors": []}
    if not response.strip():
        base["status"] = "unrated_blank"
        return base
    prompt = judge_probe.REFUSAL_PROMPT.format(question=item["prompt"], model_response=response)
    raw, error = judge_probe._call_cached(run_dir, prompt, response, settings, transport)
    if error is not None:
        base["errors"] = [{"stage": "refusal", **error}]
        return base
    base["judge_raw_sha256"] = sha256_text(raw)
    answer = judge_probe._parse_answer(raw, "yes|no")
    if answer is None:
        base["status"] = "unrated_parse"
    else:
        base["status"], base["refusal"] = "rated", answer == "yes"
    return base


def _batch_fingerprint(run_dir: Path) -> str:
    manifests = [sha256_file(path / "manifest.json") for path in finalized_batches(run_dir / "results")]
    return sha256_bytes(strict_json_bytes(manifests))


def _wilson(successes: int, total: int) -> list[float | None]:
    if not total:
        return [None, None]
    z = 1.959963984540054
    p, denominator = successes / total, 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [centre - spread, centre + spread]


def _arm_summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    decided = [row for row in rows if row["status"] == "rated"]
    refusals = sum(row["refusal"] for row in decided)
    return {"total": len(rows), "decided_count": len(decided), "refusal_count": refusals,
            "nonrefusal_count": len(decided) - refusals,
            "refusal_rate": refusals / len(decided) if decided else None,
            "wilson_95": _wilson(refusals, len(decided)),
            "null_count": sum(row["refusal"] is None for row in rows),
            "unrated_blank_count": sum(row["status"] == "unrated_blank" for row in rows),
            "unrated_parse_count": sum(row["status"] == "unrated_parse" for row in rows)}


def _paired_summary(original: list[Mapping[str, Any]], ablated: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_rank = {row["selection_rank"]: row for row in original}
    other = {row["selection_rank"]: row for row in ablated}
    pairs = [(by_rank[rank], other[rank]) for rank in sorted(by_rank)]
    decided = [(left, right) for left, right in pairs if left["status"] == right["status"] == "rated"]
    counts = {"both": 0, "original_only": 0, "abliterated_only": 0, "neither": 0}
    for left, right in decided:
        counts["both" if left["refusal"] and right["refusal"] else "original_only" if left["refusal"]
               else "abliterated_only" if right["refusal"] else "neither"] += 1
    n = counts["original_only"] + counts["abliterated_only"]
    tail = sum(math.comb(n, index) for index in range(min(counts["original_only"], counts["abliterated_only"]) + 1))
    pvalue = min(1.0, 2.0 * tail / (2 ** n)) if n else 1.0
    return {"pair_count": len(pairs), "paired_decided_count": len(decided), "paired_excluded_count": len(pairs) - len(decided),
            "counts": counts, "refusal_rate_difference_abliterated_minus_original":
            ((counts["abliterated_only"] - counts["original_only"]) / len(decided) if decided else None),
            "mcnemar_exact_two_sided_p_value": pvalue}


def build_summary(run_dir: Path, results: list[dict[str, Any]], manifest: Mapping[str, Any]) -> dict[str, Any]:
    arms = {arm: [row for row in results if row["arm"] == arm] for arm in ("original", "abliterated")}
    sources: dict[str, Any] = {}
    for source in sorted({row["source"] for row in results}):
        left = [row for row in arms["original"] if row["source"] == source]
        right = [row for row in arms["abliterated"] if row["source"] == source]
        sources[source] = {"original": _arm_summary(left), "abliterated": _arm_summary(right),
                           "paired": _paired_summary(left, right)}
    arm_summary = {arm: _arm_summary(rows) for arm, rows in arms.items()}
    original_rate, abliterated_rate = arm_summary["original"]["refusal_rate"], arm_summary["abliterated"]["refusal_rate"]
    return {"format": "paired-refusal-summary-v1", "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "selection_sha256": manifest["selection_sha256"], "paired_sample_sha256": manifest["paired_sample_sha256"],
            "result_batch_manifests_sha256": _batch_fingerprint(run_dir), "arms": arm_summary,
            "arm_refusal_rate_difference_abliterated_minus_original":
            (abliterated_rate - original_rate if abliterated_rate is not None and original_rate is not None else None),
            "paired": _paired_summary(arms["original"], arms["abliterated"]), "per_source": sources,
            "done_binding": "DONE contains final_summary_sha256"}


class _ExecutionLease:
    """Advisory OS lock; it releases on process death and blocks a second launcher."""
    def __init__(self, run_dir: Path):
        self.path = run_dir / ".execute.lock"
        self.handle: Any = None

    def __enter__(self) -> "_ExecutionLease":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            try:
                self.handle = self.path.open("x+b")
                self.handle.write(b"0")
                self.handle.flush()
                os.fsync(self.handle.fileno())
            except FileExistsError:
                self.handle = self.path.open("r+b")
        except OSError as exc:
            raise ValidationError("another paired refusal execute already holds this run lease") from exc
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - Windows is the controller target.
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            raise ValidationError("another paired refusal execute already holds this run lease") from exc
        return self

    def __exit__(self, *_unused: Any) -> bool:
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:  # pragma: no cover
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
        return False


def execute(args: argparse.Namespace, transport: Callable[[str, Mapping[str, Any]], str] | None = None) -> dict[str, Any]:
    # First validation is deliberately before credentials/transport; revalidate inside the lease.
    plan(args)
    if args.concurrency < 1:
        raise ValidationError("--concurrency must be positive")
    run_dir = Path(args.run_dir).resolve()
    with _ExecutionLease(run_dir):
        report = plan(args)
        if transport is None:
            api_key = os.environ.get("OPENROUTER_API_KEY")
            if not api_key:
                raise RuntimeError("OPENROUTER_API_KEY is required for --execute")
            transport = judge_probe.OpenRouterTransport(api_key, report["manifest"]["judge"]["endpoint"], args.timeout)
        selected, sample, manifest = _load_prepared(run_dir)
        manifest_sha256 = sha256_file(run_dir / "manifest.json")
        expected = _expected(sample, manifest["judge"], manifest_sha256)
        with RunHeartbeat(run_dir) as heartbeat:
            completed = {row["key"] for row in _final_results(run_dir, expected)}
            pending = [(arm, item, field) for item in sample for arm, field in
                       (("original", "original_response"), ("abliterated", "abliterated_response"))
                       if _key(arm, item["selection_rank"]) not in completed]
            batch_number, attempt_number = len(finalized_batches(run_dir / "results")), len(finalized_batches(run_dir / "error-attempts"))
            heartbeat.write_metric(event="paired_refusal_start", completed=len(completed), pending=len(pending))

            def publish(result: dict[str, Any]) -> None:
                nonlocal batch_number, attempt_number
                if result["status"] in FINAL_STATUSES:
                    for retry in range(5):
                        try:
                            publish_batch(run_dir / "results", "result-%05d" % batch_number, [result],
                                          key=lambda row: row["key"], required_keys=RESULT_KEYS)
                            break
                        except PermissionError:
                            if retry == 4:
                                raise
                            time.sleep(0.1 * (2 ** retry))
                    batch_number += 1
                    heartbeat.write_metric(event="result_published", key=result["key"], status=result["status"])
                else:
                    attempt = {**result, "attempt": attempt_number}
                    for retry in range(5):
                        try:
                            publish_batch(run_dir / "error-attempts", "attempt-%05d" % attempt_number, [attempt],
                                          key=lambda row: "%s:%s" % (row["attempt"], row["key"]), required_keys=ERROR_ATTEMPT_KEYS)
                            break
                        except PermissionError:
                            if retry == 4:
                                raise
                            time.sleep(0.1 * (2 ** retry))
                    attempt_number += 1
                    heartbeat.write_metric(event="error_attempt_published", key=result["key"])
            pool = ThreadPoolExecutor(max_workers=args.concurrency)
            futures: dict[Any, None] = {}
            cursor, interrupted = 0, False
            try:
                while cursor < len(pending) and len(futures) < args.concurrency:
                    arm, item, field = pending[cursor]
                    cursor += 1
                    futures[pool.submit(_result, arm, item, item[field], run_dir, manifest["judge"], manifest_sha256, transport)] = None
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        del futures[future]
                        publish(future.result())
                    while cursor < len(pending) and len(futures) < args.concurrency:
                        arm, item, field = pending[cursor]
                        cursor += 1
                        futures[pool.submit(_result, arm, item, item[field], run_dir, manifest["judge"], manifest_sha256, transport)] = None
            except KeyboardInterrupt:
                interrupted = True
                for future in futures:
                    future.cancel()
                # Only at most concurrency futures were ever submitted; retain every completed/in-flight outcome.
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        del futures[future]
                        if not future.cancelled():
                            publish(future.result())
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            if interrupted:
                heartbeat.write_metric(event="paired_refusal_interrupted", completed=len(_final_results(run_dir, expected)))
                raise KeyboardInterrupt
            final = _final_results(run_dir, expected)
            if {row["key"] for row in final} != set(expected):
                remaining = len(expected) - len(final)
                heartbeat.write_metric(event="paired_refusal_incomplete", completed=len(final), pending=remaining)
                return {"completed": len(final), "pending": remaining, "done": False}
            summary = build_summary(run_dir, final, manifest)
            summary_path = run_dir / "summary.json"
            if summary_path.exists():
                try:
                    if json.loads(summary_path.read_text(encoding="utf-8")) != summary:
                        raise ValidationError("existing final summary differs from immutable results")
                except json.JSONDecodeError as exc:
                    raise ValidationError("invalid final summary") from exc
            else:
                atomic_write_json(summary_path, summary)
            done_detail = {"status": "DONE", "row_count": len(final), "expected_row_count": 2 * SAMPLE_SIZE,
                           "final_summary_sha256": sha256_file(summary_path), "selection_sha256": manifest["selection_sha256"],
                           "result_batch_manifests_sha256": summary["result_batch_manifests_sha256"]}
            heartbeat.write_metric(event="paired_refusal_complete", **done_detail)
            mark_done(run_dir, done_detail)
            return {"completed": len(final), "pending": 0, "done": True, "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--original", type=Path, default=ORIGINAL_PATH)
    parser.add_argument("--abliterated", type=Path, default=ABLITERATED_PATH)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--concurrency", type=int, default=16)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--plan", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute(args) if args.execute else prepare(args) if args.prepare else plan(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
