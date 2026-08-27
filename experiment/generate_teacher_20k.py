"""Generate a frozen prompt manifest with a local Qwen checkpoint.

`--plan` is deliberately dependency-free: it validates evidence before a GPU model
could be imported.  Sampling layout is frozen in the run manifest; this script does
not claim stochastic equality across different batch sizes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

try:  # supports both `python -m experiment...` and direct script execution
    from .batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                           finalized_batches, iter_jsonl, mark_done, publish_batch,
                           sha256_file, sha256_text, validate_batches)
except ImportError:  # pragma: no cover
    from batch_io import (RunHeartbeat, ValidationError, assert_run_mutable, atomic_write_json,
                          finalized_batches, iter_jsonl, mark_done, publish_batch,
                          sha256_file, sha256_text, validate_batches)

ROW_KEYS = ("id", "source", "prompt", "response", "seed", "prompt_sha256",
            "response_sha256", "output_tokens", "termination", "hit_token_cap", "is_blank")
CONFIG_VERSION = "teacher-generation-v1"


def load_prompts(path: str | Path) -> list[dict[str, str]]:
    rows = list(iter_jsonl(path))
    ids: set[str] = set()
    for row in rows:
        if set(row) != {"id", "source", "prompt"}:
            raise ValidationError("frozen prompt schema must be exactly id, source, prompt")
        if not all(isinstance(row[field], str) and row[field] for field in ("id", "source", "prompt")):
            raise ValidationError("frozen prompt values must be non-empty strings")
        if row["id"] in ids:
            raise ValidationError("duplicate prompt id: %s" % row["id"])
        ids.add(row["id"])
    return rows


def _config(args: argparse.Namespace, prompts: list[dict[str, str]]) -> dict[str, Any]:
    if len(prompts) != args.expected_count:
        raise ValidationError("expected %d frozen prompts, found %d" % (args.expected_count, len(prompts)))
    if not 100 <= args.publish_size <= 250:
        raise ValidationError("--publish-size must be between 100 and 250")
    if args.microbatch_size < 1:
        raise ValidationError("--microbatch-size must be positive")
    return {
        "format": CONFIG_VERSION,
        "input_sha256": sha256_file(args.prompts),
        "prompt_count": len(prompts),
        "model_path": str(Path(args.model_path)),
        "tokenizer_path": str(Path(args.tokenizer_path)),
        "master_seed": args.seed,
        "batch_layout": {"publish_size": args.publish_size, "microbatch_size": args.microbatch_size},
        "generation": {"temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                       "thinking": False, "dtype": "bfloat16"},
        "notes": "Batch layout is frozen; batch-size-independent stochastic equality is not claimed.",
    }


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError("invalid run manifest: %s" % path) from exc
    if not isinstance(value, dict):
        raise ValidationError("invalid run manifest: %s" % path)
    return value


def _validate_existing(run_dir: Path, config: dict[str, Any], ids: Iterable[str]) -> list[dict[str, Any]]:
    assert_run_mutable(run_dir)
    existing = _read_manifest(run_dir / "manifest.json")
    if existing is not None and existing != config:
        raise ValidationError("run manifest is frozen; configuration or batch layout changed")
    return validate_batches(run_dir / "batches", key=lambda row: row["id"],
                            required_keys=ROW_KEYS, expected_keys=None)


def plan(args: argparse.Namespace) -> dict[str, Any]:
    prompts = load_prompts(args.prompts)
    config = _config(args, prompts)
    completed = _validate_existing(Path(args.run_dir), config, (row["id"] for row in prompts))
    completed_ids = {row["id"] for row in completed}
    expected_ids = {row["id"] for row in prompts}
    unexpected = completed_ids - expected_ids
    if unexpected:
        raise ValidationError("final batch contains unknown prompt IDs: %s" % sorted(unexpected)[:5])
    return {"prompt_count": len(prompts), "completed": len(completed_ids),
            "pending": len(expected_ids - completed_ids),
            "final_batches": len(finalized_batches(Path(args.run_dir) / "batches")), "config": config}


def _load_backend(args: argparse.Namespace):
    # Imports live here so plan mode cannot load a CUDA/GPU dependency.
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("execute requires CUDA; CPU fallback is intentionally forbidden")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, local_files_only=True, dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    model.eval()
    return torch, tokenizer, model


def _generate_rows(args: argparse.Namespace, prompts: list[dict[str, str]], completed: set[str]):
    torch, tokenizer, model = _load_backend(args)
    prompt_positions = {row["id"]: index for index, row in enumerate(prompts)}
    pending = [row for row in prompts if row["id"] not in completed]
    for start in range(0, len(pending), args.publish_size):
        group = pending[start:start + args.publish_size]
        output: list[dict[str, Any]] = []
        for micro_start in range(0, len(group), args.microbatch_size):
            # Sampling is deliberately one prompt at a time so the recorded per-row seed is
            # the seed actually used.  `microbatch_size` remains a bounded scheduling unit;
            # no batch-size-independent stochastic equality is claimed.
            for item in group[micro_start:micro_start + args.microbatch_size]:
                seed = args.seed + prompt_positions[item["id"]]
                torch.manual_seed(seed)
                rendered = tokenizer.apply_chat_template([{"role": "user", "content": item["prompt"]}],
                                                         tokenize=False, add_generation_prompt=True,
                                                         enable_thinking=False)
                encoded = tokenizer(rendered, return_tensors="pt")
                encoded = {name: value.to("cuda") for name, value in encoded.items()}
                with torch.inference_mode():
                    generated = model.generate(**encoded, do_sample=True, temperature=args.temperature,
                                               top_p=args.top_p, top_k=args.top_k,
                                               max_new_tokens=args.max_new_tokens,
                                               pad_token_id=tokenizer.pad_token_id,
                                               eos_token_id=tokenizer.eos_token_id)
                continuation = generated[0, encoded["input_ids"].shape[1]:].tolist()
                response = tokenizer.decode(continuation, skip_special_tokens=True,
                                            clean_up_tokenization_spaces=False)
                hit_cap = len(continuation) >= args.max_new_tokens
                termination = "eos" if continuation and continuation[-1] == tokenizer.eos_token_id else "length"
                output.append({"id": item["id"], "source": item["source"], "prompt": item["prompt"],
                               "response": response, "seed": seed,
                               "prompt_sha256": sha256_text(item["prompt"]),
                               "response_sha256": sha256_text(response),
                               "output_tokens": len(continuation), "termination": termination,
                               "hit_token_cap": hit_cap, "is_blank": not response.strip()})
        yield start // args.publish_size, output


def execute(args: argparse.Namespace) -> dict[str, Any]:
    result = plan(args)  # all corruption/config validation precedes GPU initialization
    run_dir = Path(args.run_dir)
    config = result["config"]
    with RunHeartbeat(run_dir) as heartbeat:
        if not (run_dir / "manifest.json").exists():
            atomic_write_json(run_dir / "manifest.json", config)
        prompts = load_prompts(args.prompts)
        completed = {row["id"] for row in validate_batches(
            run_dir / "batches", key=lambda row: row["id"], required_keys=ROW_KEYS
        )}
        next_number = len(finalized_batches(run_dir / "batches"))
        heartbeat.write_metric(event="generation_start", completed=len(completed), pending=len(prompts) - len(completed))
        for _, rows in _generate_rows(args, prompts, completed):
            batch_name = "batch-%05d" % next_number
            final = publish_batch(run_dir / "batches", batch_name, rows,
                                  key=lambda row: row["id"], required_keys=ROW_KEYS)
            heartbeat.write_metric(event="batch_published", batch=batch_name, rows=len(rows),
                                   sha256=sha256_file(final / "data.jsonl"))
            next_number += 1
        rows = validate_batches(run_dir / "batches", key=lambda row: row["id"],
                                required_keys=ROW_KEYS, expected_keys=(row["id"] for row in prompts))
        if any(row["is_blank"] for row in rows):
            raise ValidationError("blank generation prevents DONE")
        heartbeat.write_metric(event="generation_complete", rows=len(rows))
        mark_done(run_dir, {"status": "DONE", "row_count": len(rows)})
        return {"completed": len(rows), "pending": 0}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="frozen {id,source,prompt} JSONL")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--publish-size", "--batch-size", dest="publish_size", type=int, default=250,
                        help="immutable publication batch size (100--250)")
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--expected-count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--plan", action="store_true", help="validate/resume plan only (default)")
    mode.add_argument("--execute", action="store_true", help="load the local CUDA BF16 backend")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute(args) if args.execute else plan(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
