"""Run the unchanged coherence judge over the prepared Qwen3.5-4B pair."""
from __future__ import annotations

import argparse
import json
from typing import Any

try:
    from . import judge_coherence as core
    from .prepare_qwen35_4b_coherence import ARM_SPECS, NORMALIZED_KEYS, validate_prepared
except ImportError:  # pragma: no cover
    import judge_coherence as core
    from prepare_qwen35_4b_coherence import ARM_SPECS, NORMALIZED_KEYS, validate_prepared

COHERENCE_PROMPT = core.COHERENCE_PROMPT
FORMAT = core.FORMAT
JUDGE_ID = core.JUDGE_ID
JUDGE_TEMPERATURE = core.JUDGE_TEMPERATURE
JUDGE_MAX_TOKENS = core.JUDGE_MAX_TOKENS
JUDGE_REASONING = core.JUDGE_REASONING
CONCURRENCY = core.CONCURRENCY
RESULT_KEYS = core.RESULT_KEYS


def _call_with_contract(function: Any, *args: Any, **kwargs: Any) -> Any:
    previous_specs = core.ARM_SPECS
    previous_keys = core.NORMALIZED_KEYS
    previous_validator = core.validate_prepared
    try:
        core.ARM_SPECS = ARM_SPECS
        core.NORMALIZED_KEYS = NORMALIZED_KEYS
        core.validate_prepared = validate_prepared
        return function(*args, **kwargs)
    finally:
        core.ARM_SPECS = previous_specs
        core.NORMALIZED_KEYS = previous_keys
        core.validate_prepared = previous_validator


def plan(args: argparse.Namespace) -> dict[str, Any]:
    return _call_with_contract(core.plan, args)


def execute(args: argparse.Namespace, transport: Any = None) -> dict[str, Any]:
    return _call_with_contract(core.execute, args, transport)


def build_parser() -> argparse.ArgumentParser:
    return core.build_parser()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = execute(args) if args.execute else plan(args)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
