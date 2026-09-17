#!/usr/bin/env python3
"""Convert data/datasets/test.jsonl (Gemma messages format) into the flat
evaluation/test_cases.jsonl format scripts/evaluate.py expects, so the
generated test set can be used directly for base-vs-fine-tuned comparison.

Usage:
    python scripts/webdata/export_eval_cases.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.models import read_jsonl, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/datasets/test.jsonl", type=Path)
    parser.add_argument("--output", default="evaluation/test_cases.jsonl", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = read_jsonl(args.input)

    cases = []
    for record in records:
        messages = record["messages"]
        system_text = messages[0]["content"][0]["text"]
        last_user = messages[-2]
        assert last_user["role"] == "user"
        history = messages[1:-2]  # everything between system and the final user turn

        case = {
            "category": record["metadata"]["category"],
            "user_input": last_user["content"][0]["text"],
            "system_prompt": system_text,
        }
        if history:
            case["history"] = history
        if record["metadata"].get("expected_behavior"):
            case["expected_behavior"] = record["metadata"]["expected_behavior"]
        cases.append(case)

    write_jsonl(args.output, cases)
    print(f"Wrote {len(cases)} evaluation case(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
