#!/usr/bin/env python3
"""Validate a corporate chatbot JSONL dataset.

Checks JSON syntax, required fields, message roles, and non-empty content.
Prints dataset statistics and a list of any invalid records, then exits
non-zero if validation failed. Bad records are reported, never silently
discarded.

Usage:
    python scripts/validate_dataset.py --input data/examples/corporate_examples.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corporate_chatbot.dataset import compute_statistics, load_and_validate  # noqa: E402
from corporate_chatbot.utils import eprint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a JSONL dataset file to validate",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        eprint(f"ERROR: input file does not exist: {args.input}")
        return 1

    result = load_and_validate(args.input)

    print(f"Validating: {args.input}")
    print(f"Total records found: {result.total}")
    print(f"Valid records:       {len(result.valid_records)}")
    print(f"Invalid records:     {len({i.index for i in result.issues})}")

    if result.valid_records:
        stats = compute_statistics(result.valid_records)
        print("\n## Dataset statistics (valid records only)")
        print(f"Examples:                    {stats['num_examples']}")
        print(f"Role counts:                 {stats['role_counts']}")
        print(f"Avg user message length:     {stats['avg_user_message_chars']} chars "
              f"(n={stats['num_user_messages']})")
        print(f"Avg assistant message length:{stats['avg_assistant_message_chars']} chars "
              f"(n={stats['num_assistant_messages']})")

    if result.issues:
        print(f"\n## Validation errors ({len(result.issues)})")
        for issue in result.issues:
            print(f"  - {issue}")
        print(f"\nFAILED: {len({i.index for i in result.issues})} record(s) had errors.")
        return 1

    print("\nOK: all records passed validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
