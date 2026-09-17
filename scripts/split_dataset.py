#!/usr/bin/env python3
"""Split a processed JSONL dataset into train/validation JSONL files.

Uses a deterministic seed so the split is reproducible. Default split is
90% train / 10% validation.

Usage:
    python scripts/split_dataset.py \\
        --input data/processed/dataset.jsonl \\
        --output-dir data
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corporate_chatbot.dataset import split_records  # noqa: E402
from corporate_chatbot.utils import eprint, read_jsonl, write_jsonl  # noqa: E402

MIN_RECOMMENDED_EXAMPLES = 30
MIN_RECOMMENDED_VAL_EXAMPLES = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Processed JSONL file to split")
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Base data directory; writes <output-dir>/train/train.jsonl and "
        "<output-dir>/validation/validation.jsonl",
    )
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Validation fraction (default: 0.1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for the shuffle (default: 42)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        eprint(f"ERROR: input file does not exist: {args.input}")
        return 1

    records = read_jsonl(args.input)
    if len(records) < 2:
        eprint(f"ERROR: need at least 2 records to split, found {len(records)}")
        return 1

    if len(records) < MIN_RECOMMENDED_EXAMPLES:
        print(
            f"WARNING: only {len(records)} example(s) found. A validation split this "
            f"small is not statistically meaningful - treat eval numbers as a rough "
            f"sanity check, not a real metric. Consider adding more examples before "
            f"drawing conclusions from evaluation results."
        )

    train_records, val_records = split_records(records, val_ratio=args.val_ratio, seed=args.seed)

    if len(val_records) < MIN_RECOMMENDED_VAL_EXAMPLES:
        print(
            f"WARNING: validation split has only {len(val_records)} example(s). "
            f"eval_loss/eval metrics computed from this split will be noisy."
        )

    train_path = args.output_dir / "train" / "train.jsonl"
    val_path = args.output_dir / "validation" / "validation.jsonl"
    write_jsonl(train_path, train_records)
    write_jsonl(val_path, val_records)

    print(f"Total examples:      {len(records)}")
    print(f"Train examples:      {len(train_records)} -> {train_path}")
    print(f"Validation examples: {len(val_records)} -> {val_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
