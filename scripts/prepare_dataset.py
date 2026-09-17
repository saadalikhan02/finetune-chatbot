#!/usr/bin/env python3
"""Convert a raw JSONL dataset into a validated Hugging Face Datasets-ready
file under data/processed/.

Loads and validates the input, preserves the ``messages`` structure (no
lossy conversion), optionally renders the tokenizer's chat template into a
``text`` column for inspection, and saves the result as JSONL.

Usage:
    python scripts/prepare_dataset.py \\
        --input data/examples/corporate_examples.jsonl \\
        --output data/processed/dataset.jsonl

    # Also render the Gemma chat template into a "text" column (requires
    # transformers + network/model access, and HF_TOKEN if the model is gated):
    python scripts/prepare_dataset.py \\
        --input data/examples/corporate_examples.jsonl \\
        --output data/processed/dataset.jsonl \\
        --apply-chat-template --model-name google/gemma-3-1b-it
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from corporate_chatbot.dataset import load_and_validate  # noqa: E402
from corporate_chatbot.formatting import format_example_for_training  # noqa: E402
from corporate_chatbot.utils import eprint, load_dotenv_if_present, get_hf_token, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Raw input JSONL file")
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Where to write the processed JSONL file",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Additionally render each example through the tokenizer's chat "
        "template into a 'text' field (requires downloading the tokenizer).",
    )
    parser.add_argument(
        "--model-name",
        default="google/gemma-3-1b-it",
        help="Model/tokenizer to use when --apply-chat-template is set",
    )
    return parser.parse_args()


def main() -> int:
    load_dotenv_if_present()
    args = parse_args()

    if not args.input.exists():
        eprint(f"ERROR: input file does not exist: {args.input}")
        return 1

    result = load_and_validate(args.input)
    if not result.is_valid:
        eprint(f"ERROR: {len(result.issues)} validation error(s) found. Fix them before preparing the dataset:")
        for issue in result.issues:
            eprint(f"  - {issue}")
        return 1

    records = result.valid_records
    print(f"Loaded {len(records)} valid example(s) from {args.input}")

    if args.apply_chat_template:
        try:
            from corporate_chatbot.model import load_tokenizer
        except ImportError as e:
            eprint(f"ERROR: --apply-chat-template requires transformers to be installed: {e}")
            return 1

        print(f"Loading tokenizer: {args.model_name}")
        tokenizer = load_tokenizer(args.model_name, hf_token=get_hf_token())
        for record in records:
            record["text"] = format_example_for_training(tokenizer, record)["text"]
        print("Rendered chat-template 'text' field for each example.")

    write_jsonl(args.output, records)
    print(f"Wrote {len(records)} example(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
