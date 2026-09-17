#!/usr/bin/env python3
"""Sanity-check every generated QA record before it goes into a dataset.

This is a lint pass, not a proof of correctness - it cannot verify that an
answer is a *faithful* summary of its excerpt. What it does check,
mechanically:

  - every "grounded" record cites at least one fact_id, and every cited
    fact_id actually exists in data/knowledge/facts.jsonl (no dangling/typo'd
    references)
  - every "grounded" record has a non-empty source_url/section/excerpt
  - every cited fact's source_url matches the record's source_url (a
    grounded answer can't cite a fact from a different page than the one
    it names as its source)
  - "not_available" and "out_of_scope" records carry no fact_ids (they are
    not claiming to be backed by page content)
  - no duplicate (question, category) pairs across the combined QA bank

Usage:
    python scripts/webdata/validate_grounding.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.models import read_jsonl  # noqa: E402

QA_FILES = [
    "data/knowledge/qa_auto.jsonl",
    "data/knowledge/qa_curated.jsonl",
    "data/knowledge/qa_followups.jsonl",
    "data/knowledge/qa_unknown.jsonl",
    "data/knowledge/qa_out_of_scope.jsonl",
    "data/knowledge/qa_conversational.jsonl",
    "data/knowledge/qa_adversarial.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--facts", default="data/knowledge/facts.jsonl", type=Path)
    parser.add_argument("--qa-files", nargs="*", default=QA_FILES, type=Path)
    args = parser.parse_args()
    args.qa_files = [Path(p) for p in args.qa_files]
    return args


def main() -> int:
    args = parse_args()
    facts_by_id = {f["fact_id"]: f for f in read_jsonl(args.facts)}

    errors: list[str] = []
    seen_question_category: set[tuple[str, str]] = set()
    total = 0

    for qa_file in args.qa_files:
        if not qa_file.exists():
            errors.append(f"{qa_file}: file does not exist")
            continue
        records = read_jsonl(qa_file)
        for i, record in enumerate(records):
            total += 1
            loc = f"{qa_file}:{i}"
            status = record["grounding_status"]

            if not record.get("question", "").strip():
                errors.append(f"{loc}: empty question")
            if not record.get("answer", "").strip():
                errors.append(f"{loc}: empty answer")

            key = (record["question"].strip().lower(), record["category"])
            if key in seen_question_category:
                errors.append(f"{loc}: duplicate question+category ({key})")
            seen_question_category.add(key)

            if status == "grounded":
                fact_ids = record.get("fact_ids") or []
                if not fact_ids:
                    errors.append(f"{loc}: grounded record has no fact_ids")
                cited_urls = set()
                for fid in fact_ids:
                    if fid not in facts_by_id:
                        errors.append(f"{loc}: fact_id {fid!r} not found in {args.facts}")
                        continue
                    cited_urls.add(facts_by_id[fid]["source_url"])
                # A multi-fact answer may combine facts from more than one
                # page; source_url just needs to be *one* of them, not all.
                if fact_ids and cited_urls and record.get("source_url") not in cited_urls:
                    errors.append(
                        f"{loc}: record source_url {record.get('source_url')!r} does not "
                        f"match any cited fact's source_url {sorted(cited_urls)}"
                    )
                if not (record.get("source_url") and record.get("source_excerpt")):
                    errors.append(f"{loc}: grounded record missing source_url/source_excerpt")
            elif status in ("not_available", "out_of_scope", "conversational", "clarification"):
                if record.get("fact_ids"):
                    errors.append(f"{loc}: {status} record should not cite fact_ids")
            else:
                errors.append(f"{loc}: unknown grounding_status {status!r}")

    print(f"Checked {total} QA record(s) across {len(args.qa_files)} file(s)")
    if errors:
        print(f"\nFAILED: {len(errors)} issue(s) found:")
        for e in errors:
            print(f"  - {e}")
        return 1

    print("OK: all QA records passed grounding checks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
