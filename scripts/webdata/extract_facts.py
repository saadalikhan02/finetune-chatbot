#!/usr/bin/env python3
"""Extract atomic, classified, source-traceable facts from cleaned website
documents.

Every fact is either a verbatim string from the page, or (for the "Our
Locations" / "Close to Our Clients" blocks) a mechanical join of verbatim
strings under their own verbatim country label - never an LLM paraphrase
and never an inferred claim. See src/website_pipeline/fact_rules.py for the
exact, auditable rules used.

Usage:
    python scripts/webdata/extract_facts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.config import load_raw_website_config  # noqa: E402
from website_pipeline.fact_rules import (  # noqa: E402
    LEGAL_BOILERPLATE_PATHS,
    classify_fact_type,
    is_location_block_heading,
    is_testimonial_block_heading,
    merge_location_block,
    merge_testimonial_block,
    passes_generic_quality_filter,
)
from website_pipeline.models import Fact, fact_id_for, write_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean-dir", default="data/website/clean", type=Path)
    parser.add_argument("--website-config", default="configs/website.yaml", type=Path)
    parser.add_argument("--output", default="data/knowledge/facts.jsonl", type=Path)
    parser.add_argument(
        "--include-legal-pages",
        action="store_true",
        help="Also extract facts from legal boilerplate pages (privacy policy, "
        "terms & conditions, cookie policy). Off by default: this content is "
        "standard legal text, not FAQ-style company information, and mostly "
        "adds noise to the fact bank.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    website_config = load_raw_website_config(args.website_config)
    primary_language = website_config.get("primary_language", "en")

    clean_paths = sorted(args.clean_dir.glob("*.json"))
    if not clean_paths:
        print(f"ERROR: no clean documents found in {args.clean_dir}", file=sys.stderr)
        return 1

    facts: list[Fact] = []
    skipped_non_primary_lang = 0
    skipped_legal = 0

    for path in clean_paths:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["language"] != primary_language:
            skipped_non_primary_lang += 1
            continue

        url_path = doc["url"].split("technyxsystems.com", 1)[-1] or "/"
        if url_path in LEGAL_BOILERPLATE_PATHS and not args.include_legal_pages:
            skipped_legal += 1
            continue

        for section in doc["sections"]:
            heading = section["heading"]
            content = section["content"]

            if is_location_block_heading(heading) or is_testimonial_block_heading(heading):
                merged_items = (
                    merge_location_block(content)
                    if is_location_block_heading(heading)
                    else merge_testimonial_block(content)
                )
                for item in merged_items:
                    fact_type = classify_fact_type(doc["url"], heading, item)
                    facts.append(
                        Fact(
                            fact_id=fact_id_for(doc["url"], heading, item),
                            fact=item,
                            fact_type=fact_type,
                            source_url=doc["url"],
                            source_page_title=doc["title"],
                            source_section=heading,
                            source_excerpt=item,
                            confidence="explicit",
                        )
                    )
                continue

            for item in content:
                if heading != "Key Statistics" and not passes_generic_quality_filter(item):
                    continue
                fact_type = classify_fact_type(doc["url"], heading, item)
                facts.append(
                    Fact(
                        fact_id=fact_id_for(doc["url"], heading, item),
                        fact=item,
                        fact_type=fact_type,
                        source_url=doc["url"],
                        source_page_title=doc["title"],
                        source_section=heading,
                        source_excerpt=item,
                        confidence="explicit",
                    )
                )

    # De-duplicate identical facts that appear verbatim on more than one
    # page (e.g. the same stat block on both "/" and "/company").
    seen_ids: set[str] = set()
    deduped: list[Fact] = []
    for f in facts:
        if f.fact_id in seen_ids:
            continue
        seen_ids.add(f.fact_id)
        deduped.append(f)

    write_jsonl(args.output, deduped)

    type_counts: dict[str, int] = {}
    for f in deduped:
        type_counts[f.fact_type] = type_counts.get(f.fact_type, 0) + 1

    print(
        f"Extracted {len(deduped)} fact(s) from "
        f"{len(clean_paths) - skipped_non_primary_lang - skipped_legal} document(s)"
    )
    if skipped_non_primary_lang:
        print(f"Skipped {skipped_non_primary_lang} non-'{primary_language}' document(s)")
    if skipped_legal:
        print(f"Skipped {skipped_legal} legal boilerplate document(s) (use --include-legal-pages to include)")
    print("Fact type counts:")
    for ftype, count in sorted(type_counts.items(), key=lambda kv: -kv[1]):
        print(f"  {ftype:15s} {count}")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
