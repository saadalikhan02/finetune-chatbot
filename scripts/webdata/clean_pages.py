#!/usr/bin/env python3
"""Turn a raw crawl snapshot into structured, per-page clean documents.

Reads data/website/raw/<crawl_id>/pages.jsonl (defaults to the most recent
crawl), extracts sections (heading + content list), decodes stat counters
and obfuscated emails, detects language, and writes one JSON file per page
to data/website/clean/<document_id>.json.

Usage:
    python scripts/webdata/clean_pages.py
    python scripts/webdata/clean_pages.py --crawl-id 20260917T090842Z
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.html_clean import clean_html  # noqa: E402
from website_pipeline.lang import detect_language  # noqa: E402
from website_pipeline.models import CleanDocument, CleanSection, document_id_for, read_jsonl  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data/website/raw", type=Path)
    parser.add_argument("--crawl-id", default=None, help="Defaults to the most recent crawl")
    parser.add_argument("--output-dir", default="data/website/clean", type=Path)
    return parser.parse_args()


def latest_crawl_id(raw_root: Path) -> str:
    crawl_dirs = sorted(d.name for d in raw_root.iterdir() if d.is_dir())
    if not crawl_dirs:
        raise FileNotFoundError(f"No crawl snapshots found under {raw_root}")
    return crawl_dirs[-1]


def main() -> int:
    args = parse_args()
    crawl_id = args.crawl_id or latest_crawl_id(args.raw_root)
    pages_path = args.raw_root / crawl_id / "pages.jsonl"

    print(f"Reading raw pages from: {pages_path}")
    raw_pages = read_jsonl(pages_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = []
    for raw in raw_pages:
        if raw["status_code"] != 200 or "text/html" not in raw.get("content_type", ""):
            skipped.append((raw["url"], raw["status_code"]))
            continue

        page = clean_html(raw["html"], raw["final_url"])
        canonical = page.canonical_url or page.url
        doc_id = document_id_for(canonical)

        full_text = " ".join(
            " ".join(s.content) for s in page.sections
        ) + " " + page.title
        language, confidence = detect_language(full_text)

        doc = CleanDocument(
            document_id=doc_id,
            url=page.url,
            canonical_url=page.canonical_url,
            title=page.title,
            meta_description=page.meta_description,
            language=language,
            language_confidence=confidence,
            sections=[CleanSection(s.heading, s.level, s.content) for s in page.sections],
            structured_data=page.structured_data,
            retrieved_at=raw["retrieved_at"],
            crawl_id=crawl_id,
        )

        out_path = args.output_dir / f"{doc_id}.json"
        out_path.write_text(json.dumps(asdict(doc), indent=2, ensure_ascii=False), encoding="utf-8")
        written += 1

    print(f"Wrote {written} clean document(s) to {args.output_dir}")
    if skipped:
        print(f"Skipped {len(skipped)} non-HTML/non-200 page(s):")
        for url, status in skipped:
            print(f"  - {status}: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
