#!/usr/bin/env python3
"""Crawl the Technyx Systems public website and store raw page snapshots.

Discovers pages via sitemap.xml (recursing into sitemap indexes) plus BFS
link-following from the homepage, respects robots.txt, normalizes URLs to
avoid duplicates, and stays within the configured domain. Does not use a
browser - Technyx's pages are server-rendered HTML.

Each run gets its own crawl_id (UTC timestamp) and is written to its own
subdirectory under data/website/raw/ - previous crawls are never
overwritten, so you can compare snapshots over time.

Usage:
    python scripts/webdata/crawl_site.py --config configs/website.yaml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from website_pipeline.config import load_crawl_config  # noqa: E402
from website_pipeline.crawler import crawl  # noqa: E402
from website_pipeline.models import write_jsonl  # noqa: E402
from website_pipeline.url_utils import normalize_url  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/website.yaml", type=Path)
    parser.add_argument(
        "--output-root",
        default="data/website/raw",
        type=Path,
        help="Base directory; a new <crawl_id>/ subdirectory is created per run",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_crawl_config(args.config)

    crawl_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_root / crawl_id
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Crawl ID: {crawl_id}")
    print(f"Base URL: {config.base_url}")
    print(f"Allowed hosts: {sorted(config.allowed_hosts)}")
    print(f"Max pages: {config.max_pages}")
    print("Crawling...\n")

    pages = crawl(config, crawl_id)

    write_jsonl(output_dir / "pages.jsonl", pages)

    status_counts: dict[int, int] = {}
    for page in pages:
        status_counts[page.status_code] = status_counts.get(page.status_code, 0) + 1

    manifest = {
        "crawl_id": crawl_id,
        "started_at": crawl_id,
        "base_url": config.base_url,
        "num_pages": len(pages),
        "status_code_counts": status_counts,
        "urls": sorted({normalize_url(p.url, config.tracking_query_params) for p in pages}),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Crawled {len(pages)} page(s).")
    print(f"Status codes: {status_counts}")
    print(f"Raw pages written to: {output_dir / 'pages.jsonl'}")
    print(f"Manifest written to:  {output_dir / 'manifest.json'}")

    ok_pages = [p for p in pages if p.status_code != 200]
    if ok_pages:
        print(f"\nWARNING: {len(ok_pages)} page(s) did not return HTTP 200:")
        for p in ok_pages:
            print(f"  - {p.status_code}: {p.url}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
