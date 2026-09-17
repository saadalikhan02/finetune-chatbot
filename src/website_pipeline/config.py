"""Loads configs/website.yaml into a typed CrawlConfig."""

from __future__ import annotations

from pathlib import Path

import yaml

from .crawler import CrawlConfig


def load_crawl_config(path: str | Path) -> CrawlConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return CrawlConfig(
        base_url=raw["base_url"],
        allowed_hosts=set(raw["allowed_hosts"]),
        sitemap_paths=list(raw["sitemap_paths"]),
        excluded_path_prefixes=tuple(raw.get("excluded_path_prefixes", [])),
        tracking_query_params=set(raw.get("tracking_query_params", [])),
        user_agent=raw["user_agent"],
        request_timeout_seconds=float(raw["request_timeout_seconds"]),
        request_delay_seconds=float(raw["request_delay_seconds"]),
        max_retries=int(raw["max_retries"]),
        max_pages=int(raw["max_pages"]),
    )


def load_raw_website_config(path: str | Path) -> dict:
    """Return the raw dict too, for fields CrawlConfig doesn't carry
    (primary_language etc), used by later pipeline stages."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
