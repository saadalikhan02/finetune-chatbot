"""Crawl orchestration: sitemap discovery + robots.txt-respecting BFS link
following, normalized/deduplicated, bounded by max_pages so a bug can't
runaway-crawl the live site.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from bs4 import BeautifulSoup

from .http_client import PoliteFetcher
from .models import RawPage
from .robots_sitemap import can_fetch, discover_sitemap_urls, load_robots
from .url_utils import normalize_url, resolve_link, should_crawl


@dataclass
class CrawlConfig:
    base_url: str
    allowed_hosts: set[str]
    sitemap_paths: list[str]
    excluded_path_prefixes: tuple[str, ...]
    tracking_query_params: set[str]
    user_agent: str
    request_timeout_seconds: float
    request_delay_seconds: float
    max_retries: int
    max_pages: int


def _extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    links = []
    for a in soup.find_all("a", href=True):
        resolved = resolve_link(base_url, a["href"])
        if resolved:
            links.append(resolved)
    return links


def crawl(config: CrawlConfig, crawl_id: str) -> list[RawPage]:
    robots = load_robots(config.base_url, config.user_agent, config.request_timeout_seconds)

    sitemap_urls = discover_sitemap_urls(
        config.base_url,
        config.sitemap_paths,
        config.user_agent,
        config.request_timeout_seconds,
    )

    seed_urls = [config.base_url] + sitemap_urls
    queue: list[str] = []
    queued_normalized: set[str] = set()
    for url in seed_urls:
        norm = normalize_url(url, config.tracking_query_params)
        if norm not in queued_normalized:
            queued_normalized.add(norm)
            queue.append(url)

    visited_normalized: set[str] = set()
    pages: list[RawPage] = []

    with PoliteFetcher(
        user_agent=config.user_agent,
        timeout_seconds=config.request_timeout_seconds,
        delay_seconds=config.request_delay_seconds,
        max_retries=config.max_retries,
    ) as fetcher:
        while queue and len(pages) < config.max_pages:
            url = queue.pop(0)
            norm = normalize_url(url, config.tracking_query_params)
            if norm in visited_normalized:
                continue
            visited_normalized.add(norm)

            if not should_crawl(url, config.allowed_hosts, config.excluded_path_prefixes):
                continue
            if not can_fetch(robots, config.user_agent, url):
                continue

            result = fetcher.get(url)
            links: list[str] = []
            if result.status_code == 200 and "text/html" in result.content_type:
                links = _extract_links(result.text, result.final_url)

            pages.append(
                RawPage(
                    url=url,
                    final_url=result.final_url,
                    status_code=result.status_code,
                    retrieved_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    crawl_id=crawl_id,
                    content_type=result.content_type,
                    html=result.text,
                    links=links,
                )
            )

            for link in links:
                link_norm = normalize_url(link, config.tracking_query_params)
                if link_norm in queued_normalized or link_norm in visited_normalized:
                    continue
                if not should_crawl(link, config.allowed_hosts, config.excluded_path_prefixes):
                    continue
                queued_normalized.add(link_norm)
                queue.append(link)

    return pages
