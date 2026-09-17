"""robots.txt and sitemap.xml handling."""

from __future__ import annotations

import urllib.robotparser
from urllib.parse import urljoin
from xml.etree import ElementTree as ET

import httpx


def load_robots(base_url: str, user_agent: str, timeout: float) -> urllib.robotparser.RobotFileParser:
    """Fetch and parse robots.txt. Returns a permissive parser (allow
    everything) if robots.txt is missing or unreadable, matching standard
    crawler behavior."""
    robots_url = urljoin(base_url, "/robots.txt")
    parser = urllib.robotparser.RobotFileParser()
    parser.set_url(robots_url)
    try:
        response = httpx.get(robots_url, timeout=timeout, headers={"User-Agent": user_agent})
        if response.status_code == 200:
            parser.parse(response.text.splitlines())
        else:
            parser.parse([])  # no robots.txt -> allow all
    except httpx.HTTPError:
        parser.parse([])
    return parser


def can_fetch(parser: urllib.robotparser.RobotFileParser, user_agent: str, url: str) -> bool:
    try:
        return parser.can_fetch(user_agent, url)
    except Exception:
        # A malformed robots.txt should not block the whole crawl.
        return True


def _parse_sitemap_xml(xml_text: str) -> tuple[list[str], list[str]]:
    """Parse one sitemap document. Returns (page_urls, nested_sitemap_urls)."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return [], []

    # Namespace-agnostic: strip "{...}" prefix from tags.
    def local(tag: str) -> str:
        return tag.rsplit("}", 1)[-1]

    page_urls: list[str] = []
    nested: list[str] = []

    if local(root.tag) == "sitemapindex":
        for sitemap_el in root:
            if local(sitemap_el.tag) != "sitemap":
                continue
            for child in sitemap_el:
                if local(child.tag) == "loc" and child.text:
                    nested.append(child.text.strip())
    elif local(root.tag) == "urlset":
        for url_el in root:
            if local(url_el.tag) != "url":
                continue
            for child in url_el:
                if local(child.tag) == "loc" and child.text:
                    page_urls.append(child.text.strip())

    return page_urls, nested


def discover_sitemap_urls(
    base_url: str,
    sitemap_paths: list[str],
    user_agent: str,
    timeout: float,
    max_nested: int = 20,
) -> list[str]:
    """Fetch known sitemap paths (recursing into sitemap indexes) and return
    every page URL listed. Returns an empty list if no sitemap is found -
    the caller should fall back to link discovery from the homepage."""
    headers = {"User-Agent": user_agent}
    all_page_urls: list[str] = []
    seen_sitemaps: set[str] = set()
    to_fetch = [urljoin(base_url, path) for path in sitemap_paths]

    while to_fetch and len(seen_sitemaps) < max_nested:
        sitemap_url = to_fetch.pop(0)
        if sitemap_url in seen_sitemaps:
            continue
        seen_sitemaps.add(sitemap_url)

        try:
            response = httpx.get(
                sitemap_url, timeout=timeout, headers=headers, follow_redirects=True
            )
        except httpx.HTTPError:
            continue
        if response.status_code != 200:
            continue

        pages, nested = _parse_sitemap_xml(response.text)
        all_page_urls.extend(pages)
        to_fetch.extend(n for n in nested if n not in seen_sitemaps)

    # de-duplicate while preserving order
    seen: set[str] = set()
    result = []
    for url in all_page_urls:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result
