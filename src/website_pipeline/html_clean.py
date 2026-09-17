"""HTML -> structured, section-based content extraction.

This module is where most of the "don't hallucinate" engineering lives:
Technyx's site uses a few UI patterns that corrupt naive text extraction if
not handled explicitly:

- Animated "odometer" stat counters (clients/years/projects/markets) render
  every digit 0-9 in the DOM for a CSS transform animation; the *actual*
  resting number is only recoverable from a ``data-rest`` attribute, not
  from the visible text. See ``_decode_stat_counters``.
- Contact emails are Cloudflare-obfuscated (``__cf_email__`` spans / a
  ``/cdn-cgi/l/email-protection#<hex>`` link) - the real address is only
  recoverable by decoding a documented, deterministic XOR cipher. See
  ``_decode_cloudflare_email``.
- The header/nav/footer menu (Services/Industries/Platforms/...) repeats on
  every page and duplicate carousel/marquee content repeats within a page;
  both are stripped/deduplicated so they don't get extracted as "content".

Nothing here invents text - every string in a resulting section's content
list is copied verbatim (or, for stat counters/emails, mechanically decoded)
from the page's own HTML.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag

HEADING_TAGS = ["h1", "h2", "h3", "h4", "h5", "h6"]
# "div" is included because this site (a component-based Next.js build) uses
# plain <div>s for a lot of real content - e.g. FAQ accordion answers and
# office-address blocks are <div>s, not <p>/<li>. A div only counts as
# content if it's a *leaf* (see _is_leaf_content_tag): a div that itself
# contains another content tag is a layout wrapper and is skipped so its
# nested leaf is captured instead, avoiding double-counting whole sections.
LEAF_CONTENT_TAGS = ["p", "li", "blockquote", "dd", "dt", "td", "th", "figcaption", "div"]
FLOW_TAGS = HEADING_TAGS + LEAF_CONTENT_TAGS

BOILERPLATE_REMOVE_TAGS = ["script", "style", "noscript", "template", "svg", "form"]
BOILERPLATE_REMOVE_SEMANTIC = ["nav", "footer"]


@dataclass
class Section:
    heading: str
    level: int
    content: list[str] = field(default_factory=list)


@dataclass
class CleanedPage:
    url: str
    canonical_url: str | None
    title: str
    meta_description: str | None
    sections: list[Section]
    structured_data: dict
    internal_links: list[str]


def _decode_cloudflare_email(hex_string: str) -> str | None:
    """Decode Cloudflare's email-obfuscation cipher: the first byte is an
    XOR key applied to every subsequent byte. Publicly documented scheme,
    not a guess - this recovers exactly what the site published."""
    try:
        raw = bytes.fromhex(hex_string)
    except ValueError:
        return None
    if len(raw) < 2:
        return None
    key = raw[0]
    return "".join(chr(b ^ key) for b in raw[1:])


def _apply_cloudflare_email_decoding(soup: BeautifulSoup) -> None:
    for span in soup.select(".__cf_email__[data-cfemail]"):
        decoded = _decode_cloudflare_email(span["data-cfemail"])
        if decoded:
            span.replace_with(decoded)

    for link in soup.find_all("a", href=True):
        if "cdn-cgi/l/email-protection" not in link["href"]:
            continue
        hex_part = link["href"].rsplit("#", 1)[-1]
        decoded = _decode_cloudflare_email(hex_part)
        if decoded:
            link["href"] = f"mailto:{decoded}"
            if not link.get_text(strip=True) or "protected" in link.get_text().lower():
                link.string = decoded


def _decode_stat_counters(soup: BeautifulSoup) -> list[str]:
    """Find animated digit-reel stat blocks and decode them to plain
    strings like "30+ Clients". Returns the deduplicated list (in first-seen
    order) and removes the raw reel markup from the tree so it isn't picked
    up (garbled) by normal text extraction."""
    stats: list[str] = []
    seen: set[str] = set()

    for stat_item in soup.select('[class*="statItem"]'):
        reels = stat_item.select('[data-reel="true"][data-rest]')
        if not reels:
            continue
        try:
            digits = "".join(str(int(reel["data-rest"]) % 10) for reel in reels)
        except (ValueError, KeyError):
            continue

        has_plus = "+" in stat_item.get_text()
        label_el = stat_item.select_one('[class*="statLabel"]')
        label = label_el.get_text(strip=True) if label_el else ""

        text = f"{digits}{'+' if has_plus else ''} {label}".strip()
        if text and text not in seen:
            seen.add(text)
            stats.append(text)

        stat_item.decompose()

    return stats


def _extract_structured_data(soup: BeautifulSoup) -> dict:
    """Pull schema.org JSON-LD blocks the site itself published (Organization
    name, official social links, page description). This is first-party
    structured metadata, not an inference."""
    result: dict = {}
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(script.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            schema_type = entry.get("@type")
            if schema_type == "Organization":
                result["organization_name"] = entry.get("name")
                result["social_links"] = entry.get("sameAs", [])
            elif schema_type == "WebPage":
                result.setdefault("page_description", entry.get("description"))
    return result


def _is_leaf_content_tag(tag: Tag) -> bool:
    """True if this content tag has no nested content tag that itself holds
    real text (so we capture text at the innermost level and don't
    double-count). A nested *empty* div/span (e.g. an icon wrapper sitting
    next to a text-bearing sibling <span>) does not disqualify the parent -
    only a nested tag that would itself be captured separately does."""
    return not any(nested.get_text(strip=True) for nested in tag.find_all(LEAF_CONTENT_TAGS))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_html(html: str, url: str) -> CleanedPage:
    soup = BeautifulSoup(html, "lxml")

    # Must run before script tags are stripped below (JSON-LD lives in
    # <script type="application/ld+json">).
    structured_data = _extract_structured_data(soup)

    for tag_name in BOILERPLATE_REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title_tag = soup.find("title")
    title = _clean_text(title_tag.get_text(separator=" ")) if title_tag else ""

    meta_desc_tag = soup.find("meta", attrs={"name": "description"})
    meta_description = (
        _clean_text(meta_desc_tag["content"])
        if meta_desc_tag and meta_desc_tag.get("content")
        else structured_data.get("page_description")
    )

    canonical_tag = soup.find("link", rel="canonical")
    canonical_url = canonical_tag["href"] if canonical_tag and canonical_tag.get("href") else None

    # Collect internal links before stripping nav/footer, since footer links
    # (e.g. to service/industry/platform pages) are legitimate crawl targets
    # even though footer *text* is boilerplate we don't want as content.
    internal_links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href and not href.startswith("#"):
            internal_links.append(urljoin(url, href))

    _apply_cloudflare_email_decoding(soup)
    stat_strings = _decode_stat_counters(soup)

    for tag_name in BOILERPLATE_REMOVE_SEMANTIC:
        for tag in soup.find_all(tag_name):
            tag.decompose()
    header_tag = soup.find("header")
    if header_tag:
        header_tag.decompose()

    content_root = soup.find("main") or soup.body or soup

    sections: list[Section] = []
    current = Section(heading=title or "Overview", level=0)
    sections.append(current)

    for tag in content_root.find_all(FLOW_TAGS):
        if tag.name in HEADING_TAGS:
            heading_text = _clean_text(tag.get_text(separator=" "))
            if not heading_text:
                continue
            current = Section(heading=heading_text, level=int(tag.name[1]))
            sections.append(current)
            continue

        if not _is_leaf_content_tag(tag):
            continue
        text = _clean_text(tag.get_text(separator=" "))
        if text and text not in current.content:
            current.content.append(text)

    if stat_strings:
        sections.append(Section(heading="Key Statistics", level=0, content=stat_strings))

    # Drop empty sections, then drop exact-duplicate (heading, content)
    # sections caused by duplicated carousel/marquee markup.
    sections = [s for s in sections if s.content]
    deduped: list[Section] = []
    seen_sections: set[tuple[str, tuple[str, ...]]] = set()
    for s in sections:
        key = (s.heading, tuple(s.content))
        if key in seen_sections:
            continue
        seen_sections.add(key)
        deduped.append(s)

    return CleanedPage(
        url=url,
        canonical_url=canonical_url,
        title=title,
        meta_description=meta_description,
        sections=deduped,
        structured_data=structured_data,
        internal_links=internal_links,
    )
