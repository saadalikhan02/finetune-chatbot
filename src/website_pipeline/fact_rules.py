"""Heuristics for turning cleaned page sections into atomic, classified
facts - deliberately simple, deterministic, and auditable rather than
"smart": every fact this module produces is a verbatim (or, for the location
blocks, a mechanical join of verbatim) string taken directly from a crawled
page. Nothing is paraphrased by an LLM and nothing is inferred.
"""

from __future__ import annotations

import re

LEGAL_BOILERPLATE_PATHS = {"/privacy-policy", "/terms-conditions", "/cookie-policy"}

# Generic call-to-action / widget boilerplate that can appear right after a
# real content block (e.g. a floating "Start a Conversation" button after
# the office-locations list) with no heading of its own to separate it.
# Dropped outright rather than merged into whatever fact precedes it.
CTA_BOILERPLATE = {
    "Start a Conversation",
    "Start the conversation",
    "Get in Touch",
    "Learn More",
    "Contact Us",
    "Explore Related Capabilities",
}

FACT_TYPES = (
    "company",
    "service",
    "technology",
    "location",
    "contact",
    "portfolio",
    "client",
    "partner",
    "industry",
    "capability",
    "company_value",
    "history",
    "other",
)

KNOWN_COUNTRIES = {"UAE", "Pakistan", "USA", "Australia"}

LOCATION_BLOCK_HEADINGS = {"our locations"}
LOCATION_BLOCK_HEADING_SUBSTRINGS = ("close to our clients",)

TESTIMONIAL_BLOCK_HEADING_SUBSTRINGS = ("trusted because",)

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(\+?\d[\d\-\s]{7,}\d)")

SERVICE_URL_PREFIX = "/services"
PLATFORM_URL_PREFIX = "/platforms"
INDUSTRY_URL_PREFIX = "/industries"

SERVICE_NAMES = (
    "product & platform engineering",
    "frontend engineering",
    "digital experience platform",
    "digital experience platforms",
    "devops & cloud infrastructure",
    "ai consulting & automation",
    "creative & campaign operations",
)

VALUE_HEADINGS = (
    "own the outcome",
    "say what's true",
    "say what’s true",
    "raise the bar",
    "ai-native by default",
    "senior-led delivery",
    "built for deadlines",
    "academic meets practice",
)


def is_location_block_heading(heading: str) -> bool:
    h = heading.strip().lower()
    return h in LOCATION_BLOCK_HEADINGS or any(sub in h for sub in LOCATION_BLOCK_HEADING_SUBSTRINGS)


def merge_location_block(content: list[str]) -> list[str]:
    """Group a flat ['UAE', 'address line', 'phone', 'Pakistan', ...] list
    into one merged string per country. Country names are the only
    recognized delimiter; everything between one country name and the next
    is joined verbatim (in order) onto that country's fact."""
    groups: list[list[str]] = []
    for item in content:
        if item.strip() in CTA_BOILERPLATE:
            continue
        if item.strip() in KNOWN_COUNTRIES:
            groups.append([item.strip()])
        elif groups:
            groups[-1].append(item)
        # else: content before the first recognized country name is dropped
        # (nothing meaningful should precede it in this section).

    return [": ".join([g[0], " ".join(g[1:])]) if len(g) > 1 else g[0] for g in groups]


def is_testimonial_block_heading(heading: str) -> bool:
    h = heading.strip().lower()
    return any(sub in h for sub in TESTIMONIAL_BLOCK_HEADING_SUBSTRINGS)


def _is_attribution_line(text: str) -> bool:
    """Testimonial attributions are short "Name Title – Company" lines;
    the quotes they follow are always much longer sentences."""
    return len(text) < 70 and (" – " in text or " - " in text)


def merge_testimonial_block(content: list[str]) -> list[str]:
    """Pair each testimonial quote with the attribution line that follows
    it (name, title, and company - all verbatim from the page). Lines that
    don't form a quote/attribution pair (e.g. the section's intro sentence)
    are dropped rather than guessed at."""
    facts: list[str] = []
    pending_quote: str | None = None
    for item in content:
        if _is_attribution_line(item):
            if pending_quote is not None:
                facts.append(f"“{pending_quote}” — {item}")
                pending_quote = None
        else:
            pending_quote = item
    return facts


def passes_generic_quality_filter(text: str) -> bool:
    """Drop short UI labels/eyebrow text (e.g. "Get in Touch", "Why
    Technyx") that aren't meaningful standalone facts, while keeping real
    sentences and list items."""
    if text.strip() in CTA_BOILERPLATE:
        return False
    word_count = len(text.split())
    if word_count == 1:
        # A single token (e.g. "React", "Sitecore", "TypeScript") is almost
        # always a real product/technology name pulled from a tag/chip list,
        # not a UI label - those are multi-word ("Get in Touch").
        return len(text) >= 2
    return word_count >= 3 and len(text) >= 15


def classify_fact_type(url: str, heading: str, text: str) -> str:
    path = url.split("technyxsystems.com", 1)[-1]
    heading_l = heading.lower()
    text_l = text.lower()

    if heading_l == "key statistics":
        return "company"
    if is_testimonial_block_heading(heading):
        return "client"
    if is_location_block_heading(heading) or EMAIL_RE.search(text) or "our locations" in heading_l:
        if any(c.lower() in text_l for c in KNOWN_COUNTRIES) or any(
            city in text_l for city in ("dubai", "karachi", "mckinney", "sydney")
        ):
            return "location"
    if EMAIL_RE.search(text) or PHONE_RE.search(text) or "contact" in heading_l:
        return "contact"
    if path.startswith(SERVICE_URL_PREFIX) or any(name in heading_l for name in SERVICE_NAMES):
        return "service"
    if path.startswith(PLATFORM_URL_PREFIX) or heading_l in (
        "sitecore", "umbraco", "sitefinity", "kentico", "optimizely", "sanity", "strapi", "payload",
    ):
        return "technology"
    if "technology ecosystem" in heading_l or heading_l in ("react", "frontend"):
        return "technology"
    if path.startswith(INDUSTRY_URL_PREFIX):
        return "industry"
    if heading_l in VALUE_HEADINGS or "value" in heading_l:
        return "company_value"
    if heading_l in ("mission", "vision"):
        return "company_value"
    if any(k in text_l for k in ("started in", "since 20", "decade", "founded", "years of delivery")):
        return "history"
    if path in ("/", "/company") and heading_l in ("faqs", "overview"):
        return "company"
    if path.startswith("/agencies") or path.startswith("/brands") or path.startswith("/startups"):
        return "capability"
    return "other"
