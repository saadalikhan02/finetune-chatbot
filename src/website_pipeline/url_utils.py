"""URL normalization and filtering helpers.

Keeping this logic in one place is what makes the crawler avoid duplicate
pages (https vs http, trailing slash, www vs non-www, tracking params) and
avoid wandering off the Technyx site or into non-content URLs (mailto:,
admin paths, etc).
"""

from __future__ import annotations

from urllib.parse import urljoin, urlparse, urlunparse, parse_qsl, urlencode

NON_CONTENT_SCHEMES = {"mailto", "tel", "javascript", "data", "ftp"}

# Path prefixes that are never useful public content, regardless of what's
# configured in configs/website.yaml (kept as a hard-coded safety net).
DEFAULT_EXCLUDED_PREFIXES = (
    "/wp-admin",
    "/admin",
    "/login",
    "/logout",
    "/signin",
    "/signup",
    "/cdn-cgi",
    "/_next",
    "/api",
)

SOCIAL_MEDIA_HOSTS = (
    "facebook.com",
    "twitter.com",
    "x.com",
    "linkedin.com",
    "instagram.com",
    "youtube.com",
    "tiktok.com",
    "wa.me",
    "whatsapp.com",
)


def normalize_url(url: str, tracking_params: set[str] | None = None) -> str:
    """Normalize a URL for deduplication purposes.

    - lower-cases the scheme/host
    - forces https
    - drops a leading "www."
    - removes a trailing slash (except for the root "/")
    - drops the fragment
    - removes tracking query parameters, sorts remaining ones
    """
    tracking_params = tracking_params or set()
    parsed = urlparse(url)

    scheme = "https"
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]

    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]

    query_pairs = [
        (k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if k.lower() not in tracking_params
    ]
    query_pairs.sort()
    query = urlencode(query_pairs)

    return urlunparse((scheme, host, path, "", query, ""))


def resolve_link(base_url: str, href: str) -> str | None:
    """Resolve a possibly-relative href against a base URL. Returns None for
    non-content schemes (mailto:, tel:, javascript:, etc)."""
    href = href.strip()
    if not href or href.startswith("#"):
        return None

    parsed = urlparse(href)
    if parsed.scheme and parsed.scheme in NON_CONTENT_SCHEMES:
        return None

    return urljoin(base_url, href)


def is_internal(url: str, allowed_hosts: set[str]) -> bool:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    return host in {h.lower().removeprefix("www.") for h in allowed_hosts}


def is_social_or_external_utility(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return any(social in host for social in SOCIAL_MEDIA_HOSTS)


def is_excluded_path(url: str, excluded_prefixes: tuple[str, ...] = ()) -> bool:
    path = urlparse(url).path
    all_prefixes = tuple(DEFAULT_EXCLUDED_PREFIXES) + tuple(excluded_prefixes)
    return any(path.startswith(prefix) for prefix in all_prefixes)


def should_crawl(
    url: str,
    allowed_hosts: set[str],
    excluded_prefixes: tuple[str, ...] = (),
) -> bool:
    """Decide whether a discovered link should be queued for crawling."""
    if not url.startswith(("http://", "https://")):
        return False
    if not is_internal(url, allowed_hosts):
        return False
    if is_social_or_external_utility(url):
        return False
    if is_excluded_path(url, excluded_prefixes):
        return False
    return True
