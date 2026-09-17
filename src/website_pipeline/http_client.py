"""A small, polite HTTP fetcher: fixed user-agent, timeout, retries, and an
inter-request delay. Plain requests/httpx only - no browser rendering.
Technyx's pages are server-rendered (verified: page text is present in the
raw HTML), so Playwright is not needed for this site.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx


@dataclass
class FetchResult:
    url: str
    final_url: str
    status_code: int
    content_type: str
    text: str
    error: str | None = None


class PoliteFetcher:
    def __init__(
        self,
        user_agent: str,
        timeout_seconds: float = 20.0,
        delay_seconds: float = 0.5,
        max_retries: int = 2,
    ) -> None:
        self.user_agent = user_agent
        self.timeout_seconds = timeout_seconds
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self._client = httpx.Client(
            timeout=timeout_seconds,
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "PoliteFetcher":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def get(self, url: str) -> FetchResult:
        last_error: str | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.get(url)
                return FetchResult(
                    url=url,
                    final_url=str(response.url),
                    status_code=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                    text=response.text,
                )
            except httpx.HTTPError as e:
                last_error = str(e)
                if attempt < self.max_retries:
                    time.sleep(self.delay_seconds * (attempt + 1))
            finally:
                time.sleep(self.delay_seconds)

        return FetchResult(
            url=url, final_url=url, status_code=0, content_type="", text="", error=last_error
        )
