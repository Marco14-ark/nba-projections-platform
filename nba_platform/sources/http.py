"""HTTP fetcher with retries, backoff and raw-payload archival.

Every successful response is archived (content-addressed) before parsing, so any forecast can
later be reconstructed from exactly the bytes that were available at the time.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import requests

from ..core.timeutil import utcnow

DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) nba-platform/0.1",
    "Accept": "application/json, application/pdf, */*",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
}


class FetchError(RuntimeError):
    pass


@dataclass
class FetchResult:
    url: str
    status: int
    body: bytes
    fetched_at: str


class Fetcher:
    def __init__(self, session: requests.Session | None = None, retries: int = 3,
                 backoff_s: float = 1.5, timeout_s: float = 20.0,
                 archive: Callable[[str, str, bytes, str], str] | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.s = session or requests.Session()
        self.retries, self.backoff_s, self.timeout_s = retries, backoff_s, timeout_s
        self.archive, self.sleep = archive, sleep

    def get(self, url: str, source: str, headers: dict | None = None,
            params: dict | None = None, ok_404: bool = False) -> FetchResult | None:
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                r = self.s.get(url, headers={**DEFAULT_HEADERS, **(headers or {})},
                               params=params, timeout=self.timeout_s)
                if r.status_code == 404 and ok_404:
                    return None
                if r.status_code in (429, 500, 502, 503, 504):
                    raise FetchError(f"{r.status_code} {url}")
                if r.status_code >= 400:
                    raise FetchError(f"non-retryable {r.status_code} {url}")
                res = FetchResult(url, r.status_code, r.content, utcnow().isoformat())
                if self.archive:
                    self.archive(source, url, res.body, res.fetched_at)
                return res
            except FetchError as e:
                if "non-retryable" in str(e):
                    raise
                last = e
            except requests.RequestException as e:
                last = e
            if attempt < self.retries:
                self.sleep(self.backoff_s * (2 ** attempt))
        raise FetchError(f"failed after {self.retries + 1} attempts: {url}: {last}")
