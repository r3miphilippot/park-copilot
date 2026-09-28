"""Queue-Times client, shared by the collector and the live wait-time tool.

Data is unofficial and provided by https://queue-times.com (attribution required).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator
from typing import Any

import httpx

from app.config import QUEUE_TIMES_BASE_URL, Park

log = logging.getLogger(__name__)

HTTP_TIMEOUT_S = 10.0
USER_AGENT = "park-copilot (+https://github.com/r3miphilippot/park-copilot)"


def fetch_queue_times(
    client: httpx.Client, park: Park, *, attempts: int = 3, backoff_s: float = 2.0
) -> dict[str, Any]:
    """GET the live queue times of one park, retrying transient errors."""
    url = f"{QUEUE_TIMES_BASE_URL}/parks/{park.queue_times_id}/queue_times.json"
    for attempt in range(1, attempts + 1):
        try:
            response = client.get(url)
            response.raise_for_status()
            return response.json()
        except (httpx.TransportError, httpx.HTTPStatusError, ValueError) as exc:
            # A 4xx (except 429 "too many requests") will not fix itself: fail fast.
            if isinstance(exc, httpx.HTTPStatusError):
                status = exc.response.status_code
                if status < 500 and status != 429:
                    raise
            if attempt == attempts:
                raise
            log.warning("%s: attempt %d/%d failed (%s), retrying", park.key, attempt, attempts, exc)
            time.sleep(backoff_s * attempt)
    raise AssertionError("unreachable")


def iter_rides(payload: dict[str, Any]) -> Iterator[tuple[str | None, dict[str, Any]]]:
    """Yield (land_name, ride) pairs.

    Queue-Times puts rides in `lands[].rides[]` and sometimes also in a root `rides[]`
    (rides without a land). A ride listed twice is only yielded once.
    """
    seen: set[int] = set()
    sources = [(land.get("name"), land.get("rides", [])) for land in payload.get("lands", [])]
    sources.append((None, payload.get("rides", [])))
    for land_name, rides in sources:
        for ride in rides:
            if ride.get("id") is None or ride["id"] in seen:
                continue
            seen.add(ride["id"])
            yield land_name, ride
