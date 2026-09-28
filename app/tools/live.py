"""Tool 1: live wait times from Queue-Times (cached 5 minutes)."""

from __future__ import annotations

from datetime import datetime

import httpx
from pydantic import BaseModel

from app.clock import now_paris
from app.config import PARKS, ParkKey
from app.queue_times import HTTP_TIMEOUT_S, USER_AGENT, fetch_queue_times, iter_rides
from app.tools.common import ToolError, TTLCache, tool_guard

# Queue-Times itself refreshes about every 5 minutes: caching longer would show stale data,
# caching shorter would only hammer their free API.
_cache = TTLCache(ttl_s=5 * 60)


class RideLiveWait(BaseModel):
    ride: str
    ride_id: int
    land: str | None
    is_open: bool
    wait_time: int | None  # minutes, None when the ride is closed


class LiveWaitTimes(BaseModel):
    park: ParkKey
    park_name: str
    fetched_at: datetime  # Paris time
    park_open: bool  # False when every ride is closed
    rides: list[RideLiveWait]  # open rides first, shortest wait first
    source: str = "Queue-Times.com (unofficial data)"


@tool_guard
def get_live_wait_times(park: ParkKey) -> LiveWaitTimes | ToolError:
    """Get the CURRENT wait times (in minutes) of every ride in one park.

    Only meaningful today, while the park is open. Never use it to plan a future date:
    use get_typical_wait for that.

    Args:
        park: "disneyland_park" or "adventure_world".
    """
    cached = _cache.get(park)
    if cached is not None:
        return cached

    with httpx.Client(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}) as client:
        # Short retry: a user is waiting for the answer.
        payload = fetch_queue_times(client, PARKS[park], attempts=2, backoff_s=0.5)

    rides = [
        RideLiveWait(
            ride=ride.get("name", "?"),
            ride_id=ride["id"],
            land=land,
            is_open=bool(ride.get("is_open")),
            wait_time=ride.get("wait_time") if ride.get("is_open") else None,
        )
        for land, ride in iter_rides(payload)
    ]
    rides.sort(key=lambda r: (not r.is_open, r.wait_time or 0, r.ride))
    result = LiveWaitTimes(
        park=park,
        park_name=PARKS[park].name,
        fetched_at=now_paris(),
        park_open=any(r.is_open for r in rides),
        rides=rides,
    )
    _cache.set(park, result)
    return result
