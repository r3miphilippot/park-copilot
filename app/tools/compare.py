"""Tool 3: live wait vs usual wait at this weekday/hour ("is it a good moment for X?")."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

from app.clock import now_paris
from app.config import ParkKey
from app.tools.common import ToolError, tool_guard
from app.tools.history import DataCoverage, get_history_store
from app.tools.live import get_live_wait_times

Verdict = Literal["shorter_than_usual", "usual", "longer_than_usual", "no_history"]


class RideComparison(BaseModel):
    ride: str
    land: str | None
    live_wait: int
    typical_avg: float | None
    typical_median: float | None
    delta_minutes: float | None  # live - typical average (negative = better than usual)
    days_observed: int
    verdict: Verdict


class LiveVsTypical(BaseModel):
    park: ParkKey
    at: datetime  # Paris time used for the comparison
    weekday: int  # 0=Monday
    hour: int
    coverage: DataCoverage
    rides: list[RideComparison]  # best opportunities first (most below usual)
    note: str | None = None


def classify(live: int, typical: float) -> Verdict:
    """A gap counts only if it exceeds 5 minutes AND 20 % of the usual wait.

    Both conditions avoid calling "5 min instead of 10" a big deal, or "65 instead of 60" news.
    """
    delta = live - typical
    if abs(delta) <= max(5.0, 0.2 * typical):
        return "usual"
    return "shorter_than_usual" if delta < 0 else "longer_than_usual"


@tool_guard
def compare_live_vs_typical(park: ParkKey) -> LiveVsTypical | ToolError:
    """Compare, for each OPEN ride, the current wait with the usual wait at this same
    weekday and hour. Use it in the park, today, to recommend what to do right now.

    Args:
        park: "disneyland_park" or "adventure_world".
    """
    live = get_live_wait_times(park)
    if isinstance(live, ToolError):
        return ToolError(tool="compare_live_vs_typical", error=live.error)

    now = now_paris()
    store = get_history_store()
    # "Usual" = previous days only: today's snapshots are the live data being judged.
    start_of_today = now.replace(hour=0, minute=0, second=0, microsecond=0)
    stats = store.ride_stats(
        park=park, ride=None, weekday=now.weekday(), hour=now.hour, before=start_of_today
    )
    typical = {s.ride_id: s for s in stats}

    rows = []
    for r in live.rides:
        if not r.is_open or r.wait_time is None:
            continue
        stats = typical.get(r.ride_id)
        rows.append(
            RideComparison(
                ride=r.ride,
                land=r.land,
                live_wait=r.wait_time,
                typical_avg=stats.avg_wait if stats else None,
                typical_median=stats.median_wait if stats else None,
                delta_minutes=round(r.wait_time - stats.avg_wait, 1) if stats else None,
                days_observed=stats.days_observed if stats else 0,
                verdict=classify(r.wait_time, stats.avg_wait) if stats else "no_history",
            )
        )
    # Rides without history go last; the others from "much better than usual" to "much worse".
    rows.sort(key=lambda c: (c.delta_minutes is None, c.delta_minutes or 0))

    note = None
    if not live.park_open:
        note = "The park is closed right now (no ride open)."
    elif not typical:
        note = "No history for this weekday and hour yet: only live waits can be used."
    return LiveVsTypical(
        park=park,
        at=now,
        weekday=now.weekday(),
        hour=now.hour,
        coverage=store.coverage(),
        rides=rows,
        note=note,
    )
