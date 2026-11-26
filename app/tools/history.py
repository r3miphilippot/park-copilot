"""Tool 2: typical (historical) wait times, aggregated in MongoDB.

The data access sits behind a small `HistoryStore` interface: MongoDB in production, an
in-memory fake in unit tests and frozen fixtures in evals.

Seasonality: a queue in July is not a queue in November (school holidays, opening hours,
weather). MongoDB averages each ride per DAY, then every day counts with a weight (see
`day_weight`): high for the days close to the visit (the last weeks, for a visit soon) and
for the same period of previous years, low for the rest. Weighting in Python keeps the
queries simple and the maths unit-tested.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, Field
from pymongo.collection import Collection

from app.clock import now_paris
from app.config import ParkKey
from app.db import get_snapshots_collection
from app.tools.common import ToolError, TTLCache, tool_guard

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

RECENT_HALF_LIFE_DAYS = 30  # 30 days away from the visit: the day counts half
SEASON_HALF_LIFE_DAYS = 15  # same date last year: full weight; 15 days apart: half
HISTORY_WINDOW_DAYS = 400  # a year back is enough for "the same period last year"
CROWD_HOURS = (11, 17)  # daily crowd index: popular rides, 11:00-17:59 (the busy hours)


def day_weight(day: date, target: date) -> float:
    """How much a day of history tells about the visit day `target`, between 0 and 1.

    The best of two signals: closeness in time (for a visit next week, the last weeks) and
    season (the same time of year, any year). For a visit next July, last July counts fully
    and last month almost nothing.
    """
    closeness = 0.5 ** (abs((target - day).days) / RECENT_HALF_LIFE_DAYS)
    return max(closeness, season_weight(day, target))


def season_weight(day: date, target: date) -> float:
    gap = abs(day.timetuple().tm_yday - target.timetuple().tm_yday)
    return 0.5 ** (min(gap, 365 - gap) / SEASON_HALF_LIFE_DAYS)


def weighted_mean(values: list[tuple[float, float]]) -> float:
    """(value, weight) pairs."""
    total = sum(w for _, w in values)
    return sum(v * w for v, w in values) / total


def weighted_median(values: list[tuple[float, float]]) -> float:
    ordered = sorted(values)
    half, running = sum(w for _, w in ordered) / 2, 0.0
    for value, weight in ordered:
        running += weight
        if running >= half:
            return value
    return ordered[-1][0]


class RideStats(BaseModel):
    ride: str
    ride_id: int
    park: str
    land: str | None
    avg_wait: float  # minutes, recent weeks and same season weigh more
    median_wait: float  # minutes, median of the daily averages (same weights)
    observations: int  # number of 15-minute snapshots behind the numbers
    days_observed: int  # distinct days those snapshots come from


class DataCoverage(BaseModel):
    days: int  # distinct days with at least one snapshot, whole database
    first_day: date | None
    last_day: date | None


class TypicalWait(BaseModel):
    filters: dict[str, str | int]
    # "exact" = the requested filters; otherwise they were widened because the requested
    # weekday has no data yet (young history): "same_hour_all_days" or "all_days".
    basis: str = "exact"
    coverage: DataCoverage
    rides: list[RideStats]  # longest average wait first
    weighting: str = "recent weeks and the same period of previous years weigh more"
    note: str | None = None


class HourlyProfile(BaseModel):
    """Usual wait per ride and per hour of the day, for one park (used by the day planner)."""

    park: str
    basis: Literal["same_weekday", "all_days", "none"]
    days: int  # distinct days behind the numbers
    waits: dict[int, dict[int, float]]  # ride_id -> {hour: average wait in minutes}
    # Days of history from the same period of the year as the visit (weighted, 0 = none):
    # a low value for a far date means the waits come from another season.
    season_days: float = 0.0


class HistoryStore(Protocol):
    def ride_stats(
        self,
        *,
        park: str | None,
        ride: str | None,
        weekday: int | None,
        hour: int | None,
        before: datetime | None = None,
    ) -> list[RideStats]: ...

    def coverage(self) -> DataCoverage: ...

    def hourly_profile(
        self, *, park: str, weekday: int | None, target: date | None = None
    ) -> HourlyProfile: ...

    def daily_index(self, *, park: str, ride_ids: list[int]) -> dict[date, float]: ...


# Day of a snapshot, in Paris time (a visit at 00:30 UTC in summer belongs to the next day).
_PARIS_DAY = {
    "$dateToString": {"format": "%Y-%m-%d", "date": "$fetched_at", "timezone": "Europe/Paris"}
}


def _day(value: str) -> date:
    return date.fromisoformat(value)


class MongoHistoryStore:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        # Counting days scans the whole collection: cache it, it only changes once a day.
        self._coverage_cache = TTLCache(ttl_s=15 * 60)
        # Per-day rows of the planner and of the crowd calendar (weights applied afterwards).
        self._rows_cache = TTLCache(ttl_s=30 * 60)

    @staticmethod
    def build_stats_pipeline(
        *,
        park: str | None,
        ride: str | None,
        weekday: int | None,
        hour: int | None,
        before: datetime | None = None,
    ) -> list[dict]:
        """One row per ride and per Paris day: the day's average and number of snapshots."""
        # Only open rides with a real wait count (closed rides are stored with wait_time=None).
        match: dict = {"is_open": True, "wait_time": {"$ne": None}}
        if before is not None:
            # Lets "usual" exclude today: otherwise live waits are compared with themselves.
            match["fetched_at"] = {"$lt": before}
        if park:
            match["park"] = park
        if ride:
            # Case-insensitive partial match: "pirates" finds "Pirates of the Caribbean".
            match["ride"] = {"$regex": re.escape(ride), "$options": "i"}
        if weekday is not None:
            match["weekday"] = weekday
        if hour is not None:
            match["hour"] = hour
        return [
            {"$match": match},
            {
                "$group": {
                    "_id": {"ride_id": "$ride_id", "day": _PARIS_DAY},
                    "ride": {"$last": "$ride"},
                    "park": {"$last": "$park"},
                    "land": {"$last": "$land"},
                    "avg": {"$avg": "$wait_time"},
                    "n": {"$sum": 1},
                }
            },
        ]

    def ride_stats(self, **filters) -> list[RideStats]:
        today = now_paris().date()
        by_ride: dict[int, list[dict]] = defaultdict(list)
        for row in self.collection.aggregate(self.build_stats_pipeline(**filters)):
            by_ride[row["_id"]["ride_id"]].append(row)
        stats = []
        for ride_id, days in by_ride.items():
            values = [(row["avg"], day_weight(_day(row["_id"]["day"]), today)) for row in days]
            last = max(days, key=lambda row: row["_id"]["day"])  # most recent name and land
            stats.append(
                RideStats(
                    ride=last["ride"], ride_id=ride_id, park=last["park"], land=last["land"],
                    avg_wait=round(weighted_mean(values), 1),
                    median_wait=round(weighted_median(values), 1),
                    observations=sum(row["n"] for row in days), days_observed=len(days),
                )
            )  # fmt: skip
        return sorted(stats, key=lambda s: s.avg_wait, reverse=True)

    def coverage(self) -> DataCoverage:
        cached = self._coverage_cache.get("coverage")
        if cached is not None:
            return cached
        pipeline = [
            {"$group": {"_id": _PARIS_DAY}},
            {
                "$group": {
                    "_id": None,
                    "days": {"$sum": 1},
                    "first": {"$min": "$_id"},
                    "last": {"$max": "$_id"},
                }
            },
        ]
        docs = list(self.collection.aggregate(pipeline))
        if docs:
            d = docs[0]
            result = DataCoverage(days=d["days"], first_day=d["first"], last_day=d["last"])
        else:
            result = DataCoverage(days=0, first_day=None, last_day=None)
        self._coverage_cache.set("coverage", result)
        return result

    def _recent_match(self, park: str) -> dict:
        since = now_paris() - timedelta(days=HISTORY_WINDOW_DAYS)
        return {"park": park, "is_open": True, "wait_time": {"$ne": None},
                "fetched_at": {"$gte": since}}  # fmt: skip

    def _hourly_rows(self, park: str, weekday: int | None) -> list[dict]:
        """One row per (ride, hour, day): that day's average wait at that hour."""
        key = ("hourly", park, weekday)
        rows = self._rows_cache.get(key)
        if rows is None:
            match = self._recent_match(park)
            if weekday is not None:
                match["weekday"] = weekday
            group = {"_id": {"ride": "$ride_id", "hour": "$hour", "day": _PARIS_DAY},
                     "avg": {"$avg": "$wait_time"}}  # fmt: skip
            rows = list(self.collection.aggregate([{"$match": match}, {"$group": group}]))
            self._rows_cache.set(key, rows)
        return rows

    def hourly_profile(
        self, *, park: str, weekday: int | None, target: date | None = None
    ) -> HourlyProfile:
        """Same weekday if it has data, otherwise every day collected so far. Each day is
        weighted for the visit date `target` (recent weeks, same period of past years)."""
        today = now_paris().date()
        for basis, day in (("same_weekday", weekday), ("all_days", None)):
            if basis == "same_weekday" and weekday is None:
                continue
            rows = self._hourly_rows(park, day)
            if not rows:
                continue
            return profile_from_rows(park, basis, rows, today, target)
        return HourlyProfile(park=park, basis="none", days=0, waits={})

    def daily_index(self, *, park: str, ride_ids: list[int]) -> dict[date, float]:
        """Crowd index of each collected day: the average wait of the given (popular) rides
        during the busy hours."""
        key = ("daily", park, tuple(sorted(ride_ids)))
        cached = self._rows_cache.get(key)
        if cached is None:
            match = self._recent_match(park)
            match["ride_id"] = {"$in": list(ride_ids)}
            match["hour"] = {"$gte": CROWD_HOURS[0], "$lte": CROWD_HOURS[1]}
            group = {"_id": _PARIS_DAY, "avg": {"$avg": "$wait_time"}}
            rows = self.collection.aggregate([{"$match": match}, {"$group": group}])
            cached = {_day(row["_id"]): round(row["avg"], 1) for row in rows}
            self._rows_cache.set(key, cached)
        return cached


def profile_from_rows(
    park: str, basis: str, rows: list[dict], today: date, target: date | None
) -> HourlyProfile:
    """Weighted average per (ride, hour) of the daily averages."""
    cells: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(list)
    days: set[date] = set()
    for row in rows:
        day = _day(row["_id"]["day"])
        days.add(day)
        cells[(row["_id"]["ride"], row["_id"]["hour"])].append(
            (row["avg"], day_weight(day, target or today))
        )
    waits: dict[int, dict[int, float]] = defaultdict(dict)
    for (ride_id, hour), values in cells.items():
        waits[ride_id][hour] = round(weighted_mean(values), 1)
    season = sum(season_weight(day, target or today) for day in days)
    return HourlyProfile(park=park, basis=basis, days=len(days), waits=dict(waits),
                         season_days=round(season, 1))  # fmt: skip


@lru_cache
def get_history_store() -> HistoryStore:
    """Process-wide store. Tests and evals replace it (monkeypatch / fixtures)."""
    return MongoHistoryStore(get_snapshots_collection())


@tool_guard
def get_typical_wait(
    ride: str | None = None,
    weekday: Annotated[int, Field(ge=0, le=6)] | None = None,
    hour: Annotated[int, Field(ge=0, le=23)] | None = None,
    park: ParkKey | None = None,
) -> TypicalWait | ToolError:
    """Get the USUAL wait times from collected history: average, median, number of
    observations and number of days of data behind each estimate (the last weeks and the
    same period of previous years weigh more).

    ONE call without `ride` returns EVERY ride of the park at once, sorted from the longest to
    the shortest usual wait: use it directly for "which rides have the shortest wait on
    Saturday" (never one call per ride). Use it to plan a future visit, or to judge whether a
    live wait is short or long. Always tell the user how many days of data an estimate
    relies on.

    Args:
        ride: part of a ride name, case-insensitive (e.g. "pirates"). Omit for all rides.
        weekday: 0=Monday ... 6=Sunday. Omit for all days.
        hour: hour of the day in Paris time, 0-23 (10 means 10:00-10:59). Omit for all hours.
        park: "disneyland_park" or "adventure_world". Omit for both parks.
    """
    store = get_history_store()
    rides = store.ride_stats(park=park, ride=ride, weekday=weekday, hour=hour)
    coverage = store.coverage()

    # A young history may not cover the requested weekday yet: rather than "no data", widen
    # the filters step by step and say so.
    basis = "exact"
    if not rides and weekday is not None and coverage.days:
        for basis, wider_hour in (("same_hour_all_days", hour), ("all_days", None)):
            if basis == "same_hour_all_days" and hour is None:
                continue
            rides = store.ride_stats(park=park, ride=ride, weekday=None, hour=wider_hour)
            if rides:
                break
        else:
            basis = "exact"

    filters: dict[str, str | int] = {}
    if ride:
        filters["ride"] = ride
    if weekday is not None:
        filters["weekday"] = f"{weekday} ({WEEKDAYS[weekday]})"
    if hour is not None:
        filters["hour"] = hour
    if park:
        filters["park"] = park

    note = None
    if coverage.days == 0:
        note = "No history collected yet: rely on the park guide, do not invent wait times."
    elif not rides:
        note = "No observation matches these filters: say so, and rely on the park guide."
    elif basis != "exact":
        note = (
            f"No data yet for {WEEKDAYS[weekday]}: these are averages over all the days "
            f"collected so far ({coverage.days} day(s)). Say so: they are rough estimates."
        )
    elif coverage.days < 14:
        note = f"Only {coverage.days} day(s) of history: estimates are indicative only."
    return TypicalWait(filters=filters, basis=basis, coverage=coverage, rides=rides, note=note)
