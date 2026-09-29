"""Tool 2: typical (historical) wait times, aggregated in MongoDB.

The data access sits behind a small `HistoryStore` interface: MongoDB in production, an
in-memory fake in unit tests and frozen fixtures in evals.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from functools import lru_cache
from typing import Annotated, Protocol

from pydantic import BaseModel, Field
from pymongo.collection import Collection

from app.config import ParkKey
from app.db import get_snapshots_collection
from app.tools.common import ToolError, TTLCache, tool_guard

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class RideStats(BaseModel):
    ride: str
    ride_id: int
    park: str
    land: str | None
    avg_wait: float  # minutes
    median_wait: float  # minutes
    observations: int  # number of 15-minute snapshots behind the numbers
    days_observed: int  # distinct days those snapshots come from


class DataCoverage(BaseModel):
    days: int  # distinct days with at least one snapshot, whole database
    first_day: date | None
    last_day: date | None


class TypicalWait(BaseModel):
    filters: dict[str, str | int]
    coverage: DataCoverage
    rides: list[RideStats]  # longest average wait first
    note: str | None = None


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


# Day of a snapshot, in Paris time (a visit at 00:30 UTC in summer belongs to the next day).
_PARIS_DAY = {
    "$dateToString": {"format": "%Y-%m-%d", "date": "$fetched_at", "timezone": "Europe/Paris"}
}


class MongoHistoryStore:
    def __init__(self, collection: Collection) -> None:
        self.collection = collection
        # Counting days scans the whole collection: cache it, it only changes once a day.
        self._coverage_cache = TTLCache(ttl_s=15 * 60)

    @staticmethod
    def build_stats_pipeline(
        *,
        park: str | None,
        ride: str | None,
        weekday: int | None,
        hour: int | None,
        before: datetime | None = None,
    ) -> list[dict]:
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
                    "_id": "$ride_id",
                    "ride": {"$last": "$ride"},
                    "park": {"$last": "$park"},
                    "land": {"$last": "$land"},
                    "avg_wait": {"$avg": "$wait_time"},
                    # $median needs MongoDB 7.0+ (Atlas M0 runs 8.x).
                    "median_wait": {"$median": {"input": "$wait_time", "method": "approximate"}},
                    "observations": {"$sum": 1},
                    "days": {"$addToSet": _PARIS_DAY},
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "ride_id": "$_id",
                    "ride": 1,
                    "park": 1,
                    "land": 1,
                    "avg_wait": {"$round": ["$avg_wait", 1]},
                    "median_wait": 1,
                    "observations": 1,
                    "days_observed": {"$size": "$days"},
                }
            },
            {"$sort": {"avg_wait": -1}},
        ]

    def ride_stats(self, **filters) -> list[RideStats]:
        pipeline = self.build_stats_pipeline(**filters)
        return [RideStats(**doc) for doc in self.collection.aggregate(pipeline)]

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
    observations and number of days of data behind each estimate.

    Use it to plan a future visit, or to judge whether a live wait is short or long.
    Always tell the user how many days of data an estimate relies on.

    Args:
        ride: part of a ride name, case-insensitive (e.g. "pirates"). Omit for all rides.
        weekday: 0=Monday ... 6=Sunday. Omit for all days.
        hour: hour of the day in Paris time, 0-23 (10 means 10:00-10:59). Omit for all hours.
        park: "disneyland_park" or "adventure_world". Omit for both parks.
    """
    store = get_history_store()
    rides = store.ride_stats(park=park, ride=ride, weekday=weekday, hour=hour)
    coverage = store.coverage()

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
    elif coverage.days < 14:
        note = f"Only {coverage.days} day(s) of history: estimates are indicative only."
    return TypicalWait(filters=filters, coverage=coverage, rides=rides, note=note)
