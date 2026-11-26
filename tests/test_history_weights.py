"""Seasonal weighting of the history: recent weeks and the same period of past years count
more. MongoDB only returns one row per ride and per day; the weighting is plain Python."""

from datetime import date, datetime, timedelta

import pytest

from app.clock import frozen_now
from app.config import PARIS_TZ
from app.tools.history import (
    MongoHistoryStore,
    day_weight,
    profile_from_rows,
    weighted_mean,
    weighted_median,
)

TODAY = date(2026, 11, 26)
NOW = datetime(2026, 11, 26, 10, 0, tzinfo=PARIS_TZ)


class FakeCollection:
    """Returns preset aggregation rows and records the pipelines it received."""

    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.pipelines: list[list[dict]] = []

    def aggregate(self, pipeline: list[dict]):
        self.pipelines.append(pipeline)
        return iter(self.rows)


def test_day_weight_favours_recent_days_and_the_same_season():
    assert day_weight(TODAY - timedelta(days=1), TODAY) == pytest.approx(1, abs=0.03)
    assert day_weight(TODAY - timedelta(days=30), TODAY) == pytest.approx(0.5)
    assert day_weight(TODAY - timedelta(days=120), TODAY) < 0.1  # summer, for November
    # Planning next July: last July counts fully, this month almost nothing.
    july = date(2027, 7, 14)
    assert day_weight(TODAY, july) < 0.01
    assert day_weight(date(2026, 7, 14), july) == pytest.approx(1)
    assert day_weight(date(2026, 7, 29), july) == pytest.approx(0.5)


def test_weighted_mean_and_median():
    values = [(10.0, 1.0), (20.0, 1.0), (90.0, 0.1)]
    assert weighted_mean(values) == pytest.approx((10 + 20 + 9) / 2.1)
    assert weighted_mean([(10.0, 3.0), (40.0, 1.0)]) == pytest.approx(17.5)
    assert weighted_median(values) == 20.0
    assert weighted_median([(10.0, 0.2), (40.0, 1.0)]) == 40.0


def _day_row(ride_id: int, day: date, avg: float, n: int = 4, land: str = "Frontierland"):
    return {"_id": {"ride_id": ride_id, "day": day.isoformat()}, "ride": "Big Thunder Mountain",
            "park": "disneyland_park", "land": land, "avg": avg, "n": n}  # fmt: skip


def test_ride_stats_weight_each_day():
    rows = [
        _day_row(25, TODAY - timedelta(days=2), 40),  # this month: busy
        _day_row(25, TODAY - timedelta(days=3), 44),
        _day_row(25, date(2026, 8, 10), 90, land="Old land name"),  # summer: much busier
    ]
    collection = FakeCollection(rows)
    with frozen_now(NOW):
        [stats] = MongoHistoryStore(collection).ride_stats(
            park="disneyland_park", ride=None, weekday=5, hour=None
        )
    assert 42 < stats.avg_wait < 46  # the summer day barely counts (plain mean: 58)
    assert stats.median_wait == 44.0
    assert (stats.observations, stats.days_observed) == (12, 3)
    assert stats.land == "Frontierland"  # name and land of the most recent day
    group = collection.pipelines[0][1]["$group"]
    assert set(group["_id"]) == {"ride_id", "day"}  # one row per ride and per day


def _cell(ride: int, hour: int, day: date, avg: float) -> dict:
    return {"_id": {"ride": ride, "hour": hour, "day": day.isoformat()}, "avg": avg}


def test_hourly_profile_for_next_summer_follows_last_summer():
    rows = [_cell(25, 14, date(2026, 7, d), 80) for d in (11, 18, 25)]  # last July
    rows += [_cell(25, 14, TODAY - timedelta(days=d), 20) for d in (2, 9, 16)]  # this month
    summer = profile_from_rows("disneyland_park", "same_weekday", rows, TODAY, date(2027, 7, 17))
    winter = profile_from_rows("disneyland_park", "same_weekday", rows, TODAY, date(2026, 12, 5))
    assert summer.waits[25][14] > 60 and winter.waits[25][14] < 30
    assert summer.season_days > 2 and summer.days == 6


def test_daily_index_is_cached_per_park():
    collection = FakeCollection([{"_id": "2026-11-21", "avg": 42.04}])
    store = MongoHistoryStore(collection)
    with frozen_now(NOW):
        first = store.daily_index(park="disneyland_park", ride_ids=[25, 3])
        store.daily_index(park="disneyland_park", ride_ids=[3, 25])
    assert first == {date(2026, 11, 21): 42.0}
    assert len(collection.pipelines) == 1
    match = collection.pipelines[0][0]["$match"]
    assert match["ride_id"] == {"$in": [25, 3]} and match["hour"] == {"$gte": 11, "$lte": 17}
