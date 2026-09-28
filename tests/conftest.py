from datetime import date

import pytest

from app.tools import history, live, weather
from app.tools.history import DataCoverage, RideStats


@pytest.fixture(autouse=True)
def _clear_tool_caches():
    live._cache.clear()
    weather._cache.clear()
    yield


class FakeHistoryStore:
    """In-memory HistoryStore: returns preset stats and records the filters it received."""

    def __init__(self, stats: list[RideStats], days: int = 30) -> None:
        self.stats = stats
        self.days = days
        self.calls: list[dict] = []

    def ride_stats(self, **filters) -> list[RideStats]:
        self.calls.append(filters)
        ride = (filters.get("ride") or "").lower()
        return [s for s in self.stats if ride in s.ride.lower()]

    def coverage(self) -> DataCoverage:
        if not self.days:
            return DataCoverage(days=0, first_day=None, last_day=None)
        return DataCoverage(days=self.days, first_day=date(2026, 6, 1), last_day=date(2026, 7, 13))


def stats(ride_id: int, ride: str, avg: float, days: int = 8) -> RideStats:
    return RideStats(
        ride=ride, ride_id=ride_id, park="disneyland_park", land="Adventureland",
        avg_wait=avg, median_wait=avg, observations=days * 4, days_observed=days,
    )  # fmt: skip


@pytest.fixture
def use_store(monkeypatch):
    """Install a FakeHistoryStore everywhere the tools look it up."""

    def install(store: FakeHistoryStore) -> FakeHistoryStore:
        monkeypatch.setattr(history, "get_history_store", lambda: store)
        monkeypatch.setattr("app.tools.compare.get_history_store", lambda: store)
        return store

    return install
