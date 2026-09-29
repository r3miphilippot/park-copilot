from datetime import date

import pytest

from app.tools import history, live, weather
from app.tools.history import DataCoverage, HourlyProfile, RideStats


@pytest.fixture(autouse=True)
def _clear_tool_caches():
    live._cache.clear()
    weather._cache.clear()
    yield


class FakeHistoryStore:
    """In-memory HistoryStore: returns preset stats and records the filters it received."""

    def __init__(
        self,
        stats: list[RideStats],
        days: int = 30,
        *,
        missing_weekdays: set[int] = frozenset(),
        profile: HourlyProfile | None = None,
    ) -> None:
        self.stats = stats
        self.days = days
        self.missing_weekdays = missing_weekdays  # weekdays without any data yet
        self.profile = profile
        self.calls: list[dict] = []

    def ride_stats(self, **filters) -> list[RideStats]:
        self.calls.append(filters)
        if filters.get("weekday") in self.missing_weekdays:
            return []
        ride = (filters.get("ride") or "").lower()
        return [s for s in self.stats if ride in s.ride.lower()]

    def hourly_profile(self, *, park: str, weekday: int | None) -> HourlyProfile:
        self.calls.append({"profile": park, "weekday": weekday})
        return self.profile or HourlyProfile(park=park, basis="none", days=0, waits={})

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
        monkeypatch.setattr("app.tools.planner.get_history_store", lambda: store)
        return store

    return install
