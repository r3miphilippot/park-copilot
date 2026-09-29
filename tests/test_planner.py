from datetime import date, datetime
from types import SimpleNamespace

import pytest

from app.clock import frozen_now
from app.config import PARIS_TZ
from app.tools import ToolError, get_typical_wait, list_rides, plan_day
from app.tools.history import HourlyProfile
from app.tools.rides import CATALOG, RIDES_BY_ID
from tests.conftest import FakeHistoryStore, stats

TUESDAY_1030 = datetime(2026, 7, 14, 10, 30, tzinfo=PARIS_TZ)
SATURDAY = date(2026, 7, 18)
BASE_WAIT = {"headliner": 50, "popular": 30, "standard": 12}


def curve(hour: int) -> float:
    """Short queues at opening, peak early afternoon, lower in the evening."""
    return {9: 0.3, 10: 0.6, 11: 0.9}.get(hour, 1.2 if hour < 17 else 0.8)


def profile(park="disneyland_park", *, hours=range(9, 23), skip=(), basis="same_weekday"):
    waits = {}
    for ride in CATALOG:
        if ride.park != park or ride.kind != "ride" or ride.ride_id in skip:
            continue
        base = BASE_WAIT[ride.popularity]
        waits[ride.ride_id] = {h: round(base * curve(h), 1) for h in hours}
        if ride.single_rider_id:  # Single Rider: about a third, not running at opening
            waits[ride.single_rider_id] = {h: 0.0 if h == 9 else round(base * curve(h) / 3, 1)
                                           for h in hours}  # fmt: skip
    return HourlyProfile(park=park, basis=basis, days=12, waits=waits)


@pytest.fixture
def plan(use_store, monkeypatch):
    """plan_day with a frozen date, a fake history and a stubbed weather forecast."""

    def run(*, store_profile=None, rainy_hours=(), **kwargs):
        use_store(FakeHistoryStore([], profile=store_profile))
        forecast = SimpleNamespace(rainy_hours=list(rainy_hours))
        monkeypatch.setattr("app.tools.planner.get_weather", lambda _date: forecast)
        args = {"park": "disneyland_park", "date": SATURDAY, "start": "08:30"} | kwargs
        with frozen_now(TUESDAY_1030):
            return plan_day(**args)

    return run


def rides_of(result):
    return [s for s in result.steps if s.kind == "ride"]


# --------------------------------------------------------------------------- catalogue


def test_catalog_is_consistent():
    ids = [r.ride_id for r in CATALOG]
    assert len(ids) == len(set(ids))
    single_rider_ids = {r.single_rider_id for r in CATALOG if r.single_rider_id}
    assert not single_rider_ids & set(ids)  # a Single Rider line is not a ride of its own
    assert {r.park for r in CATALOG} == {"disneyland_park", "adventure_world"}


def test_list_rides_filters_by_park():
    result = list_rides("adventure_world")
    names = {r.name for r in result.rides}
    assert "The Twilight Zone Tower of Terror" in names
    assert "Big Thunder Mountain" not in names  # the other park
    assert not any("Jungle Cruise" in n or n == "Space Mountain" for n in names)


# --------------------------------------------------------------------------- history fallback


def test_typical_wait_widens_to_all_days_when_the_weekday_has_no_data(use_store):
    use_store(FakeHistoryStore([stats(25, "Big Thunder Mountain", 40)], days=3,
                               missing_weekdays={2}))  # fmt: skip
    result = get_typical_wait(ride="big thunder", weekday=2, hour=10)
    assert result.basis == "same_hour_all_days"
    assert result.rides and "No data yet for Wednesday" in result.note


# --------------------------------------------------------------------------- planner


def test_thrill_day_plan(plan):
    result = plan(store_profile=profile(), preference="thrill")
    rides = rides_of(result)
    names = [s.title for s in rides]
    thrills = {r.name for r in CATALOG if r.park == "disneyland_park" and r.thrill == "high"}

    assert thrills <= set(names)  # every big thrill ride, Big Thunder Mountain included
    assert result.thrill_rides_count >= 5  # and second rides of the best ones
    assert result.rides_count >= 10  # a real full day
    assert all(any(r.name == n for r in CATALOG if r.park == "disneyland_park") for n in names)
    times = [s.time for s in result.steps]
    assert times == sorted(times)
    assert result.steps[0].title.startswith("Arrival") and result.steps[0].time == "08:30"
    assert [s.title for s in result.steps if s.kind == "meal"] == ["Lunch", "Dinner"]
    assert result.steps[-1].kind == "show"  # night show at the castle
    assert all(s.expected_wait is not None for s in rides)


def test_headliners_come_first_when_queues_are_short(plan):
    result = plan(store_profile=profile(), preference="thrill")
    first_two = {s.title for s in rides_of(result)[:2]}
    assert first_two <= {r.name for r in CATALOG if r.thrill == "high"}


def test_single_rider_line_is_used_when_shorter(plan):
    alone = plan(store_profile=profile(), preference="thrill", single_rider=True)
    group = plan(store_profile=profile(), preference="thrill", single_rider=False)
    assert any(s.line == "single_rider" for s in rides_of(alone))
    assert not any(s.line == "single_rider" for s in rides_of(group))
    assert alone.total_expected_wait / alone.rides_count < (
        group.total_expected_wait / group.rides_count
    )
    # A Single Rider line at 0 min is not running yet: never scheduled as a 0-minute wait.
    assert all(s.expected_wait for s in rides_of(alone) if s.line == "single_rider")


def test_rain_moves_outdoor_rides_out_of_rainy_hours(plan):
    rainy = [f"{h:02d}:00" for h in range(13, 23)]
    result = plan(store_profile=profile(), preference="all", rainy_hours=rainy)
    in_rain = [s for s in rides_of(result) if int(s.time[:2]) >= 13]
    assert sum(s.indoor for s in in_rain) > sum(not s.indoor for s in in_rain)


def test_rides_without_recent_data_are_left_out(plan):
    crush = 32
    result = plan(store_profile=profile("adventure_world", skip={crush}), park="adventure_world")
    assert "Crush's Coaster" not in [s.title for s in rides_of(result)]
    assert any("possibly closed" in n and "Crush's Coaster" in n for n in result.notes)


def test_without_history_waits_are_unknown_never_invented(plan):
    result = plan(store_profile=None, preference="thrill")
    rides = rides_of(result)
    assert result.history_basis == "none" and rides
    assert all(s.expected_wait is None for s in rides)
    assert RIDES_BY_ID[25].name in [s.title for s in rides]  # still ordered by value
    assert any("No wait history" in n for n in result.notes)


def test_partial_first_day_does_not_fake_an_early_closing(plan):
    # History only covers 9:00-16:59 (collector started today): closing must not be 17:00.
    result = plan(store_profile=profile(hours=range(9, 17)))
    assert result.end == "22:00"


def test_afternoon_start_has_no_lunch(plan):
    result = plan(store_profile=profile(), start="14:00")
    assert "Lunch" not in [s.title for s in result.steps]
    assert result.steps[0].kind == "ride"


def test_past_date_is_refused(plan):
    result = plan(store_profile=profile(), date=date(2026, 7, 1))
    assert isinstance(result, ToolError) and "past" in result.error
