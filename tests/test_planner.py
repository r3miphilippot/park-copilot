from datetime import date, datetime
from types import SimpleNamespace

import pytest

from app.clock import frozen_now
from app.config import PARIS_TZ
from app.i18n import answer_language
from app.tools import ToolError, get_typical_wait, list_rides, plan_day
from app.tools.history import HourlyProfile
from app.tools.live import LiveWaitTimes, RideLiveWait
from app.tools.park_hours import Closure, ParkHours, Show
from app.tools.rides import CATALOG, RIDES_BY_ID
from tests.conftest import FakeHistoryStore, stats

TUESDAY_1030 = datetime(2026, 7, 14, 10, 30, tzinfo=PARIS_TZ)
SATURDAY = date(2026, 7, 18)
TODAY_1030 = datetime(2026, 7, 18, 10, 30, tzinfo=PARIS_TZ)  # the Saturday itself
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
    """plan_day with a frozen date, a fake history and stubbed weather, opening hours
    (hours=None: not published, the planner estimates them) and live waits (live=None:
    unavailable)."""

    def run(*, store_profile=None, rainy_hours=(), hours=None, live=None, now=TUESDAY_1030,
            **kwargs):  # fmt: skip
        use_store(FakeHistoryStore([], profile=store_profile))
        forecast = SimpleNamespace(rainy_hours=list(rainy_hours))
        monkeypatch.setattr("app.tools.planner.get_weather", lambda _date: forecast)
        unpublished = ToolError(tool="get_park_hours", error="not published yet")
        monkeypatch.setattr("app.tools.planner.get_park_hours", lambda **_: hours or unpublished)
        no_live = ToolError(tool="get_live_wait_times", error="unavailable")
        monkeypatch.setattr("app.tools.planner.get_live_wait_times", lambda **_: live or no_live)
        args = {"park": "disneyland_park", "date": SATURDAY, "start": "08:30"} | kwargs
        with frozen_now(now):
            return plan_day(**args)

    return run


def official(opening="09:30", closing="19:00", emh="08:30-09:30", shows=None, closed=()):
    return ParkHours(
        park="disneyland_park", park_name="Disneyland Park", date=SATURDAY, is_open=True,
        opening=opening, closing=closing, extra_magic_hours=emh, shows=shows,
        closed_rides=list(closed),
    )  # fmt: skip


def live_waits(waits: dict[int, int | None]) -> LiveWaitTimes:
    """Live snapshot: ride_id -> wait (None = closed right now)."""
    rides = [RideLiveWait(ride=f"ride {i}", ride_id=i, land=None, is_open=w is not None,
                          wait_time=w) for i, w in waits.items()]  # fmt: skip
    return LiveWaitTimes(park="disneyland_park", park_name="Disneyland Park",
                         fetched_at=TODAY_1030, park_open=True, rides=rides)  # fmt: skip


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
    # The regular line is not spelled out: models copied "standby" into the answers.
    assert all(s.line is None for s in rides_of(group))
    assert '"standby"' not in group.model_dump_json(exclude_none=True)
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


# --------------------------------------------------------------------------- opening hours


def test_official_hours_are_respected(plan):
    # A winter weekday: 19:00 closing. The history (summer) has waits until 23:00.
    result = plan(store_profile=profile(), hours=official())
    assert result.hours_source == "official" and result.park_hours == "09:30-19:00"
    assert result.steps[0].note == "Park opens at 09:30."
    assert rides_of(result)[0].time >= "09:30"  # no ride before opening for regular guests
    assert all(s.time < "19:00" for s in result.steps)
    # No published show time: the castle show shortly before closing, and nothing after it.
    assert result.steps[-1].kind == "show" and result.steps[-1].time == "18:30"
    assert "Dinner" not in [s.title for s in result.steps]  # the park closes before 20:30
    assert any("Official opening hours: 09:30-19:00" in n for n in result.notes)


def test_estimated_hours_ignore_extra_magic_hours(plan):
    # At 8:00 only a handful of rides run (Extra Magic Hours): the park opens at 9:00.
    history = profile()
    for ride_id in list(history.waits)[:3]:
        history.waits[ride_id][8] = 5.0
    result = plan(store_profile=history)
    assert result.hours_source == "estimated" and result.start == "09:00"
    assert any("estimated from the history" in n for n in result.notes)


def test_extra_magic_hours_only_plan_rides_running_early(plan):
    history = profile()
    early_rides = [r.ride_id for r in CATALOG if r.park == "disneyland_park" and r.thrill == "high"]
    for ride_id in early_rides[:2]:
        history.waits[ride_id][8] = 5.0
    result = plan(store_profile=history, hours=official(), extra_magic_hours=True, start="08:00")
    assert result.extra_magic_hours and result.start == "08:30"
    before_opening = [s for s in rides_of(result) if s.time < "09:30"]
    assert before_opening
    assert {s.title for s in before_opening} <= {RIDES_BY_ID[i].name for i in early_rides[:2]}


def test_closed_park_is_reported(plan):
    closed = ParkHours(park="disneyland_park", park_name="Disneyland Park", date=SATURDAY,
                       is_open=False)  # fmt: skip
    result = plan(store_profile=profile(), hours=closed)
    assert isinstance(result, ToolError) and "closed" in result.error


def test_today_plan_starts_now_and_fits_around_the_shows(plan):
    today_1030 = datetime(2026, 7, 18, 10, 30, tzinfo=PARIS_TZ)
    shows = [
        Show(name="Disney Tales of Magic", kind="night_show", times=["18:35"]),
        Show(name="Disney Stars on Parade", kind="parade", times=["14:20", "16:00"]),
    ]
    result = plan(store_profile=profile(), hours=official(shows=shows), now=today_1030,
                  preference="family")  # fmt: skip
    assert result.start == "10:30"
    assert result.steps[0].kind == "ride"  # already in the park: no "arrival at 08:30"
    shows_steps = [(s.time, s.title) for s in result.steps if s.kind == "show"]
    assert shows_steps == [("16:00", "Parade: Disney Stars on Parade"),
                           ("18:35", "Night show: Disney Tales of Magic")]  # fmt: skip
    # No ride overlaps a show (spot found 15-20 min before, show length after).
    for step in rides_of(result):
        start = int(step.time[:2]) * 60 + int(step.time[3:])
        end = start + (step.expected_wait or 0) + 8
        assert end <= 15 * 60 + 45 or start >= 16 * 60 + 30
        assert end <= 18 * 60 + 15
    assert not any(s.title == "Night show at the castle" for s in result.steps)


def test_parade_soon_after_an_afternoon_arrival_but_not_at_opening(plan):
    shows = [Show(name="Disney Stars on Parade", kind="parade", times=["10:00", "16:00"])]
    afternoon = datetime(2026, 7, 18, 15, 10, tzinfo=PARIS_TZ)
    later = plan(store_profile=profile(), hours=official(shows=shows), now=afternoon)
    assert [s.time for s in later.steps if s.kind == "show"] == ["16:00"]
    # The 10:00 parade falls in the first hour after the 09:30 opening: rides first.
    early = datetime(2026, 7, 18, 8, 0, tzinfo=PARIS_TZ)
    morning = plan(store_profile=profile(), hours=official(shows=shows), now=early, end="12:00")
    assert not [s for s in morning.steps if s.kind == "show"]


def test_thrill_seekers_skip_the_parade(plan):
    today_1030 = datetime(2026, 7, 18, 10, 30, tzinfo=PARIS_TZ)
    shows = [Show(name="Disney Stars on Parade", kind="parade", times=["16:00"])]
    result = plan(store_profile=profile(), hours=official(shows=shows), now=today_1030,
                  preference="thrill")  # fmt: skip
    assert not [s for s in result.steps if s.kind == "show"]


# --------------------------------------------------------------------------- closures, live, season


def test_rides_closed_that_day_are_not_planned(plan):
    closed = [Closure(ride="Big Thunder Mountain", reopens=date(2026, 7, 20)),
              Closure(ride="Phantom Manor")]  # fmt: skip
    result = plan(store_profile=profile(), hours=official(closed=closed), preference="thrill")
    titles = [s.title for s in rides_of(result)]
    assert "Big Thunder Mountain" not in titles and "Phantom Manor" not in titles
    [note] = [n for n in result.notes if "official schedule, not planned" in n]
    assert "Big Thunder Mountain (reopens 2026-07-20)" in note
    assert "Phantom Manor (no reopening date published)" in note


def test_today_uses_live_waits_then_scales_the_history(plan):
    history = profile()
    big_thunder = next(r for r in CATALOG if r.name == "Big Thunder Mountain")
    # Without the breakdown, the second ride of this plan (checked when writing the test).
    indiana = next(r for r in CATALOG if r.name.startswith("Indiana Jones"))
    usual_now = {r: by_hour[10] for r, by_hour in history.waits.items() if by_hour[10] >= 5}
    busy = {r: round(w * 1.5) for r, w in usual_now.items()}  # today: +50 % everywhere
    busy[big_thunder.ride_id], busy[indiana.ride_id] = 5, None  # short queue now; Indiana down
    result = plan(store_profile=history, live=live_waits(busy), now=TODAY_1030,
                  preference="thrill")  # fmt: skip

    assert result.live_waits and result.today_vs_usual == 1.5
    assert any("50 % busier than usual" in n for n in result.notes)
    first = rides_of(result)[0]
    assert first.title == "Big Thunder Mountain" and first.live_wait
    assert first.expected_wait == 5  # the live wait, not the usual one
    titles = [s.title for s in rides_of(result)]
    first_hour = [s.title for s in rides_of(result) if s.time < "11:30"]
    # Down right now: not in the next hour, but still planned later (it may reopen).
    assert indiana.name not in first_hour and indiana.name in titles
    later = [s for s in rides_of(result) if s.time >= "12:00" and s.expected_wait]
    assert later and not any(s.live_wait for s in later)


def test_no_live_waits_for_a_future_date(plan):
    result = plan(store_profile=profile(), live=live_waits({25: 5}))
    assert not result.live_waits and not any(s.live_wait for s in result.steps)


def test_far_date_without_history_from_that_season_says_so(plan):
    far = date(2026, 12, 19)  # five months after "now", history from the summer only
    result = plan(store_profile=profile(), date=far)
    assert any("No history yet from this time of year" in n for n in result.notes)


def test_steps_are_written_in_the_answer_language(plan):
    with answer_language("fr"):
        result = plan(store_profile=profile(), hours=official(), preference="thrill")
    titles = [s.title for s in result.steps if s.kind != "ride"]
    assert titles == [
        "Arrivée et contrôles de sécurité",
        "Déjeuner",
        "Spectacle nocturne au château",
    ]
    assert result.steps[0].note == "Ouverture du parc à 09:30."
    assert any(s.note == "deuxième tour" for s in rides_of(result))
