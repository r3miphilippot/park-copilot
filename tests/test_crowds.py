"""Crowd calendar: a small explainable model (season level x weekday x holidays)."""

from datetime import date, datetime, timedelta

import httpx
import pytest
import respx

from app import holidays
from app.clock import frozen_now
from app.config import PARIS_TZ
from app.holidays import PUBLIC_URL, SCHOOL_URL, DayOff, parse_school_holidays
from app.i18n import answer_language
from app.tools import ToolError, get_crowd_calendar
from app.tools.crowds import build_model
from tests.conftest import FakeHistoryStore

NOW = datetime(2026, 11, 26, 10, 0, tzinfo=PARIS_TZ)
TOUSSAINT = (date(2026, 10, 17), date(2026, 11, 2))  # first day off, day school resumes
NOEL = (date(2026, 12, 19), date(2027, 1, 4))
ALL_ZONES = ["Zone A", "Zone B", "Zone C"]


def _days(start: date, end: date):
    day = start
    while day < end:
        yield day
        day += timedelta(days=1)


def synthetic_history() -> dict[date, float]:
    """Since Sept 28: 20 min on weekdays, x1.5 on weekends, x1.4 during the Toussaint."""

    def index(day: date) -> float:
        weekend = 1.5 if day.weekday() >= 5 else 1
        return 20.0 * weekend * (1.4 if TOUSSAINT[0] <= day < TOUSSAINT[1] else 1)

    return {day: index(day) for day in _days(date(2026, 9, 28), date(2026, 11, 26))}


def days_off() -> dict[date, DayOff]:
    off = {}
    for (start, end), name in ((TOUSSAINT, "Vacances de la Toussaint"), (NOEL, "Vacances de Noël")):
        for day in _days(start, end):
            off[day] = DayOff(school_zones=list(ALL_ZONES), school_holiday=name)
    off.setdefault(date(2026, 12, 25), DayOff()).public_holiday = "Jour de Noël"
    return off


@pytest.fixture
def crowds(use_store, monkeypatch):
    def run(*, history=None, off=None, **kwargs):
        use_store(FakeHistoryStore([], daily=synthetic_history() if history is None else history))
        offs = days_off() if off is None else off
        monkeypatch.setattr(holidays, "days_off", offs if callable(offs) else lambda _: offs)
        monkeypatch.setattr("app.tools.crowds.published_schedule", lambda park: None)
        with frozen_now(NOW):
            return get_crowd_calendar("disneyland_park", **kwargs)

    return run


def test_model_recovers_the_weekday_and_holiday_effects():
    model = build_model(synthetic_history(), days_off())
    assert model.weekday_effect[5] > 1.15 > 1.0 > model.weekday_effect[1]
    assert model.holiday_effect > 1.2


def test_calendar_ranks_the_days_of_a_period(crowds):
    result = crowds(start=date(2026, 12, 14), end=date(2026, 12, 27))
    by_day = {d.date: d for d in result.days}
    assert len(result.days) == 14 and result.days_of_data == 59
    assert by_day[date(2026, 12, 26)].level == "very_busy"  # Saturday in the holidays
    assert by_day[date(2026, 12, 15)].level == "low"  # a Tuesday before the holidays
    # The quietest days: weekdays before the Christmas holidays.
    assert all(d < NOEL[0] and d.weekday() < 5 for d in result.best_days)
    christmas = by_day[date(2026, 12, 25)].reasons
    assert christmas[0] == "Friday" and "French public holiday (Jour de Noël)" in christmas
    assert "French school holidays (Vacances de Noël), zones A, B, C" in christmas


def test_not_enough_history_lists_the_holidays_without_levels(crowds):
    few = dict(list(synthetic_history().items())[-3:])
    result = crowds(history=few, start=date(2026, 12, 18), end=date(2026, 12, 20))
    assert all(d.level is None and d.expected_wait is None for d in result.days)
    assert "Only 3 day(s) of history" in result.note and result.best_days == []
    assert "French school holidays" in result.days[1].reasons[1]


def test_holidays_outage_keeps_the_weekday_effect(crowds):
    def down(_today):
        raise httpx.ConnectError("down")

    result = crowds(off=down, start=date(2026, 12, 14), end=date(2026, 12, 20))
    assert "holidays are unavailable" in result.note
    assert {d.level for d in result.days} - {None}


@pytest.mark.parametrize(
    ("kwargs", "expected"),
    [
        ({"start": date(2026, 11, 1)}, "in the past"),
        ({"start": date(2026, 12, 10), "end": date(2026, 12, 1)}, "before the start"),
    ],
)
def test_invalid_periods(crowds, kwargs, expected):
    result = crowds(**kwargs)
    assert isinstance(result, ToolError) and expected in result.error


def test_long_periods_are_capped(crowds):
    result = crowds(start=date(2026, 12, 1), end=date(2027, 6, 1))
    assert len(result.days) == 45 and "shortened" in result.note


# --------------------------------------------------------------------------- holidays data


def test_school_holidays_are_paris_days_until_school_resumes():
    rows = [
        {"description": "Vacances de la Toussaint", "zones": zone,
         "start_date": "2026-10-16T22:00:00+00:00", "end_date": "2026-11-01T23:00:00+00:00"}
        for zone in ("Zone C", "Zone A", "Guadeloupe")
    ]  # fmt: skip
    days = parse_school_holidays(rows)
    assert min(days) == date(2026, 10, 17) and max(days) == date(2026, 11, 1)
    assert days[date(2026, 10, 20)].school_zones == ["Zone A", "Zone C"]  # overseas left out


@respx.mock
def test_days_off_merges_school_and_public_holidays():
    school = respx.get(url__startswith=SCHOOL_URL).mock(
        return_value=httpx.Response(200, json={"results": [
            {"description": "Vacances de Noël", "zones": "Zone C",
             "start_date": "2026-12-18T23:00:00+00:00", "end_date": "2027-01-03T23:00:00+00:00"},
        ]})
    )  # fmt: skip
    respx.get(url__regex=r"https://calendrier\.api\.gouv\.fr/.*/2026\.json").mock(
        return_value=httpx.Response(200, json={"2026-12-25": "Jour de Noël"})
    )
    respx.get(url__regex=r"https://calendrier\.api\.gouv\.fr/.*/202[578]\.json").mock(
        return_value=httpx.Response(404)
    )
    off = holidays.days_off(date(2026, 11, 26))
    christmas = off[date(2026, 12, 25)]
    assert christmas.public_holiday == "Jour de Noël" and christmas.school_zones == ["Zone C"]
    assert 'zones in ("Zone A","Zone B","Zone C")' in school.calls[0].request.url.params["where"]
    assert PUBLIC_URL.startswith("https://calendrier.api.gouv.fr")


def test_levels_and_reasons_in_the_answer_language(crowds):
    with answer_language("fr"):
        result = crowds(start=date(2026, 12, 25), end=date(2026, 12, 26))
    christmas, saturday = result.days
    assert saturday.level == "very_busy" and saturday.label == "très chargée"
    assert christmas.weekday == "vendredi" and christmas.reasons == [
        "vendredi",
        "vacances scolaires (Vacances de Noël), zones A, B, C",
        "jour férié (Jour de Noël)",
    ]
