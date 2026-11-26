"""Tool 9: crowd calendar, the expected affluence of each day of a period ("which day should
we come?").

Crowd index of a collected day = average wait of the park's popular rides between 11:00 and
17:59. The expected index of a future day comes from a small model that can be explained in
one sentence:

    expected = season level x weekday effect x holiday effect

- weekday effect: mean index of that weekday / mean of all days (outside holidays);
- holiday effect: mean index on French school or public holidays / the other days;
- season level: weighted mean of the daily indexes once those effects are removed, the days
  close to the date or from the same period of previous years weighing more (same weights
  as the planner).
Each effect is shrunk toward 1 when it rests on few days. The expected index is then ranked
against the days collected so far: low / moderate / busy / very busy.
"""

from __future__ import annotations

import datetime as dt
import logging
import statistics
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel
from pymongo.errors import PyMongoError

from app import holidays
from app.clock import now_paris
from app.config import PARKS, ParkKey
from app.db import DatabaseNotConfigured
from app.holidays import DayOff
from app.tools.common import ToolError, ToolFailure, tool_guard
from app.tools.history import WEEKDAYS, day_weight, get_history_store, weighted_mean
from app.tools.park_hours import hours_on, published_schedule
from app.tools.rides import CATALOG

log = logging.getLogger(__name__)

Level = Literal["low", "moderate", "busy", "very_busy"]
LEVELS: tuple[Level, ...] = ("low", "moderate", "busy", "very_busy")
MIN_DAYS = 7  # below this, no crowd level: too little history to rank days
SHRINK_DAYS = 3  # pseudo-days pulling an effect toward 1 (no effect) when data is thin
DEFAULT_SPAN, MAX_SPAN = 14, 45  # days


class CrowdDay(BaseModel):
    date: dt.date
    weekday: str
    level: Level | None  # None: not enough history to rank this day
    expected_wait: float | None  # expected average wait of the popular rides, 11:00-18:00
    reasons: list[str]  # what drives the crowd: weekday, school or public holidays
    hours: str | None = None  # official opening hours, when already published


class CrowdCalendar(BaseModel):
    park: ParkKey
    park_name: str
    days: list[CrowdDay]
    best_days: list[dt.date]  # the quietest days of the period, best first
    days_of_data: int
    method: str = (
        "expected average wait of the popular rides (11:00-18:00), from the wait history "
        "adjusted for the weekday and French school / public holidays"
    )
    note: str | None = None
    source: str = (
        "Queue-Times.com history; holidays: data.education.gouv.fr, calendrier.api.gouv.fr"
    )


@dataclass
class CrowdModel:
    weekday_effect: dict[int, float]
    holiday_effect: float
    thresholds: list[float]  # quartiles of the estimates for the collected days
    history: dict[dt.date, float]
    off: dict[dt.date, DayOff]

    def _is_holiday(self, day: dt.date) -> bool:
        return day in self.off and self.off[day].is_holiday

    def _effects(self, day: dt.date) -> float:
        holiday = self.holiday_effect if self._is_holiday(day) else 1.0
        return self.weekday_effect[day.weekday()] * holiday

    def expected(self, day: dt.date) -> float:
        level = weighted_mean(
            [(index / self._effects(d), day_weight(d, day)) for d, index in self.history.items()]
        )
        return level * self._effects(day)

    def level(self, index: float) -> Level:
        return LEVELS[sum(index > t for t in self.thresholds)]


def _shrunk(values: list[float], reference: float) -> float:
    """Ratio mean(values) / reference, pulled toward 1 when it rests on few days."""
    if not values or reference <= 0:
        return 1.0
    ratio = statistics.mean(values) / reference
    return (len(values) * ratio + SHRINK_DAYS) / (len(values) + SHRINK_DAYS)


def build_model(history: dict[dt.date, float], off: dict[dt.date, DayOff]) -> CrowdModel:
    holiday = {d for d in history if d in off and off[d].is_holiday}
    normal = {d: v for d, v in history.items() if d not in holiday} or history
    reference = statistics.mean(normal.values())
    weekday_effect = {
        wd: _shrunk([v for d, v in normal.items() if d.weekday() == wd], reference)
        for wd in range(7)
    }
    holiday_effect = 1.0
    if holiday and len(normal) < len(history):
        holiday_effect = _shrunk(
            [history[d] / weekday_effect[d.weekday()] for d in holiday], reference
        )
    model = CrowdModel(weekday_effect=weekday_effect, holiday_effect=round(holiday_effect, 3),
                       thresholds=[], history=history, off=off)  # fmt: skip
    # Levels rank a day against the model's own estimates of the collected days (same
    # smoothing on both sides), not against the raw values.
    model.thresholds = statistics.quantiles([model.expected(d) for d in history], n=4)
    return model


def _reasons(day: dt.date, off: DayOff | None) -> list[str]:
    reasons = [WEEKDAYS[day.weekday()]]
    if off and off.school_zones:
        zones = ", ".join(z.removeprefix("Zone ") for z in off.school_zones)
        reasons.append(f"French school holidays ({off.school_holiday}), zones {zones}")
    if off and off.public_holiday:
        reasons.append(f"French public holiday ({off.public_holiday})")
    return reasons


@tool_guard
def get_crowd_calendar(
    park: ParkKey, start: dt.date | None = None, end: dt.date | None = None
) -> CrowdCalendar | ToolError:
    """Expected crowd level of each day of a period, to choose WHICH DAY to come: low /
    moderate / busy / very busy, from the wait history adjusted for the weekday and the
    French school and public holidays, with the official opening hours when published.

    Use it for "which day should we come", "is it busy during the holidays", or to compare
    dates. Say on how many days of history the levels rest.

    Args:
        park: "disneyland_park" or "adventure_world".
        start: first day of the period, YYYY-MM-DD (default: today).
        end: last day of the period, YYYY-MM-DD (default: 2 weeks after start, at most 45
            days).
    """
    today = now_paris().date()
    start = start or today
    if start < today:
        raise ToolFailure(f"{start} is in the past: start from today ({today}).")
    end = end or start + dt.timedelta(days=DEFAULT_SPAN - 1)
    if end < start:
        raise ToolFailure(f"The end ({end}) is before the start ({start}).")
    notes = []
    if (end - start).days >= MAX_SPAN:
        end = start + dt.timedelta(days=MAX_SPAN - 1)
        notes.append(f"Period shortened to {MAX_SPAN} days: ask again for later dates.")

    try:
        off = holidays.days_off(today)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("holidays unavailable: %r", exc)
        off = {}
        notes.append("French holidays are unavailable right now: weekday effect only.")
    ride_ids = [r.ride_id for r in CATALOG if r.park == park and r.kind == "ride"
                and r.popularity in ("headliner", "popular")]  # fmt: skip
    try:
        history = get_history_store().daily_index(park=park, ride_ids=ride_ids)
    except (DatabaseNotConfigured, PyMongoError) as exc:
        log.warning("history unavailable for the crowd calendar: %r", exc)
        history = {}
    history = {d: v for d, v in history.items() if d < today}  # complete days only

    model = build_model(history, off) if len(history) >= MIN_DAYS else None
    if model is None:
        notes.append(
            f"Only {len(history)} day(s) of history: not enough to rank days (at least "
            f"{MIN_DAYS}). Holidays and weekends are listed as a guide: they are usually busier."
        )
    elif len(history) < 30:
        notes.append(f"Crowd levels rest on {len(history)} days of history: indicative only.")

    schedule = published_schedule(park)  # fetched once for the whole period
    days = []
    day = start
    while day <= end:
        hours = hours_on(park, day, schedule) if schedule else None
        expected = model.expected(day) if model else None
        days.append(
            CrowdDay(
                date=day, weekday=WEEKDAYS[day.weekday()],
                level=model.level(expected) if model else None,
                expected_wait=round(expected, 1) if expected is not None else None,
                reasons=_reasons(day, off.get(day)),
                hours=f"{hours.opening}-{hours.closing}" if hours and hours.is_open else None,
            )
        )  # fmt: skip
        day += dt.timedelta(days=1)
    ranked = sorted((d for d in days if d.expected_wait is not None), key=lambda d: d.expected_wait)
    return CrowdCalendar(
        park=park, park_name=PARKS[park].name, days=days,
        best_days=[d.date for d in ranked[:3]], days_of_data=len(history),
        note=" ".join(notes) or None,
    )  # fmt: skip
