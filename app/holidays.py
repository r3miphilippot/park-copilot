"""French school holidays and public holidays, from official open data (free, no API key).

They drive the crowds at Disneyland Paris: the crowd calendar uses them to explain and
estimate the expected affluence of a day.

- School holidays: data.education.gouv.fr, dataset "fr-en-calendrier-scolaire" (zones A, B
  and C of metropolitan France; Paris is zone C).
- Public holidays: calendrier.api.gouv.fr (metropolitan France).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

import httpx

from app.config import PARIS_TZ
from app.queue_times import HTTP_TIMEOUT_S, USER_AGENT

log = logging.getLogger(__name__)

SCHOOL_URL = (
    "https://data.education.gouv.fr/api/explore/v2.1/catalog/datasets/"
    "fr-en-calendrier-scolaire/records"
)
PUBLIC_URL = "https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json"
ZONES = ("Zone A", "Zone B", "Zone C")
WINDOW_DAYS = 420  # history behind us, planning ahead: fetched once a day

_cache: dict[dt.date, dict[dt.date, DayOff]] = {}  # keyed by "today": refreshed daily


@dataclass
class DayOff:
    """What makes a day special for crowds."""

    school_zones: list[str] = field(default_factory=list)  # zones on school holidays
    school_holiday: str | None = None  # "Vacances de Noël"
    public_holiday: str | None = None  # "Jour de Noël"

    @property
    def is_holiday(self) -> bool:
        return bool(self.school_zones or self.public_holiday)


def _get(url: str, params: dict | None = None):
    response = httpx.get(
        url, params=params, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}
    )
    response.raise_for_status()
    return response.json()


def _paris_day(iso: str) -> dt.date:
    """'2026-10-16T22:00:00+00:00' is midnight on Oct 17 in Paris: the first day off."""
    return dt.datetime.fromisoformat(iso).astimezone(PARIS_TZ).date()


def parse_school_holidays(rows: list[dict]) -> dict[dt.date, DayOff]:
    """Grouped records (one per period and zone) -> every day off, with its zones."""
    days: dict[dt.date, DayOff] = {}
    for row in rows:
        if row.get("zones") not in ZONES:
            continue
        start, end = _paris_day(row["start_date"]), _paris_day(row["end_date"])
        day = start
        while day < end:  # the end date is the day school resumes
            off = days.setdefault(day, DayOff())
            if row["zones"] not in off.school_zones:
                off.school_zones.append(row["zones"])
                off.school_zones.sort()
            off.school_holiday = off.school_holiday or row.get("description")
            day += dt.timedelta(days=1)
    return days


def _school_holidays(start: dt.date, end: dt.date) -> dict[dt.date, DayOff]:
    zones = ",".join(f'"{z}"' for z in ZONES)
    params = {
        "where": f'end_date >= "{start}" and start_date <= "{end}" and zones in ({zones}) '
        'and population != "Enseignants"',
        "group_by": "description,start_date,end_date,zones",
        "limit": 100,
    }
    payload = _get(SCHOOL_URL, params)
    return parse_school_holidays(payload.get("results") or [])


def days_off(today: dt.date) -> dict[dt.date, DayOff]:
    """School and public holidays from WINDOW_DAYS before today to WINDOW_DAYS after.
    Raises httpx.HTTPError when the open data services are unreachable."""
    if today in _cache:
        return _cache[today]
    start = today - dt.timedelta(days=WINDOW_DAYS)
    end = today + dt.timedelta(days=WINDOW_DAYS)
    days = _school_holidays(start, end)
    for year in range(start.year, end.year + 1):
        try:
            public = _get(PUBLIC_URL.format(year=year))
        except httpx.HTTPStatusError:  # a year not published yet
            continue
        for iso, name in public.items():
            days.setdefault(dt.date.fromisoformat(iso), DayOff()).public_holiday = name
    _cache.clear()  # only today's copy is kept
    _cache[today] = days
    return days
