"""Tool 8: official opening hours, Extra Magic Hours and show times, from ThemeParks.wiki.

Opening hours change with the season (19:00 closing on a November weekday, 23:00 in summer):
guessing them from the wait history scheduled rides after closing time. ThemeParks.wiki relays
the official Disneyland Paris schedule about a month ahead (free, no API key). Show, parade
and night show times are only published for the current day.

Every attraction also has its own schedule: a day without opening is a closure
(refurbishment), with the reopening date when it is published.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import httpx
from pydantic import BaseModel

from app.clock import now_paris
from app.config import PARIS_TZ, PARKS, THEMEPARKS_BASE_URL, ParkKey
from app.queue_times import HTTP_TIMEOUT_S, USER_AGENT
from app.tools.common import ToolError, ToolFailure, TTLCache, tool_guard
from app.tools.rides import CATALOG

log = logging.getLogger(__name__)

# The schedule changes a few times a month at most; show times can change during the day.
_schedule_cache = TTLCache(ttl_s=6 * 60 * 60)
_shows_cache = TTLCache(ttl_s=15 * 60)
_rides_cache = TTLCache(ttl_s=12 * 60 * 60)  # one request per attraction: refreshed rarely

MAX_TIMES_PER_SHOW = 8  # keeps the payload short (character meets run every 30 minutes)
NIGHT_SHOW_WINDOW = 75  # minutes: the last show starting this close to closing ends the day
# Above this share of rides "closed", the attraction schedules are not trustworthy (not
# published yet, API problem): better no closure than a plan without rides.
MAX_CLOSED_SHARE = 0.5

ShowKind = Literal["night_show", "parade", "show", "character"]


class Show(BaseModel):
    name: str
    kind: ShowKind
    times: list[str]  # "HH:MM", Paris time, today, upcoming only


class Closure(BaseModel):
    ride: str  # catalogue name
    reopens: dt.date | None = None  # next published opening day; None = none published yet


class ParkHours(BaseModel):
    park: ParkKey
    park_name: str
    date: dt.date
    is_open: bool
    opening: str | None = None  # "HH:MM", Paris time
    closing: str | None = None  # "HH:MM"; may be past midnight ("01:00")
    # Early entry for Disney hotel guests and some annual passes: "08:30-09:30".
    extra_magic_hours: str | None = None
    special_events: list[str] = []  # separately ticketed evenings: "Halloween Party 20:30-02:00"
    closed_rides: list[Closure] = []  # closed all day per their own schedule (refurbishment)
    shows: list[Show] | None = None  # today only; None = not published for this date
    note: str | None = None
    source: str = "ThemeParks.wiki (official Disneyland Paris schedule)"


def _local(iso: str) -> dt.datetime:
    return dt.datetime.fromisoformat(iso).astimezone(PARIS_TZ)


def _hhmm(iso: str) -> str:
    return _local(iso).strftime("%H:%M")


def _span(entry: dict[str, Any]) -> str:
    return f"{_hhmm(entry['openingTime'])}-{_hhmm(entry['closingTime'])}"


def _get_json(url: str) -> dict[str, Any]:
    response = httpx.get(url, timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT})
    response.raise_for_status()
    return response.json()


def _schedule(park: ParkKey) -> list[dict[str, Any]]:
    cached = _schedule_cache.get(park)
    if cached is None:
        url = f"{THEMEPARKS_BASE_URL}/entity/{PARKS[park].themeparks_id}/schedule"
        cached = _get_json(url).get("schedule") or []
        _schedule_cache.set(park, cached)
    return cached


def parse_schedule(park: ParkKey, day: dt.date, entries: list[dict[str, Any]]) -> ParkHours:
    """Official hours of one day, from the ThemeParks.wiki schedule of the park."""
    days = sorted({e["date"] for e in entries if e.get("date")})
    # Outside the published range the park is not closed: its hours are just unknown.
    if not days or not days[0] <= day.isoformat() <= days[-1]:
        known = f" (published from {days[0]} to {days[-1]})" if days else ""
        raise ToolFailure(
            f"Official hours are not published yet for {day}{known}: say they are to be "
            "checked in the official app closer to the date."
        )
    todays = [e for e in entries if e.get("date") == day.isoformat()]
    operating = [e for e in todays if e.get("type") == "OPERATING"]
    name = PARKS[park].name
    if not operating:
        return ParkHours(park=park, park_name=name, date=day, is_open=False,
                         note="No opening on this day in the official schedule.")  # fmt: skip
    opening = min(operating, key=lambda e: e["openingTime"])["openingTime"]
    closing = max(operating, key=lambda e: e["closingTime"])["closingTime"]
    extra = [e for e in todays if e.get("type") == "EXTRA_HOURS"]
    events = [e for e in todays if e.get("type") in ("TICKETED_EVENT", "PRIVATE_EVENT")]
    return ParkHours(
        park=park, park_name=name, date=day, is_open=True,
        opening=_hhmm(opening), closing=_hhmm(closing),
        extra_magic_hours=_span(extra[0]) if extra else None,
        special_events=[f"{e.get('description') or 'Special ticketed event'} {_span(e)}"
                        for e in events],
    )  # fmt: skip


def _kind(name: str) -> ShowKind:
    lowered = name.lower()
    if lowered.startswith("meet ") or lowered.startswith("rencontre"):
        return "character"
    if "parade" in lowered or "cavalcade" in lowered:
        return "parade"
    return "show"


def parse_shows(
    live: list[dict[str, Any]], now: dt.datetime, closing: str | None = None
) -> list[Show]:
    """Today's upcoming show times. Paid "reserved viewing areas" are left out."""
    shows: list[Show] = []
    for entity in live:
        name = (entity.get("name") or "").strip()
        if entity.get("entityType") != "SHOW" or entity.get("status") != "OPERATING":
            continue
        if not name or name.lower().startswith("reserved"):
            continue
        starts = sorted(
            _local(s["startTime"]) for s in entity.get("showtimes") or [] if s.get("startTime")
        )
        times = [s.strftime("%H:%M") for s in starts if s.date() == now.date() and s > now]
        if times:
            shows.append(Show(name=name, kind=_kind(name), times=times[:MAX_TIMES_PER_SHOW]))

    # The night show: the latest show of the day, starting shortly before closing.
    if closing:
        close = _minutes_of(closing)
        late = [s for s in shows if s.kind == "show" and
                0 <= close - _minutes_of(s.times[-1]) <= NIGHT_SHOW_WINDOW]  # fmt: skip
        if late:
            night = max(late, key=lambda s: s.times[-1])
            night.kind = "night_show"
            night.times = night.times[-1:]
    order = {"night_show": 0, "parade": 1, "show": 2, "character": 3}
    shows.sort(key=lambda s: (order[s.kind], s.times[0]))
    return shows


def _minutes_of(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    total = int(hours) * 60 + int(minutes)
    return total if total >= 4 * 60 else total + 24 * 60  # 00:30 closing = after midnight


def _shows_today(park: ParkKey, now: dt.datetime, closing: str | None) -> list[Show]:
    key = (park, now.date())
    live = _shows_cache.get(key)
    if live is None:
        url = f"{THEMEPARKS_BASE_URL}/entity/{PARKS[park].themeparks_id}/live"
        live = _get_json(url).get("liveData") or []
        _shows_cache.set(key, live)
    return parse_shows(live, now, closing)


def published_schedule(park: ParkKey) -> list[dict[str, Any]] | None:
    """The park's official schedule (cached), or None when unreachable."""
    try:
        return _schedule(park)
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("schedule unavailable for %s: %r", park, exc)
        return None


def hours_on(park: ParkKey, day: dt.date, entries: list[dict[str, Any]]) -> ParkHours | None:
    """Official hours of one day, without closures nor shows; None when not published."""
    try:
        return parse_schedule(park, day, entries)
    except ToolFailure:
        return None


def normalize_name(name: str) -> str:
    """'Autopia, presented by Avis' / 'Mickey’s PhilharMagic' -> the catalogue's spelling."""
    name = re.sub(r",\s*presented by .*$", "", name, flags=re.IGNORECASE)
    name = name.replace("’", "").replace("'", "")
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9]+", " ", name).strip()


def ride_operating_days(park: ParkKey) -> dict[str, list[str]]:
    """Catalogue name -> the days the ride opens in its own official schedule (ISO dates).
    Rides that ThemeParks.wiki does not list (or names that do not match) are left out."""
    cached = _rides_cache.get(park)
    if cached is not None:
        return cached
    catalog = {normalize_name(r.name): r.name for r in CATALOG
               if r.park == park and r.kind in ("ride", "show")}  # fmt: skip
    children = _get_json(f"{THEMEPARKS_BASE_URL}/entity/{PARKS[park].themeparks_id}/children")
    matched = [
        (catalog[normalize_name(c.get("name") or "")], c["id"])
        for c in children.get("children") or []
        if c.get("entityType") == "ATTRACTION" and normalize_name(c.get("name") or "") in catalog
    ]

    def operating(entity_id: str) -> list[str]:
        schedule = _get_json(f"{THEMEPARKS_BASE_URL}/entity/{entity_id}/schedule")
        return sorted({e["date"] for e in schedule.get("schedule") or []
                       if e.get("type") == "OPERATING" and e.get("date")})  # fmt: skip

    with ThreadPoolExecutor(max_workers=8) as pool:  # ~50 small requests: in parallel
        days = list(pool.map(operating, [entity_id for _, entity_id in matched]))
    result = {name: d for (name, _), d in zip(matched, days, strict=True)}
    _rides_cache.set(park, result)
    return result


def closures_on(day: dt.date, operating: dict[str, list[str]]) -> list[Closure]:
    iso = day.isoformat()
    closed = []
    for name, days in sorted(operating.items()):
        if iso not in days:
            later = [d for d in days if d > iso]
            closed.append(Closure(ride=name, reopens=later[0] if later else None))
    if operating and len(closed) > MAX_CLOSED_SHARE * len(operating):
        log.warning("%d/%d rides closed on %s: schedules ignored", len(closed), len(operating), day)
        return []
    return closed


@tool_guard
def get_park_hours(park: ParkKey, date: dt.date) -> ParkHours | ToolError:
    """Get the OFFICIAL opening hours of one park for a day (published about a month ahead):
    opening and closing time, Extra Magic Hours (early entry for Disney hotel guests),
    special ticketed events and the rides closed that day (refurbishment, with their
    reopening date). For today only, also the upcoming times of the shows, parades and night
    show.

    Use it whenever an opening, closing, parade or show time, or a closed ride, matters:
    never guess them.

    Args:
        park: "disneyland_park" or "adventure_world".
        date: the day, YYYY-MM-DD.
    """
    now = now_paris()
    if date < now.date():
        raise ToolFailure(f"{date} is in the past: hours exist from today ({now.date()}).")
    hours = parse_schedule(park, date, _schedule(park))
    if not hours.is_open:
        return hours
    try:
        hours.closed_rides = closures_on(date, ride_operating_days(park))
    except (httpx.HTTPError, ValueError) as exc:  # the hours are still worth returning
        log.warning("ride schedules unavailable for %s: %r", park, exc)
    if date != now.date():
        hours.note = "Show and parade times are only published on the day (official app)."
        return hours
    try:
        hours.shows = _shows_today(park, now, hours.closing)
    except (httpx.HTTPError, ValueError) as exc:  # the hours are still worth returning
        log.warning("show times unavailable for %s: %r", park, exc)
        hours.note = "Show times are unavailable right now: check the official app."
    return hours
