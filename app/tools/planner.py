"""Tool 7: plan_day, a deterministic day planner.

"Most rides, least waiting" is a scheduling problem: LLMs do it badly (they invent rides and
hop between parks), plain code does it well. The LLM orchestrates and explains; this tool
optimizes.

Greedy algorithm, slot by slot: take the ride with the best value per minute spent,

    score = value(ride, preferences) / (effective wait + walk + ride time)
    effective wait = wait now - 0.5 x (average wait later today - wait now)

`value` encodes what the visitor wants (a thrill seeker values Big Thunder Mountain far above
a carousel, so a 60-minute Big Thunder beats a 20-minute carousel); a second ride of the same
attraction and an outdoor ride during rain are worth less. The effective wait favours rides
that are cheap now but will get worse later (headliners at opening).

Waits come from the collected history (usual wait per ride and per hour). Rides without data
are still scheduled, ordered by popularity, but their wait is reported as unknown: the tool
never makes up a number.

Opening hours are the official ones (ThemeParks.wiki, about a month ahead), estimated from the
history beyond that. Shows are fixed events the rides are scheduled around: today's real
parade and night show times when they are published, otherwise the castle night show shortly
before closing.
"""

from __future__ import annotations

import datetime as dt
import logging
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.clock import now_paris
from app.config import PARKS, ParkKey
from app.tools.common import ToolError, ToolFailure, tool_guard
from app.tools.history import HourlyProfile, get_history_store
from app.tools.live import get_live_wait_times
from app.tools.park_hours import ParkHours, get_park_hours
from app.tools.rides import CATALOG, Ride
from app.tools.weather import get_weather

log = logging.getLogger(__name__)

Preference = Literal["thrill", "family", "all"]

RIDE_MINUTES = 8  # boarding + ride + exit
WALK_SAME_LAND = 5
WALK_OTHER_LAND = 12
MEAL_MINUTES = 45
LUNCH_AFTER, DINNER_AFTER = 11 * 60 + 30, 18 * 60 + 15  # off-peak meals (guide)
DEFAULT_WINDOW = (9 * 60 + 30, 22 * 60)  # no official hours and no history to estimate them
LATEST_PLAUSIBLE_OPENING = 11 * 60  # earlier data than this = the history covers mornings
EARLIEST_PLAUSIBLE_CLOSING = 20 * 60  # later data than this = the history covers evenings
REPEAT_FACTOR = 0.4  # a second ride of the same attraction is worth less
RAIN_OUTDOOR_FACTOR = 0.3  # an outdoor ride during rain is worth less
# Rain expected later: do outdoor rides while it is dry, keep indoor ones for the showers
# (a purely greedy plan used up the indoor rides in the dry morning).
DRY_BEFORE_RAIN_OUTDOOR_FACTOR = 1.6
# Shows: (minutes to find a spot before, duration). Without a published time, the castle
# night show is placed shortly before closing (19:00 closing -> 18:30).
NIGHT_SHOW = (20, 25)
PARADE = (15, 30)
NIGHT_SHOW_BEFORE_CLOSING = 30
# Today: live waits are trusted for the next hour; later waits are the usual ones scaled by
# how busy today is (median live / usual ratio over enough rides, within sane bounds).
LIVE_HORIZON = 60
MIN_RIDES_FOR_TODAY_FACTOR = 5
TODAY_FACTOR_BOUNDS = (0.6, 1.8)
# A date this far ahead with little history from that time of year: say the waits may differ.
FAR_DATE_DAYS, MIN_SEASON_DAYS = 45, 2.0

# What a ride is worth to the visitor, by preference.
VALUES: dict[str, dict[str, float]] = {
    # thrill seeker: thrill level first, then popularity for the other rides
    "thrill": {"high": 100, "medium": 60, "headliner": 30, "popular": 20, "standard": 8},
    # family: gentle rides by popularity, thrill rides almost excluded
    "family": {"high": 5, "medium": 30, "headliner": 100, "popular": 70, "standard": 40},
    # no preference: popularity only
    "all": {"headliner": 100, "popular": 70, "standard": 35},
}
# Scheduling only, never shown: typical order of magnitude of a queue when there is no data,
# and how it evolves along the day (short at opening, peak early afternoon).
NO_DATA_WAIT = {"headliner": 45.0, "popular": 25.0, "standard": 10.0}
_TIME = Field(pattern=r"^\d{1,2}:\d{2}$")


class PlanStep(BaseModel):
    time: str  # "HH:MM", Paris time
    kind: Literal["ride", "meal", "show", "info"]
    title: str
    land: str | None = None
    line: Literal["standby", "single_rider"] | None = None
    expected_wait: int | None = None  # minutes, from history; None = no data for this ride
    live_wait: bool | None = None  # True: expected_wait is the current live wait
    thrill: str | None = None
    indoor: bool | None = None
    note: str | None = None


@dataclass
class Event:
    """A show at a fixed time: rides are scheduled around it."""

    at: int  # minutes since midnight
    early: int  # minutes before, to find a spot
    duration: int
    step: PlanStep

    @property
    def block_start(self) -> int:
        return self.at - self.early


class DayPlan(BaseModel):
    park: ParkKey
    park_name: str
    date: dt.date
    start: str
    end: str
    preference: Preference
    single_rider: bool
    extra_magic_hours: bool  # the plan starts with the early entry (Disney hotel guests)
    park_hours: str  # "09:30-19:00"
    hours_source: Literal["official", "estimated"]
    history_basis: str  # same_weekday | all_days | none
    days_of_data: int
    live_waits: bool  # today: the first hour uses the live waits
    today_vs_usual: float | None = None  # today: live / usual waits right now (1.3 = +30 %)
    rainy_hours: list[str] | None  # None = no forecast available
    steps: list[PlanStep]
    rides_count: int
    thrill_rides_count: int
    total_expected_wait: int  # minutes, rides with data only
    notes: list[str]


def _minutes(hhmm: str) -> int:
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def _hhmm(minutes: int) -> str:
    return f"{minutes // 60 % 24:02d}:{minutes % 60:02d}"


def _closing_minutes(opening: str, closing: str) -> int:
    """A closing at or before the opening time is after midnight (New Year's Eve: 01:00)."""
    end = _minutes(closing)
    return end + 24 * 60 if end <= _minutes(opening) else end


def _wait_at(profile: HourlyProfile, ride_id: int | None, hour: int) -> float | None:
    """Usual wait at this hour, or at the nearest hour with data; None without any data."""
    by_hour = profile.waits.get(ride_id) if ride_id is not None else None
    if not by_hour:
        return None
    nearest = min(by_hour, key=lambda h: abs(h - hour))
    return by_hour[nearest]


@dataclass
class Waits:
    """Where an expected wait comes from: today, the live wait for the next hour; then the
    usual wait (history), scaled by how busy today is compared with usual."""

    profile: HourlyProfile
    live: dict[int, int | None] | None = None  # ride_id -> live wait, None = closed right now
    live_until: int = -1  # minutes since midnight
    factor: float = 1.0  # today's crowd / usual crowd

    def _use_live(self, ride_id: int | None, minute: int | None) -> bool:
        return (self.live is not None and minute is not None and minute < self.live_until
                and ride_id in self.live)  # fmt: skip

    def at(self, ride_id: int | None, hour: int, minute: int | None = None) -> float | None:
        if self._use_live(ride_id, minute):
            return self.live[ride_id]
        usual = _wait_at(self.profile, ride_id, hour)
        return None if usual is None else usual * self.factor

    def is_live(self, ride_id: int | None, minute: int) -> bool:
        return self._use_live(ride_id, minute) and self.live[ride_id] is not None

    def closed_now(self, ride_id: int, minute: int) -> bool:
        return self._use_live(ride_id, minute) and self.live[ride_id] is None


def _no_data_wait(ride: Ride, hour: int) -> float:
    curve = 0.4 if hour < 11 else (1.2 if hour < 17 else 0.8)
    return NO_DATA_WAIT[ride.popularity] * curve


def _value(ride: Ride, preference: Preference) -> float:
    values = VALUES[preference]
    if preference != "all" and ride.thrill in values:
        return values[ride.thrill]
    return values[ride.popularity]


def _choose_line(
    ride: Ride, waits: Waits, hour: int, single_rider: bool, minute: int | None = None
):
    """(real wait or None, wait used for scheduling, line, id of that line)."""
    standby = waits.at(ride.ride_id, hour, minute)
    line, real, line_id = "standby", standby, ride.ride_id
    if single_rider and ride.single_rider_id:
        solo = waits.at(ride.single_rider_id, hour, minute)
        # A Single Rider line at 0 min is usually not operating yet (e.g. at opening).
        if solo and (standby is None or solo < standby):
            line, real, line_id = "single_rider", solo, ride.single_rider_id
    scheduling = real if real is not None else _no_data_wait(ride, hour)
    return real, scheduling, line, line_id


@tool_guard
def plan_day(
    park: ParkKey,
    date: dt.date,
    start: Annotated[str, _TIME] = "09:00",
    end: Annotated[str, _TIME] | None = None,
    preference: Preference = "all",
    single_rider: bool = False,
    extra_magic_hours: bool = False,
) -> DayPlan | ToolError:
    """Build an optimized, timed day plan in ONE park: the most rides with the least waiting,
    using the usual wait of each ride at each hour, walking between lands, off-peak meals,
    the weather (indoor rides during rain), the official opening hours and the shows
    (today's parade and night show times when published). Use it for any request of a day or
    half-day program; then present its steps faithfully (do not add rides it did not plan).

    Args:
        park: "disneyland_park" or "adventure_world" (one park per plan).
        date: day of the visit, YYYY-MM-DD.
        start: arrival time, HH:MM (e.g. "08:30").
        end: departure time, HH:MM; omit to stay until closing.
        preference: "thrill" (sensations first), "family" (gentle rides) or "all".
        single_rider: true only if the visitor is alone or accepts riding apart from the
            group: Single Rider lines are usually much shorter.
        extra_magic_hours: true only if the visitor stays at a Disney hotel (or says they
            have early entry): the plan then starts with the Extra Magic Hours.
    """
    now = now_paris()
    today = now.date()
    if date < today:
        raise ToolFailure(f"{date} is in the past: plan today ({today}) or a future date.")

    profile = _load_profile(park, date)
    notes: list[str] = []
    if profile.basis == "none":
        notes.append("No wait history available: rides are ordered by popularity, waits unknown.")
    elif profile.basis == "all_days":
        notes.append(
            f"No history yet for this weekday: waits are averages over all {profile.days} "
            "day(s) collected so far (rough estimates)."
        )
    elif profile.days < 14:
        notes.append(f"Waits based on {profile.days} day(s) of history: indicative only.")
    if (profile.basis != "none" and (date - today).days > FAR_DATE_DAYS
            and profile.season_days < MIN_SEASON_DAYS):  # fmt: skip
        notes.append("No history yet from this time of year: waits come from the last weeks, "
                     "crowds may differ (season, holidays).")  # fmt: skip

    official = _official_hours(park, date)
    if official is not None and not official.is_open:
        raise ToolFailure(f"{PARKS[park].name} is closed on {date} (official schedule).")
    early = None  # start of the Extra Magic Hours, when the visitor has them
    if official is not None:
        opening = _minutes(official.opening)
        closing = _closing_minutes(official.opening, official.closing)
        hours_note = f"Official opening hours: {official.opening}-{official.closing}"
        if official.extra_magic_hours:
            hours_note += f" (Extra Magic Hours {official.extra_magic_hours} for Disney hotels)"
            if extra_magic_hours:
                early = _minutes(official.extra_magic_hours.split("-")[0])
        notes.append(hours_note + ".")
        if extra_magic_hours and not official.extra_magic_hours:
            notes.append("No Extra Magic Hours on this day: the plan starts at opening.")
        if official.special_events:
            notes.append("Special ticketed event: " + "; ".join(official.special_events) + ".")
    else:
        opening, closing = _estimated_window(profile)
        notes.append(
            f"Official hours not published yet for this date: {_hhmm(opening)}-"
            f"{_hhmm(closing)} estimated from the history, check the official app."
        )
    if official is None or official.shows is None:
        notes.append("Parade and night show times are published on the day: check the app.")

    begin = max(_minutes(start), early or opening)
    if date == today:  # a plan for today starts now, not at opening
        begin = max(begin, now.hour * 60 + now.minute)
    finish = min(_minutes(end), closing) if end else closing
    if finish <= begin:
        if date == today and begin >= closing:
            raise ToolFailure(f"{PARKS[park].name} closes at {_hhmm(closing)} today.")
        raise ToolFailure(f"No time left between {start} and {_hhmm(finish)} to plan rides.")

    rainy_hours = _rainy_hours(date)
    if rainy_hours is None:
        notes.append("No weather forecast for this date: rain is not taken into account.")
    rainy = {int(h[:2]) for h in rainy_hours or []}

    candidates = [r for r in CATALOG if r.park == park and r.kind == "ride"]
    closed = {c.ride: c for c in official.closed_rides} if official else {}
    if closed_here := [r for r in candidates if r.name in closed]:
        candidates = [r for r in candidates if r.name not in closed]
        names = ", ".join(
            f"{r.name} (reopens {closed[r.name].reopens})" if closed[r.name].reopens
            else f"{r.name} (no reopening date published)" for r in closed_here
        )  # fmt: skip
        notes.append(f"Closed that day per the official schedule, not planned: {names}.")
    if profile.waits:
        # With history, a ride never observed open is likely closed (refurbishment): planning
        # it would send the visitor to a closed ride. Leave it out, and say so.
        silent = [r for r in candidates if r.ride_id not in profile.waits]
        candidates = [r for r in candidates if r.ride_id in profile.waits]
        if silent:
            names = ", ".join(r.name for r in silent)
            notes.append(f"Not planned, no recent data (possibly closed, check on site): {names}.")

    steps = []
    already_there = date == today and now.hour * 60 + now.minute > _minutes(start)
    if _minutes(start) < begin and not already_there:
        if official is None:
            arrival = f"Rides usually start around {_hhmm(begin)}."
        elif early is not None and begin == early:
            arrival = f"Extra Magic Hours from {_hhmm(early)}, park opens at {official.opening}."
        else:
            arrival = f"Park opens at {official.opening}."
        steps.append(PlanStep(time=_hhmm(_minutes(start)), kind="info",
                              title="Arrival and security checks", note=arrival))  # fmt: skip
    waits, today_vs_usual = Waits(profile), None
    live = _live_waits(park) if date == today else None
    if live:
        waits.live, waits.live_until = live, now.hour * 60 + now.minute + LIVE_HORIZON
        today_vs_usual = _today_vs_usual(live, profile, now.hour)
        live_note = "Live waits used for the next hour"
        if today_vs_usual is not None:
            low, high = TODAY_FACTOR_BOUNDS
            waits.factor = min(max(today_vs_usual, low), high)
            gap = round((today_vs_usual - 1) * 100)
            if abs(gap) < 10:
                live_note += "; today is about as busy as usual"
            else:
                trend = "busier" if gap > 0 else "quieter"
                live_note += (
                    f"; today is about {abs(gap)} % {trend} than usual, later waits adjusted"
                )
        notes.append(live_note + ".")

    events = _shows(park, official, preference, begin, finish, opening, closing)
    early_until = opening if begin < opening else None  # only with Extra Magic Hours
    steps += _schedule(candidates, waits, begin, finish, preference, single_rider, rainy,
                       events=events, early_until=early_until)  # fmt: skip
    if early_until is not None and not any(
        s.kind == "ride" and _minutes(s.time) < opening for s in steps
    ):
        early_until = None
        notes.append("Extra Magic Hours: no history yet of the rides running that early, so "
                     "the plan starts at opening; ask a Cast Member on site.")  # fmt: skip

    rides = [s for s in steps if s.kind == "ride"]
    return DayPlan(
        park=park, park_name=PARKS[park].name, date=date, start=_hhmm(begin), end=_hhmm(finish),
        preference=preference, single_rider=single_rider,
        extra_magic_hours=early_until is not None, park_hours=f"{_hhmm(opening)}-{_hhmm(closing)}",
        hours_source="official" if official is not None else "estimated",
        history_basis=profile.basis, live_waits=bool(live), today_vs_usual=today_vs_usual,
        days_of_data=profile.days, rainy_hours=rainy_hours, steps=steps,
        rides_count=len(rides),
        thrill_rides_count=sum(s.thrill in ("high", "medium") for s in rides),
        total_expected_wait=sum(s.expected_wait or 0 for s in rides), notes=notes,
    )  # fmt: skip


def _load_profile(park: str, date: dt.date) -> HourlyProfile:
    """Usual waits for this weekday, weighted toward the season of `date`."""
    try:
        return get_history_store().hourly_profile(park=park, weekday=date.weekday(), target=date)
    except Exception as exc:  # no history must not prevent a plan
        log.warning("history unavailable for the planner: %r", exc)
        return HourlyProfile(park=park, basis="none", days=0, waits={})


def _live_waits(park: ParkKey) -> dict[int, int | None] | None:
    """ride_id -> current wait (None = closed right now); None when the park is closed or the
    live source is unavailable (the plan then relies on the history only)."""
    live = get_live_wait_times(park=park)
    if isinstance(live, ToolError) or not live.park_open:
        return None
    return {r.ride_id: r.wait_time if r.is_open else None for r in live.rides}


def _today_vs_usual(live: dict[int, int | None], profile: HourlyProfile, hour: int):
    """Median of live / usual wait over the rides with a meaningful usual wait now."""
    ratios = [
        wait / usual
        for ride_id, wait in live.items()
        if wait is not None and (usual := _wait_at(profile, ride_id, hour)) and usual >= 5
    ]
    if len(ratios) < MIN_RIDES_FOR_TODAY_FACTOR:
        return None
    return round(statistics.median(ratios), 2)


def _official_hours(park: ParkKey, date: dt.date) -> ParkHours | None:
    """None when the schedule is not published yet for this date, or unreachable."""
    hours = get_park_hours(park=park, date=date)
    return None if isinstance(hours, ToolError) else hours


def _estimated_window(profile: HourlyProfile) -> tuple[int, int]:
    """Opening hours guessed from the history when no official schedule is published.

    Only hours when most rides had waits count (during Extra Magic Hours only a few run), and
    the history must cover mornings / evenings: a first partial day would suggest a 17:00
    closing.
    """
    opening, closing = DEFAULT_WINDOW
    rides_per_hour = Counter(h for by_hour in profile.waits.values() for h in by_hour)
    if not rides_per_hour:
        return opening, closing
    busiest = max(rides_per_hour.values())
    hours = sorted(h for h, n in rides_per_hour.items() if n >= busiest / 2)
    if hours[0] * 60 <= LATEST_PLAUSIBLE_OPENING:
        opening = hours[0] * 60
    if hours[-1] * 60 >= EARLIEST_PLAUSIBLE_CLOSING:
        closing = hours[-1] * 60 + 45
    return opening, closing


def _shows(
    park: ParkKey,
    official: ParkHours | None,
    preference: Preference,
    begin: int,
    finish: int,
    opening: int,
    closing: int,
) -> list[Event]:
    """The fixed events of the day: the night show and, unless thrills only, one parade
    (never in the first hour after opening, the best time for rides)."""
    shows = (official.shows if official else None) or []
    events: list[Event] = []
    night = next((s for s in shows if s.kind == "night_show"), None)
    if night is not None:
        step = PlanStep(time=night.times[-1], kind="show", title=f"Night show: {night.name}",
                        note="Official time today; find a spot about 20 min before.")  # fmt: skip
        events.append(Event(_minutes(night.times[-1]), *NIGHT_SHOW, step))
    elif park == "disneyland_park" and (official is None or official.shows is None):
        at = closing - NIGHT_SHOW_BEFORE_CLOSING
        note = (f"Usually shortly before closing ({_hhmm(closing)}): exact time published on "
                "the day in the official app; find a spot early.")  # fmt: skip
        step = PlanStep(time=_hhmm(at), kind="show", title="Night show at the castle", note=note)
        events.append(Event(at, *NIGHT_SHOW, step))

    if preference != "thrill":
        # The big parade rather than a smaller cavalcade; its last time that fits the visit.
        parades = sorted((s for s in shows if s.kind == "parade"),
                         key=lambda s: "parade" not in s.name.lower())  # fmt: skip
        for parade in parades[:1]:
            fitting = [t for t in parade.times
                       if max(begin, opening + 60) <= _minutes(t) - PARADE[0]
                       and _minutes(t) + PARADE[1] <= finish]  # fmt: skip
            if fitting:
                step = PlanStep(time=fitting[-1], kind="show", title=f"Parade: {parade.name}",
                                note="Official time today; find a spot about 15 min before. "
                                "Queues elsewhere drop during the parade.")  # fmt: skip
                events.append(Event(_minutes(fitting[-1]), *PARADE, step))
    # Only the shows inside the visit.
    return sorted((e for e in events if begin <= e.block_start and e.at <= finish),
                  key=lambda e: e.at)  # fmt: skip


def _rainy_hours(date: dt.date) -> list[str] | None:
    forecast = get_weather(date)
    return None if isinstance(forecast, ToolError) else forecast.rainy_hours


def _schedule(candidates, waits, begin, finish, preference, single_rider, rainy, *,
              events=(), early_until=None):  # fmt: skip
    """Greedy slot filling. `events` are shows at fixed times (sorted). Before `early_until`
    (Extra Magic Hours) only part of the rides run: those seen running in that early hour."""
    done: dict[int, int] = {}  # ride_id -> times ridden
    steps: list[PlanStep] = []
    now, land = begin, None
    had_lunch = had_dinner = False
    pending = list(events)
    early_rides = {ride_id for ride_id, by_hour in waits.profile.waits.items()
                   if begin // 60 in by_hour}  # fmt: skip

    while now < finish:
        # Time to head to a show: watch it, then carry on.
        if pending and now >= pending[0].block_start:
            event = pending.pop(0)
            steps.append(event.step)
            now, land = max(now, event.at + event.duration), None
            continue
        limit = min(finish, pending[0].block_start) if pending else finish  # next show

        # Meals only inside their off-peak windows (a plan starting at 14:00 has no lunch).
        if not had_lunch and LUNCH_AFTER <= now < 14 * 60 and now + MEAL_MINUTES <= limit:
            steps.append(PlanStep(time=_hhmm(now), kind="meal", title="Lunch",
                                  note="Quick-service, before the 12:00-14:00 peak."))  # fmt: skip
            now, had_lunch = now + MEAL_MINUTES, True
            continue
        if (not had_dinner and DINNER_AFTER <= now < 20 * 60 and finish >= 20 * 60 + 30
                and now + MEAL_MINUTES <= limit):  # fmt: skip
            steps.append(PlanStep(time=_hhmm(now), kind="meal", title="Dinner",
                                  note="Early dinner, to avoid the evening peak."))  # fmt: skip
            now, had_dinner = now + MEAL_MINUTES, True
            continue

        hour = now // 60
        early_hours = early_until is not None and now < early_until
        best, best_score, best_info = None, 0.0, None
        for ride in candidates:
            times = done.get(ride.ride_id, 0)
            # A second ride only for thrill rides of a thrill seeker; never a third.
            if times >= (2 if preference == "thrill" and ride.thrill == "high" else 1):
                continue
            if early_hours and ride.ride_id not in early_rides:
                continue  # not known to run during the Extra Magic Hours
            if waits.closed_now(ride.ride_id, now):
                continue  # closed right now (live): maybe later
            real, wait, line, line_id = _choose_line(ride, waits, hour, single_rider, now)
            walk = WALK_SAME_LAND if ride.land == land else WALK_OTHER_LAND
            if now + walk + wait + RIDE_MINUTES > limit:
                continue
            later_hours = range(hour + 1, max(hour + 2, finish // 60 + 1))
            later = [_choose_line(ride, waits, h, single_rider)[1] for h in later_hours]
            effective_wait = max(3.0, wait - 0.5 * (sum(later) / len(later) - wait))
            value = _value(ride, preference)
            if times:
                value *= REPEAT_FACTOR
            if not ride.indoor:
                if hour in rainy:
                    value *= RAIN_OUTDOOR_FACTOR
                elif any(h > hour for h in rainy):
                    value *= DRY_BEFORE_RAIN_OUTDOOR_FACTOR
            score = value / (effective_wait + walk + RIDE_MINUTES)
            if score > best_score:
                best, best_score = ride, score
                best_info = (real, wait, line, walk, waits.is_live(line_id, now))
        if best is None:
            # Nothing fits: skip ahead to the regular opening or to the next show, or stop.
            resume = [t for t in (early_until if early_hours else None,
                                  pending[0].block_start if pending else None) if t]  # fmt: skip
            if not resume:
                break
            now = min(resume)
            continue

        real, wait, line, walk, live = best_info
        steps.append(
            PlanStep(
                time=_hhmm(now + walk), kind="ride", title=best.name, land=best.land, line=line,
                expected_wait=round(real) if real is not None else None, thrill=best.thrill,
                indoor=best.indoor, live_wait=live or None,
                note="second ride" if done.get(best.ride_id) else None,
            )
        )  # fmt: skip
        done[best.ride_id] = done.get(best.ride_id, 0) + 1
        now += walk + round(wait) + RIDE_MINUTES
        land = best.land
    return steps
