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
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Annotated, Literal

from pydantic import BaseModel, Field

from app.clock import now_paris
from app.config import PARKS, ParkKey
from app.tools.common import ToolError, ToolFailure, tool_guard
from app.tools.history import HourlyProfile, get_history_store
from app.tools.rides import CATALOG, Ride
from app.tools.weather import get_weather

log = logging.getLogger(__name__)

Preference = Literal["thrill", "family", "all"]

RIDE_MINUTES = 8  # boarding + ride + exit
WALK_SAME_LAND = 5
WALK_OTHER_LAND = 12
MEAL_MINUTES = 45
LUNCH_AFTER, DINNER_AFTER = 11 * 60 + 30, 18 * 60 + 15  # off-peak meals (guide)
DEFAULT_WINDOW = (9 * 60 + 30, 22 * 60)  # when the history cannot tell opening hours
LATEST_PLAUSIBLE_OPENING = 11 * 60  # earlier data than this = the history covers mornings
EARLIEST_PLAUSIBLE_CLOSING = 20 * 60  # later data than this = the history covers evenings
REPEAT_FACTOR = 0.4  # a second ride of the same attraction is worth less
RAIN_OUTDOOR_FACTOR = 0.3  # an outdoor ride during rain is worth less
# Rain expected later: do outdoor rides while it is dry, keep indoor ones for the showers
# (a purely greedy plan used up the indoor rides in the dry morning).
DRY_BEFORE_RAIN_OUTDOOR_FACTOR = 1.6

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
    thrill: str | None = None
    indoor: bool | None = None
    note: str | None = None


class DayPlan(BaseModel):
    park: ParkKey
    park_name: str
    date: dt.date
    start: str
    end: str
    preference: Preference
    single_rider: bool
    history_basis: str  # same_weekday | all_days | none
    days_of_data: int
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
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _wait_at(profile: HourlyProfile, ride_id: int | None, hour: int) -> float | None:
    """Usual wait at this hour, or at the nearest hour with data; None without any data."""
    by_hour = profile.waits.get(ride_id) if ride_id is not None else None
    if not by_hour:
        return None
    nearest = min(by_hour, key=lambda h: abs(h - hour))
    return by_hour[nearest]


def _no_data_wait(ride: Ride, hour: int) -> float:
    curve = 0.4 if hour < 11 else (1.2 if hour < 17 else 0.8)
    return NO_DATA_WAIT[ride.popularity] * curve


def _value(ride: Ride, preference: Preference) -> float:
    values = VALUES[preference]
    if preference != "all" and ride.thrill in values:
        return values[ride.thrill]
    return values[ride.popularity]


def _choose_line(ride: Ride, profile: HourlyProfile, hour: int, single_rider: bool):
    """(real wait or None, wait used for scheduling, line)."""
    standby = _wait_at(profile, ride.ride_id, hour)
    line, real = "standby", standby
    if single_rider and ride.single_rider_id:
        solo = _wait_at(profile, ride.single_rider_id, hour)
        # A Single Rider line at 0 min is usually not operating yet (e.g. at opening).
        if solo and (standby is None or solo < standby):
            line, real = "single_rider", solo
    scheduling = real if real is not None else _no_data_wait(ride, hour)
    return real, scheduling, line


@tool_guard
def plan_day(
    park: ParkKey,
    date: dt.date,
    start: Annotated[str, _TIME] = "09:00",
    end: Annotated[str, _TIME] | None = None,
    preference: Preference = "all",
    single_rider: bool = False,
) -> DayPlan | ToolError:
    """Build an optimized, timed day plan in ONE park: the most rides with the least waiting,
    using the usual wait of each ride at each hour, walking between lands, off-peak meals and
    the weather (indoor rides during rain). Use it for any request of a day or half-day
    program; then present its steps faithfully (do not add rides it did not plan).

    Args:
        park: "disneyland_park" or "adventure_world" (one park per plan).
        date: day of the visit, YYYY-MM-DD.
        start: arrival time, HH:MM (e.g. "08:30").
        end: departure time, HH:MM; omit to stay until closing.
        preference: "thrill" (sensations first), "family" (gentle rides) or "all".
        single_rider: true only if the visitor is alone or accepts riding apart from the
            group: Single Rider lines are usually much shorter.
    """
    today = now_paris().date()
    if date < today:
        raise ToolFailure(f"{date} is in the past: plan today ({today}) or a future date.")

    profile = _load_profile(park, date.weekday())
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

    # Opening hours are not in any data source: infer them from when rides had waits, but
    # only if the history covers mornings / evenings (a first partial day would suggest a
    # 17:00 closing).
    hours = sorted({h for by_hour in profile.waits.values() for h in by_hour})
    opening, closing = DEFAULT_WINDOW
    if hours and hours[0] * 60 <= LATEST_PLAUSIBLE_OPENING:
        opening = hours[0] * 60
    if hours and hours[-1] * 60 >= EARLIEST_PLAUSIBLE_CLOSING:
        closing = hours[-1] * 60 + 45
    notes.append("Opening, parade and show times: check the official app on the day.")
    begin = max(_minutes(start), opening)
    finish = min(_minutes(end), closing) if end else closing
    if finish <= begin:
        raise ToolFailure(f"No time left between {start} and {_hhmm(finish)} to plan rides.")

    rainy_hours = _rainy_hours(date)
    if rainy_hours is None:
        notes.append("No weather forecast for this date: rain is not taken into account.")
    rainy = {int(h[:2]) for h in rainy_hours or []}

    candidates = [r for r in CATALOG if r.park == park and r.kind == "ride"]
    if profile.waits:
        # With history, a ride never observed open is likely closed (refurbishment): planning
        # it would send the visitor to a closed ride. Leave it out, and say so.
        silent = [r for r in candidates if r.ride_id not in profile.waits]
        candidates = [r for r in candidates if r.ride_id in profile.waits]
        if silent:
            names = ", ".join(r.name for r in silent)
            notes.append(f"Not planned, no recent data (possibly closed, check on site): {names}.")

    steps = []
    if _minutes(start) < begin:
        steps.append(PlanStep(time=_hhmm(_minutes(start)), kind="info",
                              title="Arrival and security checks",
                              note=f"Rides usually start around {_hhmm(begin)}."))  # fmt: skip
    steps += _schedule(candidates, profile, begin, finish, preference, single_rider, rainy)
    if park == "disneyland_park" and finish >= 20 * 60:
        show_note = "Exact time to check in the official app; arrive early."
        steps.append(PlanStep(time=_hhmm(finish), kind="show", title="Night show at the castle",
                              note=show_note))  # fmt: skip

    rides = [s for s in steps if s.kind == "ride"]
    return DayPlan(
        park=park, park_name=PARKS[park].name, date=date, start=_hhmm(begin), end=_hhmm(finish),
        preference=preference, single_rider=single_rider, history_basis=profile.basis,
        days_of_data=profile.days, rainy_hours=rainy_hours, steps=steps,
        rides_count=len(rides),
        thrill_rides_count=sum(s.thrill in ("high", "medium") for s in rides),
        total_expected_wait=sum(s.expected_wait or 0 for s in rides), notes=notes,
    )  # fmt: skip


def _load_profile(park: str, weekday: int) -> HourlyProfile:
    try:
        return get_history_store().hourly_profile(park=park, weekday=weekday)
    except Exception as exc:  # no history must not prevent a plan
        log.warning("history unavailable for the planner: %r", exc)
        return HourlyProfile(park=park, basis="none", days=0, waits={})


def _rainy_hours(date: dt.date) -> list[str] | None:
    forecast = get_weather(date)
    return None if isinstance(forecast, ToolError) else forecast.rainy_hours


def _schedule(candidates, profile, begin, finish, preference, single_rider, rainy):
    done: dict[int, int] = {}  # ride_id -> times ridden
    steps: list[PlanStep] = []
    now, land = begin, None
    had_lunch = had_dinner = False

    while now < finish:
        # Meals only inside their off-peak windows (a plan starting at 14:00 has no lunch).
        if not had_lunch and LUNCH_AFTER <= now < 14 * 60:
            steps.append(PlanStep(time=_hhmm(now), kind="meal", title="Lunch",
                                  note="Quick-service, before the 12:00-14:00 peak."))  # fmt: skip
            now, had_lunch = now + MEAL_MINUTES, True
            continue
        if not had_dinner and DINNER_AFTER <= now < 20 * 60 and finish >= 20 * 60 + 30:
            steps.append(PlanStep(time=_hhmm(now), kind="meal", title="Dinner",
                                  note="Before 19:00 to avoid the evening peak."))  # fmt: skip
            now, had_dinner = now + MEAL_MINUTES, True
            continue

        hour = now // 60
        best, best_score, best_info = None, 0.0, None
        for ride in candidates:
            times = done.get(ride.ride_id, 0)
            # A second ride only for thrill rides of a thrill seeker; never a third.
            if times >= (2 if preference == "thrill" and ride.thrill == "high" else 1):
                continue
            real, wait, line = _choose_line(ride, profile, hour, single_rider)
            walk = WALK_SAME_LAND if ride.land == land else WALK_OTHER_LAND
            if now + walk + wait + RIDE_MINUTES > finish:
                continue
            later_hours = range(hour + 1, max(hour + 2, finish // 60 + 1))
            later = [_choose_line(ride, profile, h, single_rider)[1] for h in later_hours]
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
                best, best_score, best_info = ride, score, (real, wait, line, walk)
        if best is None:
            break

        real, wait, line, walk = best_info
        steps.append(
            PlanStep(
                time=_hhmm(now + walk), kind="ride", title=best.name, land=best.land, line=line,
                expected_wait=round(real) if real is not None else None, thrill=best.thrill,
                indoor=best.indoor,
                note="second ride" if done.get(best.ride_id) else None,
            )
        )  # fmt: skip
        done[best.ride_id] = done.get(best.ride_id, 0) + 1
        now += walk + round(wait) + RIDE_MINUTES
        land = best.land
    return steps
