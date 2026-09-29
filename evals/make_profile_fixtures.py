"""Generate the frozen hourly history read by plan_day in the evals.

    uv run python -m evals.make_profile_fixtures

Synthetic but realistic day shapes, so the evals do not depend on the live database:
short queues at opening, a peak in the early afternoon, calmer evenings; headliners and
thrill rides wait longer; Single Rider lines about a third of the standby wait and closed at
opening. Regenerate only when the ride catalogue changes (then commit the JSON files).
"""

from __future__ import annotations

from pathlib import Path

from app.tools.history import HourlyProfile
from app.tools.rides import CATALOG

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BASE_WAIT = {"headliner": 50, "popular": 30, "standard": 12}
OPEN_HOURS = range(9, 23)


def day_curve(hour: int) -> float:
    return {9: 0.3, 10: 0.6, 11: 0.9}.get(hour, 1.25 if hour < 17 else 0.8)


def build(park: str) -> HourlyProfile:
    waits: dict[int, dict[int, float]] = {}
    for ride in CATALOG:
        if ride.park != park or ride.kind != "ride":
            continue
        base = BASE_WAIT[ride.popularity] * (1.2 if ride.thrill == "high" else 1.0)
        waits[ride.ride_id] = {h: round(base * day_curve(h), 1) for h in OPEN_HOURS}
        if ride.single_rider_id:
            waits[ride.single_rider_id] = {
                h: round(base * day_curve(h) / 3, 1) for h in OPEN_HOURS if h > OPEN_HOURS[0]
            }
    return HourlyProfile(park=park, basis="same_weekday", days=6, waits=waits)


def main() -> None:
    for park in ("disneyland_park", "adventure_world"):
        path = FIXTURES_DIR / f"profile_{park}.json"
        path.write_text(build(park).model_dump_json(indent=1) + "\n", encoding="utf-8")
        print(f"{path.name}: {len(build(park).waits)} series")


if __name__ == "__main__":
    main()
