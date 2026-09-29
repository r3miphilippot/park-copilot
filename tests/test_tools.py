import inspect
import json
from datetime import date, datetime
from pathlib import Path

import httpx
import pytest
import respx
from pymongo.errors import ServerSelectionTimeoutError

from app.clock import frozen_now
from app.config import PARIS_TZ, PARKS, QUEUE_TIMES_BASE_URL
from app.db import DatabaseNotConfigured
from app.tools import (
    TOOLS,
    ToolError,
    compare_live_vs_typical,
    get_live_wait_times,
    get_typical_wait,
    get_weather,
)
from app.tools import history as history_module
from app.tools.compare import classify
from app.tools.history import MongoHistoryStore
from app.tools.weather import OPEN_METEO_URL
from tests.conftest import FakeHistoryStore, stats

FIXTURES = Path(__file__).parent / "fixtures"
OPEN_PAYLOAD = json.loads((FIXTURES / "queue_times_open.json").read_text())
LIVE_URL = (
    f"{QUEUE_TIMES_BASE_URL}/parks/{PARKS['disneyland_park'].queue_times_id}/queue_times.json"
)
TUESDAY_1030 = datetime(2026, 7, 14, 10, 30, tzinfo=PARIS_TZ)


# --------------------------------------------------------------------------- contract


def test_every_tool_keeps_signature_and_docstring():
    """LangChain and FastMCP build the LLM schema from these."""
    for tool in TOOLS:
        assert tool.__doc__ and len(tool.__doc__) > 50, tool.__name__
        assert inspect.signature(tool).parameters, tool.__name__


def test_invalid_argument_returns_tool_error():
    result = get_live_wait_times("parc_asterix")
    assert isinstance(result, ToolError)
    assert result.tool == "get_live_wait_times"
    assert result.error.startswith("Invalid arguments")


# --------------------------------------------------------------------------- live


@respx.mock
def test_live_wait_times_sorted_and_cached():
    route = respx.get(LIVE_URL).mock(return_value=httpx.Response(200, json=OPEN_PAYLOAD))
    result = get_live_wait_times("disneyland_park")
    assert result.park_open is True
    assert [(r.ride_id, r.wait_time) for r in result.rides] == [
        (99, 10), (3, 25), (5, 40), (2702, None),  # open by wait, closed last
    ]  # fmt: skip
    get_live_wait_times("disneyland_park")
    assert route.call_count == 1  # second call served from the 5-minute cache


@respx.mock
def test_live_api_down_is_reported_not_raised():
    respx.get(LIVE_URL).mock(side_effect=httpx.ConnectError("down"))
    result = get_live_wait_times("disneyland_park")
    assert isinstance(result, ToolError)
    assert "unavailable" in result.error


# --------------------------------------------------------------------------- history


def test_typical_wait_reports_days_of_data(use_store):
    store = use_store(FakeHistoryStore([stats(3, "Pirates of the Caribbean", 22.5)], days=30))
    result = get_typical_wait(ride="pirates", weekday=1, hour=10)
    assert result.coverage.days == 30
    assert result.rides[0].days_observed == 8
    assert result.filters == {"ride": "pirates", "weekday": "1 (Tuesday)", "hour": 10}
    assert result.note is None
    assert store.calls == [{"park": None, "ride": "pirates", "weekday": 1, "hour": 10}]


def test_typical_wait_warns_when_history_is_short(use_store):
    use_store(FakeHistoryStore([stats(3, "Pirates", 20)], days=3))
    assert "Only 3 day(s)" in get_typical_wait().note


def test_typical_wait_without_any_history(use_store):
    use_store(FakeHistoryStore([], days=0))
    result = get_typical_wait(ride="pirates")
    assert result.rides == []
    assert "do not invent" in result.note


def test_typical_wait_rejects_bad_weekday(use_store):
    use_store(FakeHistoryStore([]))
    result = get_typical_wait(weekday=7)
    assert isinstance(result, ToolError) and "weekday" in result.error


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (DatabaseNotConfigured("no uri"), "not configured"),
        (ServerSelectionTimeoutError("timeout"), "unavailable"),
    ],
)
def test_database_problems_are_reported(monkeypatch, exc, expected):
    def boom():
        raise exc

    monkeypatch.setattr(history_module, "get_history_store", boom)
    result = get_typical_wait()
    assert isinstance(result, ToolError) and expected in result.error


def test_stats_pipeline_filters_and_escapes_ride_name():
    pipeline = MongoHistoryStore.build_stats_pipeline(
        park="disneyland_park", ride="Big Thunder (Mountain)", weekday=5, hour=14
    )
    match = pipeline[0]["$match"]
    assert match["is_open"] is True and match["wait_time"] == {"$ne": None}
    assert match["ride"] == {"$regex": r"Big\ Thunder\ \(Mountain\)", "$options": "i"}
    assert (match["park"], match["weekday"], match["hour"]) == ("disneyland_park", 5, 14)


def test_stats_pipeline_can_exclude_recent_snapshots():
    cutoff = datetime(2026, 7, 14, tzinfo=PARIS_TZ)
    match = MongoHistoryStore.build_stats_pipeline(
        park=None, ride=None, weekday=1, hour=10, before=cutoff
    )[0]["$match"]
    assert match["fetched_at"] == {"$lt": cutoff}


def test_stats_pipeline_without_filters_keeps_only_open_rides():
    match = MongoHistoryStore.build_stats_pipeline(park=None, ride=None, weekday=None, hour=None)[
        0
    ]["$match"]
    assert set(match) == {"is_open", "wait_time"}


# --------------------------------------------------------------------------- compare


@pytest.mark.parametrize(
    ("live", "typical", "verdict"),
    [
        (10, 12, "usual"),  # 2 min gap: noise
        (50, 60, "usual"),  # 10 min but < 20 % of 60
        (30, 60, "shorter_than_usual"),
        (45, 25, "longer_than_usual"),
    ],
)
def test_classify(live, typical, verdict):
    assert classify(live, typical) == verdict


@respx.mock
def test_compare_live_vs_typical(use_store):
    respx.get(LIVE_URL).mock(return_value=httpx.Response(200, json=OPEN_PAYLOAD))
    store = use_store(
        FakeHistoryStore([stats(3, "Pirates of the Caribbean", 50), stats(5, "Buzz", 20)])
    )
    with frozen_now(TUESDAY_1030):
        result = compare_live_vs_typical("disneyland_park")

    # History is looked up for the simulated moment: Tuesday (1), 10h Paris.
    assert store.calls[0]["weekday"] == 1 and store.calls[0]["hour"] == 10
    # "Usual" excludes today, otherwise the live waits would be compared with themselves.
    assert store.calls[0]["before"] == datetime(2026, 7, 14, 0, 0, tzinfo=PARIS_TZ)
    assert (result.weekday, result.hour) == (1, 10)
    assert [(r.ride, r.verdict, r.delta_minutes) for r in result.rides] == [
        ("Pirates of the Caribbean", "shorter_than_usual", -25.0),
        ("Buzz Lightyear Laser Blast", "longer_than_usual", 20.0),
        ("Standalone Ride", "no_history", None),
    ]
    assert all(r.ride != "Adventure Isle" for r in result.rides)  # closed rides skipped


@respx.mock
def test_compare_propagates_live_failure(use_store):
    use_store(FakeHistoryStore([]))
    respx.get(LIVE_URL).mock(return_value=httpx.Response(503))
    result = compare_live_vs_typical("disneyland_park")
    assert isinstance(result, ToolError) and result.tool == "compare_live_vs_typical"


# --------------------------------------------------------------------------- weather


def _hourly(rain_at: dict[int, tuple[float, int]]) -> dict:
    """Open-Meteo-like hourly block for 2026-07-15; rain_at = {hour: (mm, probability)}."""
    hours = range(24)
    return {
        "hourly": {
            "time": [f"2026-07-15T{h:02d}:00" for h in hours],
            "temperature_2m": [15 + h / 2 for h in hours],
            "apparent_temperature": [14 + h / 2 for h in hours],
            "precipitation": [rain_at.get(h, (0.0, 0))[0] for h in hours],
            "precipitation_probability": [rain_at.get(h, (0.0, 0))[1] for h in hours],
            "weather_code": [61 if h in rain_at else 1 for h in hours],
            "wind_speed_10m": [10.0 for _ in hours],
        }
    }


@respx.mock
def test_weather_rainy_afternoon():
    route = respx.get(url__startswith=OPEN_METEO_URL).mock(
        return_value=httpx.Response(200, json=_hourly({3: (5.0, 90), 14: (1.2, 80), 15: (0.0, 60)}))
    )
    with frozen_now(TUESDAY_1030):
        result = get_weather("2026-07-15")
        get_weather(date(2026, 7, 15))

    assert route.call_count == 1  # cached for one hour
    assert route.calls[0].request.url.params["start_date"] == "2026-07-15"
    assert [h.time for h in result.hours] == [f"{h:02d}:00" for h in range(8, 24)]
    assert result.rainy_hours == ["14:00", "15:00"]  # 03:00 is outside park hours
    assert result.rain_expected is True
    assert result.total_precipitation_mm == 1.2
    assert result.hours[6].conditions == "light rain"  # 14:00
    assert (result.min_temp_c, result.max_temp_c) == (19.0, 26.5)


@pytest.mark.parametrize(
    ("day", "expected"), [("2026-07-13", "in the past"), ("2026-07-30", "No forecast yet")]
)
def test_weather_out_of_range_dates(day, expected):
    with frozen_now(TUESDAY_1030):
        result = get_weather(day)
    assert isinstance(result, ToolError) and expected in result.error


@respx.mock
def test_weather_api_error_is_reported():
    respx.get(url__startswith=OPEN_METEO_URL).mock(return_value=httpx.Response(500))
    with frozen_now(TUESDAY_1030):
        result = get_weather("2026-07-15")
    assert isinstance(result, ToolError) and "unavailable" in result.error
