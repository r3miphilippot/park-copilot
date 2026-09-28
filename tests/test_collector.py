import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import mongomock
import pytest
import respx

from app.config import PARKS, QUEUE_TIMES_BASE_URL
from app.queue_times import fetch_queue_times, iter_rides
from collector.collect import build_snapshots, ensure_indexes, exit_code, run, save_snapshots

PARK = PARKS["disneyland_park"]
OTHER_PARK = PARKS["adventure_world"]
OPEN_PAYLOAD = json.loads((Path(__file__).parent / "fixtures/queue_times_open.json").read_text())
CLOSED_PAYLOAD = {
    "lands": [{"name": "Adventureland", "rides": [{"id": 3, "name": "X", "is_open": False}]}],
    "rides": [],
}
# 2026-07-14 is a Tuesday; 08:30 UTC = 10:30 in Paris (summer time, UTC+2).
NOW = datetime(2026, 7, 14, 8, 30, tzinfo=UTC)


def url(park):
    return f"{QUEUE_TIMES_BASE_URL}/parks/{park.queue_times_id}/queue_times.json"


@pytest.fixture
def collection():
    return mongomock.MongoClient()["test"]["wait_snapshots"]


# --------------------------------------------------------------------------- transform


def test_iter_rides_reads_lands_and_root_without_duplicates():
    rides = list(iter_rides(OPEN_PAYLOAD))
    assert [r["id"] for _, r in rides] == [3, 2702, 5, 99]
    assert dict((r["id"], land) for land, r in rides)[99] is None


def test_build_snapshots_fields():
    docs = build_snapshots(PARK, OPEN_PAYLOAD, NOW)
    pirates = next(d for d in docs if d["ride_id"] == 3)
    assert pirates == {
        "park": "disneyland_park",
        "land": "Adventureland",
        "ride": "Pirates of the Caribbean",
        "ride_id": 3,
        "is_open": True,
        "wait_time": 25,
        "source_updated_at": datetime(2026, 7, 14, 8, 25, tzinfo=UTC),
        "fetched_at": NOW,
        "weekday": 1,  # Tuesday
        "hour": 10,  # Paris time, not UTC
    }


def test_closed_ride_has_no_wait_time():
    docs = build_snapshots(PARK, OPEN_PAYLOAD, NOW)
    isle = next(d for d in docs if d["ride_id"] == 2702)
    assert isle["is_open"] is False and isle["wait_time"] is None


def test_winter_time_hour_is_paris_time():
    payload = {
        "rides": [
            {
                "id": 1,
                "name": "A",
                "is_open": True,
                "wait_time": 5,
                "last_updated": "2026-12-01T09:00:00Z",
            }
        ]
    }
    [doc] = build_snapshots(PARK, payload, NOW)
    assert doc["hour"] == 10  # UTC+1 in winter


def test_missing_source_timestamp_falls_back_to_fetch_time():
    payload = {"rides": [{"id": 1, "name": "A", "is_open": True, "wait_time": 5}]}
    [doc] = build_snapshots(PARK, payload, NOW)
    assert doc["source_updated_at"] == NOW


def test_closed_park_produces_nothing():
    assert build_snapshots(PARK, CLOSED_PAYLOAD, NOW) == []
    assert build_snapshots(PARK, {"lands": [], "rides": []}, NOW) == []


# --------------------------------------------------------------------------- store


def test_save_is_idempotent(collection):
    ensure_indexes(collection)
    docs = build_snapshots(PARK, OPEN_PAYLOAD, NOW)
    assert save_snapshots(collection, docs) == 4
    # Same source data fetched again 15 minutes later: nothing new.
    assert save_snapshots(collection, build_snapshots(PARK, OPEN_PAYLOAD, NOW)) == 0
    assert collection.count_documents({}) == 4


def test_indexes_are_created(collection):
    ensure_indexes(collection)
    info = collection.index_information()
    assert {"ride_weekday_hour", "fetched_at", "ride_source_updated_unique"} <= set(info)
    assert info["ride_source_updated_unique"]["unique"] is True


# --------------------------------------------------------------------------- fetch


@respx.mock
def test_fetch_retries_on_server_error():
    route = respx.get(url(PARK)).mock(
        side_effect=[httpx.Response(503), httpx.Response(200, json=OPEN_PAYLOAD)]
    )
    with httpx.Client() as client:
        assert fetch_queue_times(client, PARK, backoff_s=0) == OPEN_PAYLOAD
    assert route.call_count == 2


@respx.mock
def test_fetch_retries_on_timeout_then_gives_up():
    route = respx.get(url(PARK)).mock(side_effect=httpx.ConnectTimeout("boom"))
    with httpx.Client() as client, pytest.raises(httpx.ConnectTimeout):
        fetch_queue_times(client, PARK, attempts=3, backoff_s=0)
    assert route.call_count == 3


@respx.mock
def test_fetch_does_not_retry_client_errors():
    route = respx.get(url(PARK)).mock(return_value=httpx.Response(404))
    with httpx.Client() as client, pytest.raises(httpx.HTTPStatusError):
        fetch_queue_times(client, PARK, backoff_s=0)
    assert route.call_count == 1


# --------------------------------------------------------------------------- orchestration


@respx.mock
def test_run_one_park_open_one_closed(collection):
    respx.get(url(PARK)).mock(return_value=httpx.Response(200, json=OPEN_PAYLOAD))
    respx.get(url(OTHER_PARK)).mock(return_value=httpx.Response(200, json=CLOSED_PAYLOAD))
    with httpx.Client() as client:
        summary = run(client, collection, now=NOW, backoff_s=0)
    assert summary["disneyland_park"] == {"status": "ok", "rides": 4, "inserted": 4}
    assert summary["adventure_world"]["status"] == "closed"
    assert collection.count_documents({"park": "adventure_world"}) == 0
    assert exit_code(summary) == 0


@respx.mock
def test_partial_failure_is_not_fatal(collection):
    respx.get(url(PARK)).mock(return_value=httpx.Response(200, json=OPEN_PAYLOAD))
    respx.get(url(OTHER_PARK)).mock(return_value=httpx.Response(500))
    with httpx.Client() as client:
        summary = run(client, collection, now=NOW, backoff_s=0)
    assert summary["adventure_world"]["status"] == "error"
    assert exit_code(summary) == 0


@respx.mock
def test_total_failure_exits_non_zero(collection):
    respx.get(url(PARK)).mock(return_value=httpx.Response(500))
    respx.get(url(OTHER_PARK)).mock(side_effect=httpx.ConnectError("down"))
    with httpx.Client() as client:
        summary = run(client, collection, now=NOW, backoff_s=0)
    assert exit_code(summary) == 1
