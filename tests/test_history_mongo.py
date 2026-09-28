"""Integration test of the real MongoDB aggregation ($median, Paris days).

Runs only when a real MONGODB_URI is configured (local .env); skipped in CI.
Works in a throwaway collection, dropped at the end.
"""

import uuid
from datetime import UTC, datetime

import pytest
from pymongo import MongoClient

from app.config import get_settings
from app.tools.history import MongoHistoryStore

settings = get_settings()
pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not settings.mongodb_uri or "<" in settings.mongodb_uri,
        reason="needs a real MONGODB_URI",
    ),
]


def snap(ride_id, ride, wait, fetched_at, weekday=1, hour=10, is_open=True):
    return {
        "park": "disneyland_park", "land": "Adventureland", "ride": ride, "ride_id": ride_id,
        "is_open": is_open, "wait_time": wait if is_open else None,
        "source_updated_at": fetched_at, "fetched_at": fetched_at,
        "weekday": weekday, "hour": hour,
    }  # fmt: skip


@pytest.fixture
def collection():
    client = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000, tz_aware=True)
    coll = client[settings.mongodb_db][f"it_{uuid.uuid4().hex[:8]}"]
    yield coll
    coll.drop()
    client.close()


def test_real_aggregation(collection):
    collection.insert_many(
        [
            # Pirates: 3 Tuesdays at 10h -> avg 20, median 20, 3 days.
            snap(3, "Pirates of the Caribbean", 10, datetime(2026, 7, 7, 8, 0, tzinfo=UTC)),
            snap(3, "Pirates of the Caribbean", 20, datetime(2026, 7, 14, 8, 0, tzinfo=UTC)),
            snap(3, "Pirates of the Caribbean", 30, datetime(2026, 7, 21, 8, 0, tzinfo=UTC)),
            # Closed snapshot: must be ignored.
            snap(
                3,
                "Pirates of the Caribbean",
                0,
                datetime(2026, 7, 21, 8, 15, tzinfo=UTC),
                is_open=False,
            ),
            # Other hour: filtered out by hour=10.
            snap(
                3, "Pirates of the Caribbean", 90, datetime(2026, 7, 21, 13, 0, tzinfo=UTC), hour=15
            ),
            # 22:30 UTC on July 21 is already July 22 in Paris -> a 4th distinct day.
            snap(5, "Buzz Lightyear", 15, datetime(2026, 7, 21, 22, 30, tzinfo=UTC), hour=0),
        ]
    )
    store = MongoHistoryStore(collection)

    [pirates] = store.ride_stats(park=None, ride="PIRATES", weekday=1, hour=10)
    assert (pirates.avg_wait, pirates.median_wait) == (20.0, 20.0)
    assert (pirates.observations, pirates.days_observed) == (3, 3)

    coverage = store.coverage()
    assert coverage.days == 4
    assert (str(coverage.first_day), str(coverage.last_day)) == ("2026-07-07", "2026-07-22")
