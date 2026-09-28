"""Wait-time collector: Queue-Times -> MongoDB collection `wait_snapshots`.

Runs every 15 minutes from .github/workflows/collect.yml. One document per ride per run,
which later feeds the "typical wait" aggregations (by ride, weekday and hour).

Usage:
    uv run python -m collector.collect             # fetch and write to MongoDB
    uv run python -m collector.collect --dry-run   # fetch and print, no database needed

Design notes:
- Idempotent: each document is keyed on (ride_id, source_updated_at). Running the job twice
  on the same Queue-Times data stores nothing new (unique index, duplicates skipped).
- Robust: HTTP timeout, simple retry with backoff on network/5xx/429 errors, and one park
  failing does not stop the other. The exit code is non-zero only on total failure.
- A closed park (every ride closed) writes nothing, so nights don't pollute the averages.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

import httpx
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

from app.config import PARIS_TZ, PARKS, Park, get_settings
from app.db import SNAPSHOTS_COLLECTION
from app.queue_times import HTTP_TIMEOUT_S, USER_AGENT, fetch_queue_times, iter_rides

log = logging.getLogger("collector")


# --------------------------------------------------------------------------- transform


def _parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value).astimezone(UTC)
    except ValueError:
        return None


def build_snapshots(park: Park, payload: dict[str, Any], fetched_at: datetime) -> list[dict]:
    """Turn one Queue-Times payload into `wait_snapshots` documents.

    Returns an empty list when the park is closed (no ride open).
    """
    rides = list(iter_rides(payload))
    if not any(ride.get("is_open") for _, ride in rides):
        return []

    docs = []
    for land, ride in rides:
        is_open = bool(ride.get("is_open"))
        # When the source timestamp is missing we fall back to our own fetch time,
        # so the idempotency key (ride_id, source_updated_at) is never null.
        observed_at = _parse_utc(ride.get("last_updated")) or fetched_at
        # weekday/hour are bucketed on the moment the wait was observed, in Paris time.
        local = observed_at.astimezone(PARIS_TZ)
        docs.append(
            {
                "park": park.key,
                "land": land,
                "ride": ride.get("name"),
                "ride_id": ride["id"],
                "is_open": is_open,
                # Queue-Times reports 0 for closed rides: store None so a closed ride can
                # never drag an average down.
                "wait_time": ride.get("wait_time") if is_open else None,
                "source_updated_at": observed_at,
                "fetched_at": fetched_at,
                "weekday": local.weekday(),  # 0 = Monday ... 6 = Sunday
                "hour": local.hour,
            }
        )
    return docs


# --------------------------------------------------------------------------- store


def ensure_indexes(collection: Collection) -> None:
    """Create indexes (no-op when they already exist)."""
    # Serves get_typical_wait(ride, weekday, hour) aggregations.
    collection.create_index(
        [("ride_id", ASCENDING), ("weekday", ASCENDING), ("hour", ASCENDING)],
        name="ride_weekday_hour",
    )
    # Serves "how many days of data" and time-window queries.
    collection.create_index([("fetched_at", ASCENDING)], name="fetched_at")
    # Idempotency key.
    collection.create_index(
        [("ride_id", ASCENDING), ("source_updated_at", ASCENDING)],
        name="ride_source_updated_unique",
        unique=True,
    )


DUPLICATE_KEY = 11000


def save_snapshots(collection: Collection, docs: list[dict]) -> int:
    """Insert documents, skipping those already stored; returns how many were new.

    `ordered=False` keeps inserting after a duplicate, and the unique index on
    (ride_id, source_updated_at) rejects the rows we already have.
    """
    if not docs:
        return 0
    try:
        return len(collection.insert_many(docs, ordered=False).inserted_ids)
    except BulkWriteError as exc:
        # Duplicates are expected on re-runs; anything else is a real error.
        if any(e.get("code") != DUPLICATE_KEY for e in exc.details.get("writeErrors", [])):
            raise
        return exc.details.get("nInserted", 0)


# --------------------------------------------------------------------------- orchestration


def run(
    client: httpx.Client,
    collection: Collection | None,
    *,
    parks: list[Park] | None = None,
    now: datetime | None = None,
    backoff_s: float = 2.0,
) -> dict[str, dict]:
    """Collect every park. `collection=None` means dry run (nothing written).

    Returns a per-park summary: {"status": "ok" | "closed" | "error", ...}.
    """
    fetched_at = now or datetime.now(UTC)
    summary: dict[str, dict] = {}
    for park in parks or list(PARKS.values()):
        try:
            payload = fetch_queue_times(client, park, backoff_s=backoff_s)
            docs = build_snapshots(park, payload, fetched_at)
            if not docs:
                summary[park.key] = {"status": "closed", "rides": 0, "inserted": 0}
                continue
            inserted = save_snapshots(collection, docs) if collection is not None else 0
            summary[park.key] = {"status": "ok", "rides": len(docs), "inserted": inserted}
        except Exception as exc:  # one park failing must not stop the others
            log.error("%s: collection failed: %r", park.key, exc)
            summary[park.key] = {"status": "error", "error": repr(exc)}
    return summary


def exit_code(summary: dict[str, dict]) -> int:
    """Non-zero only when every park failed (a closed park is a success)."""
    return 1 if all(s["status"] == "error" for s in summary.values()) else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="fetch only, do not write")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    collection = None
    mongo = None
    if not args.dry_run:
        settings = get_settings()
        if not settings.mongodb_uri:
            log.error("MONGODB_URI is not set (use --dry-run to test without a database)")
            return 2
        mongo = MongoClient(settings.mongodb_uri, serverSelectionTimeoutMS=10_000, tz_aware=True)
        collection = mongo[settings.mongodb_db][SNAPSHOTS_COLLECTION]
        try:
            ensure_indexes(collection)
        except Exception as exc:
            log.error("MongoDB unreachable: %r", exc)
            return 1

    try:
        with httpx.Client(timeout=HTTP_TIMEOUT_S, headers={"User-Agent": USER_AGENT}) as client:
            summary = run(client, collection)
    finally:
        if mongo is not None:
            mongo.close()

    print(json.dumps({"event": "collect_summary", "parks": summary}))
    return exit_code(summary)


if __name__ == "__main__":
    sys.exit(main())
