"""MongoDB access (Atlas M0). The client is created lazily, once per process."""

from __future__ import annotations

from functools import lru_cache

from pymongo import MongoClient
from pymongo.collection import Collection

from app.config import get_settings

SNAPSHOTS_COLLECTION = "wait_snapshots"


class DatabaseNotConfigured(RuntimeError):
    pass


@lru_cache
def get_mongo_client() -> MongoClient:
    uri = get_settings().mongodb_uri
    if not uri:
        raise DatabaseNotConfigured("MONGODB_URI is not set")
    # Short server selection timeout: an unreachable DB must fail fast inside a chat request.
    return MongoClient(uri, serverSelectionTimeoutMS=5_000, tz_aware=True)


def get_snapshots_collection() -> Collection:
    return get_mongo_client()[get_settings().mongodb_db][SNAPSHOTS_COLLECTION]
