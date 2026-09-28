"""Central configuration: environment settings, park IDs and timezone.

Every setting comes from environment variables (or a local `.env` file), never from code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
from zoneinfo import ZoneInfo

from pydantic_settings import BaseSettings, SettingsConfigDict

# All "human" times (weekday, hour, opening hours, planning vs in-park mode) are Paris time.
# Timestamps are stored in UTC and converted with this zone.
PARIS_TZ = ZoneInfo("Europe/Paris")

QUEUE_TIMES_BASE_URL = "https://queue-times.com"


@dataclass(frozen=True)
class Park:
    key: str  # stable slug stored in MongoDB and used as tool argument
    queue_times_id: int
    name: str


# Type used by tool arguments: the LLM can only pass one of these values.
ParkKey = Literal["disneyland_park", "adventure_world"]

# IDs taken from https://queue-times.com/parks.json (group "Walt Disney Attractions").
PARKS: dict[str, Park] = {
    "disneyland_park": Park("disneyland_park", 4, "Disneyland Park"),
    "adventure_world": Park("adventure_world", 28, "Disney Adventure World"),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # MongoDB Atlas (M0 free cluster)
    mongodb_uri: str = ""
    mongodb_db: str = "park_copilot"


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process."""
    return Settings()
