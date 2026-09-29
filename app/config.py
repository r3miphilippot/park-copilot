"""Central configuration: environment settings, park IDs and timezone.

Every setting comes from environment variables (or a local `.env` file), never from code.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal
from zoneinfo import ZoneInfo

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# All "human" times (weekday, hour, opening hours, planning vs in-park mode) are Paris time.
# Timestamps are stored in UTC and converted with this zone.
PARIS_TZ = ZoneInfo("Europe/Paris")

QUEUE_TIMES_BASE_URL = "https://queue-times.com"

BASE_DIR = Path(__file__).resolve().parent.parent  # repository root


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
    # Absolute path: the MCP server may be launched from another directory (Claude Desktop).
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # MongoDB Atlas (M0 free cluster)
    mongodb_uri: str = ""
    mongodb_db: str = "park_copilot"

    # LLM: main provider + automatic fallback on 429 / timeout / 5xx.
    # Groq quotas are per model, so another Groq model is a valid fallback on Hugging Face
    # (where Ollama is not available). Locally: FALLBACK_PROVIDER=ollama FALLBACK_MODEL=qwen3.
    llm_provider: Literal["groq", "ollama"] = "groq"
    llm_model: str = "openai/gpt-oss-120b"
    fallback_provider: Literal["groq", "ollama", "none"] = "groq"
    fallback_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"
    llm_timeout_s: float = 30.0
    llm_max_retry_wait_s: float = 5.0  # wait for `retry-after` only when it is this short
    agent_max_iterations: int = 6  # LLM calls per user message before forcing an answer

    # API
    # Comma-separated in the environment: ALLOWED_ORIGINS=https://a.com,http://localhost:3000
    allowed_origins: Annotated[list[str], NoDecode] = [
        "https://remiphilippot.vercel.app",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:7860",
    ]
    rate_limit_per_minute: int = 10  # per client IP, on /chat
    # Protects the Groq free quota (~1000 requests/day/model, each chat = 3-4 LLM calls).
    daily_request_cap: int = 200
    max_threads: int = 500  # conversations kept in memory (oldest evicted first)
    # Protects /metrics/prometheus (scraped by Grafana Cloud). Empty = open (local dev).
    metrics_token: str = ""
    log_level: str = "INFO"

    # Observability: Langfuse (free cloud tier) is enabled only when both keys are set.
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    # RAG: local embeddings (FastEmbed, no API) + in-memory Chroma rebuilt at startup.
    # Measured on 14 French questions over the guide: all-MiniLM-L6-v2 finds the right
    # section in its top 3 for 14/14 (multilingual MiniLM-L12: 12/14) while using ~340 MB of
    # RAM for the whole app instead of ~770 MB, which fits the 512 MB free hosting tiers.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    knowledge_dir: Path = BASE_DIR / "knowledge"
    fastembed_cache_dir: Path = BASE_DIR / ".fastembed_cache"


@lru_cache
def get_settings() -> Settings:
    """Settings are read once per process."""
    return Settings()
