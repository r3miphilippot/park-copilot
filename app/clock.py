"""Single source of "now", so evals and tests can simulate a date/time.

with frozen_now(datetime(2026, 7, 14, 10, 30, tzinfo=PARIS_TZ)):
    ...  # every tool and the agent see this moment
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import UTC, datetime

from app.config import PARIS_TZ

# A ContextVar (not a global) so concurrent requests/evals don't see each other's frozen time.
_frozen: ContextVar[datetime | None] = ContextVar("frozen_now", default=None)


def now_paris() -> datetime:
    """Current time in Europe/Paris (timezone-aware)."""
    return (_frozen.get() or datetime.now(UTC)).astimezone(PARIS_TZ)


@contextmanager
def frozen_now(moment: datetime) -> Iterator[None]:
    if moment.tzinfo is None:
        raise ValueError("frozen_now() needs a timezone-aware datetime")
    token = _frozen.set(moment)
    try:
        yield
    finally:
        _frozen.reset(token)
