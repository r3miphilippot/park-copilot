"""In-memory protections for the free tiers: per-IP rate limit, global daily cap, and a
bound on the number of conversations kept in memory.

In memory is enough: the app runs as a single process on one Hugging Face Space.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from datetime import date


class RateLimiter:
    """Sliding window: at most `limit` hits per `window_s` seconds for each key (client IP)."""

    def __init__(self, limit: int, window_s: float = 60.0) -> None:
        self.limit = limit
        self.window_s = window_s
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def hit(self, key: str, now: float | None = None) -> tuple[bool, int]:
        """Record a hit. Returns (allowed, seconds until the next hit is allowed)."""
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits.setdefault(key, deque())
            while hits and hits[0] <= now - self.window_s:
                hits.popleft()
            if len(hits) >= self.limit:
                return False, max(1, round(hits[0] + self.window_s - now))
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients, keeps memory bounded
                self._hits = {k: v for k, v in self._hits.items() if v}
            return True, 0


class DailyCap:
    """Global number of chats per (Paris) day, to keep the Groq free quota for everyone."""

    def __init__(self, cap: int) -> None:
        self.cap = cap
        self.day: date | None = None
        self.count = 0
        self._lock = threading.Lock()

    def try_acquire(self, today: date) -> bool:
        with self._lock:
            if today != self.day:
                self.day, self.count = today, 0
            if self.count >= self.cap:
                return False
            self.count += 1
            return True


class ThreadRegistry:
    """Least-recently-used conversations. Returns the thread ids to forget when full,
    so the in-memory checkpointer cannot grow forever."""

    def __init__(self, max_threads: int) -> None:
        self.max_threads = max_threads
        self._threads: OrderedDict[str, None] = OrderedDict()
        self._lock = threading.Lock()

    def touch(self, thread_id: str) -> list[str]:
        with self._lock:
            self._threads[thread_id] = None
            self._threads.move_to_end(thread_id)
            evicted = []
            while len(self._threads) > self.max_threads:
                evicted.append(self._threads.popitem(last=False)[0])
            return evicted
