# =============================================================================
# Live Detection Abuse Protection
# =============================================================================
# Small in-process rate limiter for the live-frame endpoint.
#
# IMPORTANT SCALING NOTE: this state lives in the memory of a single process.
# On Vercel each serverless instance has its own memory, so this is a
# best-effort guard against a runaway browser loop, not a distributed quota.
# It is intentionally NOT the source of truth - the database is. Real
# enforcement of the minimum frame interval is also done per session against the
# database, which survives cold starts.
# =============================================================================

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """
    Fixed-size sliding window counter.

    Thread-safe, O(1) amortised, and self-pruning so memory cannot grow without
    bound during a long-running process.
    """

    def __init__(self, window_seconds=60, max_requests=40, max_keys=20000):
        self.window_seconds = window_seconds
        self.max_requests = max_requests
        self.max_keys = max_keys
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key):
        """
        Record a hit for ``key`` and report whether it is permitted.

        Returns ``(allowed, retry_after_seconds)``.
        """
        now = time.monotonic()
        cutoff = now - self.window_seconds

        with self._lock:
            if len(self._hits) > self.max_keys:
                self._prune(now, cutoff)

            bucket = self._hits[key]
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            if len(bucket) >= self.max_requests:
                retry_after = max(1, int(bucket[0] + self.window_seconds - now))
                return False, retry_after

            bucket.append(now)
            return True, 0

    def _prune(self, now, cutoff):
        """Drop keys with no activity inside the current window."""
        stale = [key for key, bucket in self._hits.items() if not bucket or bucket[-1] < cutoff]
        for key in stale:
            del self._hits[key]

    def reset(self):
        with self._lock:
            self._hits.clear()


# Shared limiter instance for the live detection endpoints. Built from
# configuration rather than hard-coded defaults, so RATE_LIMIT_WINDOW_SECONDS and
# RATE_LIMIT_MAX_REQUESTS in the environment actually take effect.
def build_default_limiter():
    from config import Config

    return SlidingWindowLimiter(
        window_seconds=Config.RATE_LIMIT_WINDOW_SECONDS,
        max_requests=Config.RATE_LIMIT_MAX_REQUESTS,
    )


live_frame_limiter = build_default_limiter()


class RateLimited(Exception):
    """Raised when a caller exceeds the configured request budget."""

    def __init__(self, retry_after):
        super().__init__('Too many requests')
        self.retry_after = retry_after
