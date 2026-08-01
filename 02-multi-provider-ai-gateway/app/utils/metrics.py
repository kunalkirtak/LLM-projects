"""
In-process metrics store.

A minimal, thread-safe-enough (single-process asyncio) metrics tracker
that aggregates gateway usage so the /metrics endpoint can report
totals without needing an external time-series database. In a real
production deployment this would be swapped for Prometheus counters /
histograms, but the interface below would stay the same, which is the
point: callers depend on `metrics_store`, not on how it's implemented.
"""

import time
from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ProviderMetrics:
    """Running totals for a single provider."""

    request_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    fallback_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    retry_count: int = 0

    @property
    def average_latency_ms(self) -> float:
        if self.request_count == 0:
            return 0.0
        return self.total_latency_ms / self.request_count

    def to_dict(self) -> dict:
        return {
            "request_count": self.request_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "fallback_count": self.fallback_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "average_latency_ms": round(self.average_latency_ms, 2),
            "retry_count": self.retry_count,
        }


class MetricsStore:
    """
    Aggregates per-provider metrics across the process lifetime.

    Guarded by a plain threading.Lock rather than an asyncio.Lock because
    updates are short, synchronous dict/attribute mutations -- cheap
    enough that lock contention is a non-issue, and it keeps the store
    usable from both sync and async call sites without extra plumbing.
    """

    def __init__(self) -> None:
        self._lock = Lock()
        self._providers: dict[str, ProviderMetrics] = {}
        self._started_at = time.time()

    def record(
        self,
        *,
        provider: str,
        success: bool,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
        retry_count: int,
        fallback_used: bool,
    ) -> None:
        with self._lock:
            stats = self._providers.setdefault(provider, ProviderMetrics())
            stats.request_count += 1
            if success:
                stats.success_count += 1
            else:
                stats.failure_count += 1
            if fallback_used:
                stats.fallback_count += 1
            stats.total_input_tokens += input_tokens
            stats.total_output_tokens += output_tokens
            stats.total_cost_usd += cost_usd
            stats.total_latency_ms += latency_ms
            stats.retry_count += retry_count

    def snapshot(self) -> dict:
        with self._lock:
            uptime_seconds = time.time() - self._started_at
            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "providers": {
                    name: stats.to_dict() for name, stats in self._providers.items()
                },
                "totals": self._totals_locked(),
            }

    def _totals_locked(self) -> dict:
        """Aggregate across all providers. Caller must already hold the lock."""
        total = ProviderMetrics()
        for stats in self._providers.values():
            total.request_count += stats.request_count
            total.success_count += stats.success_count
            total.failure_count += stats.failure_count
            total.fallback_count += stats.fallback_count
            total.total_input_tokens += stats.total_input_tokens
            total.total_output_tokens += stats.total_output_tokens
            total.total_cost_usd += stats.total_cost_usd
            total.total_latency_ms += stats.total_latency_ms
            total.retry_count += stats.retry_count
        return total.to_dict()


# Process-wide singleton. Imported by routes and the gateway.
metrics_store = MetricsStore()
