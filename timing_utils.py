from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


@dataclass
class TimingEvent:
    name: str
    elapsed_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


class TimerRegistry:
    def __init__(self) -> None:
        self.events: list[TimingEvent] = []
        self.totals: dict[str, float] = {}

    @contextmanager
    def timed(self, name: str, **metadata: Any) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self.events.append(TimingEvent(name=name, elapsed_sec=elapsed, metadata=metadata))
            self.totals[name] = self.totals.get(name, 0.0) + elapsed

    def summary(self) -> dict[str, Any]:
        return {
            "totals_sec": {name: round(value, 4) for name, value in self.totals.items()},
            "events": [
                {
                    "name": event.name,
                    "elapsed_sec": round(event.elapsed_sec, 4),
                    "metadata": event.metadata,
                }
                for event in self.events
            ],
        }


def format_duration(seconds: float) -> str:
    return f"{seconds:.1f}s"
