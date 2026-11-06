"""Real (measured) performance metrics."""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager


class StageTimer:
    def __init__(self) -> None:
        self._t: dict[str, float] = {}
        self._start = time.perf_counter()

    @contextmanager
    def __call__(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._t[name] = self._t.get(name, 0.0) + (time.perf_counter() - t0) * 1000

    def result(self) -> dict[str, float]:
        r = dict(self._t)
        r["total"] = (time.perf_counter() - self._start) * 1000
        return r


class RollingMetrics:
    """FPS from timestamps of the last N frames; latency as EMA and p95 over a window."""

    def __init__(self, window: int = 90) -> None:
        self._lock = threading.Lock()
        self._stamps: deque[float] = deque(maxlen=window)
        self._lat: deque[float] = deque(maxlen=window)
        self.stages: dict[str, float] = {}
        self.dropped = 0
        self.frames = 0
        self.capture_fps = 0.0
        self._cap_stamps: deque[float] = deque(maxlen=window)

    def on_capture(self) -> None:
        with self._lock:
            self._cap_stamps.append(time.perf_counter())

    def on_frame(self, latency_ms: float, stages: dict[str, float]) -> None:
        with self._lock:
            self._stamps.append(time.perf_counter())
            self._lat.append(latency_ms)
            self.frames += 1
            for k, v in stages.items():
                self.stages[k] = v if k not in self.stages else self.stages[k] * 0.9 + v * 0.1

    def on_drop(self, n: int = 1) -> None:
        with self._lock:
            self.dropped += n

    @staticmethod
    def _fps(stamps: deque) -> float:
        if len(stamps) < 2:
            return 0.0
        span = stamps[-1] - stamps[0]
        if time.perf_counter() - stamps[-1] > 2.0:  # stalled
            return 0.0
        return (len(stamps) - 1) / span if span > 0 else 0.0

    def snapshot(self) -> dict:
        with self._lock:
            lat = sorted(self._lat)
            return {
                "fps": round(self._fps(self._stamps), 2),
                "capture_fps": round(self._fps(self._cap_stamps), 2),
                "latency_ms": round(sum(lat) / len(lat), 1) if lat else 0.0,
                "latency_p95_ms": round(lat[int(len(lat) * 0.95) - 1], 1) if len(lat) >= 20 else None,
                "stages_ms": {k: round(v, 1) for k, v in self.stages.items()},
                "frames": self.frames,
                "dropped": self.dropped,
            }

    def reset(self) -> None:
        with self._lock:
            self._stamps.clear()
            self._lat.clear()
            self._cap_stamps.clear()
            self.stages.clear()
            self.dropped = 0
            self.frames = 0
