"""Inference session factory with automatic fallback.

``ResilientSession`` wraps an ONNX Runtime session. If creating it on the
selected backend fails, or if an inference call later raises (driver reset,
out-of-memory, unsupported op), it transparently rebuilds the session on the
next verified backend, ending at CPU, and records the event so diagnostics can
show it.
"""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

from .providers import PRIORITY, BackendSelection, provider_list, session_options

log = logging.getLogger(__name__)


@dataclass
class FallbackEvent:
    model: str
    from_backend: str
    to_backend: str
    reason: str


@dataclass
class RuntimeState:
    """Process-wide record of the active backend and any fallbacks."""
    selection: Optional[BackendSelection] = None
    events: list[FallbackEvent] = field(default_factory=list)
    listeners: list[Callable[[FallbackEvent], None]] = field(default_factory=list)

    def record(self, ev: FallbackEvent) -> None:
        self.events.append(ev)
        del self.events[:-50]
        for cb in list(self.listeners):
            try:
                cb(ev)
            except Exception:  # noqa: BLE001
                log.exception("fallback listener failed")


STATE = RuntimeState()


def _chain(selection: BackendSelection) -> list[str]:
    verified = [c.id for c in selection.checks if c.verified]
    chain = [selection.selected] + [p for p in PRIORITY if p in verified and p != selection.selected]
    if "cpu" not in chain:
        chain.append("cpu")
    return chain


class ResilientSession:
    def __init__(self, model_path: str | Path, selection: BackendSelection, threads: Optional[int] = None):
        import onnxruntime as ort

        self._ort = ort
        self.model_path = str(model_path)
        self.name = Path(model_path).name
        self.selection = selection
        self._threads = threads if threads is not None else selection.intra_op_threads
        self._chain = _chain(selection)
        self._lock = threading.Lock()
        self.backend = ""
        self.sess = None
        self._build(0)

    def _build(self, start: int) -> None:
        last_err = None
        for i in range(start, len(self._chain)):
            pid = self._chain[i]
            try:
                threads = self._threads if pid == "cpu" or pid == self.selection.selected else 0
                self.sess = self._ort.InferenceSession(self.model_path, sess_options=session_options(pid, threads),
                                                       providers=provider_list(pid))
                if self.backend and self.backend != pid:
                    why = last_err or f"runtime error: {getattr(self, '_runtime_error', '')}"
                    STATE.record(FallbackEvent(self.name, self.backend, pid, str(why)[:300]))
                elif i > 0:
                    STATE.record(FallbackEvent(self.name, self._chain[0], pid, str(last_err)[:300]))
                self.backend = pid
                self._idx = i
                return
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                log.warning("session for %s failed on %s: %s", self.name, pid, exc)
        raise RuntimeError(f"could not create a session for {self.name}: {last_err}")

    # passthroughs
    def get_inputs(self):
        return self.sess.get_inputs()

    def get_outputs(self):
        return self.sess.get_outputs()

    def run(self, output_names: Optional[list[str]], feeds: dict[str, np.ndarray]) -> list[Any]:
        try:
            if self.backend == "directml":
                with self._lock:  # DirectML sessions are not safe for concurrent Run()
                    return self.sess.run(output_names, feeds)
            return self.sess.run(output_names, feeds)
        except Exception as exc:  # noqa: BLE001
            if self._idx + 1 >= len(self._chain):
                raise
            prev = self.backend
            log.error("inference on %s failed for %s (%s); falling back", prev, self.name, exc)
            self._runtime_error = exc
            self._build(self._idx + 1)  # records the FallbackEvent (backend changes from prev)
            return self.sess.run(output_names, feeds)
