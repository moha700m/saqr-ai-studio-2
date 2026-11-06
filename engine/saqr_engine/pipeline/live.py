"""Real-time pipeline: capture thread -> latest-frame slot -> processing thread -> output slot.

Design goals (fixing issues seen in the reference projects):
* Single-slot buffers: the processor always works on the newest frame, stale
  frames are dropped and counted, so latency never accumulates.
* No queue joins / blocking puts: stop() can never deadlock; threads are
  daemonic and joined with a timeout.
* Camera switch = stop capture thread + reopen, processing keeps its state.
* Works with any frame source (camera index, video file, callable) so the same
  pipeline is testable without a webcam.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional, Union

import cv2
import numpy as np

from ..safety.watermark import draw_badge
from .metrics import RollingMetrics

log = logging.getLogger(__name__)

Source = Union[int, str, Callable[[], Optional[np.ndarray]]]


@dataclass
class OutputFrame:
    frame: np.ndarray
    seq: int
    processed: bool
    faces: int
    timestamp: float


class _Slot:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self.item = None
        self.seq = 0

    def put(self, item) -> bool:
        """Store item; returns True if an unconsumed item was overwritten (dropped)."""
        with self._cond:
            dropped = self.item is not None
            self.item = item
            self.seq += 1
            self._cond.notify_all()
            return dropped

    def take(self, timeout: float):
        with self._cond:
            if self.item is None:
                self._cond.wait(timeout)
            item, self.item = self.item, None
            return item

    def peek(self):
        with self._cond:
            return self.item


class LiveSession:
    def __init__(self, engine, source: Source = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 process: bool = False, mirror: bool = True, nsfw_interval_s: float = 2.0,
                 on_error: Optional[Callable[[str], None]] = None):
        self.engine = engine
        self.source = source
        self.width, self.height, self.fps = width, height, fps
        self.processing = process
        self.mirror = mirror
        self.metrics = RollingMetrics()
        self.on_error = on_error
        self.nsfw_interval_s = nsfw_interval_s
        self.sinks: list[Callable[[np.ndarray], None]] = []  # e.g. virtual camera (always badged)
        self._in = _Slot()
        self._out_lock = threading.Lock()
        self._latest: Optional[OutputFrame] = None
        self._stop = threading.Event()
        self._cap_stop = threading.Event()
        self._cap_thread: Optional[threading.Thread] = None
        self._proc_thread: Optional[threading.Thread] = None
        self._cap = None
        self._blocked_until = 0.0
        self._last_nsfw = 0.0
        self.error: Optional[str] = None
        self.capture_size: tuple[int, int] = (0, 0)

    # ------------------------------------------------------------------ lifecycle
    @property
    def running(self) -> bool:
        return self._proc_thread is not None and self._proc_thread.is_alive()

    def start(self) -> None:
        if self.running:
            return
        self._stop.clear()
        self.metrics.reset()
        self.engine.reset_tracking()
        self._start_capture()
        self._proc_thread = threading.Thread(target=self._process_loop, name="saqr-process", daemon=True)
        self._proc_thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        self._stop_capture(timeout)
        if self._proc_thread:
            self._proc_thread.join(timeout)
        self._proc_thread = None

    def switch_source(self, source: Source) -> None:
        self._stop_capture()
        self.source = source
        self.engine.reset_tracking()
        if not self._stop.is_set():
            self._start_capture()

    def set_processing(self, enabled: bool) -> None:
        self.processing = enabled
        self.engine.reset_tracking()

    # ------------------------------------------------------------------ capture
    def _open(self):
        if callable(self.source):
            return None
        if isinstance(self.source, int):
            from ..camera.devices import open_camera

            return open_camera(self.source, self.width, self.height, self.fps)
        cap = cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open {self.source}")
        return cap

    def _start_capture(self) -> None:
        self._cap_stop.clear()
        self._cap_thread = threading.Thread(target=self._capture_loop, name="saqr-capture", daemon=True)
        self._cap_thread.start()

    def _stop_capture(self, timeout: float = 3.0) -> None:
        self._cap_stop.set()
        if self._cap_thread:
            self._cap_thread.join(timeout)
        self._cap_thread = None

    def _fail(self, msg: str) -> None:
        self.error = msg
        log.error(msg)
        if self.on_error:
            try:
                self.on_error(msg)
            except Exception:  # noqa: BLE001
                pass

    def _capture_loop(self) -> None:
        try:
            cap = self._open()
        except Exception as exc:  # noqa: BLE001
            self._fail(str(exc))
            return
        self._cap = cap
        is_file = isinstance(self.source, str)
        file_fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if is_file else 0
        fails = 0
        try:
            while not self._cap_stop.is_set():
                t0 = time.perf_counter()
                if callable(self.source):
                    frame = self.source()
                    ok = frame is not None
                else:
                    ok, frame = cap.read()
                    if not ok and is_file:  # loop video files (used as a test camera)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                if not ok or frame is None:
                    fails += 1
                    if fails > 50:
                        self._fail("camera stopped delivering frames")
                        return
                    time.sleep(0.02)
                    continue
                fails = 0
                if self.mirror and not is_file and not callable(self.source):
                    frame = cv2.flip(frame, 1)
                self.capture_size = (frame.shape[1], frame.shape[0])
                self.metrics.on_capture()
                if self._in.put((frame, time.perf_counter())):
                    self.metrics.on_drop()
                if is_file or callable(self.source):  # pace non-camera sources at their own FPS
                    delay = 1.0 / (file_fps or self.fps) - (time.perf_counter() - t0)
                    if delay > 0:
                        time.sleep(delay)
        finally:
            if cap is not None:
                cap.release()
            self._cap = None

    # ------------------------------------------------------------------ processing
    def _process_loop(self) -> None:
        while not self._stop.is_set():
            item = self._in.take(timeout=0.2)
            if item is None:
                continue
            frame, t_cap = item
            processed = False
            faces = 0
            stages: dict[str, float] = {}
            out = frame
            if self.processing and self.engine.sources:
                try:
                    scale = self.engine.cfg.processing_scale
                    work = frame if scale >= 0.999 else cv2.resize(frame, None, fx=scale, fy=scale,
                                                                   interpolation=cv2.INTER_AREA)
                    res = self.engine.process(work)
                    out = res.frame if scale >= 0.999 else cv2.resize(res.frame, (frame.shape[1], frame.shape[0]))
                    processed, faces, stages = res.swapped > 0, len(res.faces), res.timings
                except Exception as exc:  # noqa: BLE001 - keep the preview alive, surface the error
                    self._fail(f"processing error: {exc}")
                    out = frame
                if processed:
                    out = self._safety(out)
                    draw_badge(out)
            self.metrics.on_frame((time.perf_counter() - t_cap) * 1000, stages)
            with self._out_lock:
                self._latest = OutputFrame(out, self.metrics.frames, processed, faces, time.time())
            for sink in list(self.sinks):
                try:
                    # processed frames already carry the badge; untouched camera frames are passed as-is
                    sink(out)
                except Exception as exc:  # noqa: BLE001
                    self._fail(f"output error: {exc}")
                    self.sinks.remove(sink)

    def _safety(self, frame: np.ndarray) -> np.ndarray:
        now = time.time()
        if now - self._last_nsfw >= self.nsfw_interval_s:
            self._last_nsfw = now
            try:
                if self.engine.check_nsfw(frame) >= 0.8:
                    self._blocked_until = now + 5.0
            except Exception as exc:  # noqa: BLE001
                log.warning("nsfw check failed: %s", exc)
        if now < self._blocked_until:
            blurred = cv2.GaussianBlur(frame, (0, 0), 25)
            cv2.putText(blurred, "Blocked by safety filter", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                        (255, 255, 255), 2, cv2.LINE_AA)
            return blurred
        return frame

    def latest(self) -> Optional[OutputFrame]:
        with self._out_lock:
            return self._latest
