"""Reusable widgets and a background-task helper."""
from __future__ import annotations

import traceback
from typing import Any, Callable, Optional

import cv2
import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QSizePolicy, QSlider, QVBoxLayout, QWidget)


def bgr_to_qimage(img: np.ndarray) -> QImage:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def read_image(path: str) -> Optional[np.ndarray]:
    data = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


class ImageView(QLabel):
    """Aspect-preserving image display."""

    def __init__(self, placeholder: str = "", min_h: int = 240):
        super().__init__(placeholder)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(min_h)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#0e0e15;border:1px dashed #2a2a38;border-radius:10px;color:#77778c;")
        self._pix: Optional[QPixmap] = None

    def set_bgr(self, img: Optional[np.ndarray]) -> None:
        if img is None:
            self._pix = None
            self.clear()
            return
        self._pix = QPixmap.fromImage(bgr_to_qimage(img))
        self._rescale()

    def resizeEvent(self, e):  # noqa: N802
        super().resizeEvent(e)
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.setPixmap(self._pix.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                            Qt.TransformationMode.SmoothTransformation))


def card(title: str = "", subtitle: str = "") -> tuple[QFrame, QVBoxLayout]:
    f = QFrame()
    f.setObjectName("card")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(16, 14, 16, 14)
    lay.setSpacing(10)
    if title:
        t = QLabel(title)
        t.setObjectName("h2")
        lay.addWidget(t)
    if subtitle:
        s = QLabel(subtitle)
        s.setObjectName("muted")
        s.setWordWrap(True)
        lay.addWidget(s)
    return f, lay


class LabeledSlider(QWidget):
    """0..1 float slider with a live value label."""
    valueChanged = Signal(float)

    def __init__(self, label: str, value: float, tooltip: str = ""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.name = QLabel(label)
        self.name.setMinimumWidth(130)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(int(round(value * 100)))
        self.val = QLabel(f"{value:.2f}")
        self.val.setMinimumWidth(36)
        lay.addWidget(self.name)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.val)
        if tooltip:
            self.setToolTip(tooltip)
        self.slider.valueChanged.connect(self._changed)

    def _changed(self, v: int) -> None:
        self.val.setText(f"{v / 100:.2f}")
        self.valueChanged.emit(v / 100)

    def value(self) -> float:
        return self.slider.value() / 100


class _Signals(QObject):
    done = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class Task(QRunnable):
    """Run ``fn`` on the global thread pool; results come back on the GUI thread via signals."""

    def __init__(self, fn: Callable[..., Any], *args, with_progress: bool = False, **kw):
        super().__init__()
        self.fn, self.args, self.kw = fn, args, kw
        self.signals = _Signals()
        self.with_progress = with_progress
        self.setAutoDelete(True)

    def run(self) -> None:
        try:
            if self.with_progress:
                self.kw["progress"] = lambda *p: self.signals.progress.emit(p)
            res = self.fn(*self.args, **self.kw)
            self.signals.done.emit(res)
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            self.signals.failed.emit(str(exc) or exc.__class__.__name__)


_ACTIVE: set = set()


def run_task(fn, *args, on_done=None, on_error=None, on_progress=None, **kw) -> Task:
    t = Task(fn, *args, with_progress=on_progress is not None, **kw)
    _ACTIVE.add(t.signals)  # keep the signal object alive until it fires

    def _finish(*_a):
        _ACTIVE.discard(t.signals)

    if on_done:
        t.signals.done.connect(on_done)
    if on_error:
        t.signals.failed.connect(on_error)
    if on_progress:
        t.signals.progress.connect(on_progress)
    t.signals.done.connect(_finish)
    t.signals.failed.connect(_finish)
    QThreadPool.globalInstance().start(t)
    return t
