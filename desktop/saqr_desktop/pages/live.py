"""Live Camera page."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QCheckBox, QComboBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout,
                               QWidget)

from saqr_engine.camera.devices import list_cameras
from saqr_engine.pipeline.live import LiveSession

from ..widgets import ImageView, card, run_task

RESOLUTIONS = [(640, 480), (960, 540), (1280, 720), (1920, 1080)]


class LivePage(QWidget):
    def __init__(self, ctx, status):
        super().__init__()
        self.ctx, self.status = ctx, status
        self.session: LiveSession | None = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Live Camera")
        h.setObjectName("h1")
        lay.addWidget(h)
        note = QLabel("The camera never starts on its own. Swapped output always carries an \"AI-altered\" badge.")
        note.setObjectName("muted")
        lay.addWidget(note)

        bar = QHBoxLayout()
        self.cams = QComboBox()
        self.cams.setMinimumWidth(220)
        bar.addWidget(self.cams)
        ref = QPushButton("Refresh")
        ref.clicked.connect(self.refresh_cameras)
        bar.addWidget(ref)
        self.res = QComboBox()
        for w, hgt in RESOLUTIONS:
            self.res.addItem(f"{w}×{hgt}", (w, hgt))
        self.res.setCurrentIndex(2)
        bar.addWidget(self.res)
        self.fps = QComboBox()
        for f in (15, 24, 30, 60):
            self.fps.addItem(f"{f} fps", f)
        self.fps.setCurrentIndex(2)
        bar.addWidget(self.fps)
        self.mirror = QCheckBox("Mirror")
        self.mirror.setChecked(True)
        bar.addWidget(self.mirror)
        bar.addStretch(1)
        self.start_btn = QPushButton("Start camera")
        self.start_btn.setObjectName("primary")
        self.start_btn.clicked.connect(self.toggle_camera)
        bar.addWidget(self.start_btn)
        lay.addLayout(bar)

        bar2 = QHBoxLayout()
        self.swap_btn = QPushButton("Start face swap")
        self.swap_btn.setCheckable(True)
        self.swap_btn.setEnabled(False)
        self.swap_btn.toggled.connect(self.toggle_processing)
        bar2.addWidget(self.swap_btn)
        bar2.addWidget(QLabel("Processing scale"))
        self.scale = QComboBox()
        for s in (1.0, 0.75, 0.5):
            self.scale.addItem(f"{int(s * 100)}%", s)
        self.scale.currentIndexChanged.connect(lambda _i: self.ctx.cfg.update(processing_scale=self.scale.currentData()))
        bar2.addWidget(self.scale)
        bar2.addStretch(1)
        self.metrics = QLabel("—")
        self.metrics.setObjectName("badge")
        bar2.addWidget(self.metrics)
        lay.addLayout(bar2)

        self.view = ImageView("Camera is off", min_h=360)
        lay.addWidget(self.view, 1)
        c, cl = card()
        self.hint = QLabel("")
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        cl.addWidget(self.hint)
        lay.addWidget(c)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.refresh_cameras()

    def refresh_cameras(self) -> None:
        self.cams.clear()
        self.cams.addItem("Searching…")
        run_task(list_cameras, on_done=self._cams, on_error=lambda e: self._cams([]))

    def _cams(self, cams) -> None:
        self.cams.clear()
        for c in cams:
            self.cams.addItem(c.name, c.index)
        if not cams:
            self.cams.addItem("No camera found", -1)

    def on_sources_changed(self) -> None:
        has = bool(self.ctx.engine.sources) if self.ctx.selection else False
        self.swap_btn.setEnabled(has and self.session is not None)
        if not has and self.swap_btn.isChecked():
            self.swap_btn.setChecked(False)

    def toggle_camera(self) -> None:
        if self.session:
            self.stop()
            return
        idx = self.cams.currentData()
        if idx is None or idx < 0:
            QMessageBox.information(self, "No camera", "No camera was found. Connect one and press Refresh.")
            return
        w, h = self.res.currentData()
        try:
            eng = self.ctx.engine
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Engine not ready", str(exc))
            return
        self.session = LiveSession(eng, idx, w, h, self.fps.currentData(), process=False,
                                   mirror=self.mirror.isChecked())
        self.ctx.live = self.session
        self.session.start()
        self.start_btn.setText("Stop camera")
        self.timer.start(15)
        self.on_sources_changed()
        self.hint.setText("Preview only — nothing is being processed." if not eng.sources else
                          "Press “Start face swap” to begin processing.")

    def toggle_processing(self, on: bool) -> None:
        if not self.session:
            return
        if on and not self.ctx.models_ready():
            QMessageBox.information(self, "Models needed", "Download the AI models first (Models page).")
            self.swap_btn.setChecked(False)
            return
        self.session.set_processing(on)
        self.swap_btn.setText("Stop face swap" if on else "Start face swap")
        sel = self.ctx.selection
        if on and sel and sel.selected == "cpu":
            self.hint.setText("Running on the CPU: expect a low frame rate. A GPU gives real-time results. "
                              "Try the Fast preset and a 50% processing scale.")
        elif on:
            self.hint.setText(f"Processing on {sel.label if sel else '?'}.")

    def stop(self) -> None:
        if self.session:
            self.session.stop()
        self.session = None
        self.ctx.live = None
        self.timer.stop()
        self.swap_btn.setChecked(False)
        self.swap_btn.setEnabled(False)
        self.start_btn.setText("Start camera")
        self.view.set_bgr(None)
        self.view.setText("Camera is off")
        self.metrics.setText("—")

    def _tick(self) -> None:
        s = self.session
        if not s:
            return
        if s.error and not s.running:
            msg = s.error
            self.stop()
            QMessageBox.warning(self, "Camera", msg)
            return
        o = s.latest()
        if o is not None:
            self.view.set_bgr(o.frame)
        m = s.metrics.snapshot()
        txt = f"{m['fps']:.1f} fps · {m['latency_ms']:.0f} ms"
        if s.processing:
            txt += f" · camera {m['capture_fps']:.0f} fps · dropped {m['dropped']}"
        self.metrics.setText(txt)
        if s.error:
            self.status(s.error)
            s.error = None
