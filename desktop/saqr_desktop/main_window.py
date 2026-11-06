"""Main window: sidebar navigation + stacked pages + status bar mode indicator."""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QButtonGroup, QHBoxLayout, QLabel, QMainWindow, QMessageBox, QPushButton,
                               QStackedWidget, QStatusBar, QVBoxLayout, QWidget)

from saqr_engine import config

from .pages.dashboard import DashboardPage, SettingsPage
from .pages.face_studio import FaceStudioPage
from .pages.hardware import HardwarePage
from .pages.image_studio import ImageStudioPage
from .pages.live import LivePage
from .pages.models import ModelsPage
from .widgets import run_task

NAV = [("dashboard", "Dashboard"), ("face", "Face Studio"), ("live", "Live Camera"), ("image", "Image Studio"),
       ("models", "Models"), ("hardware", "Hardware"), ("settings", "Privacy && Settings")]


class MainWindow(QMainWindow):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.setWindowTitle(f"{config.APP_NAME} {config.VERSION}")
        self.resize(1360, 860)
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        side = QWidget()
        side.setObjectName("sidebar")
        side.setFixedWidth(210)
        sl = QVBoxLayout(side)
        sl.setContentsMargins(10, 0, 10, 12)
        brand = QLabel("SAQR <span style='color:#22d3ee'>AI</span> Studio")
        brand.setObjectName("brand")
        brand.setTextFormat(Qt.TextFormat.RichText)
        sl.addWidget(brand)
        self.stack = QStackedWidget()
        self.group = QButtonGroup(self)
        self.pages: dict[str, QWidget] = {}
        self.buttons: dict[str, QPushButton] = {}

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.mode = QLabel("LOCAL")
        self.mode.setObjectName("modeLocal")
        self.mode.setToolTip("All processing happens on this computer. Nothing is uploaded.")
        self.backend = QLabel("Detecting hardware…")
        self.backend.setObjectName("badge")
        self.status_bar.addPermanentWidget(self.backend)
        self.status_bar.addPermanentWidget(self.mode)

        st = self.status_bar.showMessage
        status = lambda m: st(m, 6000)  # noqa: E731
        self.dashboard = DashboardPage(ctx, self.go)
        self.face = FaceStudioPage(ctx, status)
        self.live = LivePage(ctx, status)
        self.image = ImageStudioPage(ctx, status)
        self.models = ModelsPage(ctx, status, self._models_changed)
        self.hardware = HardwarePage(ctx, status, self._backend_changed)
        self.settings = SettingsPage(ctx, status)
        for key, w in (("dashboard", self.dashboard), ("face", self.face), ("live", self.live),
                       ("image", self.image), ("models", self.models), ("hardware", self.hardware),
                       ("settings", self.settings)):
            self.pages[key] = w
            self.stack.addWidget(w)
        for key, label in NAV:
            b = QPushButton(label)
            b.setObjectName("nav")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, k=key: self.go(k))
            self.group.addButton(b)
            self.buttons[key] = b
            sl.addWidget(b)
        sl.addStretch(1)
        foot = QLabel("AI-altered output is always labelled.")
        foot.setObjectName("muted")
        foot.setWordWrap(True)
        sl.addWidget(foot)
        h.addWidget(side)
        h.addWidget(self.stack, 1)
        self.face.sourcesChanged.connect(self.live.on_sources_changed)
        self.face.sourcesChanged.connect(self.image.on_sources_changed)
        self.go("dashboard")
        self.setEnabled_pages(False)
        run_task(ctx.detect, on_done=self._detected, on_error=self._detect_failed)

    def setEnabled_pages(self, on: bool) -> None:  # noqa: N802
        for k in ("face", "live", "image"):
            self.buttons[k].setEnabled(on)

    def go(self, key: str) -> None:
        self.stack.setCurrentWidget(self.pages[key])
        self.buttons[key].setChecked(True)
        if key == "models":
            self.models.refresh()

    def _detected(self, _r) -> None:
        sel = self.ctx.selection
        self.backend.setText(f"AI: {sel.label}")
        self.setEnabled_pages(True)
        self.dashboard.refresh()
        self.hardware.refresh()
        self.models.refresh()
        if not self.ctx.models_ready():
            self.go("models")
            self.status_bar.showMessage("Download the AI models to get started.", 10000)

    def _detect_failed(self, e: str) -> None:
        QMessageBox.critical(self, "Hardware detection failed", e)

    def _models_changed(self) -> None:
        self.dashboard.refresh()

    def _backend_changed(self) -> None:
        self.backend.setText(f"AI: {self.ctx.selection.label}")
        self.dashboard.refresh()
        self.face.refresh()

    def closeEvent(self, e):  # noqa: N802
        self.live.stop()
        if self.ctx.selection:
            try:
                self.ctx.engine.clear_sources()  # never keep faces beyond the session
            except Exception:  # noqa: BLE001
                pass
        super().closeEvent(e)
