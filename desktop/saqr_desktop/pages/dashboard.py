"""Dashboard + Privacy/Settings pages."""
from __future__ import annotations

import shutil

from PySide6.QtWidgets import QCheckBox, QGridLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget

from saqr_engine import config

from ..widgets import card


class DashboardPage(QWidget):
    def __init__(self, ctx, go):
        super().__init__()
        self.ctx, self.go = ctx, go
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Welcome to SAQR AI Studio")
        h.setObjectName("h1")
        lay.addWidget(h)
        s = QLabel("Local-first face studio. Your camera and images are processed on this computer.")
        s.setObjectName("muted")
        lay.addWidget(s)
        grid = QGridLayout()
        grid.setSpacing(14)
        self.hw_c, hl = card("Your computer")
        self.hw = QLabel("Detecting hardware…")
        self.hw.setWordWrap(True)
        hl.addWidget(self.hw)
        b = QPushButton("Hardware details")
        b.clicked.connect(lambda: go("hardware"))
        hl.addWidget(b)
        grid.addWidget(self.hw_c, 0, 0)
        self.m_c, ml = card("AI models")
        self.models = QLabel("…")
        self.models.setWordWrap(True)
        ml.addWidget(self.models)
        b = QPushButton("Open Models")
        b.clicked.connect(lambda: go("models"))
        ml.addWidget(b)
        grid.addWidget(self.m_c, 0, 1)
        c, sl = card("Get started")
        for label, page in (("1. Add your source face", "face"), ("2. Go live with the camera", "live"),
                            ("3. Or edit an image", "image")):
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, p=page: go(p))
            sl.addWidget(b)
        grid.addWidget(c, 0, 2)
        lay.addLayout(grid)
        c, rl = card("Responsible use", "Use only your own face or faces you have permission to use. Never use SAQR "
                                        "to impersonate, deceive, defraud or harass anyone, or to make intimate "
                                        "imagery. Live output always shows an “AI-altered” badge.")
        lay.addWidget(c)
        lay.addStretch(1)

    def refresh(self) -> None:
        hw, sel = self.ctx.hardware, self.ctx.selection
        if hw and sel:
            g = hw.best_gpu
            self.hw.setText(f"<b>{hw.cpu.model}</b><br>{g.name if g else 'No GPU detected'}<br>"
                            f"RAM {hw.system.ram_total_mb / 1024:.0f} GB<br>AI backend: <b>{sel.label}</b>")
        miss = self.ctx.models.missing_required()
        self.models.setText("All required models installed ✓" if not miss else
                            "<b>Models needed.</b> Open Models to download them (about 1.1 GB).")


class SettingsPage(QWidget):
    def __init__(self, ctx, status):
        super().__init__()
        self.ctx, self.status = ctx, status
        st = ctx.settings.settings
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Privacy & Settings")
        h.setObjectName("h1")
        lay.addWidget(h)
        c, pl = card("Processing mode", "LOCAL: everything runs on this computer (default). Cloud and hybrid "
                                        "processing are off: no image, video or camera frame is ever uploaded.")
        self.mode = QLabel("Current mode: LOCAL")
        self.mode.setObjectName("modeLocal")
        pl.addWidget(self.mode)
        lay.addWidget(c)
        c, dl = card("Defaults")
        self.cam = QCheckBox("Start the camera automatically (off by default)")
        self.cam.setChecked(st.auto_start_camera)
        self.cam.toggled.connect(lambda v: ctx.settings.update(auto_start_camera=v))
        self.wm = QCheckBox("Visible AI badge on exported images (live/virtual camera output always has it)")
        self.wm.setChecked(st.export_watermark)
        self.wm.toggled.connect(lambda v: ctx.settings.update(export_watermark=v))
        self.meta = QCheckBox("Embed AI-generated metadata in exported files")
        self.meta.setChecked(st.export_metadata)
        self.meta.toggled.connect(lambda v: ctx.settings.update(export_metadata=v))
        for w in (self.cam, self.wm, self.meta):
            dl.addWidget(w)
        lay.addWidget(c)
        c, ml = card("Media cleanup", "SAQR keeps no recordings. Source faces live in memory only. Temporary files "
                                      "are deleted after each job; you can also clear them here.")
        b = QPushButton("Delete temporary files now")
        b.clicked.connect(self.cleanup)
        ml.addWidget(b)
        b2 = QPushButton("Forget all source faces in this session")
        b2.clicked.connect(self.forget)
        ml.addWidget(b2)
        lay.addWidget(c)
        c, il = card("About")
        il.addWidget(QLabel(f"SAQR AI Studio {config.VERSION} · AGPL-3.0 · data folder: {config.data_dir()}"))
        lay.addWidget(c)
        lay.addStretch(1)

    def cleanup(self) -> None:
        d = config.temp_dir()
        n = sum(1 for _ in d.rglob("*") if _.is_file())
        shutil.rmtree(d, ignore_errors=True)
        config.temp_dir()
        self.status(f"Deleted {n} temporary file(s)")

    def forget(self) -> None:
        if self.ctx.selection:
            self.ctx.engine.clear_sources()
        QMessageBox.information(self, "Done", "All source faces were removed from memory.")
