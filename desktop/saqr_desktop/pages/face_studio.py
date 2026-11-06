"""Face Studio: consented source library + every processing control.

The controls edit the shared ProcessConfig, so they apply live to the camera
pipeline and to Image Studio.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QGridLayout,
                               QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
                               QRadioButton, QScrollArea, QVBoxLayout, QWidget)

from saqr_engine.pipeline.config import PRESETS
from saqr_engine.safety.consent import CONSENT_TEXT, attest

from ..widgets import LabeledSlider, bgr_to_qimage, card, read_image, run_task


class ConsentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Confirm consent")
        self.setMinimumWidth(480)
        lay = QVBoxLayout(self)
        t = QLabel(CONSENT_TEXT)
        t.setWordWrap(True)
        lay.addWidget(t)
        self.self_rb = QRadioButton("This is my own face")
        self.perm_rb = QRadioButton("I have explicit permission from the person shown")
        lay.addWidget(self.self_rb)
        lay.addWidget(self.perm_rb)
        self.check = QCheckBox("I confirm the statement above")
        lay.addWidget(self.check)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.ok = bb.button(QDialogButtonBox.StandardButton.Ok)
        self.ok.setEnabled(False)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for w in (self.self_rb, self.perm_rb, self.check):
            w.toggled.connect(self._upd)

    def _upd(self) -> None:
        self.ok.setEnabled(self.check.isChecked() and (self.self_rb.isChecked() or self.perm_rb.isChecked()))

    def basis(self) -> str:
        return "self" if self.self_rb.isChecked() else "permission"


class FaceStudioPage(QWidget):
    sourcesChanged = Signal()

    def __init__(self, ctx, status):
        super().__init__()
        self.ctx, self.status = ctx, status
        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Face Studio")
        h.setObjectName("h1")
        outer.addWidget(h)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        body = QWidget()
        grid = QGridLayout(body)
        grid.setSpacing(14)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        # ---- sources
        c, lay = card("Source faces", "Only use your own face, or a face you have explicit permission to use. "
                                      "Faces are kept in memory only and are cleared when you close the app.")
        self.list = QListWidget()
        self.list.setIconSize(QPixmap(64, 64).size())
        self.list.setMinimumHeight(180)
        self.list.currentItemChanged.connect(self._select)
        lay.addWidget(self.list)
        row = QHBoxLayout()
        add = QPushButton("Import source image…")
        add.setObjectName("primary")
        add.clicked.connect(self.import_source)
        rm = QPushButton("Remove")
        rm.clicked.connect(self.remove_source)
        clr = QPushButton("Clear all")
        clr.clicked.connect(self.clear_sources)
        row.addWidget(add)
        row.addWidget(rm)
        row.addWidget(clr)
        lay.addLayout(row)
        grid.addWidget(c, 0, 0)

        # ---- quality / faces
        c2, l2 = card("Quality & faces")
        r = QHBoxLayout()
        r.addWidget(QLabel("Preset"))
        self.preset = QComboBox()
        self.preset.addItems(["fast", "balanced", "quality"])
        self.preset.setCurrentText(ctx.settings.settings.preset)
        self.preset.currentTextChanged.connect(self._preset)
        r.addWidget(self.preset, 1)
        l2.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel("Faces"))
        self.selection = QComboBox()
        self.selection.addItem("Largest face", "largest")
        self.selection.addItem("All faces", "all")
        self.selection.currentIndexChanged.connect(lambda _i: self._set(face_selection=self.selection.currentData()))
        r.addWidget(self.selection, 1)
        l2.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel("Detector size"))
        self.det = QComboBox()
        for s in (320, 480, 640, 800):
            self.det.addItem(f"{s}px", s)
        self.det.currentIndexChanged.connect(lambda _i: self._set(det_size=self.det.currentData()))
        r.addWidget(self.det, 1)
        l2.addLayout(r)
        r = QHBoxLayout()
        r.addWidget(QLabel("Detect every"))
        self.interval = QComboBox()
        for n in (1, 2, 3, 5):
            self.interval.addItem(f"{n} frame(s)", n)
        self.interval.currentIndexChanged.connect(lambda _i: self._set(detect_interval=self.interval.currentData()))
        r.addWidget(self.interval, 1)
        l2.addLayout(r)
        self.enhance = QCheckBox("Face enhancement (GFPGAN) — slow on CPU")
        self.enhance.toggled.connect(lambda v: self._set(enhance=v))
        l2.addWidget(self.enhance)
        self.enh_blend = LabeledSlider("Enhance strength", ctx.cfg.enhance_blend)
        self.enh_blend.valueChanged.connect(lambda v: self._set(enhance_blend=v))
        l2.addWidget(self.enh_blend)
        self.landmarks = QCheckBox("Precise face shape (106 landmarks) — needed for region masks")
        self.landmarks.toggled.connect(lambda v: self._set(landmarks=v))
        l2.addWidget(self.landmarks)
        grid.addWidget(c2, 0, 1)

        # ---- masks
        c3, l3 = card("Masks & regions", "Choose which parts of the original face stay visible.")
        self.mouth = LabeledSlider("Keep original mouth", ctx.cfg.mouth_mask,
                                   "Keeps your real lip movement (good for talking)")
        self.mouth.valueChanged.connect(lambda v: self._set(mouth_mask=v))
        self.eyes = LabeledSlider("Keep original eyes", ctx.cfg.eye_mask, "Keeps your real gaze and blinks")
        self.eyes.valueChanged.connect(lambda v: self._set(eye_mask=v))
        self.forehead = LabeledSlider("Forehead coverage", ctx.cfg.forehead, "0 = stop at the eyebrows")
        self.forehead.valueChanged.connect(lambda v: self._set(forehead=v))
        self.feather = LabeledSlider("Edge softness", ctx.cfg.mask_feather / 0.4)
        self.feather.valueChanged.connect(lambda v: self._set(mask_feather=v * 0.4))
        for w in (self.mouth, self.eyes, self.forehead, self.feather):
            l3.addWidget(w)
        grid.addWidget(c3, 1, 0)

        # ---- blending
        c4, l4 = card("Blending & colour")
        self.opacity = LabeledSlider("Face blend", ctx.cfg.opacity)
        self.opacity.valueChanged.connect(lambda v: self._set(opacity=v))
        l4.addWidget(self.opacity)
        r = QHBoxLayout()
        r.addWidget(QLabel("Colour correction"))
        self.color = QComboBox()
        self.color.addItem("Match skin tone (LAB)", "lab")
        self.color.addItem("Histogram match", "histogram")
        self.color.addItem("Off", "none")
        self.color.currentIndexChanged.connect(lambda _i: self._set(color_correction=self.color.currentData()))
        r.addWidget(self.color, 1)
        l4.addLayout(r)
        self.cstrength = LabeledSlider("Colour strength", ctx.cfg.color_strength)
        self.cstrength.valueChanged.connect(lambda v: self._set(color_strength=v))
        self.sharp = LabeledSlider("Sharpness", ctx.cfg.sharpness)
        self.sharp.valueChanged.connect(lambda v: self._set(sharpness=v))
        self.smooth = LabeledSlider("Motion smoothing", ctx.cfg.temporal_smoothing / 0.9)
        self.smooth.valueChanged.connect(lambda v: self._set(temporal_smoothing=v * 0.9))
        self.poisson = QCheckBox("Seamless (Poisson) blending — slower")
        self.poisson.toggled.connect(lambda v: self._set(poisson=v))
        for w in (self.cstrength, self.sharp, self.smooth, self.poisson):
            l4.addWidget(w)
        grid.addWidget(c4, 1, 1)
        self.sync_from_cfg()

    # ------------------------------------------------------------------ config
    def _set(self, **kw) -> None:
        self.ctx.cfg.update(**kw)

    def _preset(self, name: str) -> None:
        self.ctx.cfg.update(**PRESETS[name])
        self.ctx.settings.update(preset=name)
        self.sync_from_cfg()
        self.status(f"Preset: {name}")

    def sync_from_cfg(self) -> None:
        c = self.ctx.cfg
        widgets = (self.det, self.interval, self.enhance, self.landmarks, self.color, self.poisson, self.selection)
        for w in widgets:
            w.blockSignals(True)
        self.det.setCurrentIndex(max(0, self.det.findData(c.det_size)))
        self.interval.setCurrentIndex(max(0, self.interval.findData(c.detect_interval)))
        self.enhance.setChecked(c.enhance)
        self.landmarks.setChecked(c.landmarks)
        self.color.setCurrentIndex(max(0, self.color.findData(c.color_correction)))
        self.poisson.setChecked(c.poisson)
        self.selection.setCurrentIndex(max(0, self.selection.findData(c.face_selection)))
        for w in widgets:
            w.blockSignals(False)
        self.sharp.slider.setValue(int(c.sharpness * 100))

    # ------------------------------------------------------------------ sources
    def import_source(self) -> None:
        if not self.ctx.models_ready():
            QMessageBox.information(self, "Models needed", "Download the AI models first (Models page).")
            return
        path, _ = QFileDialog.getOpenFileName(self, "Choose a source face", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        img = read_image(path)
        if img is None:
            QMessageBox.warning(self, "Cannot open", "This image could not be read.")
            return
        dlg = ConsentDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        consent = attest(dlg.basis())
        self.status("Analysing source face…")
        import os

        name = os.path.splitext(os.path.basename(path))[0][:40]
        run_task(self.ctx.engine.add_source, img, consent, name,
                 on_done=self._added, on_error=lambda e: QMessageBox.warning(self, "Source not added", e))

    def _added(self, src) -> None:
        self.refresh()
        self.status(f"Source added: {src.name}")
        self.sourcesChanged.emit()

    def refresh(self) -> None:
        eng = self.ctx.engine
        self.list.blockSignals(True)
        self.list.clear()
        for s in eng.sources.values():
            it = QListWidgetItem(QIcon(QPixmap.fromImage(bgr_to_qimage(s.thumbnail))), s.name)
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            self.list.addItem(it)
            if s.id == eng.active_source:
                self.list.setCurrentItem(it)
        self.list.blockSignals(False)

    def _select(self, cur, _prev) -> None:
        if cur is not None:
            self.ctx.engine.set_active_source(cur.data(Qt.ItemDataRole.UserRole))
            self.sourcesChanged.emit()

    def remove_source(self) -> None:
        it = self.list.currentItem()
        if it:
            self.ctx.engine.remove_source(it.data(Qt.ItemDataRole.UserRole))
            self.refresh()
            self.sourcesChanged.emit()

    def clear_sources(self) -> None:
        self.ctx.engine.clear_sources()
        self.refresh()
        self.sourcesChanged.emit()
