"""Image Studio: single and batch image processing with export."""
from __future__ import annotations

import os
import time
from pathlib import Path

import cv2
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox, QProgressBar,
                               QPushButton, QSpinBox, QVBoxLayout, QWidget)

from saqr_engine.safety.watermark import draw_badge, write_image

from ..widgets import ImageView, read_image, run_task


class ImageStudioPage(QWidget):
    def __init__(self, ctx, status):
        super().__init__()
        self.ctx, self.status = ctx, status
        self.target = None
        self.result = None
        self.target_path = ""
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Image Studio")
        h.setObjectName("h1")
        lay.addWidget(h)
        sub = QLabel("Pick a source face in Face Studio, then open the image to edit. Processing happens on this "
                     "computer; nothing is uploaded.")
        sub.setObjectName("muted")
        sub.setWordWrap(True)
        lay.addWidget(sub)
        bar = QHBoxLayout()
        op = QPushButton("Open image…")
        op.clicked.connect(self.open)
        bar.addWidget(op)
        self.run_btn = QPushButton("Swap face")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.run)
        bar.addWidget(self.run_btn)
        bar.addStretch(1)
        self.fmt = QComboBox()
        self.fmt.addItems(["PNG", "JPEG"])
        bar.addWidget(self.fmt)
        bar.addWidget(QLabel("JPEG quality"))
        self.quality = QSpinBox()
        self.quality.setRange(50, 100)
        self.quality.setValue(95)
        bar.addWidget(self.quality)
        self.scale = QComboBox()
        for label, s in (("Original size", 1.0), ("75%", 0.75), ("50%", 0.5), ("200% (upscale)", 2.0)):
            self.scale.addItem(label, s)
        bar.addWidget(self.scale)
        self.wm = QCheckBox("Visible AI badge")
        self.wm.setChecked(ctx.settings.settings.export_watermark)
        self.meta = QCheckBox("AI metadata tag")
        self.meta.setChecked(ctx.settings.settings.export_metadata)
        bar.addWidget(self.wm)
        bar.addWidget(self.meta)
        self.save_btn = QPushButton("Export…")
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save)
        bar.addWidget(self.save_btn)
        batch = QPushButton("Batch folder…")
        batch.clicked.connect(self.batch)
        bar.addWidget(batch)
        lay.addLayout(bar)
        views = QHBoxLayout()
        self.before = ImageView("Input")
        self.after = ImageView("Output")
        views.addWidget(self.before, 1)
        views.addWidget(self.after, 1)
        lay.addLayout(views, 1)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.info = QLabel("")
        self.info.setObjectName("muted")
        lay.addWidget(self.info)

    def on_sources_changed(self) -> None:
        self.run_btn.setEnabled(self.target is not None and bool(self.ctx.engine.sources))

    def open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open image", "", "Images (*.png *.jpg *.jpeg *.webp *.bmp)")
        if not path:
            return
        img = read_image(path)
        if img is None:
            QMessageBox.warning(self, "Cannot open", "This image could not be read.")
            return
        self.target, self.target_path, self.result = img, path, None
        self.before.set_bgr(img)
        self.after.set_bgr(None)
        self.after.setText("Output")
        self.save_btn.setEnabled(False)
        self.on_sources_changed()

    def _process(self, img):
        eng = self.ctx.engine
        t0 = time.perf_counter()
        res = eng.process(img, force_detect=True)
        dt = (time.perf_counter() - t0) * 1000
        blocked = res.swapped > 0 and eng.check_nsfw(res.frame) >= 0.8
        return res, dt, blocked

    def run(self) -> None:
        if not self.ctx.models_ready():
            QMessageBox.information(self, "Models needed", "Download the AI models first (Models page).")
            return
        self.run_btn.setEnabled(False)
        self.status("Processing…")
        self.ctx.engine.reset_tracking()
        run_task(self._process, self.target, on_done=self._done, on_error=self._err)

    def _err(self, e: str) -> None:
        self.run_btn.setEnabled(True)
        QMessageBox.warning(self, "Processing failed", e)

    def _done(self, out) -> None:
        res, dt, blocked = out
        self.run_btn.setEnabled(True)
        if blocked:
            QMessageBox.warning(self, "Blocked", "The result was blocked by the NSFW safety filter.")
            return
        if res.swapped == 0:
            self.info.setText("No face found in this image.")
            return
        self.result = res.frame
        self.after.set_bgr(self.result)
        self.save_btn.setEnabled(True)
        be = self.ctx.engine.backends().get("swapper", "?")
        self.info.setText(f"{res.swapped} face(s) swapped in {dt:.0f} ms on {be}.")
        self.status("Done")

    def _finalise(self, img):
        out = img.copy()
        s = self.scale.currentData()
        if s != 1.0:
            out = cv2.resize(out, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        if self.wm.isChecked():
            draw_badge(out)
        return out

    def save(self) -> None:
        if self.result is None:
            return
        ext = ".png" if self.fmt.currentText() == "PNG" else ".jpg"
        base = Path(self.target_path).stem + "_saqr" + ext
        path, _ = QFileDialog.getSaveFileName(self, "Export", str(Path.home() / base), f"*{ext}")
        if not path:
            return
        self.ctx.settings.update(export_watermark=self.wm.isChecked(), export_metadata=self.meta.isChecked())
        write_image(Path(path), self._finalise(self.result), metadata=self.meta.isChecked(),
                    quality=self.quality.value())
        self.status(f"Saved {path}")

    def batch(self) -> None:
        if not self.ctx.engine.sources:
            QMessageBox.information(self, "No source", "Add a source face in Face Studio first.")
            return
        src = QFileDialog.getExistingDirectory(self, "Folder with images")
        if not src:
            return
        dst = QFileDialog.getExistingDirectory(self, "Output folder")
        if not dst:
            return
        files = [f for f in sorted(os.listdir(src)) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp", ".bmp"))]
        self.progress.setVisible(True)
        self.progress.setRange(0, len(files))
        ext = ".png" if self.fmt.currentText() == "PNG" else ".jpg"
        wm, meta, q = self.wm.isChecked(), self.meta.isChecked(), self.quality.value()

        def work(progress):
            done = skipped = 0
            for i, f in enumerate(files):
                img = read_image(os.path.join(src, f))
                if img is not None:
                    res, _dt, blocked = self._process(img)
                    if res.swapped and not blocked:
                        out = self._finalise(res.frame) if wm else res.frame
                        write_image(Path(dst) / (Path(f).stem + "_saqr" + ext), out, metadata=meta, quality=q)
                        done += 1
                    else:
                        skipped += 1
                progress(i + 1)
            return done, skipped

        run_task(work, on_progress=lambda p: self.progress.setValue(p[0]),
                 on_done=lambda r: (self.progress.setVisible(False),
                                    self.info.setText(f"Batch finished: {r[0]} saved, {r[1]} skipped.")),
                 on_error=self._err)
