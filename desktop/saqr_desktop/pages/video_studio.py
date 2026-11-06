"""Video Studio: timeline preview, trim, quality, progress, cancel and MP4 export."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
                               QProgressBar, QPushButton, QSlider, QSpinBox, QVBoxLayout, QWidget)

from saqr_engine.pipeline.video import VideoCancelled, VideoJob, ffmpeg_path, probe, process_video, read_frame_at

from ..widgets import ImageView, run_task


class VideoStudioPage(QWidget):
    def __init__(self, ctx, status):
        super().__init__()
        self.ctx, self.status = ctx, status
        self.path = ""
        self.info = None
        self.cancel = threading.Event()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Video Studio")
        h.setObjectName("h1")
        lay.addWidget(h)
        sub = QLabel("Processed on this computer with FFmpeg. The original audio is kept.")
        sub.setObjectName("muted")
        lay.addWidget(sub)
        bar = QHBoxLayout()
        op = QPushButton("Open video...")
        op.clicked.connect(self.open)
        bar.addWidget(op)
        bar.addWidget(QLabel("From"))
        self.start = QDoubleSpinBox()
        self.start.setSuffix(" s")
        self.start.setDecimals(1)
        bar.addWidget(self.start)
        bar.addWidget(QLabel("To"))
        self.end = QDoubleSpinBox()
        self.end.setSuffix(" s")
        self.end.setDecimals(1)
        bar.addWidget(self.end)
        self.res = QComboBox()
        for label, s in (("Original resolution", 1.0), ("75%", 0.75), ("50%", 0.5)):
            self.res.addItem(label, s)
        bar.addWidget(self.res)
        bar.addWidget(QLabel("Quality"))
        self.crf = QSpinBox()
        self.crf.setRange(14, 32)
        self.crf.setValue(20)
        self.crf.setToolTip("CRF: lower = better quality, bigger file")
        bar.addWidget(self.crf)
        self.enc = QComboBox()
        for label, e in (("Auto encoder (GPU if available)", "auto"), ("CPU x264", "libx264"),
                         ("NVIDIA NVENC", "h264_nvenc"), ("AMD AMF", "h264_amf"), ("Intel QSV", "h264_qsv")):
            self.enc.addItem(label, e)
        bar.addWidget(self.enc)
        self.audio = QCheckBox("Keep audio")
        self.audio.setChecked(True)
        bar.addWidget(self.audio)
        self.wm = QCheckBox("AI badge")
        self.wm.setChecked(ctx.settings.settings.export_watermark)
        bar.addWidget(self.wm)
        self.meta = QCheckBox("AI metadata")
        self.meta.setChecked(ctx.settings.settings.export_metadata)
        bar.addWidget(self.meta)
        bar.addStretch(1)
        lay.addLayout(bar)
        views = QHBoxLayout()
        self.input_view = ImageView("Input preview")
        self.output_view = ImageView("Output preview")
        views.addWidget(self.input_view, 1)
        views.addWidget(self.output_view, 1)
        lay.addLayout(views, 1)
        self.timeline = QSlider(Qt.Orientation.Horizontal)
        self.timeline.setEnabled(False)
        self.timeline.sliderReleased.connect(self.scrub)
        lay.addWidget(self.timeline)
        row = QHBoxLayout()
        self.prev_btn = QPushButton("Preview swap at this frame")
        self.prev_btn.setEnabled(False)
        self.prev_btn.clicked.connect(self.preview_frame)
        row.addWidget(self.prev_btn)
        self.run_btn = QPushButton("Export MP4...")
        self.run_btn.setObjectName("primary")
        self.run_btn.setEnabled(False)
        self.run_btn.clicked.connect(self.export)
        row.addWidget(self.run_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setObjectName("danger")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel.set)
        row.addWidget(self.cancel_btn)
        row.addStretch(1)
        self.meta_lbl = QLabel("")
        self.meta_lbl.setObjectName("muted")
        row.addWidget(self.meta_lbl)
        lay.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.msg = QLabel("" if ffmpeg_path() else "FFmpeg was not found. Reinstall SAQR AI Studio.")
        self.msg.setObjectName("muted")
        lay.addWidget(self.msg)

    def on_sources_changed(self) -> None:
        ok = bool(self.path) and bool(self.ctx.engine.sources)
        self.run_btn.setEnabled(ok)
        self.prev_btn.setEnabled(ok)

    def open(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open video", "", "Videos (*.mp4 *.mov *.mkv *.avi *.webm)")
        if not path:
            return
        try:
            self.info = probe(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Cannot open", str(exc))
            return
        self.path = path
        i = self.info
        for sb in (self.start, self.end):
            sb.setRange(0, max(0.1, i.duration))
        self.start.setValue(0)
        self.end.setValue(i.duration)
        self.timeline.setRange(0, int(i.duration * 10))
        self.timeline.setValue(0)
        self.timeline.setEnabled(True)
        self.meta_lbl.setText(f"{i.width}x{i.height} · {i.fps:g} fps · {i.duration:.1f} s · "
                              f"{'audio' if i.has_audio else 'no audio'}")
        self.scrub()
        self.on_sources_changed()

    def scrub(self) -> None:
        if self.path:
            self.input_view.set_bgr(read_frame_at(self.path, self.timeline.value() / 10))

    def preview_frame(self) -> None:
        frame = read_frame_at(self.path, self.timeline.value() / 10, max_w=4096)
        if frame is None:
            return
        self.ctx.engine.reset_tracking()
        run_task(lambda: self.ctx.engine.process(frame, force_detect=True),
                 on_done=lambda r: self.output_view.set_bgr(r.frame),
                 on_error=lambda e: QMessageBox.warning(self, "Preview failed", e))

    def export(self) -> None:
        if not self.ctx.models_ready():
            QMessageBox.information(self, "Models needed", "Download the AI models first (Models page).")
            return
        out, _ = QFileDialog.getSaveFileName(self, "Export video", str(Path.home() / (Path(self.path).stem + "_saqr.mp4")),
                                             "MP4 (*.mp4)")
        if not out:
            return
        job = VideoJob(self.path, out, start=self.start.value(),
                       end=self.end.value() if self.end.value() < self.info.duration - 0.05 else 0.0,
                       scale=self.res.currentData(), crf=self.crf.value(), encoder=self.enc.currentData(),
                       keep_audio=self.audio.isChecked(), watermark=self.wm.isChecked(), metadata=self.meta.isChecked())
        self.cancel.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 1000)
        self.run_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        def work(progress):
            return process_video(self.ctx.engine, job, progress=lambda d, t, f: progress("p", d, t, f),
                                 cancel=self.cancel, preview=lambda fr: progress("f", fr))

        def prog(p):
            if p[0] == "p":
                _k, d, t, f = p
                self.progress.setValue(int(d / max(1, t) * 1000))
                left = (t - d) / f if f > 0 else 0
                self.msg.setText(f"{d}/{t} frames · {d / max(1, t) * 100:.0f}% · {f:.1f} fps · about {left:.0f} s left")
            else:
                self.output_view.set_bgr(p[1])

        run_task(work, on_progress=prog, on_done=self._done, on_error=self._err)

    def _finish(self) -> None:
        self.progress.setVisible(False)
        self.cancel_btn.setEnabled(False)
        self.on_sources_changed()

    def _done(self, r) -> None:
        self._finish()
        self.msg.setText(f"Saved {r['output']} — {r['frames']} frames in {r['seconds']} s ({r['fps']} fps, "
                         f"encoder {r['encoder']}, faces swapped in {r['swapped_frames']} frames)")
        self.status("Video exported")

    def _err(self, e: str) -> None:
        self._finish()
        if "cancelled" in e:
            self.msg.setText("Export cancelled. Temporary files were deleted.")
        else:
            QMessageBox.warning(self, "Export failed", e)


__all__ = ["VideoStudioPage", "VideoCancelled"]
