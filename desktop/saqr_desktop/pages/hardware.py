"""Hardware diagnostics page."""
from __future__ import annotations

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (QComboBox, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QPlainTextEdit,
                               QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from saqr_engine.runtime.providers import LABELS, PRIORITY
from saqr_engine.runtime.session import STATE

from ..widgets import card


class HardwarePage(QWidget):
    def __init__(self, ctx, status, on_backend_change):
        super().__init__()
        self.ctx, self.status, self.on_backend_change = ctx, status, on_backend_change
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Hardware")
        h.setObjectName("h1")
        lay.addWidget(h)
        grid = QGridLayout()
        grid.setSpacing(14)
        self.cpu_c, cl = card("Processor")
        self.cpu = QLabel("…")
        self.cpu.setWordWrap(True)
        cl.addWidget(self.cpu)
        grid.addWidget(self.cpu_c, 0, 0)
        self.gpu_c, gl = card("Graphics")
        self.gpu = QLabel("…")
        self.gpu.setWordWrap(True)
        gl.addWidget(self.gpu)
        grid.addWidget(self.gpu_c, 0, 1)
        self.sys_c, sl = card("System")
        self.sys = QLabel("…")
        self.sys.setWordWrap(True)
        sl.addWidget(self.sys)
        grid.addWidget(self.sys_c, 0, 2)
        lay.addLayout(grid)

        c, bl = card("AI backend", "Chosen automatically after testing each backend on this computer. "
                                   "Advanced users can force one; SAQR falls back automatically if it fails.")
        row = QHBoxLayout()
        self.active = QLabel("…")
        self.active.setObjectName("h2")
        row.addWidget(self.active, 1)
        self.force = QComboBox()
        self.force.addItem("Automatic (recommended)", "")
        for pid in PRIORITY:
            self.force.addItem(LABELS[pid], pid)
        row.addWidget(self.force)
        apply = QPushButton("Apply")
        apply.clicked.connect(self.apply)
        row.addWidget(apply)
        bl.addLayout(row)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Backend", "In this build", "Hardware", "Verified", "Details"])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(220)
        bl.addWidget(self.table)
        lay.addWidget(c)
        c2, pl = card("Live performance & errors")
        self.perf = QLabel("Start the camera to see real FPS and latency.")
        pl.addWidget(self.perf)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(140)
        pl.addWidget(self.log)
        lay.addWidget(c2, 1)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(1000)

    def refresh(self) -> None:
        hw, sel = self.ctx.hardware, self.ctx.selection
        if hw is None or sel is None:
            return
        c = hw.cpu
        simd = ", ".join(x for x, ok in (("AVX", c.avx), ("AVX2", c.avx2), ("AVX-512", c.avx512)) if ok) or "none"
        self.cpu.setText(f"<b>{c.model}</b><br>{c.vendor.upper()} · {c.architecture}<br>"
                         f"{c.physical_cores} cores / {c.logical_cores} threads<br>SIMD: {simd}")
        if hw.gpus:
            self.gpu.setText("<br><br>".join(
                f"<b>{g.name}</b><br>{g.vendor.upper()} · "
                f"{'integrated' if g.integrated else 'dedicated' if g.integrated is False else 'type unknown'}"
                f"<br>VRAM: {g.vram_mb or '?'} MB · driver {g.driver_version or '?'}"
                + (f" · compute {g.compute_capability}" if g.compute_capability else "") for g in hw.gpus))
        else:
            self.gpu.setText("No GPU detected")
        s = hw.system
        self.sys.setText(f"<b>{s.os} {s.os_release}</b><br>{s.os_version}<br>{'64-bit' if s.is_64bit else '32-bit'}"
                         f"<br>RAM {s.ram_total_mb / 1024:.1f} GB ({s.ram_available_mb / 1024:.1f} GB free)"
                         f"<br>Disk free {s.disk_free_mb / 1024:.0f} GB")
        self.active.setText(f"Active: {sel.label}" + (" (forced)" if sel.forced else " (automatic)")
                            + (f" — {sel.fallback_reason}" if sel.fallback_reason else ""))
        self.force.setCurrentIndex(max(0, self.force.findData(self.ctx.settings.settings.forced_backend)))
        self.table.setRowCount(len(sel.checks))
        for r, ch in enumerate(sel.checks):
            vals = [ch.label, "yes" if ch.installed else "no", "ok" if ch.hardware_ok else "no",
                    (f"yes ({ch.latency_ms:.1f} ms)" if ch.latency_ms else "yes") if ch.verified else "no", ch.reason]
            for cc, v in enumerate(vals):
                self.table.setItem(r, cc, QTableWidgetItem(v))
        self.table.resizeColumnsToContents()
        errs = list(hw.errors) + [f"fallback: {e.model} {e.from_backend} → {e.to_backend}: {e.reason}"
                                  for e in STATE.events]
        self.log.setPlainText("\n".join(errs) if errs else "No hardware or backend errors.")

    def apply(self) -> None:
        self.ctx.set_backend(self.force.currentData())
        self.refresh()
        self.on_backend_change()
        self.status(f"Backend: {self.ctx.selection.label}")

    def _tick(self) -> None:
        if not self.isVisible():
            return
        live = self.ctx.live
        if live is None:
            return
        m = live.metrics.snapshot()
        eng = self.ctx.engine
        stages = ", ".join(f"{k} {v:.0f} ms" for k, v in m["stages_ms"].items())
        self.perf.setText(f"<b>{m['fps']:.1f} FPS</b> · latency {m['latency_ms']:.0f} ms"
                          + (f" (p95 {m['latency_p95_ms']:.0f} ms)" if m["latency_p95_ms"] else "")
                          + f" · camera {m['capture_fps']:.0f} FPS · dropped {m['dropped']}<br>"
                          f"Stages: {stages or '—'}<br>Models on: {eng.backends()} · swapper {eng.swapper_id}")
