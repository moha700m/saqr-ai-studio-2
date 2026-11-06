"""Model Manager page."""
from __future__ import annotations

import threading

from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QHeaderView, QLabel, QMessageBox, QProgressBar, QPushButton,
                               QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ..widgets import card, run_task


class ModelsPage(QWidget):
    def __init__(self, ctx, status, on_change):
        super().__init__()
        self.ctx, self.status, self.on_change = ctx, status, on_change
        self.cancel = threading.Event()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 20, 24, 20)
        h = QLabel("Models")
        h.setObjectName("h1")
        lay.addWidget(h)
        c, cl = card("Licences", "SAQR does not ship model weights. They are downloaded from their public source "
                                 "and checked against a SHA-256 checksum. The face-analysis and face-swap models from "
                                 "InsightFace are licensed for NON-COMMERCIAL research use only.")
        self.accept = QCheckBox("I accept the licence of every model I download, including the non-commercial "
                                "InsightFace terms")
        cl.addWidget(self.accept)
        lay.addWidget(c)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Model", "Role", "Size", "Status", "Licence", ""])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        lay.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.dl_btn = QPushButton("Download recommended models")
        self.dl_btn.setObjectName("primary")
        self.dl_btn.clicked.connect(self.download_recommended)
        row.addWidget(self.dl_btn)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel.set)
        row.addWidget(self.cancel_btn)
        verify = QPushButton("Verify checksums")
        verify.clicked.connect(self.verify_all)
        row.addWidget(verify)
        test = QPushButton("Test on active backend")
        test.clicked.connect(self.test_all)
        row.addWidget(test)
        row.addStretch(1)
        self.usage = QLabel("")
        self.usage.setObjectName("muted")
        row.addWidget(self.usage)
        lay.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        lay.addWidget(self.progress)
        self.msg = QLabel("")
        self.msg.setObjectName("muted")
        self.msg.setWordWrap(True)
        lay.addWidget(self.msg)
        self.refresh()

    def refresh(self) -> None:
        mm = self.ctx.models
        items = mm.list()
        self.table.setRowCount(len(items))
        for r, st in enumerate(items):
            s = st.spec
            vals = [s.title, s.role, f"{s.size / 1e6:.0f} MB",
                    ("Installed" + (" ✓ verified" if st.verified else "")) if st.installed else "Not installed",
                    s.license]
            for c, v in enumerate(vals):
                it = QTableWidgetItem(v)
                it.setToolTip(s.license_url if c == 4 else v)
                self.table.setItem(r, c, it)
            btn = QPushButton("Remove" if st.installed else "Download")
            btn.clicked.connect(lambda _=False, mid=s.id, inst=st.installed: self.remove(mid) if inst
                                else self.download([mid]))
            self.table.setCellWidget(r, 5, btn)
        self.table.resizeColumnsToContents()
        self.usage.setText(f"{mm.disk_usage() / 1e9:.2f} GB on disk")
        miss = mm.missing_required()
        self.msg.setText("All required models are installed." if not miss else "Missing: " + ", ".join(miss))

    def recommended(self) -> list[str]:
        mm = self.ctx.models
        backend = self.ctx.selection.selected if self.ctx.selection else "cpu"
        ids = [m for m, s in mm.specs.items() if s.required]
        ids.append(mm.recommended_swapper(backend))
        ids.append("gfpgan_1.4")
        return [i for i in ids if not mm.path_for(i)]

    def download_recommended(self) -> None:
        ids = self.recommended()
        if not ids:
            self.msg.setText("Everything recommended is already installed.")
            return
        self.download(ids)

    def download(self, ids: list[str]) -> None:
        if not self.accept.isChecked():
            QMessageBox.information(self, "Licence", "Tick the licence checkbox first.")
            return
        mm = self.ctx.models
        mm.accept_license(ids)
        total = sum(mm.spec(i).size for i in ids)
        self.cancel.clear()
        self.progress.setVisible(True)
        self.progress.setRange(0, 1000)
        self.dl_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        def work(progress):
            base = 0
            for mid in ids:
                mm.download(mid, progress=lambda m, d, t, b=base: progress(m, b + d, total), cancel=self.cancel)
                base += mm.spec(mid).size
            return len(ids)

        def prog(p):
            mid, done, tot = p
            self.progress.setValue(int(done / max(1, tot) * 1000))
            self.msg.setText(f"Downloading {mid}: {done / 1e6:.0f} / {tot / 1e6:.0f} MB")

        run_task(work, on_progress=prog, on_done=self._dl_done, on_error=self._dl_err)

    def _dl_done(self, n) -> None:
        self.progress.setVisible(False)
        self.dl_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.refresh()
        self.msg.setText(f"Downloaded and verified {n} model(s).")
        self.on_change()

    def _dl_err(self, e: str) -> None:
        self.progress.setVisible(False)
        self.dl_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.refresh()
        self.msg.setText(f"Download stopped: {e}. Run it again to resume.")

    def remove(self, mid: str) -> None:
        if QMessageBox.question(self, "Remove model", f"Delete {mid} from disk?") == QMessageBox.StandardButton.Yes:
            self.ctx.models.remove(mid)
            self.refresh()
            self.on_change()

    def verify_all(self) -> None:
        mm = self.ctx.models
        self.msg.setText("Verifying checksums…")
        run_task(lambda: {s.spec.id: mm.verify(s.spec.id) for s in mm.list() if s.installed},
                 on_done=lambda r: (self.refresh(), self.msg.setText(
                     "Checksums: " + ", ".join(f"{k} {'OK' if v else 'MISMATCH'}" for k, v in r.items()))),
                 on_error=lambda e: self.msg.setText(e))

    def test_all(self) -> None:
        mm, sel = self.ctx.models, self.ctx.selection
        self.msg.setText("Testing models…")
        run_task(lambda: {s.spec.id: mm.test(s.spec.id, sel) for s in mm.list() if s.installed},
                 on_done=lambda r: self.msg.setText(" · ".join(
                     f"{k}: {v['backend']} {v['infer_ms']} ms" if v.get("ok") else f"{k}: FAILED {v.get('error')}"
                     for k, v in r.items())),
                 on_error=lambda e: self.msg.setText(e))
