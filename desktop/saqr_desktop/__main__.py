"""SAQR AI Studio desktop entry point.

Extra flags used by the Windows installer (no window is shown):
    --probe-json PATH      write hardware report + verified backend to PATH
    --selftest-json PATH   load every installed model, run one inference, write results
    --cli ...              run the ``saqr`` command line (e.g. --cli models list)
"""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import sys
from pathlib import Path


def _watchdog(log_path: Path | None) -> None:
    """Never hang forever in headless mode: after SAQR_WATCHDOG seconds (default 900) dump every
    thread's stack to the log and exit, so installers and CI always get control back."""
    import faulthandler

    secs = float(os.environ.get("SAQR_WATCHDOG", "900"))
    if secs <= 0:
        return
    fh = open(log_path, "a", encoding="utf-8") if log_path else sys.stderr  # noqa: SIM115 - lives until exit
    if fh is None:  # windowed build without a console
        return
    # C-level watchdog (works even if a native call holds the GIL) ...
    faulthandler.dump_traceback_later(secs, exit=True, file=fh)

    # ... plus a Python-level one a little earlier with readable thread names
    import threading
    import traceback

    def fire() -> None:
        try:
            fh.write(f"\nWATCHDOG: still running after {secs - 5:.0f} s, thread stacks:\n")
            names = {t.ident: t.name for t in threading.enumerate()}
            for ident, frame in sys._current_frames().items():
                fh.write(f"--- thread {names.get(ident, ident)}\n{''.join(traceback.format_stack(frame))}")
            fh.flush()
        finally:
            os._exit(3)

    t = threading.Timer(max(1.0, secs - 5), fire)
    t.daemon = True
    t.start()


def _step(log: Path | None, msg: str) -> None:
    if log:
        import time

        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"{time.strftime('%H:%M:%S')} {msg}\n")


def _headless(argv: list[str]) -> int | None:
    if "--cli" in argv:
        from saqr_engine.config import data_dir

        _watchdog(data_dir() / "cli_watchdog.log")
        from saqr_engine.cli import main

        return main(argv[argv.index("--cli") + 1:])
    for flag in ("--probe-json", "--selftest-json"):
        if flag in argv:
            out = Path(argv[argv.index(flag) + 1])
            log = out.with_suffix(".log")
            log.write_text("", encoding="utf-8")
            _watchdog(log)
            _step(log, f"start {flag}")
            from saqr_engine.config import SettingsStore
            from saqr_engine.hardware import detect_hardware
            from saqr_engine.models.manager import ModelManager
            from saqr_engine.runtime.providers import select_backend

            hw = detect_hardware()
            _step(log, f"hardware detected in {hw.duration_ms:.0f} ms")
            sel = select_backend(hw, forced=SettingsStore().settings.forced_backend or None)
            _step(log, f"backend selected: {sel.selected}")
            data: dict = {"hardware": hw.to_dict(), "backend": sel.to_dict()}
            if flag == "--selftest-json":
                mm = ModelManager()
                data["models"] = {}
                for s in mm.list():
                    _step(log, f"testing {s.spec.id} (installed={s.installed})")
                    data["models"][s.spec.id] = (mm.test(s.spec.id, sel) if s.installed
                                                 else {"ok": False, "error": "not installed"})
                    _step(log, f"  -> {data['models'][s.spec.id]}")
                data["ok"] = all(v.get("ok") for k, v in data["models"].items()
                                 if mm.status(k).installed)
            # short human summary for the installer's final page
            g = hw.best_gpu
            data["summary"] = (f"CPU: {hw.cpu.model} ({hw.cpu.physical_cores} cores)\n"
                               f"GPU: {g.name if g else 'none detected'}"
                               f"{f' ({g.vram_mb} MB)' if g and g.vram_mb else ''}\n"
                               f"RAM: {hw.system.ram_total_mb / 1024:.0f} GB\n"
                               f"AI backend: {sel.label}")
            out.write_text(json.dumps(data, indent=2), encoding="utf-8")
            summary = data["summary"]
            if "models" in data:
                okc = sum(1 for v in data["models"].values() if v.get("ok"))
                summary += f"\nModels working: {okc} of {len(data['models'])}"
            out.with_suffix(".txt").write_text(summary, encoding="utf-8")
            _step(log, "done")
            return 0 if data.get("ok", True) else 1
    return None


def main() -> int:
    multiprocessing.freeze_support()
    logging.basicConfig(level=os.environ.get("SAQR_LOG", "WARNING"))
    rc = _headless(sys.argv[1:])
    if rc is not None:
        return rc
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from .context import AppContext
    from .main_window import MainWindow
    from .theme import QSS

    app = QApplication(sys.argv)
    app.setApplicationName("SAQR AI Studio")
    app.setStyleSheet(QSS)
    icon = Path(__file__).with_name("assets") / "saqr.png"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    win = MainWindow(AppContext())
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
