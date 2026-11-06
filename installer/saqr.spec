# PyInstaller spec - builds "dist/SAQR AI Studio/" (one-folder, windowed).
# Run from the repo root:  pyinstaller installer/saqr.spec --noconfirm
import glob
import os
import sys

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ENGINE = os.path.join(ROOT, "engine")
DESKTOP = os.path.join(ROOT, "desktop")
# make our packages importable *before* the collect_* helpers run (they import the packages)
sys.path[:0] = [ENGINE, DESKTOP]
ICON = os.path.join(DESKTOP, "saqr_desktop", "assets", "saqr.ico")


def package_data(base, package):
    """Every non-Python file inside the package, mapped to its in-bundle folder."""
    out = []
    pkg_dir = os.path.join(base, package)
    for path in glob.glob(os.path.join(pkg_dir, "**", "*"), recursive=True):
        if os.path.isfile(path) and not path.endswith((".py", ".pyc")) and "__pycache__" not in path:
            rel = os.path.relpath(os.path.dirname(path), base)
            out.append((path, rel))
    return out


datas = package_data(ENGINE, "saqr_engine") + package_data(DESKTOP, "saqr_desktop")
required = {"registry.json", "probe_conv.onnx", "saqr.png"}
missing = required - {os.path.basename(src) for src, _ in datas}
if missing:
    raise SystemExit(f"spec: required data files missing: {missing}")
print("spec: bundling data files:", sorted(os.path.basename(s) for s, _ in datas))

binaries = collect_dynamic_libs("onnxruntime")
hidden = collect_submodules("saqr_engine") + collect_submodules("saqr_desktop") + ["cv2_enumerate_cameras"]

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[ENGINE, DESKTOP],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "torch", "tensorflow", "keras", "IPython", "pytest",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SAQR AI Studio", console=False,
          icon=ICON if sys.platform == "win32" else None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SAQR AI Studio")
