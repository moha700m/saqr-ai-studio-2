# PyInstaller spec — builds "dist/SAQR AI Studio/" (one-folder, windowed).
# Run from the repo root:  pyinstaller installer/saqr.spec --noconfirm
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))
ICON = os.path.join(ROOT, "desktop", "saqr_desktop", "assets", "saqr.ico")

datas = collect_data_files("saqr_engine") + collect_data_files("saqr_desktop")
binaries = collect_dynamic_libs("onnxruntime")
hidden = collect_submodules("saqr_engine") + collect_submodules("saqr_desktop") + ["cv2_enumerate_cameras"]

a = Analysis(
    [os.path.join(SPECPATH, "launcher.py")],
    pathex=[os.path.join(ROOT, "engine"), os.path.join(ROOT, "desktop")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hidden,
    excludes=["tkinter", "matplotlib", "torch", "tensorflow", "keras", "IPython", "pytest",
              "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.Qt3DCore", "PySide6.QtQuick",
              "PySide6.QtQml", "PySide6.QtPdf", "PySide6.QtCharts", "PySide6.QtDataVisualization"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="SAQR AI Studio", console=False, icon=ICON,
          version=os.path.join(SPECPATH, "version_info.txt") if os.path.exists(os.path.join(SPECPATH, "version_info.txt")) else None)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SAQR AI Studio")
