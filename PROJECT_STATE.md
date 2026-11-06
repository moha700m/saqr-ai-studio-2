# Project state

| Area | Status |
|---|---|
| Hardware detection (CPU/GPU/RAM/disk/OS) | Done, unit-tested |
| Verified backend selection + runtime fallback | Done, unit-tested |
| Model Manager (licence gate, resumable download, SHA-256) | Done, unit-tested |
| Face pipeline (detect, embed, landmarks, swap, enhance, masks, colour, tracking, mapping) | Done; real-model test passes on CPU |
| Live camera pipeline | Done; tested with file/synthetic sources (no webcam in CI) |
| Desktop app (PySide6) | Dashboard, Face Studio, Live Camera, Image Studio, Models, Hardware, Privacy |
| Windows installer (PyInstaller + Inno Setup) + CI install/uninstall test | Built and tested by GitHub Actions |
| Video Studio (FFmpeg) | Not started |
| OBS virtual camera | Not started (pyvirtualcam is GPLv2; own writer planned) |
| Web app / Cloudflare | Not started |
| Desktop↔web bridge, cloud mode | Not started |
