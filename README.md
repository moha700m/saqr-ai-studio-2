# SAQR AI Studio

A local-first, hardware-aware live face studio for Windows: real-time camera face swap, an image studio, automatic
GPU/CPU engine selection and a one-click installer. It is an original AGPL-3.0 implementation in the same class
as Deep-Live-Cam.

> **Status: early (0.1).** What works now: the Windows installer, the desktop app (Face Studio, Live Camera, Image
> Studio, Models, Hardware, Privacy), and the CLI. Still in progress: video studio, OBS virtual camera, web app,
> desktop↔web bridge, cloud mode. See [PROJECT_STATE.md](PROJECT_STATE.md).

## Install (Windows 10/11, 64-bit)

1. Download **`SAQR-AI-Setup.exe`** from the [latest release](https://github.com/moha700m/saqr-ai-studio-2/releases/latest).
2. Run it. If SmartScreen warns about an unsigned app, choose **More info → Run anyway**.
3. The installer:
   - detects your CPU, GPU and RAM
   - shows the model licences, then downloads the AI models (about 840 MB, or about 1.2 GB with the enhancer); every
     file is SHA-256 verified
   - creates desktop and Start-menu shortcuts and an uninstaller
   - tests the AI engine on your hardware, then starts the app

You do not need Python, Git, CUDA or FFmpeg.

## Using it

1. **Face Studio**: import a source face. You must confirm it is your face, or that you have the person's permission.
2. **Live Camera**: pick a camera, press *Start camera*, then *Start face swap*. Swapped output always shows an
   "AI-altered" badge.
3. **Image Studio**: open an image, press *Swap face*, then export as PNG or JPEG. You can also process a whole folder
   at once.

## Hardware and AI engine

SAQR tests every available engine on your machine and picks the fastest one that actually works. If it fails at
runtime, SAQR falls back automatically. The order it tries is:

CUDA → TensorRT (opt-in) → DirectML → ROCm → OpenVINO → CoreML → CPU

The Windows build uses DirectML, which runs on NVIDIA, AMD and Intel GPUs, with the CPU as fallback. The Hardware
page shows what was detected, why each engine was or wasn't chosen, and live FPS and latency.

Measured on a 4-core CPU with no GPU: about 1.4 FPS live. Real-time use needs a GPU.

## Run from source

```bash
pip install "./engine[cpu]" PySide6 pillow
```
```bash
PYTHONPATH=engine:desktop python -m saqr_desktop
```

CLI commands: `saqr probe`, `saqr models download`, `saqr swap -s face.jpg -t photo.jpg -o out.png`, `saqr selftest`.

## Responsible use and licences

- Only use your own face or faces you have explicit permission to use.
- Do not use SAQR to impersonate, deceive, defraud or harass anyone, or to create intimate imagery.
- Live output is always labelled "AI-altered".
- The NSFW filter is always on.

Code: AGPL-3.0. The face models are **non-commercial research only**; see [MODEL_LICENSES.md](MODEL_LICENSES.md).
