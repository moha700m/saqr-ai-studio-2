"""Camera discovery and opening with the right OS backend.

Windows: DirectShow first (fast open, MJPG support, device names), then Media Foundation.
Linux:   V4L2 forced (fixes preview stalls seen with the generic backend).
macOS:   AVFoundation.
"""
from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Optional

import cv2

log = logging.getLogger(__name__)


@dataclass
class CameraDevice:
    index: int
    name: str
    backend: int

    def to_dict(self) -> dict:
        return {"index": self.index, "name": self.name, "backend": backend_name(self.backend)}


def backend_name(b: int) -> str:
    return {cv2.CAP_DSHOW: "DirectShow", cv2.CAP_MSMF: "MediaFoundation", cv2.CAP_V4L2: "V4L2",
            cv2.CAP_AVFOUNDATION: "AVFoundation", cv2.CAP_ANY: "auto"}.get(b, str(b))


def preferred_backends() -> list[int]:
    if sys.platform == "win32":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    if sys.platform == "darwin":
        return [cv2.CAP_AVFOUNDATION]
    return [cv2.CAP_V4L2, cv2.CAP_ANY]


def list_cameras(max_probe: int = 6) -> list[CameraDevice]:
    """Return available cameras. Uses OS device names when the optional
    ``cv2_enumerate_cameras`` package is present, otherwise probes indices."""
    try:
        from cv2_enumerate_cameras import enumerate_cameras  # MIT, optional

        out, seen = [], set()
        for b in preferred_backends():
            for c in enumerate_cameras(b):
                key = (c.name, c.index)
                if key in seen:
                    continue
                seen.add(key)
                out.append(CameraDevice(c.index, c.name, b))
            if out:
                return out
    except Exception:  # noqa: BLE001 - package missing or backend unsupported
        pass
    if sys.platform.startswith("linux"):
        import glob
        import re

        devs = []
        for p in sorted(glob.glob("/dev/video*")):
            m = re.search(r"(\d+)$", p)
            if not m:
                continue
            idx = int(m.group(1))
            name = p
            try:
                with open(f"/sys/class/video4linux/video{idx}/name", encoding="utf-8") as fh:
                    name = fh.read().strip()
            except OSError:
                pass
            devs.append(CameraDevice(idx, name, cv2.CAP_V4L2))
        return devs
    found = []
    b = preferred_backends()[0]
    for i in range(max_probe):
        cap = cv2.VideoCapture(i, b)
        ok = cap.isOpened()
        cap.release()
        if ok:
            found.append(CameraDevice(i, f"Camera {i}", b))
    return found


def open_camera(index: int, width: int = 1280, height: int = 720, fps: int = 30,
                backend: Optional[int] = None) -> cv2.VideoCapture:
    backends = [backend] if backend is not None else preferred_backends()
    last = None
    for b in backends:
        cap = cv2.VideoCapture(index, b)
        if not cap.isOpened():
            cap.release()
            last = b
            continue
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok, _ = cap.read()
        if ok:
            return cap
        cap.release()
        last = b
    raise RuntimeError(f"could not open camera {index} (last backend tried: {backend_name(last) if last else '-'})")
