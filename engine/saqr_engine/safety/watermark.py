"""Synthetic-media disclosure: visible badge + embedded metadata.

* Virtual-camera / live output: badge is always drawn (cannot be disabled).
* Exports: badge and metadata are on by default; the user may turn them off for exports only.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

BADGE_TEXT = "AI-altered"
METADATA_TEXT = "AI-generated / AI-altered media created with SAQR AI Studio"


def draw_badge(frame: np.ndarray, text: str = BADGE_TEXT) -> np.ndarray:
    """Draw a small semi-transparent badge in the bottom-right corner (in place)."""
    h, w = frame.shape[:2]
    scale = max(0.4, min(w, h) / 900.0)
    thick = max(1, int(round(scale * 2)))
    (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(8 * scale) + 4
    x1, y1 = w - tw - pad * 2 - 8, h - th - base - pad * 2 - 8
    x2, y2 = w - 8, h - 8
    if x1 < 0 or y1 < 0:
        return frame
    roi = frame[y1:y2, x1:x2]
    overlay = np.zeros_like(roi)
    overlay[:] = (60, 20, 40)
    cv2.addWeighted(overlay, 0.6, roi, 0.4, 0, dst=roi)
    cv2.putText(frame, text, (x1 + pad, y2 - pad - base // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 230, 120), thick, cv2.LINE_AA)
    return frame


def write_image(path: Path, img: np.ndarray, metadata: bool = True, quality: int = 95) -> Path:
    """Write PNG/JPEG. With ``metadata`` the file carries an AI-generated disclosure tag."""
    path = Path(path)
    ext = path.suffix.lower()
    if metadata:
        try:
            from PIL import Image, PngImagePlugin

            rgb = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if ext == ".png":
                info = PngImagePlugin.PngInfo()
                info.add_text("Description", METADATA_TEXT)
                info.add_text("Software", "SAQR AI Studio")
                info.add_text("ai_generated", "true")
                rgb.save(path, pnginfo=info)
            else:
                exif = Image.Exif()
                exif[0x010E] = METADATA_TEXT  # ImageDescription
                exif[0x0131] = "SAQR AI Studio"  # Software
                rgb.save(path, quality=quality, exif=exif)
            return path
        except ImportError:
            pass
    params = [cv2.IMWRITE_JPEG_QUALITY, quality] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext or ".png", img, params)
    if not ok:
        raise IOError(f"could not encode {path}")
    path.write_bytes(buf.tobytes())
    return path
