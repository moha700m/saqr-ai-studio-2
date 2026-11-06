"""ROI-limited paste-back of an aligned crop into the full frame."""
from __future__ import annotations

import cv2
import numpy as np


def paste_back(frame: np.ndarray, crop: np.ndarray, mask: np.ndarray, M: np.ndarray,
               opacity: float = 1.0, poisson: bool = False) -> np.ndarray:
    """Blend ``crop`` (aligned via ``M`` frame->crop) into ``frame`` in place, touching only its ROI.

    Only the bounding box of the warped crop is processed, so cost scales with face
    size instead of frame size (important for 1080p/4K live input).
    """
    h, w = frame.shape[:2]
    cs = crop.shape[0]
    IM = cv2.invertAffineTransform(M)
    corners = np.array([[0, 0], [cs, 0], [0, cs], [cs, cs]], np.float32) @ IM[:, :2].T + IM[:, 2]
    x0, y0 = np.floor(corners.min(0)).astype(int) - 2
    x1, y1 = np.ceil(corners.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x1 <= x0 or y1 <= y0:
        return frame
    rw, rh = x1 - x0, y1 - y0
    IMr = IM.copy()
    IMr[:, 2] -= (x0, y0)
    warped = cv2.warpAffine(crop, IMr, (rw, rh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    wmask = cv2.warpAffine(mask, IMr, (rw, rh), flags=cv2.INTER_LINEAR, borderValue=0)
    wmask = np.clip(wmask * float(opacity), 0, 1)
    roi = frame[y0:y1, x0:x1]
    if poisson and wmask.max() > 0.5:
        bin_mask = (wmask > 0.5).astype(np.uint8) * 255
        ys, xs = np.nonzero(bin_mask)
        if len(xs) > 32 and xs.min() > 0 and ys.min() > 0 and xs.max() < rw - 1 and ys.max() < rh - 1:
            center = (int((xs.min() + xs.max()) / 2), int((ys.min() + ys.max()) / 2))
            try:
                cloned = cv2.seamlessClone(warped, roi, bin_mask, center, cv2.NORMAL_CLONE)
                warped = cloned
            except cv2.error:
                pass
    m3 = wmask[:, :, None]
    roi[:] = (warped.astype(np.float32) * m3 + roi.astype(np.float32) * (1.0 - m3)).astype(np.uint8)
    return frame
