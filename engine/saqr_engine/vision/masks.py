"""Region masks built from the 106-point landmark layout.

Layout (InsightFace 2d106): 0-32 jaw contour, 33-42 eye A, 43-51 brow A,
52-71 mouth, 72-86 nose, 87-96 eye B, 97-105 brow B.
"""
from __future__ import annotations

import cv2
import numpy as np

CONTOUR = list(range(0, 33))
EYE_A = list(range(33, 43))
BROW_A = list(range(43, 52))
MOUTH = list(range(52, 72))
NOSE = list(range(72, 87))
EYE_B = list(range(87, 97))
BROW_B = list(range(97, 106))


def _poly(mask: np.ndarray, pts: np.ndarray, value: float = 1.0) -> None:
    hull = cv2.convexHull(np.round(pts).astype(np.int32))
    cv2.fillConvexPoly(mask, hull, value, lineType=cv2.LINE_AA)


def _dilate_pts(pts: np.ndarray, factor: float) -> np.ndarray:
    c = pts.mean(0, keepdims=True)
    return c + (pts - c) * factor


def ellipse_mask(size: int, feather: float = 0.12) -> np.ndarray:
    m = np.zeros((size, size), np.float32)
    cv2.ellipse(m, (size // 2, int(size * 0.52)), (int(size * 0.40), int(size * 0.47)), 0, 0, 360, 1.0, -1,
                lineType=cv2.LINE_AA)
    return feather_mask(m, feather)


def feather_mask(m: np.ndarray, feather: float, erode: float = 0.0) -> np.ndarray:
    size = m.shape[0]
    if erode > 0:
        k = max(1, int(size * erode))
        m = cv2.erode(m, np.ones((k, k), np.uint8))
    if feather > 0:
        k = max(3, int(size * feather) | 1)
        m = cv2.GaussianBlur(m, (k, k), 0)
    return np.clip(m, 0.0, 1.0)


def face_region_mask(size: int, lm: np.ndarray, forehead: float = 0.5, feather: float = 0.08,
                     erode: float = 0.02) -> np.ndarray:
    """Swap region: jaw contour + brows, optionally extended upward over the forehead.

    ``forehead`` 0 = stop at the brows, 1 = extend by ~0.8x brow-to-nose-tip distance.
    """
    m = np.zeros((size, size), np.float32)
    brows = lm[BROW_A + BROW_B]
    pts = [lm[CONTOUR], brows]
    if forehead > 0:
        nose_tip = lm[NOSE].max(axis=0)  # lowest nose point (largest y)
        brow_c = brows.mean(0)
        up = brow_c - np.array([brow_c[0], nose_tip[1]])
        pts.append(brows + up * 0.8 * forehead)
    _poly(m, np.concatenate(pts))
    return feather_mask(m, feather, erode)


def region_keep_mask(size: int, lm: np.ndarray, mouth: float = 0.0, eyes: float = 0.0,
                     mouth_scale: float = 1.15, eye_scale: float = 1.4, feather: float = 0.05) -> np.ndarray:
    """Mask of regions where the ORIGINAL frame should show through (value = keep strength)."""
    keep = np.zeros((size, size), np.float32)
    if mouth > 0:
        m = np.zeros_like(keep)
        _poly(m, _dilate_pts(lm[MOUTH], mouth_scale))
        keep = np.maximum(keep, feather_mask(m, feather) * mouth)
    if eyes > 0:
        m = np.zeros_like(keep)
        _poly(m, _dilate_pts(lm[EYE_A], eye_scale))
        _poly(m, _dilate_pts(lm[EYE_B], eye_scale))
        keep = np.maximum(keep, feather_mask(m, feather) * eyes)
    return keep
