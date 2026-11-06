"""Colour matching of the generated face to the target frame."""
from __future__ import annotations

import cv2
import numpy as np


def lab_transfer(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """Match LAB mean/std of ``src`` to ``ref`` inside ``mask`` (Reinhard transfer)."""
    sel = mask > 0.5
    if sel.sum() < 16 or strength <= 0:
        return src
    s = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    r = cv2.cvtColor(ref, cv2.COLOR_BGR2LAB).astype(np.float32)
    s_mean, s_std = s[sel].mean(0), s[sel].std(0) + 1e-6
    r_mean, r_std = r[sel].mean(0), r[sel].std(0) + 1e-6
    out = (s - s_mean) * (r_std / s_std) + r_mean
    out = s + (out - s) * float(strength)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def histogram_match(src: np.ndarray, ref: np.ndarray, mask: np.ndarray, strength: float = 1.0) -> np.ndarray:
    sel = mask > 0.5
    if sel.sum() < 16 or strength <= 0:
        return src
    out = src.copy()
    for c in range(3):
        s_vals = src[:, :, c][sel]
        r_vals = ref[:, :, c][sel]
        s_hist = np.bincount(s_vals, minlength=256).astype(np.float64)
        r_hist = np.bincount(r_vals, minlength=256).astype(np.float64)
        s_cdf = np.cumsum(s_hist) / s_hist.sum()
        r_cdf = np.cumsum(r_hist) / r_hist.sum()
        lut = np.interp(s_cdf, r_cdf, np.arange(256)).astype(np.float32)
        mapped = lut[src[:, :, c]]
        out[:, :, c] = np.clip(src[:, :, c] + (mapped - src[:, :, c]) * strength, 0, 255).astype(np.uint8)
    return out


def unsharp(img: np.ndarray, amount: float, radius: float = 1.2) -> np.ndarray:
    if amount <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
