"""Similarity alignment to canonical 5-point templates."""
from __future__ import annotations

import cv2
import numpy as np

# ArcFace 112x112 reference points (left eye, right eye, nose, left mouth, right mouth)
ARCFACE_112 = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
# FFHQ 512x512 reference points used by GFPGAN-family restorers
FFHQ_512 = np.array([[192.98138, 239.94708], [318.90277, 240.19366], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], dtype=np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation + uniform scale + translation), returns 2x3."""
    src = src.astype(np.float64)
    dst = dst.astype(np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(cov) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    var_s = (sc ** 2).sum() / len(src)
    scale = (S * d).sum() / var_s if var_s > 0 else 1.0
    t = mu_d - scale * R @ mu_s
    M = np.zeros((2, 3), np.float64)
    M[:, :2] = scale * R
    M[:, 2] = t
    return M.astype(np.float32)


def arcface_template(size: int) -> np.ndarray:
    """ArcFace template scaled to ``size`` (112 for recognition, 128 for inswapper)."""
    ratio = size / 112.0
    return ARCFACE_112 * ratio


def estimate_norm(kps: np.ndarray, size: int = 112) -> np.ndarray:
    return umeyama(np.asarray(kps, np.float32), arcface_template(size))


def norm_crop(img: np.ndarray, kps: np.ndarray, size: int = 112) -> tuple[np.ndarray, np.ndarray]:
    M = estimate_norm(kps, size)
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return crop, M


def ffhq_crop(img: np.ndarray, kps: np.ndarray, size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    M = umeyama(np.asarray(kps, np.float32), FFHQ_512 * (size / 512.0))
    crop = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return crop, M


def transform_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float32)
    return pts @ M[:, :2].T + M[:, 2]
