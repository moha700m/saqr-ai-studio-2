"""Lightweight face tracker.

Between detector runs, the 5 keypoints (and 106 landmarks when present) are
propagated with pyramidal Lucas-Kanade optical flow. This keeps alignment fresh
on every frame at a fraction of the detector cost. IoU matching gives faces a
stable ``track_id`` across detections (used for identity-mapping caches).
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from ..vision.face import Face


def iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


class FaceTracker:
    def __init__(self) -> None:
        self.faces: list[Face] = []
        self._next_id = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._last_detect = -10**9
        self._lk = dict(winSize=(21, 21), maxLevel=3,
                        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))

    def reset(self) -> None:
        self.faces = []
        self._prev_gray = None
        self._last_detect = -10**9

    def needs_detection(self, frame_idx: int, interval: int) -> bool:
        if not self.faces or self._prev_gray is None:
            self._last_detect = frame_idx
            return True
        if frame_idx - self._last_detect >= interval:
            self._last_detect = frame_idx
            return True
        return False

    def update_detections(self, frame: np.ndarray, faces: list[Face], smoothing: float = 0.0) -> list[Face]:
        used = set()
        for f in faces:
            best, best_iou = None, 0.3
            for p in self.faces:
                if p.track_id in used:
                    continue
                v = iou(f.bbox, p.bbox)
                if v > best_iou:
                    best, best_iou = p, v
            if best is not None:
                f.track_id = best.track_id
                used.add(best.track_id)
                if smoothing > 0:
                    f.kps = best.kps * smoothing + f.kps * (1 - smoothing)
            else:
                f.track_id = self._next_id
                self._next_id += 1
        self.faces = faces
        return faces

    def track(self, frame: np.ndarray) -> Optional[list[Face]]:
        if self._prev_gray is None or not self.faces:
            return None
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != self._prev_gray.shape:
            return None
        out = []
        for f in self.faces:
            pts = f.kps if f.landmarks is None else np.vstack([f.kps, f.landmarks])
            p0 = pts.reshape(-1, 1, 2).astype(np.float32)
            p1, st, _ = cv2.calcOpticalFlowPyrLK(self._prev_gray, gray, p0, None, **self._lk)
            if p1 is None or st is None or st[:5].sum() < 5:
                return None
            p1 = p1.reshape(-1, 2)
            shift = p1[:5] - f.kps
            if np.abs(shift).max() > f.size * 0.5:  # implausible jump
                return None
            nf = Face(bbox=f.bbox + np.tile(shift.mean(0), 2), kps=p1[:5], score=f.score,
                      landmarks=p1[5:] if f.landmarks is not None else None, embedding=f.embedding,
                      track_id=f.track_id)
            out.append(nf)
        self.faces = out
        return out

    def remember(self, frame: np.ndarray) -> None:
        self._prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
