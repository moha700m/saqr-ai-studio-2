from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Face:
    bbox: np.ndarray  # (4,) x1,y1,x2,y2 in frame pixels
    kps: np.ndarray  # (5,2) eyes, nose, mouth corners
    score: float
    landmarks: Optional[np.ndarray] = None  # (106,2)
    embedding: Optional[np.ndarray] = None  # (512,) raw ArcFace output
    track_id: int = -1
    extra: dict = field(default_factory=dict)

    @property
    def normed_embedding(self) -> Optional[np.ndarray]:
        if self.embedding is None:
            return None
        n = np.linalg.norm(self.embedding)
        return self.embedding / n if n > 0 else self.embedding

    @property
    def size(self) -> float:
        return float(max(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.bbox[0] + self.bbox[2]) / 2, (self.bbox[1] + self.bbox[3]) / 2], np.float32)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
