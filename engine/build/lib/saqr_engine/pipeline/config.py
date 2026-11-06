"""Processing configuration and quality presets."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any


@dataclass
class ProcessConfig:
    # what to do
    swap: bool = True
    enhance: bool = False
    enhance_blend: float = 0.8  # 0..1 mix of restored face over swapped face
    face_selection: str = "largest"  # largest | all | mapped
    max_faces: int = 4
    map_threshold: float = 0.35  # cosine similarity for identity mapping
    # detection / tracking
    det_size: int = 640
    det_score: float = 0.5
    detect_interval: int = 1  # run the detector every N frames, track in between
    landmarks: bool = True  # 106-pt landmarks: needed for region masks
    # masking
    mask_feather: float = 0.08  # fraction of crop size
    mask_erode: float = 0.02
    forehead: float = 0.5  # 0 = keep original forehead, 1 = swap up to the hairline estimate
    mouth_mask: float = 0.0  # 0..1 strength of keeping the original mouth
    mouth_scale: float = 1.15
    eye_mask: float = 0.0  # 0..1 strength of keeping the original eyes
    # blending / colour
    opacity: float = 1.0
    color_correction: str = "lab"  # none | lab | histogram
    color_strength: float = 0.7
    poisson: bool = False
    sharpness: float = 0.0  # 0..1 unsharp amount on the face region
    temporal_smoothing: float = 0.0  # 0..0.9 keypoint smoothing for live video
    # output
    processing_scale: float = 1.0  # downscale input before processing (live)
    target_fps: int = 30

    def update(self, **kw: Any) -> "ProcessConfig":
        names = {f.name for f in fields(self)}
        for k, v in kw.items():
            if k not in names:
                raise KeyError(f"unknown config field {k}")
            cur = getattr(self, k)
            if isinstance(cur, bool):
                v = bool(v)
            elif isinstance(cur, int) and not isinstance(cur, bool):
                v = int(v)
            elif isinstance(cur, float):
                v = float(v)
            setattr(self, k, v)
        self._clamp()
        return self

    def _clamp(self) -> None:
        c = lambda v, lo, hi: max(lo, min(hi, v))  # noqa: E731
        self.enhance_blend = c(self.enhance_blend, 0.0, 1.0)
        self.det_size = int(c(self.det_size // 32 * 32, 160, 1280))
        self.detect_interval = int(c(self.detect_interval, 1, 30))
        self.max_faces = int(c(self.max_faces, 1, 20))
        for k in ("forehead", "mouth_mask", "eye_mask", "opacity", "color_strength", "sharpness"):
            setattr(self, k, c(getattr(self, k), 0.0, 1.0))
        self.mask_feather = c(self.mask_feather, 0.0, 0.4)
        self.mask_erode = c(self.mask_erode, 0.0, 0.2)
        self.temporal_smoothing = c(self.temporal_smoothing, 0.0, 0.9)
        self.processing_scale = c(self.processing_scale, 0.25, 1.0)
        self.target_fps = int(c(self.target_fps, 1, 120))
        if self.color_correction not in ("none", "lab", "histogram"):
            self.color_correction = "lab"
        if self.face_selection not in ("largest", "all", "mapped"):
            self.face_selection = "largest"

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, dict[str, Any]] = {
    # Fast: low det resolution, tracking between detections, no landmarks/enhancer
    "fast": dict(det_size=320, detect_interval=3, landmarks=False, enhance=False, color_correction="lab",
                 processing_scale=0.75, poisson=False, sharpness=0.0),
    "balanced": dict(det_size=480, detect_interval=2, landmarks=True, enhance=False, color_correction="lab",
                     processing_scale=1.0, poisson=False, sharpness=0.15),
    "quality": dict(det_size=640, detect_interval=1, landmarks=True, enhance=True, enhance_blend=0.8,
                    color_correction="lab", processing_scale=1.0, poisson=False, sharpness=0.1),
}


def preset(name: str, base: ProcessConfig | None = None) -> ProcessConfig:
    cfg = base or ProcessConfig()
    return cfg.update(**PRESETS[name])
