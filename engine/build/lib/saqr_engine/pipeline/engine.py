"""FaceEngine: owns the models, the consented source-face library, identity
mapping and the per-frame processing graph.

    detect (every N frames) ─┬─> landmarks ─> region masks
            track (between) ─┘
    source latent (cached) ─> inswapper ─> colour match ─> paste-back
                                              └─> optional GFPGAN pass ─> sharpen
"""
from __future__ import annotations

import hashlib
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..models.manager import ModelManager
from ..runtime.providers import BackendSelection
from ..safety.consent import ConsentRequired, ConsentRecord
from ..vision import color, masks
from ..vision.blend import paste_back
from ..vision.face import Face, cosine
from ..vision.models import GFPGAN, SCRFD, ArcFace, InSwapper, Landmark106, NSFWClassifier
from .config import ProcessConfig
from .metrics import StageTimer
from .tracker import FaceTracker

log = logging.getLogger(__name__)


class NoFaceFound(ValueError):
    pass


class ContentBlocked(PermissionError):
    pass


@dataclass
class SourceFace:
    id: str
    name: str
    embedding: np.ndarray  # raw ArcFace embedding (memory only; never written to disk)
    thumbnail: np.ndarray  # small BGR crop for the UI
    consent: ConsentRecord
    latent: Optional[np.ndarray] = None
    image_sha256: str = ""

    def public(self) -> dict:
        return {"id": self.id, "name": self.name, "consent": self.consent.to_dict(), "image_sha256": self.image_sha256}


@dataclass
class MappingRule:
    """Replace faces resembling ``reference`` with ``source_id``."""
    id: str
    reference: np.ndarray
    source_id: str
    label: str = ""


@dataclass
class FrameResult:
    frame: np.ndarray
    faces: list[Face]
    timings: dict[str, float]
    swapped: int = 0
    blocked: bool = False


NSFW_THRESHOLD = 0.80


class FaceEngine:
    def __init__(self, manager: ModelManager, selection: BackendSelection, cfg: Optional[ProcessConfig] = None):
        self.manager = manager
        self.selection = selection
        self.cfg = cfg or ProcessConfig()
        self._lock = threading.RLock()
        self._detector: Optional[SCRFD] = None
        self._recognizer: Optional[ArcFace] = None
        self._landmarker: Optional[Landmark106] = None
        self._swapper: Optional[InSwapper] = None
        self._enhancer: Optional[GFPGAN] = None
        self._nsfw: Optional[NSFWClassifier] = None
        self.swapper_id: Optional[str] = None
        self.sources: dict[str, SourceFace] = {}
        self.active_source: Optional[str] = None
        self.rules: list[MappingRule] = []
        self.tracker = FaceTracker()
        self._frame_idx = 0
        self._identity_cache: dict[int, tuple[int, Optional[str]]] = {}  # track_id -> (frame_idx, source_id)

    # ------------------------------------------------------------------ model loading
    def _path(self, model_id: str):
        p = self.manager.path_for(model_id)
        if not p:
            raise FileNotFoundError(f"model '{model_id}' is not installed — open the Model Manager to download it")
        return p

    @property
    def detector(self) -> SCRFD:
        if self._detector is None:
            self._detector = SCRFD(self._path("scrfd_10g"), self.selection, self.cfg.det_size, self.cfg.det_score)
        self._detector.det_size = self.cfg.det_size
        self._detector.score_thresh = self.cfg.det_score
        return self._detector

    @property
    def recognizer(self) -> ArcFace:
        if self._recognizer is None:
            self._recognizer = ArcFace(self._path("arcface_w600k_r50"), self.selection)
        return self._recognizer

    @property
    def landmarker(self) -> Landmark106:
        if self._landmarker is None:
            self._landmarker = Landmark106(self._path("landmark_2d106"), self.selection)
        return self._landmarker

    @property
    def swapper(self) -> InSwapper:
        if self._swapper is None:
            sid = self.manager.pick_installed_swapper(self.selection.selected)
            if not sid:
                raise FileNotFoundError("no face-swap model installed — download inswapper_128 in the Model Manager")
            self._swapper = InSwapper(self._path(sid), self.selection)
            self.swapper_id = sid
            for s in self.sources.values():
                s.latent = None
        return self._swapper

    @property
    def enhancer(self) -> GFPGAN:
        if self._enhancer is None:
            self._enhancer = GFPGAN(self._path("gfpgan_1.4"), self.selection)
        return self._enhancer

    @property
    def nsfw(self) -> NSFWClassifier:
        if self._nsfw is None:
            self._nsfw = NSFWClassifier(self._path("nsfw_vit_int8"))
        return self._nsfw

    def warmup(self) -> dict[str, float]:
        """Load models and run one inference each so the first live frame isn't slow."""
        t = {}
        dummy = np.full((480, 640, 3), 127, np.uint8)
        for name, fn in (("detector", lambda: self.detector.detect(dummy)),
                         ("recognizer", lambda: self.recognizer),
                         ("landmarks", lambda: self.landmarker),
                         ("swapper", lambda: self.swapper),
                         ("nsfw", lambda: self.nsfw.probability(dummy))):
            t0 = time.perf_counter()
            fn()
            t[name] = round((time.perf_counter() - t0) * 1000, 1)
        if self.cfg.enhance:
            t0 = time.perf_counter()
            self.enhancer  # noqa: B018 - property load
            t["enhancer"] = round((time.perf_counter() - t0) * 1000, 1)
        return t

    def backends(self) -> dict[str, str]:
        out = {}
        for name, m in (("detector", self._detector), ("recognizer", self._recognizer),
                        ("landmarks", self._landmarker), ("swapper", self._swapper), ("enhancer", self._enhancer),
                        ("nsfw", self._nsfw)):
            if m is not None:
                out[name] = m.session.backend
        return out

    # ------------------------------------------------------------------ safety
    def check_nsfw(self, img: np.ndarray) -> float:
        return self.nsfw.probability(img)

    # ------------------------------------------------------------------ sources
    def add_source(self, img: np.ndarray, consent: Optional[ConsentRecord], name: str = "") -> SourceFace:
        """Register a source identity. Requires an explicit consent attestation."""
        if consent is None or not consent.valid:
            raise ConsentRequired("confirm that you have the consent of the person in the source image")
        if self.check_nsfw(img) >= NSFW_THRESHOLD:
            raise ContentBlocked("the source image was flagged by the NSFW safety filter")
        faces = self.detector.detect(img, max_num=1)
        if not faces:
            raise NoFaceFound("no face detected in the source image")
        f = faces[0]
        self.recognizer.embed(img, f)
        x1, y1, x2, y2 = f.bbox.astype(int)
        pad = int(0.25 * max(x2 - x1, y2 - y1))
        thumb = img[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad]
        thumb = cv2.resize(thumb, (160, int(160 * thumb.shape[0] / max(1, thumb.shape[1]))))
        sf = SourceFace(id=uuid.uuid4().hex[:12], name=name or f"Source {len(self.sources) + 1}",
                        embedding=f.embedding.copy(), thumbnail=thumb, consent=consent,
                        image_sha256=hashlib.sha256(img.tobytes()).hexdigest()[:16])
        with self._lock:
            self.sources[sf.id] = sf
            if self.active_source is None:
                self.active_source = sf.id
        return sf

    def remove_source(self, sid: str) -> None:
        with self._lock:
            self.sources.pop(sid, None)
            self.rules = [r for r in self.rules if r.source_id != sid]
            if self.active_source == sid:
                self.active_source = next(iter(self.sources), None)
            self._identity_cache.clear()

    def clear_sources(self) -> None:
        with self._lock:
            self.sources.clear()
            self.rules.clear()
            self.active_source = None
            self._identity_cache.clear()

    def set_active_source(self, sid: str) -> None:
        if sid not in self.sources:
            raise KeyError(sid)
        self.active_source = sid

    def _latent(self, sid: str) -> np.ndarray:
        s = self.sources[sid]
        if s.latent is None:
            s.latent = self.swapper.source_latent(Face(bbox=np.zeros(4), kps=np.zeros((5, 2)), score=1.0,
                                                       embedding=s.embedding))
        return s.latent

    # ------------------------------------------------------------------ mapping
    def find_targets(self, img: np.ndarray) -> list[Face]:
        """Detect and embed every face in an image/frame (used to build mapping rules)."""
        faces = self.detector.detect(img, max_num=self.cfg.max_faces)
        for f in faces:
            self.recognizer.embed(img, f)
        return faces

    def add_rule(self, reference: Face, source_id: str, label: str = "") -> MappingRule:
        if source_id not in self.sources:
            raise KeyError(source_id)
        if reference.embedding is None:
            raise ValueError("reference face has no embedding")
        r = MappingRule(uuid.uuid4().hex[:10], reference.embedding.copy(), source_id, label)
        with self._lock:
            self.rules.append(r)
            self._identity_cache.clear()
        return r

    def clear_rules(self) -> None:
        with self._lock:
            self.rules.clear()
            self._identity_cache.clear()

    def _source_for(self, img: np.ndarray, face: Face) -> Optional[str]:
        if self.cfg.face_selection != "mapped":
            return self.active_source
        cached = self._identity_cache.get(face.track_id)
        if cached and face.track_id >= 0 and self._frame_idx - cached[0] < 15:
            return cached[1]
        if face.embedding is None:
            self.recognizer.embed(img, face)
        best, best_sim = None, self.cfg.map_threshold
        for r in self.rules:
            sim = cosine(face.embedding, r.reference)
            if sim > best_sim:
                best, best_sim = r.source_id, sim
        if face.track_id >= 0:
            self._identity_cache[face.track_id] = (self._frame_idx, best)
        return best

    # ------------------------------------------------------------------ processing
    def detect(self, frame: np.ndarray, force: bool = False) -> list[Face]:
        cfg = self.cfg
        max_num = 1 if cfg.face_selection == "largest" else cfg.max_faces
        do_detect = force or self.tracker.needs_detection(self._frame_idx, cfg.detect_interval)
        if do_detect:
            faces = self.detector.detect(frame, max_num=max_num)
            faces = self.tracker.update_detections(frame, faces, cfg.temporal_smoothing)
        else:
            faces = self.tracker.track(frame)
            if faces is None:  # tracking lost -> detect now
                faces = self.tracker.update_detections(frame, self.detector.detect(frame, max_num=max_num),
                                                       cfg.temporal_smoothing)
        return faces

    def process(self, frame: np.ndarray, force_detect: bool = False) -> FrameResult:
        """Process one BGR frame. Returns a new frame (input is not modified)."""
        cfg = self.cfg
        timer = StageTimer()
        out = frame.copy()
        with self._lock:
            self._frame_idx += 1
            with timer("detect"):
                faces = self.detect(frame, force_detect)
            if not cfg.swap or not self.sources or not faces:
                self.tracker.remember(frame)
                return FrameResult(out, faces, timer.result())
            swapped = 0
            for face in faces:
                sid = self._source_for(frame, face)
                if sid is None or sid not in self.sources:
                    continue
                if cfg.landmarks and face.landmarks is None:
                    with timer("landmarks"):
                        self.landmarker.predict(frame, face)
                with timer("swap"):
                    fake, M = self.swapper.swap(frame, face, self._latent(sid))
                with timer("blend"):
                    self._blend(out, frame, face, fake, M)
                if cfg.enhance:
                    with timer("enhance"):
                        self._enhance(out, face)
                swapped += 1
            self.tracker.remember(frame)
            return FrameResult(out, faces, timer.result(), swapped)

    def _crop_masks(self, face: Face, M: np.ndarray, size: int) -> np.ndarray:
        cfg = self.cfg
        if face.landmarks is not None:
            lm = face.landmarks @ M[:, :2].T + M[:, 2]
            mask = masks.face_region_mask(size, lm, cfg.forehead, cfg.mask_feather, cfg.mask_erode)
            if cfg.mouth_mask > 0 or cfg.eye_mask > 0:
                keep = masks.region_keep_mask(size, lm, cfg.mouth_mask, cfg.eye_mask, cfg.mouth_scale)
                mask = mask * (1.0 - keep)
        else:
            mask = masks.ellipse_mask(size, max(cfg.mask_feather, 0.1))
        return mask

    def _blend(self, out: np.ndarray, orig: np.ndarray, face: Face, fake: np.ndarray, M: np.ndarray) -> None:
        cfg = self.cfg
        size = fake.shape[0]
        mask = self._crop_masks(face, M, size)
        # upscale crop + mask 2x before paste-back for smoother edges on large faces
        if face.size > size * 1.5:
            fake = cv2.resize(fake, (size * 2, size * 2), interpolation=cv2.INTER_CUBIC)
            mask = cv2.resize(mask, (size * 2, size * 2), interpolation=cv2.INTER_LINEAR)
            S = np.array([[2, 0, 0], [0, 2, 0]], np.float32)
            M = np.vstack([M, [0, 0, 1]])
            M = (np.vstack([S, [0, 0, 1]]) @ M)[:2].astype(np.float32)
            size *= 2
        if cfg.color_correction != "none":
            ref = cv2.warpAffine(orig, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fn = color.lab_transfer if cfg.color_correction == "lab" else color.histogram_match
            fake = fn(fake, ref, mask, cfg.color_strength)
        if cfg.sharpness > 0:
            fake = color.unsharp(fake, cfg.sharpness * 1.5)
        paste_back(out, fake, mask, M, cfg.opacity, cfg.poisson)

    def _enhance(self, out: np.ndarray, face: Face) -> None:
        restored, M = self.enhancer.enhance(out, face)
        size = restored.shape[0]
        mask = masks.ellipse_mask(size, 0.1)
        if face.landmarks is not None:
            lm = face.landmarks @ M[:, :2].T + M[:, 2]
            mask = masks.face_region_mask(size, lm, self.cfg.forehead, 0.08, 0.02)
            if self.cfg.mouth_mask > 0 or self.cfg.eye_mask > 0:
                mask *= 1.0 - masks.region_keep_mask(size, lm, self.cfg.mouth_mask, self.cfg.eye_mask,
                                                     self.cfg.mouth_scale)
        paste_back(out, restored, mask, M, self.cfg.enhance_blend)

    def reset_tracking(self) -> None:
        with self._lock:
            self.tracker.reset()
            self._identity_cache.clear()
            self._frame_idx = 0
