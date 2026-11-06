"""ONNX model wrappers: SCRFD detector, ArcFace recognizer, 106-pt landmarks,
inswapper face swapper, GFPGAN enhancer and the NSFW classifier.

All pre/post-processing is implemented here directly on top of ONNX Runtime
(no dependency on the insightface Python package).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..runtime.providers import BackendSelection
from ..runtime.session import ResilientSession
from .align import ffhq_crop, norm_crop, transform_points
from .face import Face

log = logging.getLogger(__name__)


def _nms(dets: np.ndarray, thresh: float) -> list[int]:
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    """SCRFD-10GF detector (3 strides, 2 anchors, with 5 keypoints)."""

    STRIDES = (8, 16, 32)
    NUM_ANCHORS = 2

    def __init__(self, path: Path, selection: BackendSelection, det_size: int = 640,
                 score_thresh: float = 0.5, nms_thresh: float = 0.4):
        self.session = ResilientSession(path, selection)
        self.input_name = self.session.get_inputs()[0].name
        self.det_size = det_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self._anchor_cache: dict[tuple[int, int, int], np.ndarray] = {}

    def _anchors(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._anchor_cache:
            ys, xs = np.mgrid[:h, :w]
            centers = (np.stack([xs, ys], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
            centers = np.repeat(centers, self.NUM_ANCHORS, axis=0)
            self._anchor_cache[key] = centers
        return self._anchor_cache[key]

    def detect(self, img: np.ndarray, max_num: int = 0) -> list[Face]:
        size = self.det_size
        h0, w0 = img.shape[:2]
        scale = min(size / w0, size / h0)
        nw, nh = max(1, int(round(w0 * scale))), max(1, int(round(h0 * scale)))
        canvas = np.zeros((size, size, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128.0, (size, size), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(None, {self.input_name: blob})
        fmc = len(self.STRIDES)
        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.STRIDES):
            scores = outs[idx].reshape(-1)
            bbox = outs[idx + fmc].reshape(-1, 4) * stride
            kps = outs[idx + fmc * 2].reshape(-1, 10) * stride
            fh, fw = size // stride, size // stride
            centers = self._anchors(fh, fw, stride)
            pos = np.where(scores >= self.score_thresh)[0]
            if pos.size == 0:
                continue
            c = centers[pos]
            b = bbox[pos]
            boxes_l.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1], c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], -1))
            k = kps[pos].reshape(-1, 5, 2) + c[:, None, :]
            kps_l.append(k)
            scores_l.append(scores[pos])
        if not scores_l:
            return []
        scores = np.concatenate(scores_l)
        boxes = np.concatenate(boxes_l) / scale
        kpss = np.concatenate(kps_l) / scale
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)
        keep = _nms(dets, self.nms_thresh)
        faces = [Face(bbox=dets[i, :4], kps=kpss[i].astype(np.float32), score=float(dets[i, 4])) for i in keep]
        faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if max_num > 0:
            faces = faces[:max_num]
        return faces


class ArcFace:
    def __init__(self, path: Path, selection: BackendSelection):
        self.session = ResilientSession(path, selection)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        crop, _ = norm_crop(img, face.kps, 112)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        emb = self.session.run(None, {self.input_name: blob})[0][0]
        face.embedding = emb.astype(np.float32)
        return face.embedding


class Landmark106:
    SIZE = 192

    def __init__(self, path: Path, selection: BackendSelection):
        self.session = ResilientSession(path, selection)
        self.input_name = self.session.get_inputs()[0].name

    def predict(self, img: np.ndarray, face: Face) -> np.ndarray:
        x1, y1, x2, y2 = face.bbox
        w, h = x2 - x1, y2 - y1
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        s = self.SIZE / (max(w, h) * 1.5)
        M = np.array([[s, 0, self.SIZE / 2 - cx * s], [0, s, self.SIZE / 2 - cy * s]], np.float32)
        crop = cv2.warpAffine(img, M, (self.SIZE, self.SIZE), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (self.SIZE, self.SIZE), (0, 0, 0), swapRB=True)
        pred = self.session.run(None, {self.input_name: blob})[0].reshape(-1, 2)
        pred = (pred + 1.0) * (self.SIZE // 2)
        IM = cv2.invertAffineTransform(M)
        face.landmarks = transform_points(pred, IM)
        return face.landmarks


def load_emap(model_path: Path) -> np.ndarray:
    """Read inswapper's 512x512 embedding map (last 512x512 initializer); cached as .emap.npy."""
    cache = Path(model_path).with_suffix(".emap.npy")
    if cache.is_file():
        return np.load(cache)
    import onnx
    from onnx import numpy_helper

    model = onnx.load(str(model_path))
    emap = None
    for init in model.graph.initializer:
        if list(init.dims) == [512, 512]:
            emap = numpy_helper.to_array(init).astype(np.float32)
    if emap is None:
        raise ValueError(f"no 512x512 emap initializer in {model_path}")
    try:
        np.save(cache, emap)
    except OSError:
        pass
    return emap


class InSwapper:
    SIZE = 128

    def __init__(self, path: Path, selection: BackendSelection):
        self.session = ResilientSession(path, selection)
        names = [i.name for i in self.session.get_inputs()]
        self.target_name, self.source_name = names[0], names[1]
        self.emap = load_emap(path)

    def source_latent(self, source: Face) -> np.ndarray:
        latent = source.normed_embedding.reshape(1, -1) @ self.emap
        latent /= np.linalg.norm(latent)
        return latent.astype(np.float32)

    def swap(self, img: np.ndarray, target: Face, latent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Returns (swapped 128x128 BGR uint8 crop, 2x3 affine frame->crop)."""
        crop, M = norm_crop(img, target.kps, self.SIZE)
        blob = cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (self.SIZE, self.SIZE), (0, 0, 0), swapRB=True)
        out = self.session.run(None, {self.target_name: blob, self.source_name: latent})[0][0]
        fake = np.clip(out.transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)[:, :, ::-1]
        return np.ascontiguousarray(fake), M


class GFPGAN:
    SIZE = 512

    def __init__(self, path: Path, selection: BackendSelection):
        self.session = ResilientSession(path, selection)
        self.input_name = self.session.get_inputs()[0].name

    def enhance(self, img: np.ndarray, face: Face) -> tuple[np.ndarray, np.ndarray]:
        """Returns (restored 512x512 BGR crop, affine frame->crop)."""
        crop, M = ffhq_crop(img, face.kps, self.SIZE)
        x = crop[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None]
        out = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]
        out = np.clip((out.transpose(1, 2, 0) + 1.0) * 127.5, 0, 255).astype(np.uint8)[:, :, ::-1]
        return np.ascontiguousarray(out), M


class NSFWClassifier:
    """ViT classifier (labels: normal, nsfw)."""

    def __init__(self, path: Path, selection: Optional[BackendSelection] = None):
        from ..runtime.providers import BackendSelection as BS, provider_list

        sel = BS("cpu", provider_list("cpu"), [], intra_op_threads=2)  # tiny & rare: keep GPU free
        self.session = ResilientSession(path, sel)
        self.input_name = self.session.get_inputs()[0].name

    def probability(self, img: np.ndarray) -> float:
        x = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32) / 255.0
        x = ((x - 0.5) / 0.5).transpose(2, 0, 1)[None]
        logits = self.session.run(None, {self.input_name: np.ascontiguousarray(x)})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())
