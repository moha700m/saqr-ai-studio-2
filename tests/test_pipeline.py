import time

import cv2
import numpy as np
import pytest

from conftest import FIXTURES, models_dir
from saqr_engine.pipeline.live import LiveSession
from saqr_engine.pipeline.metrics import RollingMetrics
from saqr_engine.safety.consent import ConsentRecord, ConsentRequired, attest
from saqr_engine.safety.watermark import draw_badge, write_image
from saqr_engine.vision import masks
from saqr_engine.vision.align import ARCFACE_112, estimate_norm, transform_points, umeyama


def test_umeyama_recovers_similarity():
    a = np.deg2rad(20)
    M = np.array([[1.3 * np.cos(a), -1.3 * np.sin(a), 15], [1.3 * np.sin(a), 1.3 * np.cos(a), -7]], np.float32)
    src = ARCFACE_112 + 10
    dst = transform_points(src, M)
    assert np.allclose(umeyama(src, dst), M, atol=1e-3)
    assert np.allclose(transform_points(ARCFACE_112, estimate_norm(ARCFACE_112)), ARCFACE_112, atol=1e-3)


def test_region_masks():
    lm = np.zeros((106, 2), np.float32)
    rng = np.random.default_rng(0)
    lm[:] = rng.uniform(30, 98, (106, 2))
    lm[masks.MOUTH] = rng.uniform(55, 75, (20, 2)) + [0, 25]
    full = masks.face_region_mask(128, lm, forehead=1.0)
    none = masks.face_region_mask(128, lm, forehead=0.0)
    assert full.sum() >= none.sum() > 0
    keep = masks.region_keep_mask(128, lm, mouth=1.0)
    assert keep.max() > 0.9 and keep[:40].max() == 0


def test_consent_required():
    assert not ConsentRecord(attested=False, basis="self").valid
    assert not ConsentRecord(attested=True, basis="guess").valid
    assert attest("self").valid


def test_badge_and_metadata(tmp_path):
    img = np.zeros((480, 640, 3), np.uint8)
    out = draw_badge(img.copy())
    assert out[-40:, -160:].sum() > 0 and out[:100].sum() == 0
    p = write_image(tmp_path / "a.png", out)
    from PIL import Image

    assert "SAQR" in Image.open(p).info.get("Description", "")


class FakeEngine:
    def __init__(self, delay=0.05):
        self.delay = delay
        self.sources = {"s": object()}
        self.calls = 0

        class C:
            processing_scale = 1.0
        self.cfg = C()

    def reset_tracking(self):
        pass

    def process(self, frame, force_detect=False):
        self.calls += 1
        time.sleep(self.delay)

        class R:
            pass
        r = R()
        r.frame, r.swapped, r.faces, r.timings = frame + 1, 1, [1], {"swap": self.delay * 1000}
        return r

    def check_nsfw(self, frame):
        return 0.0


def test_live_session_drops_stale_frames_and_stops_cleanly():
    frames = iter(range(10**9))

    def src():
        next(frames)
        return np.zeros((120, 160, 3), np.uint8)

    eng = FakeEngine(delay=0.1)  # 10 fps processing vs 60 fps source
    ls = LiveSession(eng, src, fps=60, process=True)
    ls.start()
    time.sleep(2.0)
    m = ls.metrics.snapshot()
    t0 = time.time()
    ls.stop()
    assert time.time() - t0 < 2.0 and not ls.running
    assert 5 <= m["fps"] <= 12, m
    assert m["dropped"] > 30  # stale frames dropped, not queued
    assert m["latency_ms"] < 400  # no latency build-up
    out = ls.latest()
    assert out.processed and out.frame[-30:, -100:].sum() > 0  # badge present on processed output


def test_live_preview_only_does_not_process():
    eng = FakeEngine()
    ls = LiveSession(eng, lambda: np.zeros((120, 160, 3), np.uint8), fps=30, process=False)
    ls.start()
    time.sleep(0.5)
    ls.set_processing(True)
    time.sleep(0.5)
    ls.stop()
    assert eng.calls > 0


def test_metrics_fps():
    m = RollingMetrics()
    for _ in range(31):
        m.on_frame(10.0, {"a": 1.0})
        time.sleep(1 / 100)
    assert 60 < m.snapshot()["fps"] < 110


@pytest.mark.models
def test_real_swap_changes_identity():
    md = models_dir()
    if md is None:
        pytest.skip("set SAQR_MODELS to run model tests")
    import os

    os.environ["SAQR_MODELS"] = str(md)
    from saqr_engine.models.manager import ModelManager
    from saqr_engine.pipeline.config import preset
    from saqr_engine.pipeline.engine import FaceEngine
    from saqr_engine.runtime.providers import select_backend
    from saqr_engine.vision.face import cosine

    eng = FaceEngine(ModelManager(md), select_backend(None), preset("balanced"))
    with pytest.raises(ConsentRequired):
        eng.add_source(cv2.imread(str(FIXTURES / "painting_youth_met.jpg")), None)
    src = eng.add_source(cv2.imread(str(FIXTURES / "painting_youth_met.jpg")), attest("permission"))
    tgt = cv2.imread(str(FIXTURES / "painting_lady_1760.jpg"))
    before = eng.find_targets(tgt)[0]
    res = eng.process(tgt, force_detect=True)
    after = eng.find_targets(res.frame)[0]
    assert res.swapped == 1
    assert cosine(before.embedding, src.embedding) < 0.3
    assert cosine(after.embedding, src.embedding) > 0.6
    assert eng.check_nsfw(res.frame) < 0.5
