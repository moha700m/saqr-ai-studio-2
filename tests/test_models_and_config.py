import hashlib
import http.server
import threading
from functools import partial

import pytest

from saqr_engine import config
from saqr_engine.models.manager import (ChecksumMismatch, DownloadCancelled, LicenseNotAccepted, ModelManager,
                                        ModelSpec, load_registry)
from saqr_engine.pipeline.config import PRESETS, ProcessConfig, preset


def test_registry_is_complete():
    reg = load_registry()
    ids = {m.id for m in reg}
    assert {"scrfd_10g", "arcface_w600k_r50", "landmark_2d106", "inswapper_128", "nsfw_vit_int8"} <= ids
    for m in reg:
        assert len(m.sha256) == 64 and m.url.startswith("https://") and m.license


@pytest.fixture()
def server(tmp_path):
    payload = b"saqr-model-bytes" * 5000
    (tmp_path / "srv").mkdir()
    (tmp_path / "srv" / "m.onnx").write_bytes(payload)
    handler = partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_path / "srv"))
    handler.log_message = lambda *a: None
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_port}/m.onnx", payload
    httpd.shutdown()


def spec(url, payload, sha=None):
    return ModelSpec(id="m", file="m.onnx", role="detector", title="t", url=url, size=len(payload),
                     sha256=sha or hashlib.sha256(payload).hexdigest(), license="test", license_url="",
                     commercial=True, required=True)


def test_download_requires_licence_then_verifies(tmp_path, server):
    url, payload = server
    store = config.SettingsStore(tmp_path / "s.json")
    mm = ModelManager(tmp_path / "models", store, [spec(url, payload)])
    with pytest.raises(LicenseNotAccepted):
        mm.download("m")
    mm.accept_license(["m"])
    seen = []
    p = mm.download("m", progress=lambda i, d, t: seen.append(d))
    assert p.read_bytes() == payload and seen[-1] == len(payload)
    assert mm.verify("m") and mm.status("m").installed
    assert mm.remove("m") and not mm.status("m").installed


def test_checksum_mismatch_rejected(tmp_path, server):
    url, payload = server
    store = config.SettingsStore(tmp_path / "s.json")
    mm = ModelManager(tmp_path / "models", store, [spec(url, payload, sha="0" * 64)])
    mm.accept_license(["m"])
    with pytest.raises(ChecksumMismatch):
        mm.download("m")
    assert not (tmp_path / "models" / "m.onnx").exists()


def test_download_cancel(tmp_path, server):
    url, payload = server
    store = config.SettingsStore(tmp_path / "s.json")
    mm = ModelManager(tmp_path / "models", store, [spec(url, payload)])
    mm.accept_license(["m"])
    ev = threading.Event()
    ev.set()
    with pytest.raises(DownloadCancelled):
        mm.download("m", cancel=ev)


def test_installer_licence_marker(tmp_path):
    mm = ModelManager(tmp_path / "models", None, load_registry())
    assert not mm.license_accepted("scrfd_10g")
    (config.data_dir() / "accepted_licenses.txt").write_text("scrfd_10g\r\ninswapper_128\r\n")
    assert mm.license_accepted("scrfd_10g") and not mm.license_accepted("gfpgan_1.4")


def test_swapper_choice_per_backend():
    mm = ModelManager(registry=load_registry())
    assert mm.recommended_swapper("cuda") == "inswapper_128_fp16"
    assert mm.recommended_swapper("directml") == "inswapper_128"
    assert mm.compatible("inswapper_128_fp16", "directml")[0] is False


def test_privacy_defaults(tmp_path):
    s = config.SettingsStore(tmp_path / "s.json")
    st = s.settings
    assert st.processing_mode == "local" and not st.cloud_enabled
    assert not st.auto_upload and not st.auto_record and not st.auto_start_camera
    s.update(processing_mode="cloud")  # cannot switch to cloud while cloud is disabled
    assert s.settings.processing_mode == "local"
    s.update(auto_upload=True)
    assert s.settings.auto_upload is False


def test_config_clamps_and_presets():
    c = ProcessConfig().update(opacity=3, det_size=1000, detect_interval=0, color_correction="bogus")
    assert c.opacity == 1.0 and c.det_size == 992 and c.detect_interval == 1 and c.color_correction == "lab"
    with pytest.raises(KeyError):
        c.update(nope=1)
    for name in PRESETS:
        preset(name)
