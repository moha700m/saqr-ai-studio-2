import numpy as np
import pytest

from saqr_engine.hardware.report import CPUInfo, GPUInfo, HardwareReport, SystemInfo
from saqr_engine.runtime import providers as P
from saqr_engine.runtime.session import STATE, ResilientSession


def hw(gpus=(), os_name="Windows", cpu_vendor="amd"):
    return HardwareReport(cpu=CPUInfo(vendor=cpu_vendor, physical_cores=8, logical_cores=16),
                          gpus=list(gpus), system=SystemInfo(os=os_name))


NV = GPUInfo(vendor="nvidia", name="RTX 3060", vram_mb=12288, driver_version="31.0.15.5222",
             integrated=False, compute_capability="8.6")


def test_driver_parsing():
    assert P._parse_driver("31.0.15.5222") == (552, 22)
    assert P._parse_driver("560.94") == (560, 94)
    assert P._parse_driver(None) is None


def test_hardware_gate():
    assert P.hardware_gate("cuda", hw([NV]))[0]
    assert not P.hardware_gate("cuda", hw([]))[0]
    old = GPUInfo(vendor="nvidia", name="GTX 760", driver_version="27.21.14.5671", compute_capability="3.0")
    ok, why = P.hardware_gate("cuda", hw([old]))
    assert not ok and "older" in why
    assert P.hardware_gate("directml", hw([NV]))[0]
    assert not P.hardware_gate("directml", hw([NV], os_name="Linux"))[0]
    assert not P.hardware_gate("coreml", hw())[0]


def check(pid, verified):
    return P.ProviderCheck(pid, P.PROVIDERS[pid], P.LABELS[pid], True, True, verified=verified)


def test_priority_and_forced_fallback():
    checks = [check("cuda", True), check("directml", True), check("cpu", True)]
    assert P.select_backend(hw([NV]), checks=checks).selected == "cuda"
    checks = [check("cuda", False), check("directml", True), check("cpu", True)]
    s = P.select_backend(hw([NV]), checks=checks)
    assert s.selected == "directml"
    s = P.select_backend(hw([NV]), forced="cuda", checks=checks)
    assert s.selected == "directml" and "cuda" in s.fallback_reason
    s = P.select_backend(hw([NV]), forced="cpu", checks=checks)
    assert s.selected == "cpu" and s.forced


def test_tensorrt_is_opt_in():
    checks = [check("tensorrt", True), check("cuda", True), check("cpu", True)]
    assert P.select_backend(hw([NV]), checks=checks).selected == "cuda"
    assert P.select_backend(hw([NV]), checks=checks, include_tensorrt=True).selected == "tensorrt"


def test_cpu_provider_really_verifies():
    ok, ms, why = P.verify_provider("cpu", 1)
    assert ok, why
    assert np.allclose(P._reference_conv(np.zeros((1, 3, 16, 16), np.float32)), 0.5)


def test_resilient_session_runtime_fallback(monkeypatch):
    sel = P.BackendSelection("cpu", P.provider_list("cpu"), [check("cpu", True)])
    s = ResilientSession(P.PROBE_MODEL, sel)
    x = np.ones((1, 3, 16, 16), np.float32)
    assert s.run(None, {"x": x})[0].shape == (1, 4, 16, 16)
    # pretend we were on a GPU that dies mid-run: the session must rebuild on CPU and still answer
    s._chain = ["directml", "cpu"]
    s._idx = 0
    s.backend = "directml"

    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("device removed")

    s.sess = Broken()
    n = len(STATE.events)
    out = s.run(None, {"x": x})
    assert out[0].shape == (1, 4, 16, 16)
    assert s.backend == "cpu"
    assert len(STATE.events) == n + 1 and "device removed" in STATE.events[-1].reason


def test_session_fails_cleanly_when_nothing_works():
    sel = P.BackendSelection("cpu", P.provider_list("cpu"), [])
    with pytest.raises(RuntimeError):
        ResilientSession("does-not-exist.onnx", sel)
