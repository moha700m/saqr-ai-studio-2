"""Execution-provider discovery, verification and selection.

The selector never trusts ``get_available_providers()`` alone: a provider is
only *usable* after it created a session for a tiny Conv graph and produced the
numerically correct result. Priority (best first):

    CUDA  >  TensorRT (opt-in)  >  DirectML  >  ROCm / MIGraphX  >  OpenVINO  >  CoreML  >  CPU

Hardware facts from :mod:`saqr_engine.hardware` gate the candidates, e.g. CUDA
is skipped when no NVIDIA GPU is present or the driver is too old for the CUDA
12 runtime bundled with onnxruntime-gpu.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..hardware.report import HardwareReport

log = logging.getLogger(__name__)

PROBE_MODEL = Path(__file__).with_name("probe_conv.onnx")

# Canonical ids <-> ONNX Runtime provider names
PROVIDERS: dict[str, str] = {
    "cuda": "CUDAExecutionProvider",
    "tensorrt": "TensorrtExecutionProvider",
    "directml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "migraphx": "MIGraphXExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "cpu": "CPUExecutionProvider",
}
PRIORITY = ["cuda", "tensorrt", "directml", "rocm", "migraphx", "openvino", "coreml", "cpu"]
LABELS = {
    "cuda": "NVIDIA CUDA", "tensorrt": "NVIDIA TensorRT", "directml": "DirectML (DirectX 12 GPU)",
    "rocm": "AMD ROCm", "migraphx": "AMD MIGraphX", "openvino": "Intel OpenVINO", "coreml": "Apple CoreML",
    "cpu": "Optimised CPU",
}
# onnxruntime-gpu 1.2x wheels are built against CUDA 12.x which needs driver >= 525.60 (Linux) / 528.33 (Windows)
MIN_NVIDIA_DRIVER = (528, 33)


@dataclass
class ProviderCheck:
    id: str
    ort_name: str
    label: str
    installed: bool
    hardware_ok: bool
    verified: bool = False
    latency_ms: Optional[float] = None
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class BackendSelection:
    selected: str
    providers: list[Any]  # list passed to InferenceSession(providers=...)
    checks: list[ProviderCheck] = field(default_factory=list)
    forced: bool = False
    fallback_reason: str = ""
    intra_op_threads: int = 0

    @property
    def label(self) -> str:
        return LABELS.get(self.selected, self.selected)

    def to_dict(self) -> dict:
        return {"selected": self.selected, "label": self.label, "forced": self.forced,
                "fallback_reason": self.fallback_reason, "intra_op_threads": self.intra_op_threads,
                "checks": [c.to_dict() for c in self.checks]}


def _ort():
    import onnxruntime as ort  # imported lazily: installers ship different ORT flavours
    return ort


def installed_providers() -> list[str]:
    try:
        return list(_ort().get_available_providers())
    except Exception as exc:  # noqa: BLE001
        log.error("onnxruntime unavailable: %s", exc)
        return []


def _parse_driver(v: Optional[str]) -> Optional[tuple[int, int]]:
    if not v:
        return None
    parts = v.split(".")
    try:
        if len(parts) >= 4 and len(parts[-1]) >= 4:  # Windows style 31.0.15.2833 -> 528.33
            tail = (parts[-2] + parts[-1])[-5:]
            return int(tail[:3]), int(tail[3:])
        return int(parts[0]), int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        return None


def hardware_gate(pid: str, hw: Optional[HardwareReport]) -> tuple[bool, str]:
    """Is the hardware plausibly able to run this provider? (verification still follows)."""
    if hw is None or pid == "cpu":
        return True, ""
    vendors = {g.vendor for g in hw.gpus if not g.is_virtual}
    if pid in ("cuda", "tensorrt"):
        nv = hw.nvidia_gpus
        if not nv:
            return False, "no NVIDIA GPU detected"
        drv = _parse_driver(nv[0].driver_version)
        if drv and drv < MIN_NVIDIA_DRIVER:
            return False, f"NVIDIA driver {nv[0].driver_version} older than {MIN_NVIDIA_DRIVER[0]}.{MIN_NVIDIA_DRIVER[1]}"
        cc = nv[0].compute_capability
        if cc and float(cc) < 5.0:
            return False, f"compute capability {cc} < 5.0 not supported by CUDA 12"
        return True, ""
    if pid == "directml":
        if hw.system.os != "Windows":
            return False, "DirectML is Windows-only"
        if not (vendors & {"nvidia", "amd", "intel"}):
            return False, "no DirectX 12 capable GPU detected"
        return True, ""
    if pid in ("rocm", "migraphx"):
        return ("amd" in vendors, "" if "amd" in vendors else "no AMD GPU detected")
    if pid == "openvino":
        return (hw.cpu.vendor == "intel" or "intel" in vendors, "requires Intel CPU/GPU/NPU")
    if pid == "coreml":
        return (hw.system.os == "Darwin", "CoreML is macOS-only")
    return True, ""


def provider_options(pid: str) -> dict:
    if pid == "cuda":
        return {"cudnn_conv_algo_search": "HEURISTIC", "arena_extend_strategy": "kSameAsRequested"}
    if pid == "tensorrt":
        cache = Path(os.environ.get("SAQR_TRT_CACHE", Path.home() / ".saqr" / "trt_cache"))
        cache.mkdir(parents=True, exist_ok=True)
        return {"trt_fp16_enable": True, "trt_engine_cache_enable": True, "trt_engine_cache_path": str(cache)}
    if pid == "directml":
        return {"device_id": 0}
    if pid == "openvino":
        return {"device_type": "AUTO:GPU,NPU,CPU"}
    if pid == "coreml":
        return {"ModelFormat": "MLProgram", "MLComputeUnits": "ALL"}
    return {}


def provider_list(pid: str) -> list[Any]:
    """Provider list for InferenceSession, always ending with CPU for unsupported ops."""
    lst: list[Any] = []
    if pid == "tensorrt":
        lst.append((PROVIDERS["tensorrt"], provider_options("tensorrt")))
        lst.append((PROVIDERS["cuda"], provider_options("cuda")))
    elif pid != "cpu":
        lst.append((PROVIDERS[pid], provider_options(pid)))
    lst.append("CPUExecutionProvider")
    return lst


def session_options(pid: str, threads: int = 0):
    ort = _ort()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3
    if pid == "directml":
        so.enable_mem_pattern = False  # required by the DirectML EP
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    if threads:
        so.intra_op_num_threads = threads
    return so


def default_threads(pid: str, hw: Optional[HardwareReport]) -> int:
    if pid != "cpu":
        return 1 if pid in ("directml", "openvino") else 2
    phys = (hw.cpu.physical_cores if hw else 0) or (os.cpu_count() or 4)
    return max(1, min(phys, 16))


def verify_provider(pid: str, threads: int = 0) -> tuple[bool, Optional[float], str]:
    """Create a session on the probe graph and check the numeric result."""
    ort = _ort()
    try:
        sess = ort.InferenceSession(str(PROBE_MODEL), sess_options=session_options(pid, threads),
                                    providers=provider_list(pid))
    except Exception as exc:  # noqa: BLE001
        return False, None, f"session creation failed: {exc}"[:300]
    active = sess.get_providers()
    if pid != "cpu" and PROVIDERS["cuda" if pid == "tensorrt" else pid] not in active and PROVIDERS[pid] not in active:
        return False, None, f"provider silently fell back to {active}"
    x = np.arange(1 * 3 * 16 * 16, dtype=np.float32).reshape(1, 3, 16, 16) / 100.0
    try:
        t0 = time.perf_counter()
        (y,) = sess.run(None, {"x": x})
        dt = (time.perf_counter() - t0) * 1000
    except Exception as exc:  # noqa: BLE001
        return False, None, f"inference failed: {exc}"[:300]
    expected = _reference_conv(x)
    if y.shape != expected.shape or not np.allclose(y, expected, rtol=1e-2, atol=1e-2):
        return False, dt, "numeric check failed"
    return True, dt, "ok"


def _reference_conv(x: np.ndarray) -> np.ndarray:
    # probe graph: 3x3 conv, 3->4 channels, weights w[o,i,ky,kx] = (o+1)*0.1, bias 0.5, pad 1, then Relu
    pad = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    s = sum(pad[:, :, ky:ky + 16, kx:kx + 16] for ky in range(3) for kx in range(3)).sum(axis=1)
    out = np.stack([(o + 1) * 0.1 * s + 0.5 for o in range(4)], axis=1)
    return np.maximum(out, 0).astype(np.float32)


def check_all(hw: Optional[HardwareReport], include_tensorrt: bool = False) -> list[ProviderCheck]:
    avail = set(installed_providers())
    checks = []
    for pid in PRIORITY:
        if pid == "tensorrt" and not include_tensorrt:
            if PROVIDERS[pid] in avail:
                checks.append(ProviderCheck(pid, PROVIDERS[pid], LABELS[pid], True, True,
                                            reason="available; opt-in (engine build takes minutes on first run)"))
            continue
        installed = PROVIDERS[pid] in avail
        hw_ok, why = hardware_gate(pid, hw)
        c = ProviderCheck(pid, PROVIDERS[pid], LABELS[pid], installed, hw_ok)
        if not installed:
            c.reason = "not included in this build's onnxruntime package"
        elif not hw_ok:
            c.reason = why
        else:
            c.verified, c.latency_ms, c.reason = verify_provider(pid, default_threads(pid, hw))
        checks.append(c)
    return checks


def select_backend(hw: Optional[HardwareReport] = None, forced: Optional[str] = None,
                   include_tensorrt: bool = False, checks: Optional[list[ProviderCheck]] = None) -> BackendSelection:
    checks = checks if checks is not None else check_all(hw, include_tensorrt or forced == "tensorrt")
    by_id = {c.id: c for c in checks}
    fallback_reason = ""
    if forced:
        c = by_id.get(forced)
        if c and c.verified:
            return BackendSelection(forced, provider_list(forced), checks, True, "",
                                    default_threads(forced, hw))
        fallback_reason = f"requested backend '{forced}' unusable: {c.reason if c else 'unknown provider'}"
    for pid in PRIORITY:
        if pid == "tensorrt" and not (include_tensorrt or forced == "tensorrt"):
            continue
        c = by_id.get(pid)
        if c and c.verified:
            return BackendSelection(pid, provider_list(pid), checks, False, fallback_reason, default_threads(pid, hw))
    # CPU always works in a sane install; if even that failed, still return it so errors surface at session time
    return BackendSelection("cpu", provider_list("cpu"), checks, False,
                            fallback_reason or "no provider verified; using CPU", default_threads("cpu", hw))
