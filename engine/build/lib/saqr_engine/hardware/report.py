from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class CPUInfo:
    vendor: str = "unknown"
    model: str = "unknown"
    architecture: str = "unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    max_frequency_mhz: Optional[float] = None
    flags: list[str] = field(default_factory=list)

    @property
    def avx(self) -> bool:
        return "avx" in self.flags

    @property
    def avx2(self) -> bool:
        return "avx2" in self.flags

    @property
    def avx512(self) -> bool:
        return any(f.startswith("avx512") for f in self.flags)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(avx=self.avx, avx2=self.avx2, avx512=self.avx512)
        # flags list is long; keep only the interesting SIMD ones in the summary
        d["flags"] = sorted(f for f in self.flags if f.startswith(("avx", "sse", "fma", "f16c", "neon", "amx")))
        return d


@dataclass
class GPUInfo:
    vendor: str  # "nvidia" | "amd" | "intel" | "apple" | "microsoft" | "unknown"
    name: str
    vram_mb: Optional[int] = None
    driver_version: Optional[str] = None
    integrated: Optional[bool] = None
    compute_capability: Optional[str] = None  # NVIDIA only
    source: str = "unknown"  # which probe produced this entry
    apis: list[str] = field(default_factory=list)  # e.g. ["cuda", "directml", "vulkan"]

    @property
    def is_virtual(self) -> bool:
        n = self.name.lower()
        return any(k in n for k in ("basic render", "basic display", "remote display", "virtual", "llvmpipe", "parsec", "citrix"))


@dataclass
class SystemInfo:
    os: str = "unknown"
    os_version: str = "unknown"
    os_release: str = "unknown"
    is_64bit: bool = False
    machine: str = "unknown"
    ram_total_mb: int = 0
    ram_available_mb: int = 0
    disk_free_mb: int = 0
    python_version: str = ""


@dataclass
class HardwareReport:
    cpu: CPUInfo
    gpus: list[GPUInfo]
    system: SystemInfo
    errors: list[str] = field(default_factory=list)
    duration_ms: float = 0.0

    @property
    def nvidia_gpus(self) -> list[GPUInfo]:
        return [g for g in self.gpus if g.vendor == "nvidia"]

    @property
    def dedicated_gpus(self) -> list[GPUInfo]:
        return [g for g in self.gpus if g.integrated is False and not g.is_virtual]

    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        real = [g for g in self.gpus if not g.is_virtual]
        if not real:
            return None
        return max(real, key=lambda g: ((g.integrated is False), g.vram_mb or 0))

    def to_dict(self) -> dict:
        return {
            "cpu": self.cpu.to_dict(),
            "gpus": [asdict(g) | {"is_virtual": g.is_virtual} for g in self.gpus],
            "system": asdict(self.system),
            "errors": list(self.errors),
            "duration_ms": round(self.duration_ms, 1),
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "HardwareReport":
        cpu_d = dict(d.get("cpu", {}))
        for k in ("avx", "avx2", "avx512"):
            cpu_d.pop(k, None)
        gpus = []
        for g in d.get("gpus", []):
            g = dict(g)
            g.pop("is_virtual", None)
            gpus.append(GPUInfo(**g))
        return cls(cpu=CPUInfo(**cpu_d), gpus=gpus, system=SystemInfo(**d.get("system", {})),
                   errors=list(d.get("errors", [])), duration_ms=d.get("duration_ms", 0.0))
