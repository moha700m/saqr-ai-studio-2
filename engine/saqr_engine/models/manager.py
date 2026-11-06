"""Model Manager: list, download (resumable, SHA-256 verified), remove, update,
inspect licences, show requirements and test compatibility.

Weights are never bundled with SAQR. They are fetched from their public source
only after the user accepted the model's licence.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional

from .. import config

REGISTRY_PATH = Path(__file__).with_name("registry.json")
CHUNK = 1 << 20

ProgressCb = Callable[[str, int, int], None]  # (model_id, done_bytes, total_bytes)


class LicenseNotAccepted(PermissionError):
    pass


class ChecksumMismatch(ValueError):
    pass


class DownloadCancelled(RuntimeError):
    pass


@dataclass
class ModelSpec:
    id: str
    file: str
    role: str
    title: str
    url: str
    size: int
    sha256: str
    license: str
    license_url: str
    commercial: bool
    required: bool
    min_ram_mb: int = 0
    min_vram_mb: int = 0
    providers: list[str] = field(default_factory=list)


@dataclass
class ModelStatus:
    spec: ModelSpec
    installed: bool
    path: Optional[str]
    size_on_disk: int = 0
    verified: Optional[bool] = None  # None = not checked in this session

    def to_dict(self) -> dict:
        d = asdict(self.spec)
        d.update(installed=self.installed, path=self.path, size_on_disk=self.size_on_disk, verified=self.verified)
        return d


def load_registry(path: Path = REGISTRY_PATH) -> list[ModelSpec]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [ModelSpec(**m) for m in data["models"]]


def sha256_file(path: Path, cancel: Optional[threading.Event] = None) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK * 4):
            if cancel and cancel.is_set():
                raise DownloadCancelled(str(path))
            h.update(chunk)
    return h.hexdigest()


class ModelManager:
    def __init__(self, root: Optional[Path] = None, settings: Optional[config.SettingsStore] = None,
                 registry: Optional[list[ModelSpec]] = None):
        self.root = Path(root) if root else config.models_dir()
        self.root.mkdir(parents=True, exist_ok=True)
        self.settings = settings
        self.specs = {m.id: m for m in (registry or load_registry())}
        self._verified: dict[str, bool] = {}
        self._extra_dirs = [Path(p) for p in os.environ.get("SAQR_MODEL_PATHS", "").split(os.pathsep) if p]

    # ------------------------------------------------------------------ queries
    def spec(self, model_id: str) -> ModelSpec:
        return self.specs[model_id]

    def path_for(self, model_id: str) -> Optional[Path]:
        spec = self.specs[model_id]
        for d in [self.root, *self._extra_dirs]:
            p = d / spec.file
            if p.is_file() and p.stat().st_size == spec.size:
                return p
        return None

    def status(self, model_id: str) -> ModelStatus:
        spec = self.specs[model_id]
        p = self.path_for(model_id)
        return ModelStatus(spec, p is not None, str(p) if p else None, p.stat().st_size if p else 0,
                           self._verified.get(model_id))

    def list(self, role: Optional[str] = None) -> list[ModelStatus]:
        return [self.status(m) for m, s in self.specs.items() if role is None or s.role == role]

    def missing_required(self) -> list[str]:
        miss = [m for m, s in self.specs.items() if s.required and self.path_for(m) is None]
        if not any(self.path_for(m) for m, s in self.specs.items() if s.role == "swapper"):
            miss.append("swapper (inswapper_128 or inswapper_128_fp16)")
        return miss

    def requirements(self, model_id: str) -> dict:
        s = self.specs[model_id]
        return {"min_ram_mb": s.min_ram_mb, "min_vram_mb": s.min_vram_mb, "providers": s.providers,
                "download_mb": round(s.size / 1e6, 1)}

    def compatible(self, model_id: str, backend: str, ram_mb: int = 0, vram_mb: Optional[int] = None) -> tuple[bool, str]:
        s = self.specs[model_id]
        if backend not in s.providers:
            return False, f"{s.title} is not recommended on {backend}"
        if ram_mb and s.min_ram_mb and ram_mb < s.min_ram_mb:
            return False, f"needs {s.min_ram_mb} MB RAM, {ram_mb} MB available"
        if backend != "cpu" and vram_mb is not None and s.min_vram_mb and vram_mb < s.min_vram_mb:
            return False, f"needs {s.min_vram_mb} MB VRAM, GPU has {vram_mb} MB"
        return True, "ok"

    def recommended_swapper(self, backend: str) -> str:
        """FP16 graph runs best on CUDA/TensorRT; elsewhere FP32 avoids cast overhead and precision issues."""
        return "inswapper_128_fp16" if backend in ("cuda", "tensorrt") else "inswapper_128"

    def pick_installed_swapper(self, backend: str) -> Optional[str]:
        pref = self.recommended_swapper(backend)
        other = "inswapper_128" if pref.endswith("fp16") else "inswapper_128_fp16"
        for m in (pref, other):
            if self.path_for(m):
                return m
        return None

    # ------------------------------------------------------------------ licences
    def license_accepted(self, model_id: str) -> bool:
        if os.environ.get("SAQR_ACCEPT_MODEL_LICENSES") == "1":  # CI after explicit acceptance
            return True
        if self.settings and model_id in self.settings.settings.accepted_licenses:
            return True
        # written by the Windows installer after the user ticked the licence page
        marker = config.data_dir() / "accepted_licenses.txt"
        try:
            return model_id in marker.read_text(encoding="utf-8").split()
        except OSError:
            return False

    def accept_license(self, model_ids: Iterable[str]) -> None:
        if not self.settings:
            return
        acc = set(self.settings.settings.accepted_licenses) | set(model_ids)
        self.settings.update(accepted_licenses=sorted(acc))

    # ------------------------------------------------------------------ mutation
    def verify(self, model_id: str, cancel: Optional[threading.Event] = None) -> bool:
        p = self.path_for(model_id)
        ok = bool(p) and sha256_file(p, cancel) == self.specs[model_id].sha256
        self._verified[model_id] = ok
        return ok

    def download(self, model_id: str, progress: Optional[ProgressCb] = None,
                 cancel: Optional[threading.Event] = None, timeout: float = 30.0, retries: int = 3) -> Path:
        spec = self.specs[model_id]
        if not self.license_accepted(model_id):
            raise LicenseNotAccepted(f"licence for {model_id} not accepted: {spec.license}")
        dest = self.root / spec.file
        if dest.is_file() and dest.stat().st_size == spec.size and self.verify(model_id, cancel):
            return dest
        part = dest.with_suffix(dest.suffix + ".part")
        for attempt in range(retries):
            try:
                self._fetch(spec, part, progress, cancel, timeout)
                break
            except DownloadCancelled:
                raise
            except Exception:  # noqa: BLE001 - network errors: retry with backoff, keep partial file
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)
        digest = sha256_file(part, cancel)
        if digest != spec.sha256:
            part.unlink(missing_ok=True)
            raise ChecksumMismatch(f"{model_id}: expected {spec.sha256}, got {digest}")
        os.replace(part, dest)
        self._verified[model_id] = True
        return dest

    def _fetch(self, spec: ModelSpec, part: Path, progress: Optional[ProgressCb],
               cancel: Optional[threading.Event], timeout: float) -> None:
        have = part.stat().st_size if part.exists() else 0
        if have > spec.size:
            part.unlink()
            have = 0
        if have == spec.size:
            return
        req = urllib.request.Request(spec.url, headers={"User-Agent": f"SAQR-AI-Studio/{config.VERSION}",
                                                        "Range": f"bytes={have}-"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - fixed https URLs from registry
            if resp.status == 200 and have:  # server ignored Range: restart
                have = 0
                mode = "wb"
            else:
                mode = "ab"
            with open(part, mode) as fh:
                done = have
                while True:
                    if cancel and cancel.is_set():
                        raise DownloadCancelled(spec.id)
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    fh.write(chunk)
                    done += len(chunk)
                    if progress:
                        progress(spec.id, done, spec.size)
        if part.stat().st_size != spec.size:
            raise IOError(f"incomplete download for {spec.id}: {part.stat().st_size}/{spec.size}")

    def remove(self, model_id: str) -> bool:
        p = self.root / self.specs[model_id].file
        removed = False
        for f in (p, p.with_suffix(p.suffix + ".part"), p.with_suffix(".emap.npy")):
            if f.exists():
                f.unlink()
                removed = True
        self._verified.pop(model_id, None)
        return removed

    def update(self, model_id: str, **kw) -> Path:
        """Re-download if the on-disk file does not match the registry checksum."""
        if self.path_for(model_id) and self.verify(model_id):
            return self.path_for(model_id)  # type: ignore[return-value]
        self.remove(model_id)
        return self.download(model_id, **kw)

    def test(self, model_id: str, selection) -> dict:
        """Load the model on the active backend and run one dummy inference."""
        import numpy as np

        from ..runtime.session import ResilientSession

        p = self.path_for(model_id)
        if not p:
            return {"ok": False, "error": "not installed"}
        t0 = time.perf_counter()
        try:
            s = ResilientSession(p, selection)
            feeds = {}
            for i in s.get_inputs():
                shape = [d if isinstance(d, int) and d > 0 else (640 if "input.1" == i.name and model_id == "scrfd_10g" else 1)
                         for d in i.shape]
                if model_id == "nsfw_vit_int8":
                    shape = [1, 3, 224, 224]
                feeds[i.name] = np.zeros(shape, dtype=np.float32)
            load_ms = (time.perf_counter() - t0) * 1000
            t1 = time.perf_counter()
            s.run(None, feeds)
            return {"ok": True, "backend": s.backend, "load_ms": round(load_ms, 1),
                    "infer_ms": round((time.perf_counter() - t1) * 1000, 1)}
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "error": str(exc)[:300]}

    def disk_usage(self) -> int:
        return sum(f.stat().st_size for f in self.root.glob("*.onnx"))

    def import_file(self, model_id: str, src: Path) -> Path:
        """Install a manually downloaded file after checksum verification (offline installs)."""
        spec = self.specs[model_id]
        if sha256_file(src) != spec.sha256:
            raise ChecksumMismatch(f"{src} does not match {model_id}")
        dest = self.root / spec.file
        shutil.copy2(src, dest)
        self._verified[model_id] = True
        return dest
