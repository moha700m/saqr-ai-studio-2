"""Cross-platform hardware probes.

Windows: PowerShell CIM (Win32_Processor / Win32_VideoController) + registry
         HardwareInformation.qwMemorySize for accurate VRAM (WMI AdapterRAM is a
         32-bit field capped at 4 GiB) + NVML for NVIDIA driver/VRAM.
Linux:   /proc/cpuinfo, lspci, /sys/class/drm, NVML.
macOS:   sysctl + system_profiler.

Probes run with short timeouts and never raise; errors land in report.errors.
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Callable, Optional

from .report import CPUInfo, GPUInfo, HardwareReport, SystemInfo

# Runner indirection makes every probe unit-testable with canned outputs.
Runner = Callable[[list[str], float], Optional[str]]

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


def _run(cmd: list[str], timeout: float = 8.0) -> Optional[str]:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                             creationflags=_NO_WINDOW, errors="replace")
        if out.returncode != 0:
            return None
        return out.stdout
    except (OSError, subprocess.SubprocessError):
        return None


def vendor_from_name(name: str, pci_vendor_id: Optional[str] = None) -> str:
    vid = (pci_vendor_id or "").lower().replace("0x", "")
    if vid == "10de":
        return "nvidia"
    if vid in ("1002", "1022"):
        return "amd"
    if vid == "8086":
        return "intel"
    n = name.lower()
    if any(k in n for k in ("nvidia", "geforce", "quadro", "tesla", "rtx", "gtx")):
        return "nvidia"
    if any(k in n for k in ("amd", "radeon", "ati ")):
        return "amd"
    if any(k in n for k in ("intel", "iris", "uhd graphics", "arc ")):
        return "intel"
    if "apple" in n:
        return "apple"
    if "microsoft" in n:
        return "microsoft"
    return "unknown"


def guess_integrated(vendor: str, name: str) -> Optional[bool]:
    n = name.lower()
    if vendor == "nvidia":
        return False
    if vendor == "intel":
        # Discrete Arc cards carry a model number (A770, A380, B580); "Arc Graphics" alone is the iGPU.
        return not re.search(r"arc(\(tm\))?\s*(pro\s*)?[ab]\d{2,3}", n)
    if vendor == "amd":
        # "Radeon(TM) Graphics", "Radeon Vega 8 Graphics", 680M/780M are APUs
        if re.search(r"radeon\(tm\) graphics|vega \d+ graphics|\b\d{3}m\b|radeon graphics", n):
            return True
        if re.search(r"\b(rx|pro|w)\s?\d{3,4}", n) or "instinct" in n:
            return False
        return None
    if vendor == "apple":
        return True
    return None


# ----------------------------------------------------------------------------- CPU

def probe_cpu(errors: list[str]) -> CPUInfo:
    import psutil

    info = CPUInfo(architecture=platform.machine() or "unknown",
                   logical_cores=psutil.cpu_count(logical=True) or 0,
                   physical_cores=psutil.cpu_count(logical=False) or 0)
    try:
        freq = psutil.cpu_freq()
        if freq and freq.max:
            info.max_frequency_mhz = float(freq.max)
    except Exception:  # noqa: BLE001 - psutil raises varied errors in containers
        pass
    try:
        import cpuinfo  # py-cpuinfo

        ci = cpuinfo.get_cpu_info()
        info.model = ci.get("brand_raw") or info.model
        info.vendor = _normalise_cpu_vendor(ci.get("vendor_id_raw") or ci.get("vendor_id") or "")
        info.flags = sorted(set(ci.get("flags", [])))
        info.architecture = ci.get("arch_string_raw") or info.architecture
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cpu: py-cpuinfo failed: {exc}")
        if sys.platform.startswith("linux"):
            _cpu_from_proc(info)
    if info.vendor == "unknown":
        info.vendor = _normalise_cpu_vendor(info.model)
    return info


def _normalise_cpu_vendor(s: str) -> str:
    s = s.lower()
    if "intel" in s:
        return "intel"
    if "amd" in s or "authenticamd" in s:
        return "amd"
    if "apple" in s:
        return "apple"
    if "arm" in s or "qualcomm" in s or "snapdragon" in s:
        return "arm"
    return "unknown"


def _cpu_from_proc(info: CPUInfo) -> None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
            txt = fh.read()
        m = re.search(r"model name\s*:\s*(.+)", txt)
        if m:
            info.model = m.group(1).strip()
        m = re.search(r"^flags\s*:\s*(.+)$", txt, re.M)
        if m:
            info.flags = sorted(set(m.group(1).split()))
    except OSError:
        pass


# ----------------------------------------------------------------------------- GPU

def probe_nvml(errors: list[str]) -> list[GPUInfo]:
    try:
        import pynvml  # provided by nvidia-ml-py
    except ImportError:
        return []
    gpus: list[GPUInfo] = []
    try:
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - no NVIDIA driver present: normal on most machines
        return []
    try:
        drv = pynvml.nvmlSystemGetDriverVersion()
        drv = drv.decode() if isinstance(drv, bytes) else drv
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            name = name.decode() if isinstance(name, bytes) else name
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            try:
                maj, mnr = pynvml.nvmlDeviceGetCudaComputeCapability(h)
                cc = f"{maj}.{mnr}"
            except Exception:  # noqa: BLE001
                cc = None
            gpus.append(GPUInfo(vendor="nvidia", name=name, vram_mb=int(mem.total // (1024 * 1024)),
                                driver_version=drv, integrated=False, compute_capability=cc,
                                source="nvml", apis=["cuda"]))
    except Exception as exc:  # noqa: BLE001
        errors.append(f"gpu: NVML query failed: {exc}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            pass
    return gpus


_PS_GPU = (
    "$ErrorActionPreference='SilentlyContinue';"
    "$vc=Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID,AdapterCompatibility;"
    "$reg=@();"
    "Get-ChildItem 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}' |"
    " ForEach-Object { $p=Get-ItemProperty $_.PSPath; if($p.DriverDesc){ $reg+=[pscustomobject]@{Name=$p.DriverDesc;"
    " Mem=$p.'HardwareInformation.qwMemorySize'; MemDw=$p.'HardwareInformation.MemorySize'} } };"
    "[pscustomobject]@{vc=@($vc);reg=@($reg)} | ConvertTo-Json -Depth 4 -Compress"
)


def parse_windows_gpus(ps_json: str) -> list[GPUInfo]:
    data = json.loads(ps_json)
    vcs = data.get("vc") or []
    if isinstance(vcs, dict):
        vcs = [vcs]
    regs = data.get("reg") or []
    if isinstance(regs, dict):
        regs = [regs]
    reg_mem: dict[str, int] = {}
    for r in regs:
        mem = r.get("Mem")
        if isinstance(mem, list):  # REG_BINARY sometimes arrives as a byte list
            mem = int.from_bytes(bytes(mem), "little")
        if not mem:
            mdw = r.get("MemDw")
            if isinstance(mdw, list):
                mdw = int.from_bytes(bytes(mdw), "little")
            mem = mdw
        if r.get("Name") and mem:
            reg_mem[str(r["Name"]).strip()] = int(mem)
    out: list[GPUInfo] = []
    for v in vcs:
        name = str(v.get("Name") or "unknown").strip()
        pnp = str(v.get("PNPDeviceID") or "")
        m = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp)
        vendor = vendor_from_name(name, m.group(1) if m else None)
        vram = reg_mem.get(name) or v.get("AdapterRAM")
        vram_mb = int(vram) // (1024 * 1024) if vram else None
        apis = ["directml"] if vendor in ("nvidia", "amd", "intel") else []
        if vendor == "nvidia":
            apis.insert(0, "cuda")
        out.append(GPUInfo(vendor=vendor, name=name, vram_mb=vram_mb,
                           driver_version=v.get("DriverVersion"),
                           integrated=guess_integrated(vendor, name), source="wmi", apis=apis))
    return out


def probe_windows_gpus(errors: list[str], run: Runner = _run) -> list[GPUInfo]:
    ps = shutil.which("powershell") or shutil.which("pwsh") or "powershell"
    out = run([ps, "-NoProfile", "-NonInteractive", "-Command", _PS_GPU], 15.0)
    if not out:
        errors.append("gpu: Windows CIM query returned nothing")
        return []
    try:
        return parse_windows_gpus(out)
    except (ValueError, TypeError) as exc:
        errors.append(f"gpu: could not parse CIM output: {exc}")
        return []


def parse_lspci(text: str) -> list[GPUInfo]:
    out = []
    for line in text.splitlines():
        if not re.search(r"VGA compatible|3D controller|Display controller", line):
            continue
        name = line.split(":", 2)[-1].strip()
        m = re.search(r"\[([0-9a-f]{4}):[0-9a-f]{4}\]", line)
        vendor = vendor_from_name(name, m.group(1) if m else None)
        name = re.sub(r"\s*\[[0-9a-f]{4}:[0-9a-f]{4}\]", "", name)
        name = re.sub(r"\s*\(rev [0-9a-f]+\)", "", name)
        out.append(GPUInfo(vendor=vendor, name=name, integrated=guess_integrated(vendor, name), source="lspci"))
    return out


def probe_linux_gpus(errors: list[str], run: Runner = _run) -> list[GPUInfo]:
    if shutil.which("lspci"):
        txt = run(["lspci", "-nn"], 5.0)
        if txt:
            return parse_lspci(txt)
    # Fallback: /sys/class/drm vendor ids
    gpus = []
    try:
        for card in sorted(os.listdir("/sys/class/drm")):
            if not re.fullmatch(r"card\d+", card):
                continue
            try:
                with open(f"/sys/class/drm/{card}/device/vendor", encoding="ascii") as fh:
                    vid = fh.read().strip()
            except OSError:
                continue
            vendor = vendor_from_name("", vid)
            gpus.append(GPUInfo(vendor=vendor, name=f"{vendor.upper()} GPU ({card})", source="drm"))
    except OSError:
        pass
    return gpus


def probe_macos_gpus(errors: list[str], run: Runner = _run) -> list[GPUInfo]:
    out = run(["system_profiler", "SPDisplaysDataType", "-json"], 15.0)
    if not out:
        return []
    try:
        data = json.loads(out).get("SPDisplaysDataType", [])
    except ValueError:
        return []
    gpus = []
    for d in data:
        name = d.get("sppci_model", "unknown")
        vendor = vendor_from_name(name)
        vram = d.get("spdisplays_vram") or d.get("spdisplays_vram_shared") or ""
        m = re.search(r"(\d+)\s*(GB|MB)", str(vram))
        vram_mb = (int(m.group(1)) * (1024 if m.group(2) == "GB" else 1)) if m else None
        gpus.append(GPUInfo(vendor=vendor, name=name, vram_mb=vram_mb, integrated=vendor in ("apple", "intel"),
                            source="system_profiler", apis=["coreml", "metal"] if vendor == "apple" else []))
    return gpus


def merge_gpus(primary: list[GPUInfo], secondary: list[GPUInfo]) -> list[GPUInfo]:
    """Merge NVML (accurate) entries into OS-level entries, de-duplicating by name."""
    merged = list(primary)
    for g in secondary:
        match = next((p for p in merged if _same_gpu(p.name, g.name)), None)
        if match is None:
            merged.append(g)
            continue
        match.driver_version = match.driver_version or g.driver_version
        match.vram_mb = match.vram_mb or g.vram_mb
        match.apis = sorted(set(match.apis) | set(g.apis), key=lambda a: ["cuda", "tensorrt", "directml"].index(a)
                            if a in ("cuda", "tensorrt", "directml") else 9)
    return merged


def _same_gpu(a: str, b: str) -> bool:
    norm = lambda s: re.sub(r"[^a-z0-9]", "", s.lower().replace("nvidia", "").replace("geforce", ""))  # noqa: E731
    na, nb = norm(a), norm(b)
    return bool(na) and (na == nb or na in nb or nb in na)


# ----------------------------------------------------------------------------- system

def probe_system(errors: list[str], disk_path: Optional[str] = None) -> SystemInfo:
    import psutil

    vm = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage(disk_path or os.path.expanduser("~"))
        free = disk.free
    except OSError as exc:
        errors.append(f"system: disk usage failed: {exc}")
        free = 0
    os_version = platform.version()
    if sys.platform == "win32":
        try:
            os_version = platform.win32_ver()[1] or os_version
        except Exception:  # noqa: BLE001
            pass
    return SystemInfo(os=platform.system(), os_version=os_version, os_release=platform.release(),
                      is_64bit=sys.maxsize > 2**32 and platform.machine().lower() in
                      ("amd64", "x86_64", "arm64", "aarch64"),
                      machine=platform.machine(), ram_total_mb=vm.total // (1024 * 1024),
                      ram_available_mb=vm.available // (1024 * 1024), disk_free_mb=free // (1024 * 1024),
                      python_version=platform.python_version())


# ----------------------------------------------------------------------------- entry

def detect_hardware(run: Runner = _run, disk_path: Optional[str] = None) -> HardwareReport:
    t0 = time.perf_counter()
    errors: list[str] = []
    cpu = probe_cpu(errors)
    nv = probe_nvml(errors)
    if sys.platform == "win32":
        os_gpus = probe_windows_gpus(errors, run)
    elif sys.platform == "darwin":
        os_gpus = probe_macos_gpus(errors, run)
    else:
        os_gpus = probe_linux_gpus(errors, run)
    gpus = merge_gpus(nv, os_gpus)
    system = probe_system(errors, disk_path)
    return HardwareReport(cpu=cpu, gpus=gpus, system=system, errors=errors,
                          duration_ms=(time.perf_counter() - t0) * 1000)
