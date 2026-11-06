"""Hardware Detection Engine.

Inspects CPU, GPU(s), memory, disk and OS without requiring any user knowledge,
and produces a serialisable :class:`HardwareReport` that the backend selector
consumes. Every probe is best-effort: a failing probe records an error string
instead of raising, so detection never blocks application start-up.
"""
from .report import CPUInfo, GPUInfo, SystemInfo, HardwareReport
from .probe import detect_hardware

__all__ = ["CPUInfo", "GPUInfo", "SystemInfo", "HardwareReport", "detect_hardware"]
