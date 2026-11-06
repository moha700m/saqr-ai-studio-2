"""Shared application state for the desktop UI."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from saqr_engine.config import SettingsStore
from saqr_engine.hardware import HardwareReport, detect_hardware
from saqr_engine.models.manager import ModelManager
from saqr_engine.pipeline.config import ProcessConfig, preset
from saqr_engine.pipeline.engine import FaceEngine
from saqr_engine.runtime.providers import BackendSelection, select_backend
from saqr_engine.runtime.session import STATE

log = logging.getLogger(__name__)


class AppContext:
    def __init__(self) -> None:
        self.settings = SettingsStore()
        self.hardware: Optional[HardwareReport] = None
        self.selection: Optional[BackendSelection] = None
        self.models = ModelManager(settings=self.settings)
        self.cfg: ProcessConfig = preset(self.settings.settings.preset if self.settings.settings.preset
                                         in ("fast", "balanced", "quality") else "balanced")
        self._engine: Optional[FaceEngine] = None
        self._lock = threading.Lock()
        self.live = None  # LiveSession owned by the Live page

    def detect(self) -> None:
        """Hardware probe + backend verification (slow: run off the GUI thread)."""
        self.hardware = detect_hardware()
        s = self.settings.settings
        self.selection = select_backend(self.hardware, forced=s.forced_backend or None,
                                        include_tensorrt=s.enable_tensorrt)
        STATE.selection = self.selection

    def set_backend(self, forced: str) -> None:
        self.settings.update(forced_backend=forced)
        s = self.settings.settings
        self.selection = select_backend(self.hardware, forced=forced or None, include_tensorrt=s.enable_tensorrt)
        STATE.selection = self.selection
        with self._lock:
            old = self._engine
            self._engine = None
        if old is not None:  # carry the consented sources over to the new engine
            new = self.engine
            new.sources = old.sources
            new.active_source = old.active_source
            new.rules = old.rules
            for src in new.sources.values():
                src.latent = None

    @property
    def engine(self) -> FaceEngine:
        with self._lock:
            if self._engine is None:
                if self.selection is None:
                    raise RuntimeError("hardware detection has not finished")
                self._engine = FaceEngine(self.models, self.selection, self.cfg)
            return self._engine

    def models_ready(self) -> bool:
        return not self.models.missing_required()
