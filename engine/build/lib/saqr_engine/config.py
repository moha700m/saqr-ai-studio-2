"""Application paths and persisted settings.

Privacy-first defaults: local processing on, cloud off, no automatic uploads,
recording or camera start-up. Settings are stored as JSON in the per-user data
directory; nothing biometric is persisted.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_NAME = "SAQR AI Studio"
APP_ID = "saqr"
VERSION = "0.1.0"


def data_dir() -> Path:
    env = os.environ.get("SAQR_HOME")
    if env:
        p = Path(env)
    elif sys.platform == "win32":
        p = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / APP_NAME
    elif sys.platform == "darwin":
        p = Path.home() / "Library/Application Support" / APP_NAME
    else:
        p = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local/share")) / APP_ID
    p.mkdir(parents=True, exist_ok=True)
    return p


def models_dir() -> Path:
    p = Path(os.environ.get("SAQR_MODELS", data_dir() / "models"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def temp_dir() -> Path:
    p = data_dir() / "tmp"
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_dir() -> Path:
    p = Path(os.environ.get("SAQR_OUTPUT", Path.home() / "Videos" / "SAQR" if sys.platform == "win32"
                            else Path.home() / "SAQR"))
    return p


@dataclass
class Settings:
    processing_mode: str = "local"  # local | cloud | hybrid
    cloud_enabled: bool = False
    cloud_url: str = ""
    cloud_api_key: str = ""  # stored locally only; never sent anywhere except cloud_url
    auto_upload: bool = False  # fixed False: there is no automatic upload code path
    auto_record: bool = False
    auto_start_camera: bool = False
    bridge_enabled: bool = False
    bridge_port: int = 47800
    forced_backend: str = ""  # "" = automatic
    enable_tensorrt: bool = False
    accepted_licenses: list[str] = field(default_factory=list)
    export_watermark: bool = True
    export_metadata: bool = True
    preset: str = "balanced"
    last_camera: int = -1
    keep_temp_files: bool = False
    language: str = "en"

    def __post_init__(self) -> None:
        self.auto_upload = False


class SettingsStore:
    def __init__(self, path: Path | None = None):
        self.path = path or data_dir() / "settings.json"
        self._lock = threading.Lock()
        self.settings = self._load()

    def _load(self) -> Settings:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            known = {k: v for k, v in raw.items() if k in Settings.__dataclass_fields__}
            return Settings(**known)
        except (OSError, ValueError, TypeError):
            return Settings()

    def save(self) -> None:
        with self._lock:
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(asdict(self.settings), indent=2), encoding="utf-8")
            os.replace(tmp, self.path)

    def update(self, **changes: Any) -> Settings:
        for k, v in changes.items():
            if k not in Settings.__dataclass_fields__:
                raise KeyError(k)
            setattr(self.settings, k, v)
        self.settings.auto_upload = False
        if self.settings.processing_mode in ("cloud", "hybrid") and not self.settings.cloud_enabled:
            self.settings.processing_mode = "local"
        self.save()
        return self.settings
