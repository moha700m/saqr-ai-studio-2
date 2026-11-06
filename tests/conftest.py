import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "engine"), str(ROOT / "desktop")]
FIXTURES = ROOT / "tests" / "fixtures"


def pytest_configure(config):
    config.addinivalue_line("markers", "models: needs the real ONNX model files (set SAQR_MODELS)")


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("SAQR_HOME", str(tmp_path / "home"))
    yield


def models_dir():
    d = os.environ.get("SAQR_MODELS")
    return Path(d) if d and Path(d, "det_10g.onnx").exists() else None
