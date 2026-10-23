"""Omite los tests que necesitan los datos si no se han descargado (ver scripts/download_data.py)."""
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "demanda_limpia.csv"
RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "train.csv"


def pytest_collection_modifyitems(config, items):
    if DATA.exists() and RAW.exists():
        return
    skip = pytest.mark.skip(reason="Datos no disponibles: ejecuta `python scripts/download_data.py`")
    for item in items:
        if "datos" in item.keywords or _needs_data(item):
            item.add_marker(skip)


def _needs_data(item) -> bool:
    try:
        src = Path(item.fspath).read_text(encoding="utf-8")
    except OSError:
        return False
    return "demanda_limpia.csv" in src or "data/raw" in src or "DATA" in src
