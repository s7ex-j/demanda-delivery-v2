from __future__ import annotations

import json
import os
import pickle
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    for candidate in [start.resolve(), *start.resolve().parents]:
        if (candidate / "src" / "modeling.py").exists() and (candidate / "data").is_dir():
            if (candidate / "data" / "demanda_limpia.csv").exists():
                return candidate
            if (candidate / "data" / "processed" / "demanda_limpia.csv").exists():
                return candidate
    raise RuntimeError(
        "No se pudo localizar la raíz del proyecto. Se esperaba "
        "src/modeling.py y data/demanda_limpia.csv (o data/processed/demanda_limpia.csv)."
    )


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = find_project_root(SCRIPT_DIR)
DATA_CSV = PROJECT_ROOT / "data" / "demanda_limpia.csv"
if not DATA_CSV.exists():
    DATA_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia.csv"

REPORTS_DIR = PROJECT_ROOT / "reports"
OUT_DIR = REPORTS_DIR / "diagnostics"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    required = {"center_id", "meal_id", "week", "checkout_price", "num_orders"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"Faltan columnas requeridas: {sorted(missing)}")
    return df


def save_npz_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def atomic_save_npy(path: Path, obj: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npy")
    np.save(tmp, obj, allow_pickle=True)
    os.replace(tmp, path)


def save_result(obj: Any, filename: str) -> Path:
    primary = OUT_DIR / filename
    atomic_save_npy(primary, obj)
    return primary


def save_json(obj: Any, filename: str) -> Path:
    def conv(x):
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        if isinstance(x, np.ndarray):
            return conv(x.tolist())
        if isinstance(x, np.integer):
            return int(x)
        if isinstance(x, np.floating):
            return float(x)
        if isinstance(x, np.bool_):
            return bool(x)
        return x
    primary = OUT_DIR / filename
    tmp = primary.with_suffix(primary.suffix + ".tmp")
    tmp.write_text(json.dumps(conv(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, primary)
    return primary


def backup_existing_outputs(run_dir: Path, patterns: list[str]) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = run_dir / "_vm_backups" / ts
    backup.mkdir(parents=True, exist_ok=True)
    for pattern in patterns:
        for d in (OUT_DIR,):
            for p in d.glob(pattern):
                if p.is_file():
                    target = backup / d.name / p.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, target)
    return backup


def progress(iteration: int, total: int, t0: float) -> str:
    elapsed = time.time() - t0
    rate = elapsed / max(iteration, 1)
    remain = rate * max(total - iteration, 0)
    return f"{iteration}/{total} | {elapsed/60:.1f} min | ETA {remain/60:.1f} min"


def cluster_multiplicities(df: pd.DataFrame, clusters: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict[Any, int]]:
    sampled = rng.choice(clusters, size=len(clusters), replace=True)
    mult = {c: 0 for c in clusters.tolist()}
    for c in sampled:
        mult[c] += 1
    weights = df["center_id"].map(mult).to_numpy(dtype=float)
    return weights, mult
