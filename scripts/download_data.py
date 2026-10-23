#!/usr/bin/env python3
"""
Descarga los datos de Kaggle, verifica su integridad y genera data/processed/demanda_limpia.csv.

Los datos NO se redistribuyen en este repositorio (pertenecen a la competencia
Genpact / Analytics Vidhya "Food Demand Forecasting", publicada en Kaggle).

Requisitos: cuenta de Kaggle y credenciales de la API (~/.kaggle/kaggle.json) o
descarga manual de los CSV en data/raw/.

Uso:
    python scripts/download_data.py            # descarga (si falta), verifica y procesa
    python scripts/download_data.py --no-download
"""
import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "demanda_limpia.csv"
DATASET = "kannanaikkal/food-demand-forecasting"
SHA256 = {  # archivos exactos usados en el análisis
    "train.csv": "a2ed2d7c6905d63a9361d8dd35bc06c8a597831f7185644e0f33ad7bfe20cc81",
    "meal_info.csv": "9d8045af391c492f531c928d7f3d58581c80d0594776d1f4e53c8b29bbff8f75",
    "fulfilment_center_info.csv": "c874f74fa392b2c4dd24a167f32a5bee68a444b8bf151acf295cd5d66aa0517f",
}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    missing = [f for f in SHA256 if not (RAW / f).exists()]
    if missing and not a.no_download:
        print(f"Descargando {DATASET} con la API de Kaggle...")
        subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-p", str(RAW), "--unzip"], check=True)
    for f, h in SHA256.items():
        p = RAW / f
        if not p.exists():
            sys.exit(f"Falta {p}. Descárgalo manualmente desde https://www.kaggle.com/datasets/{DATASET}")
        status = "OK" if sha256(p) == h else "DISTINTO (otra versión del dataset; los resultados pueden variar)"
        print(f"{f}: {status}")
    sys.path.insert(0, str(ROOT / "src"))
    from data_prep import pipeline_completo
    df = pipeline_completo(str(RAW))
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print(f"Generado {PROCESSED.relative_to(ROOT)}: {df.shape[0]:,} filas x {df.shape[1]} columnas")


if __name__ == "__main__":
    main()
