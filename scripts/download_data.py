"""
Descarga el dataset completo "Food Demand Forecasting" desde Kaggle
y lo deja listo en data/raw/.

Requisitos previos:
    1. Cuenta de Kaggle (gratis).
    2. Token de API descargado desde https://www.kaggle.com/settings
       (botón "Create New Token" -> descarga kaggle.json).
    3. Colocar kaggle.json en:
         - Linux/Mac: ~/.kaggle/kaggle.json
         - Windows:   C:\\Users\\<tu_usuario>\\.kaggle\\kaggle.json
    4. pip install kaggle (ya incluido en environment.yml / requirements.txt)

Uso:
    python scripts/download_data.py

Resultado esperado en data/raw/:
    train.csv
    fulfilment_center_info.csv
    meal_info.csv
"""

import shutil
import sys
import zipfile
from pathlib import Path

DATASET_SLUG = "kannanaikkal/food-demand-forecasting"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

EXPECTED_FILES = {
    "train.csv",
    "fulfilment_center_info.csv",
    "meal_info.csv",
}


def check_kaggle_credentials() -> None:
    """Verifica que exista el token de Kaggle antes de intentar descargar."""
    kaggle_json_locations = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path.home() / "AppData" / "Roaming" / "kaggle" / "kaggle.json",  # Windows alt
    ]
    if not any(p.exists() for p in kaggle_json_locations):
        sys.exit(
            "No se encontró kaggle.json. Descárgalo desde "
            "https://www.kaggle.com/settings (Create New Token) y colócalo en "
            "~/.kaggle/kaggle.json antes de correr este script."
        )


def download_dataset() -> None:
    """Descarga y descomprime el dataset completo en data/raw/."""
    # Import diferido: si kaggle no está instalado, el mensaje de check_kaggle_credentials
    # o el ImportError son más claros que un fallo silencioso al importar arriba del archivo.
    from kaggle.api.kaggle_api_extended import KaggleApi

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    api = KaggleApi()
    api.authenticate()

    print(f"Descargando dataset '{DATASET_SLUG}' en {RAW_DIR} ...")
    api.dataset_download_files(DATASET_SLUG, path=str(RAW_DIR), unzip=False)

    # El archivo descargado tiene el nombre del último segmento del slug + .zip
    zip_name = DATASET_SLUG.split("/")[-1] + ".zip"
    zip_path = RAW_DIR / zip_name

    if not zip_path.exists():
        # A veces Kaggle nombra el zip distinto; buscamos cualquier .zip nuevo en RAW_DIR
        zips = list(RAW_DIR.glob("*.zip"))
        if not zips:
            sys.exit("La descarga no generó ningún archivo .zip. Revisa tus credenciales de Kaggle.")
        zip_path = zips[0]

    print(f"Descomprimiendo {zip_path.name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(RAW_DIR)

    zip_path.unlink()  # limpiar el zip después de extraer

    _flatten_nested_dirs()
    _verify_files()


def _flatten_nested_dirs() -> None:
    """Algunos datasets de Kaggle extraen en una subcarpeta; la aplanamos si aplica."""
    subdirs = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    for sub in subdirs:
        for f in sub.iterdir():
            shutil.move(str(f), str(RAW_DIR / f.name))
        sub.rmdir()


def _verify_files() -> None:
    present = {p.name for p in RAW_DIR.glob("*.csv")}
    missing = EXPECTED_FILES - present
    if missing:
        print(f"Advertencia: no se encontraron estos archivos esperados: {missing}")
        print(f"Archivos presentes en data/raw/: {sorted(present)}")
    else:
        print("Descarga completa. Archivos verificados en data/raw/:")
        for f in sorted(present):
            size_mb = (RAW_DIR / f).stat().st_size / (1024 * 1024)
            print(f"  - {f} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    check_kaggle_credentials()
    download_dataset()
