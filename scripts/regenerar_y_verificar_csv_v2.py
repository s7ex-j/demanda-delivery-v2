"""
ETAPA 2 del plan de corrección (Auditoría v2.1 / 2.2):
Regenera data/processed/demanda_limpia_v2.csv con el pipeline actualizado
(log(num_orders) en vez de log1p) y verifica todos los checkpoints.

Uso:
    python scripts/regenerar_y_verificar_csv_v2.py

No depende del directorio de trabajo: resuelve rutas desde la raíz del repo.
"""
import sys
import os
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd

# Resolver rutas absolutas desde la raíz del proyecto
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_prep import pipeline_completo  # noqa: E402

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "demanda_limpia_v2.csv"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    print("=" * 70)
    print("ETAPA 2 — Regeneración y verificación del dataset procesado")
    print("=" * 70)

    # 1. Ejecutar pipeline
    print(f"\n[1/8] Ejecutando pipeline_completo(raw_dir={RAW_DIR}) ...")
    df = pipeline_completo(str(RAW_DIR))

    # 2. Guardar CSV v2
    print(f"[2/8] Guardando CSV -> {OUT_CSV}")
    df.to_csv(OUT_CSV, index=False)

    # 3. Verificar N
    nesperado = 456_548
    n = df.shape[0]
    assert n == nesperado, f"N MISMATCH: {n} != {nesperado}"
    print(f"[3/8] N = {n:,} filas (esperado {nesperado:,})  OK")

    # 4. Columnas
    cols_esperadas = {
        "id", "week", "center_id", "meal_id", "checkout_price", "base_price",
        "emailer_for_promotion", "homepage_featured", "num_orders",
        "category", "cuisine", "city_code", "region_code", "center_type",
        "op_area", "variacion_precio", "variacion_precio_pct", "hubo_aumento",
        "log_checkout_price", "log_num_orders",
    }
    faltan = cols_esperadas - set(df.columns)
    assert not faltan, f"Faltan columnas: {faltan}"
    print(f"[4/8] Columnas ({len(df.columns)}) completas  OK")

    # 5. Duplicados
    dups = df.duplicated(subset=["week", "center_id", "meal_id"]).sum()
    assert dups == 0, f"DUPLICADOS en (week,center,meal): {dups}"
    print(f"[5/8] Duplicados (week,center_id,meal_id) = 0  OK")

    # 6. Nulos en variables clave
    nulos = df[["checkout_price", "base_price", "num_orders", "log_num_orders"]].isna().sum()
    assert nulos.sum() == 0, f"NULLS:\n{nulos}"
    print(f"[6/8] Nulos en variables clave = 0  OK")

    # 7. min(num_orders) y verificación log vs log1p
    min_orders = int(df["num_orders"].min())
    assert min_orders == 13, f"MIN num_orders != 13: {min_orders}"
    # Verificar que log_num_orders = log(num_orders) y NO log1p(num_orders)
    diff_log = (df["log_num_orders"] - np.log(df["num_orders"])).abs().max()
    diff_log1p = (df["log_num_orders"] - np.log1p(df["num_orders"])).abs().max()
    assert diff_log < 1e-12, f"log_num_orders NO es log(): diff={diff_log}"
    print(f"[7/8] min(num_orders) = {min_orders} (sin ceros)  OK")
    print(f"      log_num_orders = log(num_orders):   diff_max = {diff_log:.2e}  OK")
    print(f"      log_num_orders = log1p(num_orders): diff_max = {diff_log1p:.2e}  (debe ser GRANDE)")

    # 8. Concordancia con raw (train.csv)
    import pandas as pd
    train = pd.read_csv(RAW_DIR / "train.csv")
    assert len(train) == n, f"train.csv rows {len(train)} != N {n}"
    assert df["num_orders"].sum() == train["num_orders"].sum(), "Suma de num_orders no coincide con raw"
    print(f"[8/8] Concordancia con train.csv: filas y suma de pedidos  OK")

    # Hash del nuevo CSV
    h = sha256_file(OUT_CSV)
    print("\n" + "=" * 70)
    print("RESULTADO ETAPA 2 — TODOS LOS CHECKPOINTS OK")
    print("=" * 70)
    print(f"CSV v2:        {OUT_CSV}")
    print(f"SHA256:        {h}")
    print(f"N:             {n:,}")
    print(f"min(pedidos):  {min_orders}")
    print(f"Transformación: log(num_orders)  [NO log1p]")
    print("\nRegistra este hash en el plan 2.2(1).md como baseline de datos v2.")


if __name__ == "__main__":
    main()
