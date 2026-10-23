#!/usr/bin/env python3
"""
Runner para ejecutar placebo_temporal_valido() del proyecto activo.
NO modifica src/modeling.py.
"""

import sys
import json
import hashlib
import pandas as pd
import numpy as np
from pathlib import Path

# Importar la función exacta del proyecto activo
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.modeling import placebo_temporal_valido

def main():
    # 1. Cargar dataset exacto del proyecto
    dataset_path = Path(__file__).resolve().parents[2] / "data" / "processed" / "demanda_limpia.csv"
    df = pd.read_csv(dataset_path)

    # 2. Hashes de trazabilidad
    with open(dataset_path, 'rb') as f:
        dataset_sha256 = hashlib.sha256(f.read()).hexdigest()
    with open(Path(__file__).resolve().parents[2] / "src" / "modeling.py", "rb") as f:
        script_sha256 = hashlib.sha256(f.read()).hexdigest()

    # 3. Ejecutar la función EXACTA con parámetros fijos
    result = placebo_temporal_valido(df, n_iteraciones=500, seed=42)

    # 4. Validaciones de integridad
    print("=== VALIDACIONES ===")
    print(f"Dataset: {dataset_path} (sha256={dataset_sha256[:16]}...)")
    print(f"Script: src/modeling.py (sha256={script_sha256[:16]}...)")
    print(f"N obs: {len(df)}")
    print(f"N iteraciones válidas: {len(result['elasticidades_placebo'])}")
    print(f"Seed: 42")
    print(f"NaN en placebos: {np.isnan(result['elasticidades_placebo']).sum()}")
    print(f"Función: placebo_temporal_valido")
    print(f"Tipo: {result.get('tipo', 'NO ESPECIFICADO')}")
    print()
    print("=== RESULTADO ===")
    print(f"elasticidad_real: {result['elasticidad_real']:.6f}")
    print(f"media_placebo: {result['media_placebo']:.6f}")
    print(f"std_placebo: {result['std_placebo']:.6f}")
    print(f"p_valor: {result['p_valor']:.6f}")
    print(f"rechaza_H0_05: {result['rechaza_H0_05']}")

    # 5. Verificaciones críticas
    assert len(result['elasticidades_placebo']) == 500, f"Esperado 500 réplicas válidas, got {len(result['elasticidades_placebo'])}"
    assert np.isnan(result['elasticidades_placebo']).sum() == 0, "Hay NaN en resultados"
    assert result.get('tipo') == 'permutación_precio_dentro_entidad', f"Tipo inesperado: {result.get('tipo')}"

    # 6. Preparar JSON completo con metadatos
    output = {
        "provenance": {
            "script": "src/modeling.py",
            "function": "placebo_temporal_valido",
            "script_sha256": script_sha256,
            "dataset": "data/processed/demanda_limpia.csv",
            "dataset_sha256": dataset_sha256
        },
        "parameters": {
            "n_iteraciones": 500,
            "seed": 42,
            "n_obs": int(len(df))
        },
        "results": {
            "elasticidad_real": float(result['elasticidad_real']),
            "media_placebo": float(result['media_placebo']),
            "std_placebo": float(result['std_placebo']),
            "p_valor": float(result['p_valor']),
            "rechaza_H0_05": bool(result['rechaza_H0_05']),
            "n_iteraciones_validas": len(result['elasticidades_placebo']),
            "tipo": result.get('tipo'),
            "nota_metodologica": result.get('nota', '')
        },
        "validation": {
            "exactamente_500_replicas": len(result['elasticidades_placebo']) == 500,
            "sin_nan": bool(np.isnan(result['elasticidades_placebo']).sum() == 0),
            "seed_42": True,
            "funcion_exacta": "placebo_temporal_valido",
            "dataset_exacto": "data/processed/demanda_limpia.csv"
        }
    }

    # 7. Persistir JSON en reports/certified
    json_path = Path(__file__).resolve().parents[2] / "reports" / "certified" / "placebo_temporal_valido.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Resultado guardado en: {json_path}")


if __name__ == "__main__":
    main()