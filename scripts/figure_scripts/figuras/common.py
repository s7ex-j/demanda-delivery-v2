from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_project_root():
    """Obtiene el directorio raíz del proyecto usando .git, pyproject.toml o MANIFEST.json como marcadores."""
    start_path = Path(__file__).resolve().parent
    for depth in range(5):  # Límite de 5 niveles para evitar bucles infinitos
        current_path = start_path.parents[depth]
        if (current_path / ".git").exists() or (current_path / "pyproject.toml").exists() or (current_path / "MANIFEST.json").exists():
            return current_path
    raise FileNotFoundError("Raíz del proyecto no encontrada dentro de 5 niveles.")

# figuras/ está dos niveles abajo de scripts/figure_scripts/, así que root está en parents[3]
ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "processed" / "demanda_limpia.csv"
REPORTS = ROOT / "reports"
FIG_DIR = ROOT / "Informe" / "figuras"

def load_df():
    return pd.read_csv(DATA)

def load_json(name):
    with open(REPORTS / name, encoding="utf-8") as f:
        return json.load(f)

def save(fig, name, dpi=160):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(path)
    return path

def apply_grid(ax):
    ax.grid(True, axis="both", alpha=0.25)
    return ax

# Main OLS log-log with controls from active results
OLS_LOG_LOG_CONTROLS = -0.759
