"""Tests 3.3.0: cifras corregidas en la auditoría previa a la publicación (requieren los datos)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling import modelo_quantile_regression, modelo_semilog  # noqa: E402

DATA = ROOT / "data" / "processed" / "demanda_limpia.csv"
TEX = ROOT / "Informe" / "informe.tex"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


def test_semilog_elasticidad_equivalente_e_ic(df):
    m = modelo_semilog(df).model.fit(cov_type="HC3")
    p = df["checkout_price"].mean()
    b = m.params["checkout_price"]
    lo, hi = m.conf_int().loc["checkout_price"]
    assert b * p == pytest.approx(-1.0742, abs=5e-5)
    # IC de la elasticidad equivalente (beta x precio medio) del informe: [-1,086; -1,062]
    assert lo * p == pytest.approx(-1.086, abs=1e-3)
    assert hi * p == pytest.approx(-1.062, abs=1e-3)


@pytest.mark.parametrize("tau,coef,lo,hi", [(0.5, -0.7748, -0.786, -0.763), (0.9, -1.0068, -1.017, -0.997)])
def test_cuantiles_ic_analiticos(df, tau, coef, lo, hi):
    q = modelo_quantile_regression(df, tau)
    ci = q.conf_int().loc["log_checkout_price"]
    assert q.params["log_checkout_price"] == pytest.approx(coef, abs=5e-5)
    assert ci[0] == pytest.approx(lo, abs=5e-4)
    assert ci[1] == pytest.approx(hi, abs=5e-4)
