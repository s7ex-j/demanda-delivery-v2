#!/usr/bin/env python3
"""
Tests V3.2.5: cada cifra IV / Shea / CF / interacción que reporta el informe debe
reproducirse desde el código y los artefactos certificados con tolerancias estrictas,
y el informe no debe contener las cifras obsoletas detectadas en la auditoría externa.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modeling import (construir_instrumentos, modelo_iv_2sls, modelo_iv_control_function,  # noqa: E402
                      modelo_interaccion_precio_promo)

DATA = ROOT / "data" / "processed" / "demanda_limpia.csv"
CERT = ROOT / "reports" / "certified"
TEX = ROOT / "Informe" / "informe.tex"
CTRL = "emailer_for_promotion + homepage_featured + hubo_aumento + C(category) + C(center_type)"


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def df_iv(df):
    return construir_instrumentos(df).dropna(subset=["log_precio_prom_otros_centros"]).copy()


def test_iv_principal_y_ee_agrupado(df_iv):
    from linearmodels.iv import IV2SLS
    res, d = modelo_iv_2sls(df_iv)
    assert len(d) == 456_544
    assert res.params["log_checkout_price"] == pytest.approx(-0.84138, abs=1e-4)
    assert res.params["emailer_for_promotion"] == pytest.approx(0.4928, abs=1e-4)
    assert res.params["homepage_featured"] == pytest.approx(0.6458, abs=1e-4)
    f = ("log_num_orders ~ 1 + " + CTRL +
         " + [log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]")
    cl = IV2SLS.from_formula(f, d).fit(cov_type="clustered", clusters=d["center_id"])
    assert cl.std_errors["log_checkout_price"] == pytest.approx(0.0502, abs=2e-4)


def test_hansen_j_especificacion_principal(df_iv):
    from linearmodels.iv import IVGMM
    f = ("log_num_orders ~ 1 + " + CTRL +
         " + [log_checkout_price ~ log_base_price + log_precio_prom_otros_centros]")
    j = IVGMM.from_formula(f, df_iv).fit(cov_type="robust").j_stat
    assert j.stat == pytest.approx(33.28, abs=0.01)
    jc = IVGMM.from_formula(f, df_iv, weight_type="clustered", clusters=df_iv["center_id"]).fit(
        cov_type="clustered", clusters=df_iv["center_id"]).j_stat
    assert jc.stat == pytest.approx(0.923, abs=0.01)
    assert jc.pval > 0.05


def test_shea_y_primera_etapa(df_iv):
    full = smf.ols(f"log_checkout_price ~ log_base_price + log_precio_prom_otros_centros + {CTRL}", df_iv).fit()
    red = smf.ols(f"log_checkout_price ~ {CTRL}", df_iv).fit()
    assert 1 - full.ssr / red.ssr == pytest.approx(0.9226, abs=1e-4)          # Shea (1 endógena)
    f_joint = ((red.ssr - full.ssr) / 2) / (full.ssr / full.df_resid)
    assert f_joint == pytest.approx(2.719e6, rel=1e-3)


def test_cf_valores_del_informe(df_iv):
    r = modelo_iv_control_function(df_iv)
    assert r["v_hat_coef"] == pytest.approx(1.0002, abs=1e-4)
    assert r["test_exogeneidad"]["F"] == pytest.approx(3177.94, abs=0.5)
    assert r["test_no_linealidad_condicional"]["F"] == pytest.approx(3335.41, abs=0.5)
    assert r["elasticidad_precio_medio"] == pytest.approx(-1.5041, abs=1e-4)
    assert r["elasticidad_precio_mediana"] == pytest.approx(-1.3945, abs=1e-4)
    assert r["coef_lineal_equivalente_2sls"] == pytest.approx(-0.84138, abs=1e-4)  # CF lineal == 2SLS


def test_interaccion_precio_promo(df):
    m, w, _ = modelo_interaccion_precio_promo(df)
    assert m.params["log_checkout_price_c"] == pytest.approx(-0.7481, abs=1e-4)
    assert m.params["log_checkout_price_c:emailer_for_promotion"] == pytest.approx(0.3284, abs=1e-4)
    assert m.params["log_checkout_price_c:homepage_featured"] == pytest.approx(-0.2204, abs=1e-4)
    assert w["estadistico_F"] == pytest.approx(652.28, abs=0.01)
    assert w["grados_libertad"][1] == 456_526


def test_bootstrap_iv_controles_certificado():
    meta = json.loads((CERT / "wild_bootstrap_iv_2sls_controles.json").read_text(encoding="utf-8"))
    coefs = np.load(CERT / "wild_bootstrap_iv_2sls_controles.npz")["coefs"]
    assert meta["coef_base"] == pytest.approx(-0.84138, abs=1e-4)      # especificación CON controles
    assert coefs.size == 2000 and np.isfinite(coefs).all()
    assert coefs.mean() == pytest.approx(meta["coef_media"], abs=1e-12)
    assert np.percentile(coefs, 2.5) == pytest.approx(meta["ic_95_inferior"], abs=1e-12)
    assert coefs.std(ddof=1) == pytest.approx(0.0504, abs=2e-4)


def test_bootstrap_cf_replicas_reproducen_certificado():
    old = json.loads((CERT / "bootstrap_vhat_canonical.json").read_text(encoding="utf-8"))
    rep = np.load(CERT / "bootstrap_cf_canonical_replicas.npz")
    v = rep["v_hat"]
    assert v.size == 5000 and np.isfinite(v).all()
    assert v.mean() == pytest.approx(old["v_hat"]["media"], abs=1e-6)
    assert np.percentile(v, 2.5) == pytest.approx(old["v_hat"]["ic95_inf"], abs=1e-6)
    assert np.percentile(v, 97.5) == pytest.approx(old["v_hat"]["ic95_sup"], abs=1e-6)
    assert rep["f_exog"].mean() == pytest.approx(old["test_exogeneidad"]["media"], abs=1e-3)


def test_informe_sin_cifras_obsoletas():
    tex = TEX.read_text(encoding="utf-8")
    for obsoleta in ["Shea Partial F", "7,03", "0,7517", "17\\,253", "3\\,198 ",
                     "</threeparttable>", "2\\,991,4", "3\\,492,3", "-0,9746", "0,5028", "0,6492",
                     "456\\,528", "1,22\\%", "2,77\\%"]:
        assert obsoleta not in tex, f"cifra obsoleta en el informe: {obsoleta}"
    # -0,9604 y J=324,5 solo pueden aparecer como referencia histórica de la especificación sin controles
    for line in tex.splitlines():
        if "-0,9604" in line or "324,5" in line:
            assert ("sin controles" in line or "anteriores" in line or "histórica" in line), line[:160]
