"""Comprobaciones de texto del informe que no requieren los datos."""
from pathlib import Path

TEX = Path(__file__).resolve().parents[1] / "Informe" / "informe.tex"


def test_sin_ic_cuantiles_bootstrap_no_reproducibles():
    tex = TEX.read_text(encoding="utf-8")
    for obsoleta in ["[-0,697, -0,649]", "$-0,697$", "$-1,021$", "se ejecutó en Azure",
                     "[-1,069, -1,056]", "$-0,9583$", "ratio $\\approx 1,03$", "V3.2.5)}"]:
        assert obsoleta not in tex, obsoleta


def test_bootstrap_b200_marcado_historico():
    tex = TEX.read_text(encoding="utf-8")
    i = tex.find("$B=200$")
    assert i > 0 and "referencia histórica" in tex[max(0, i - 400):i]
