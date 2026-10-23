#!/usr/bin/env python3
"""
Script to reproduce the F=3607.28 nested F-test statistic.

The test compares:
- Restricted model: OLS log-log with controls (log_checkout_price + controls)
- Unrestricted model: RCS OLS with controls (rcs_1, rcs_2, rcs_3 + intercept + controls)

The RCS specification uses 4 explicit knots at P5/P35/P65/P95 of log_checkout_price,
3 spline bases (rcs_1 linear, rcs_2, rcs_3 nonlinear), separate model intercept,
constraints='center', and -1 to remove redundant constant.

H0: rcs_2 = 0 and rcs_3 = 0 (elasticity is constant)
H1: At least one of rcs_2, rcs_3 is non-zero (non-linear elasticity)

Expected: F = 3607.28, df_num = 2, df_den = 456526, p ≈ 0
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from patsy import dmatrix, build_design_matrices
from pathlib import Path


def build_rcs_design(log_price_full: np.ndarray):
    """Build canonical RCS design on full sample (fixed knots, fixed design_info)."""
    knots = np.percentile(log_price_full, [5, 35, 65, 95])
    k0, k1, k2, k3 = knots

    formula = (f"cr(log_checkout_price, "
               f"knots=({float(k1):.15f}, {float(k2):.15f}), "
               f"lower_bound={float(k0):.15f}, "
               f"upper_bound={float(k3):.15f}, "
               "constraints='center') - 1")

    Z = dmatrix(formula, {'log_checkout_price': log_price_full}, return_type='dataframe')
    Z.columns = ['rcs_1', 'rcs_2', 'rcs_3']
    return Z.design_info, knots


def run_f_test():
    """Run the nested F-test and return results."""
    # Load FULL dataset
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    DATA = PROJECT_ROOT / 'data' / 'processed' / 'demanda_limpia.csv'
    df = pd.read_csv(DATA)

    print(f"Dataset shape: {df.shape}")
    print(f"Expected N: 456,548")

    # Build RCS design on FULL sample (critical: knots fixed on full sample)
    log_price_full = df['log_checkout_price'].to_numpy()
    design_info, knots = build_rcs_design(log_price_full)

    print(f"\nRCS knots (log): {knots}")
    print(f"RCS knots (price): {[float(np.exp(k)) for k in knots]}")

    # Apply RCS to full dataset
    S = build_design_matrices([design_info],
                               {'log_checkout_price': log_price_full},
                               return_type='dataframe')[0]
    S.columns = ['rcs_1', 'rcs_2', 'rcs_3']
    for c in S.columns:
        df[c] = S[c].to_numpy()

    # Restricted model: OLS log-log with controls
    ll_formula = ('log_num_orders ~ log_checkout_price + emailer_for_promotion '
                  '+ homepage_featured + hubo_aumento + C(category) + C(center_type)')
    ll_model = smf.ols(ll_formula, data=df).fit()

    # Unrestricted model: RCS OLS with controls (3 bases + intercept + controls)
    rcs_formula = ('log_num_orders ~ 1 + rcs_1 + rcs_2 + rcs_3 '
                   '+ emailer_for_promotion + homepage_featured + hubo_aumento '
                   '+ C(category) + C(center_type)')
    rcs_model = smf.ols(rcs_formula, data=df).fit()

    # Manual F-test calculation (RSS difference / q) / (RSS_u / (n - k_u))
    RSS_r = ll_model.ssr
    RSS_u = rcs_model.ssr
    n = ll_model.nobs
    k_u = len(rcs_model.params)
    q = 2  # number of restrictions (rcs_2 = 0, rcs_3 = 0)

    F_stat = (RSS_r - RSS_u) / q / (RSS_u / (n - k_u))

    # Also compute via wald_test on unrestricted model
    param_names = list(rcs_model.params.index)
    R = np.zeros((2, len(rcs_model.params)))
    R[0, param_names.index('rcs_2')] = 1
    R[1, param_names.index('rcs_3')] = 1
    wald = rcs_model.wald_test(R)
    F_wald = float(wald.fvalue.item()) if hasattr(wald.fvalue, 'item') else float(wald.fvalue)

    # Results
    results = {
        'F_manual': F_stat,
        'F_wald': F_wald,
        'df_num': q,
        'df_den': n - k_u,
        'RSS_restricted': RSS_r,
        'RSS_unrestricted': RSS_u,
        'n_obs': n,
        'k_unrestricted': k_u,
        'log_log_coef': float(ll_model.params.get('log_checkout_price', np.nan)),
        'rcs_1_coef': float(rcs_model.params.get('rcs_1', np.nan)),
        'rcs_2_coef': float(rcs_model.params.get('rcs_2', np.nan)),
        'rcs_3_coef': float(rcs_model.params.get('rcs_3', np.nan)),
        'knots_log': knots.tolist(),
        'knots_price': [float(np.exp(k)) for k in knots],
    }

    print(f"\n=== NESTED F-TEST RESULTS ===")
    print(f"F (manual RSS diff): {results['F_manual']:.6f}")
    print(f"F (wald_test OLS):   {results['F_wald']:.6f}")
    print(f"Expected (paper):    3607.28")
    print(f"Difference:          {abs(results['F_manual'] - 3607.28):.6f}")
    print(f"df_num: {results['df_num']}, df_den: {results['df_den']}")
    print(f"N obs: {results['n_obs']:.0f}")
    print(f"RSS restricted: {results['RSS_restricted']:.6f}")
    print(f"RSS unrestricted: {results['RSS_unrestricted']:.6f}")
    print(f"Log-log coef: {results['log_log_coef']:.6f}")
    print(f"RCS coefs: rcs_1={results['rcs_1_coef']:.6f}, rcs_2={results['rcs_2_coef']:.6f}, rcs_3={results['rcs_3_coef']:.6f}")

    # Validation
    match = abs(results['F_manual'] - 3607.28) < 0.01
    print(f"\nREPRODUCTION {'SUCCESSFUL' if match else 'FAILED'}: F matches paper within 0.01")

    return results


if __name__ == '__main__':
    run_f_test()