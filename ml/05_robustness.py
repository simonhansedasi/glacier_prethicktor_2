"""
05_robustness.py — Supplementary robustness checks.

1. GlaThiDa permutation test: verify one-tailed permutation p ≈ OLS p / 2
2. Reports power calculation explicitly under spatial autocorrelation reduction

Run from repo root:
    conda run -n glacierml python ml/05_robustness.py
"""
import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_PATH

SEED = 378
N_PERM = 10_000


def permutation_test(x, y, n_perm=N_PERM, seed=SEED):
    rng = np.random.default_rng(seed)
    obs_slope, _, _, obs_p_twotail, _ = stats.linregress(x, y)
    null_slopes = np.array([
        stats.linregress(rng.permutation(x), y)[0]
        for _ in range(n_perm)
    ])
    p_onetail = (null_slopes >= obs_slope).mean()
    return obs_slope, obs_p_twotail, p_onetail, null_slopes


# ---------------------------------------------------------------------------
# 1. GlaThiDa permutation test
# ---------------------------------------------------------------------------

def check_glathida_pvalue():
    pkl = os.path.join(RESULTS_PATH, 'analysis_df.pkl')
    if not os.path.exists(pkl):
        print(f'analysis_df.pkl not found at {pkl} — run 03_constraint_analysis.py first.')
        return

    df = pd.read_pickle(pkl)
    df = df.dropna(subset=['thickness_excess_m', 'water_depth_m'])
    x = df['water_depth_m'].values
    y = df['thickness_excess_m'].values

    slope, p_ols_2tail, p_perm_1tail, null_slopes = permutation_test(x, y)

    p_ols_1tail = p_ols_2tail / 2.0

    print('=== GlaThiDa depth-scaling: p-value equivalence check ===')
    print(f'  n                     = {len(df)}')
    print(f'  OLS slope             = {slope:.4f}')
    print(f'  OLS p (two-tailed)    = {p_ols_2tail:.4f}')
    print(f'  OLS p (one-tailed)    = {p_ols_1tail:.4f}  [= two-tailed / 2]')
    print(f'  Permutation p (one-tailed, {N_PERM:,} perms) = {p_perm_1tail:.4f}')
    diff = abs(p_ols_1tail - p_perm_1tail)
    print(f'  Difference            = {diff:.4f}  '
          f'({"consistent" if diff < 0.05 else "INCONSISTENT — investigate"})')
    print()


# ---------------------------------------------------------------------------
# 2. Power calculation under spatial autocorrelation
# ---------------------------------------------------------------------------

def power_under_autocorrelation():
    """
    Reproduce power calculation with explicit effective-n reduction.
    Under assumption of k independent spatial clusters, effective n = k.
    """
    pkl = os.path.join(RESULTS_PATH, 'modeled_analysis_df.pkl')
    if not os.path.exists(pkl):
        print('modeled_analysis_df.pkl not found — run 04_hab_modeled.py first.')
        return

    df = pd.read_pickle(pkl)
    df = df.dropna(subset=['thickness_excess_m', 'water_depth_m'])

    n_full   = len(df)
    sigma    = df['thickness_excess_m'].std()
    x        = df['water_depth_m'].values
    sx       = np.std(x, ddof=1)

    print('=== Power calculation under spatial autocorrelation reduction ===')
    print(f'  Full n = {n_full:,}, sigma = {sigma:.1f} m, SD(depth) = {sx:.1f} m')
    print()

    from scipy.stats import norm, t as tdist
    alpha_sig = 0.05

    for target_alpha in [0.10, 0.30]:
        print(f'  --- Target slope alpha = {target_alpha} ---')
        for n_eff in [n_full, 1000, 500, 200, 100, 50]:
            se = sigma / (sx * np.sqrt(n_eff - 1))
            ncp = target_alpha / se
            crit = tdist.ppf(1 - alpha_sig / 2, df=n_eff - 2)
            power = 1 - tdist.cdf(crit - ncp, df=n_eff - 2) + tdist.cdf(-crit - ncp, df=n_eff - 2)
            print(f'    n_eff={n_eff:6,}  SE={se:.4f}  power={power:.4f}')
        print()


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    check_glathida_pvalue()
    power_under_autocorrelation()
