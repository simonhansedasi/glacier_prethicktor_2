"""
07_volume_estimate.py — First-order estimate of missing ice volume from absent HAB constraint.

Estimates how much ice volume the Farinotti consensus likely underestimates for
water-terminating glaciers, given that its models never enforce a flotation
boundary condition at calving termini.

Three methods are presented:

  Method 1 — Conservative observational bound:
    Apply the matched-pair mean thickness excess (60.2 m) uniformly to all
    water-terminating glaciers. Does not account for depth-scaling; treats all
    glaciers equally regardless of terminus depth. This is a lower bound because
    the matched-pair sample has a mean depth of ~200 m; deeper glaciers are
    expected to carry proportionally more missing thickness.

  Method 2 — Observed depth-scaling:
    Apply the observed OLS slope (α = 0.066) to each glacier's terminus depth.
    Also a lower bound — α = 0.066 is the underpowered point estimate from n=17.

  Method 3 — Flowline model (physically valid glaciers):
    Use the 1D perfect-plasticity analytical model to predict Δh for each
    glacier from its length (Lmax) and terminus depth. Only applied where the
    model is physically valid: A < B (flotation terminus thickness < land divide
    thickness). Glaciers failing this test are small glaciers in very deep water
    where either the model breaks down or GEBCO depth data is unreliable at the
    terminus location.

All volume estimates are converted to sea-level equivalent (SLE) using:
    SLE (mm) = volume (km³) × 0.9 × 1e6 / ocean_area (km²)
where ocean_area = 3.618e8 km².

NOTE: These are structural diagnostics from a missing boundary condition,
not validated volume corrections. The true bias depends on how much the
HAB signal propagates from terminus to glacier-mean thickness in real
calving systems — a question the present data cannot fully resolve.

Run from the repo root:
    python ml/07_volume_estimate.py
"""
import os
import sys
import pickle
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_PATH

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_I   = 917.0      # kg m⁻³  ice density
RHO_W   = 1025.0     # kg m⁻³  seawater density
GRAV    = 9.81       # m s⁻²
OCEAN   = 3.618e8    # km²  ocean surface area (for SLE conversion)

# Observed values from the paper (hardcoded for transparency)
MATCHED_PAIR_MEAN_DH = 60.2   # m — matched-pair mean thickness excess (n=17)
OBS_ALPHA            = 0.066  # observed OLS slope Δh ~ α·d (n=17, underpowered)


# ---------------------------------------------------------------------------
# Flowline model (1D perfect plasticity, exact analytical solution)
# See ml/06_flowline_model.py for derivation.
# ---------------------------------------------------------------------------

def divide_thickness(L_m, tau_Pa):
    """B = sqrt(2 τ_0 L / ρ_i g)  — land glacier divide thickness (m)."""
    return np.sqrt(2.0 * tau_Pa * L_m / (RHO_I * GRAV))


def flotation_thickness(d_m):
    """A = (ρ_w / ρ_i) d  — minimum terminus thickness for grounding (m)."""
    return (RHO_W / RHO_I) * d_m


def mean_thickness_excess(L_m, tau_Pa, d_m):
    """
    Exact mean thickness excess Δh̄ = h̄_water − h̄_land.

    Physical validity requires A < B:
      if terminus flotation thickness ≥ land divide thickness,
      the simple plasticity model is unphysical for that glacier.
    Returns NaN where invalid.
    """
    A = flotation_thickness(d_m)
    B = divide_thickness(L_m, tau_Pa)
    valid = A < B
    result = np.full_like(d_m, np.nan, dtype=float)
    result[valid] = (
        (2.0 / (3.0 * B[valid]**2))
        * ((A[valid]**2 + B[valid]**2)**1.5 - A[valid]**3)
        - (2.0 / 3.0) * B[valid]
    )
    return np.clip(result, 0, None)


def vol_to_sle(vol_km3):
    """Convert ice volume (km³) to sea-level equivalent (mm)."""
    return vol_km3 * 0.9 / OCEAN * 1e6


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    path = os.path.join(RESULTS_PATH, 'terminus_depths.pkl')
    with open(path, 'rb') as f:
        df = pickle.load(f)

    n_raw = len(df)
    df = df[
        df['water_depth_m'].notna() &
        df['Lmax'].notna() &
        (df['Lmax'] > 0)
    ].copy()

    print(f'Loaded terminus_depths.pkl: {n_raw:,} total water-terminating glaciers')
    print(f'After filtering (valid Lmax + depth): {len(df):,} glaciers')
    print(f'Dropped (missing Lmax or depth): {n_raw - len(df):,}')
    print()
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    df = load_data()

    L_m     = df['Lmax'].values          # metres (RGI Lmax is already in metres)
    d_m     = df['water_depth_m'].values
    area_m2 = df['Area'].values * 1e6    # km² → m²

    A = flotation_thickness(d_m)
    B = divide_thickness(L_m, tau_Pa=100e3)
    valid = A < B

    print(f'Glacier population summary:')
    print(f'  Total glaciers with data:   {len(df):,}')
    print(f'  Total area:                 {df["Area"].sum():,.0f} km²')
    print(f'  Mean terminus depth:        {d_m.mean():.1f} m')
    print(f'  Depth range:                {d_m.min():.0f}–{d_m.max():.0f} m')
    print(f'  Mean Lmax:                  {L_m.mean()/1e3:.1f} km')
    print()
    print(f'  Physically valid for flowline model (A < B, τ=100 kPa): {valid.sum():,}')
    print(f'  Invalid (small glacier / deep water — A ≥ B):            {(~valid).sum():,}')
    print(f'    Invalid: mean depth {d_m[~valid].mean():.0f} m, '
          f'mean Lmax {L_m[~valid].mean()/1e3:.1f} km, '
          f'mean area {df["Area"].values[~valid].mean():.1f} km²')
    print()

    # ------------------------------------------------------------------
    # Method 1: flat matched-pair correction
    # ------------------------------------------------------------------
    print('=' * 60)
    print('METHOD 1 — Conservative observational bound')
    print(f'  Flat Δh = {MATCHED_PAIR_MEAN_DH} m (matched-pair mean, n=17)')
    print('=' * 60)
    vol1 = MATCHED_PAIR_MEAN_DH * area_m2.sum() / 1e9
    print(f'  Missing volume: {vol1:,.0f} km³')
    print(f'  SLE equivalent: {vol_to_sle(vol1):.1f} mm')
    print(f'  As % of Farinotti total (~170,000 km³): {vol1/170000*100:.1f}%')
    print()

    # ------------------------------------------------------------------
    # Method 2: observed slope
    # ------------------------------------------------------------------
    print('=' * 60)
    print('METHOD 2 — Observed depth-scaling (underpowered lower bound)')
    print(f'  Δh = α·d, α = {OBS_ALPHA} (OLS slope from n=17; known underestimate)')
    print('=' * 60)
    dh2  = OBS_ALPHA * d_m
    vol2 = (dh2 * area_m2).sum() / 1e9
    print(f'  Mean Δh: {dh2.mean():.1f} m')
    print(f'  Missing volume: {vol2:,.0f} km³')
    print(f'  SLE equivalent: {vol_to_sle(vol2):.1f} mm')
    print(f'  As % of Farinotti total: {vol2/170000*100:.1f}%')
    print()

    # ------------------------------------------------------------------
    # Method 3: flowline model (valid glaciers only)
    # ------------------------------------------------------------------
    print('=' * 60)
    print('METHOD 3 — Flowline model (physically valid glaciers only)')
    print('  Invalid glaciers excluded (set to zero) — conservative.')
    print('=' * 60)
    dh3 = mean_thickness_excess(L_m, 100e3, d_m)
    dh3_valid = np.where(valid, dh3, 0.0)
    vol3 = (dh3_valid * area_m2).sum() / 1e9
    print(f'  Valid glacier mean Δh: {np.nanmean(dh3[valid]):.1f} m')
    print(f'  Missing volume (valid only): {vol3:,.0f} km³')
    print(f'  SLE equivalent: {vol_to_sle(vol3):.1f} mm')
    print(f'  As % of Farinotti total: {vol3/170000*100:.1f}%')
    print()

    # Sensitivity to yield stress
    print('  Sensitivity to basal yield stress (valid glaciers only):')
    for tau_kPa in [50, 75, 100, 150, 200]:
        dh_s = mean_thickness_excess(L_m, tau_kPa * 1e3, d_m)
        v_s  = valid_s = flotation_thickness(d_m) < divide_thickness(L_m, tau_kPa * 1e3)
        dh_s = np.where(v_s, dh_s, 0.0)
        vol_s = (dh_s * area_m2).sum() / 1e9
        print(f'    τ₀ = {tau_kPa:>3} kPa: '
              f'valid n={v_s.sum():,}, '
              f'missing vol={vol_s:,.0f} km³, '
              f'SLE={vol_to_sle(vol_s):.1f} mm')
    print()

    # ------------------------------------------------------------------
    # Depth-bin breakdown (method 3, tau=100 kPa)
    # ------------------------------------------------------------------
    print('=' * 60)
    print('DEPTH-BIN BREAKDOWN — Method 3 (τ=100 kPa, valid glaciers)')
    print('=' * 60)
    bins   = [0, 50, 100, 200, 400, np.inf]
    labels = ['0–50 m', '50–100 m', '100–200 m', '200–400 m', '>400 m']
    df['depth_bin']   = pd.cut(d_m, bins=bins, labels=labels)
    df['dh_flowline'] = dh3_valid
    df['vol_km3']     = dh3_valid * area_m2 / 1e9

    summary = df.groupby('depth_bin', observed=True).agg(
        n           = ('RGIId',           'count'),
        mean_depth  = ('water_depth_m',   'mean'),
        mean_dh     = ('dh_flowline',     lambda x: x[x > 0].mean()),
        total_area  = ('Area',            'sum'),
        missing_vol = ('vol_km3',         'sum'),
    ).round(1)
    summary['SLE_mm'] = vol_to_sle(summary['missing_vol']).round(2)
    print(summary.to_string())
    print()
    print(f'  Total missing volume: {df["vol_km3"].sum():,.0f} km³')
    print(f'  Total SLE:            {vol_to_sle(df["vol_km3"].sum()):.1f} mm')
    print()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('=' * 60)
    print('SUMMARY')
    print('=' * 60)
    print(f'  Method 1 (obs flat):     {vol1:>8,.0f} km³   ~{vol_to_sle(vol1):.0f} mm SLE')
    print(f'  Method 2 (obs slope):    {vol2:>8,.0f} km³   ~{vol_to_sle(vol2):.0f} mm SLE')
    print(f'  Method 3 (flowline):     {vol3:>8,.0f} km³   ~{vol_to_sle(vol3):.0f} mm SLE')
    print()
    print(f'  Farinotti total (~170,000 km³):')
    print(f'    Method 1 represents {vol1/170000*100:.1f}% of total glacier volume')
    print(f'    Method 3 represents {vol3/170000*100:.1f}% of total glacier volume')
    print()
    print('NOTE: These are first-order diagnostics of a structural omission,')
    print('not validated volume corrections. Method 3 (flowline) is anchored')
    print('by the matched-pair observation at the representative geometry')
    print('(L=20km, d=200m → Δh=60.3m, matching the observed 60.2m).')
