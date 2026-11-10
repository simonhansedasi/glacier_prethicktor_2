"""
06_flowline_model.py — 1D perfect-plasticity flowline model.

Tests whether a flotation boundary condition at the calving terminus
produces a glacier-mean thickness excess that scales with water depth,
validating the Lc/L scaling argument used in Section 2 of the paper.

Physics
-------
Perfect plasticity (constant basal yield stress τ_0) on a horizontal bed.
The force balance gives an exact analytical thickness profile:

    h(x)² = h_T² + (2 τ_0 / ρ_i g)(L − x)

where x ∈ [0, L] is distance from the divide, and h_T is terminus
ice thickness.

Boundary conditions:
  Land glacier   :  h_T = 0
  Water glacier  :  h_T = (ρ_w / ρ_i) · d   (flotation minimum)

Mean thickness (exact analytical integral):
  h̄_land  = (2/3) B
  h̄_water = (2 / 3B²) [(A² + B²)^{3/2} − A³]

where
  A = (ρ_w / ρ_i) · d          (flotation terminus thickness)
  B = sqrt(2 τ_0 L / ρ_i g)    (divide thickness for land glacier)

Thickness excess:
  Δh = h̄_water − h̄_land

This is computed over an ensemble of realistic glacier geometries
(length L, yield stress τ_0, terminus depth d) to obtain the
distribution of the effective slope α = dΔh/dd and the implied Lc/L.

Key question for the paper
--------------------------
Is the observed α ≈ 0.07 (with 95% CI [−0.122, 0.337]) physically
consistent with this mechanism? i.e., does the perfect-plasticity
model predict α values that overlap the observed CI?

Outputs
-------
  flowline_results.png    — 4-panel figure (saved to FIGURES_PATH)
  flowline_summary.csv    — key numbers (saved to RESULTS_PATH)

Key printed numbers:
  • Δh at representative (L, τ_0, d) combinations
  • Ensemble-wide regression slope and distribution of α
  • Fraction of ensemble with α inside observed CI [−0.122, 0.337]

Run from the repo root:
    python ml/06_flowline_model.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_PATH, FIGURES_PATH

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
RHO_I   = 917.0       # kg m⁻³  ice density
RHO_W   = 1025.0      # kg m⁻³  seawater (use 1000 for lake)
GRAV    = 9.81        # m s⁻²

# Observed constraints from the paper (used for comparison)
OBS_ALPHA   = 0.066            # OLS slope from n=17 glaciers
OBS_CI_LO   = -0.122           # bootstrap 95% CI lower
OBS_CI_HI   =  0.337           # bootstrap 95% CI upper
OBS_MEAN_DH =  60.2            # matched-pair mean excess (m)
OBS_MEAN_D  =  220.0           # rough mean terminus depth in observed sample (m)

SEED = 378
rng  = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Core analytical model
# ---------------------------------------------------------------------------

def divide_thickness(L_m, tau_Pa):
    """B = sqrt(2 τ_0 L / ρ_i g)  — divide thickness for land glacier (m)."""
    return np.sqrt(2.0 * tau_Pa * L_m / (RHO_I * GRAV))


def flotation_thickness(d_m, rho_w=RHO_W):
    """A = (ρ_w / ρ_i) d  — minimum ice thickness at tidal terminus (m)."""
    return (rho_w / RHO_I) * d_m


def mean_thickness_land(B):
    """h̄_land = (2/3) B."""
    return (2.0 / 3.0) * B


def mean_thickness_water(A, B):
    """
    h̄_water = (2 / 3B²) [(A² + B²)^{3/2} − A³]

    Derived by integrating h(u) = sqrt(A² + B²u) over u ∈ [0, 1]
    (u = 1 − x/L).
    """
    return (2.0 / (3.0 * B**2)) * ((A**2 + B**2)**1.5 - A**3)


def thickness_excess(L_m, tau_Pa, d_m, rho_w=RHO_W):
    """Δh = h̄_water − h̄_land  (m)."""
    A  = flotation_thickness(d_m, rho_w)
    B  = divide_thickness(L_m, tau_Pa)
    return mean_thickness_water(A, B) - mean_thickness_land(B)


def thickness_profile(L_m, tau_Pa, d_m, n_pts=500, rho_w=RHO_W):
    """
    Return x-coordinates and thickness profiles for land and water glaciers.
    x = 0 is divide, x = L is terminus.
    """
    A  = flotation_thickness(d_m, rho_w)
    B  = divide_thickness(L_m, tau_Pa)
    x  = np.linspace(0, L_m, n_pts)
    u  = 1.0 - x / L_m               # u = 1 at divide, 0 at terminus
    h_land  = B * np.sqrt(u)
    h_water = np.sqrt(A**2 + B**2 * u)
    return x, h_land, h_water


def coupling_length(L_m, tau_Pa, d_m, frac=0.5, rho_w=RHO_W):
    """
    Distance from terminus at which the thickness excess falls to
    `frac` of its terminus value.  Computed by bisection.
    Returns Lc in metres.
    """
    A = flotation_thickness(d_m, rho_w)
    B = divide_thickness(L_m, tau_Pa)

    # excess at terminus = A (h_water(L) - h_land(L) = A - 0)
    target = frac * A

    def excess_at_dist(dist_from_term):
        u = dist_from_term / L_m
        return np.sqrt(A**2 + B**2 * u) - B * np.sqrt(u)

    # find where excess_at_dist == target  (it's monotone increasing from 0)
    # search on [0, L]
    if excess_at_dist(L_m) <= target:
        return L_m   # signal never decays to target — whole glacier length

    lo, hi = 0.0, L_m
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if excess_at_dist(mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def build_ensemble(n=50_000):
    """
    Sample (L, τ_0, d) from realistic distributions.

    L   : log-uniform 1–200 km  (RGI water-terminating glaciers span ~1–180 km)
    τ_0 : uniform 50–200 kPa    (temperate to polythermal; Cuffey & Paterson §7.3)
    d   : uniform 10–600 m      (observed depth range in dataset)
    """
    print(f'  Sampling {n:,} (L, τ₀, d) combinations ...')
    L_m   = np.exp(rng.uniform(np.log(1e3), np.log(200e3), n))   # m
    tau   = rng.uniform(50e3, 200e3, n)                           # Pa
    d_m   = rng.uniform(10, 600, n)                               # m

    print('  Computing Δh (vectorised) ...')
    dh    = thickness_excess(L_m, tau, d_m)
    h_bar_land = mean_thickness_land(divide_thickness(L_m, tau))

    print(f'  Done. Mean Δh = {dh.mean():.1f} m, mean α_local = {(dh/d_m).mean():.3f}')
    return pd.DataFrame({
        'L_km':        L_m / 1e3,
        'tau_kPa':     tau / 1e3,
        'd_m':         d_m,
        'dh_m':        dh,
        'h_bar_land':  h_bar_land,
        'alpha_local': dh / d_m,           # local ratio Δh/d
    })


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def fit_slope(d, dh):
    slope, intercept, r, p, se = stats.linregress(d, dh)
    return slope, intercept, r**2, p


def per_glacier_slope():
    """
    For each (L, τ) bin, fit slope of Δh vs d across the d range.
    Returns a DataFrame of per-geometry α estimates.
    """
    L_vals   = np.logspace(np.log10(1), np.log10(200), 20)
    tau_vals = np.linspace(50, 200, 10)
    n_total  = len(L_vals) * len(tau_vals)

    rows = []
    with tqdm(total=n_total, desc='  Per-geometry slopes', unit='combo') as pbar:
        for L in L_vals:
            for tau in tau_vals:
                d_vals  = np.linspace(10, 600, 100)
                dh_vals = thickness_excess(L * 1e3, tau * 1e3, d_vals)
                if dh_vals.max() < 1:
                    pbar.update(1)
                    continue
                s, _, r2, _ = fit_slope(d_vals, dh_vals)
                lc = coupling_length(L * 1e3, tau * 1e3, 200.0)  # at d=200m
                rows.append({
                    'L_km':    L,
                    'tau_kPa': tau,
                    'alpha':   s,
                    'lc_L':    lc / (L * 1e3),
                    'r2':      r2,
                })
                pbar.update(1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def make_figures(df_ens, df_geom):
    print('  Rendering 4-panel figure ...')
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle('Perfect-plasticity flowline model: flotation BC → mean-thickness excess',
                 fontsize=11)

    # ------------------------------------------------------------------ #
    # Panel A: thickness profiles for a representative glacier            #
    # ------------------------------------------------------------------ #
    ax = axes[0, 0]
    L_rep, tau_rep = 20e3, 100e3   # 20 km, 100 kPa
    for d, col in [(100, '#2196F3'), (200, '#00897B'), (400, '#E53935')]:
        x, h_land, h_water = thickness_profile(L_rep, tau_rep, d)
        ax.plot(x / 1e3, h_land,  color='gray',  lw=1.5, ls='--',
                label='Land (d=0)' if d == 100 else '_nolegend_')
        ax.plot(x / 1e3, h_water, color=col, lw=1.8,
                label=f'Tidewater d={d} m')
        ax.axhline((RHO_W/RHO_I)*d, color=col, lw=0.8, ls=':')

    ax.set_xlabel('Distance from divide (km)')
    ax.set_ylabel('Ice thickness (m)')
    ax.set_title(f'A  Thickness profiles\n(L={L_rep/1e3:.0f} km, τ₀={tau_rep/1e3:.0f} kPa)')
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel B: Δh vs d for a range of glacier lengths                     #
    # ------------------------------------------------------------------ #
    ax = axes[0, 1]
    d_range = np.linspace(0, 600, 300)
    tau_fixed = 100e3
    colors_b  = plt.cm.viridis_r(np.linspace(0.1, 0.9, 6))
    for L_km, col in zip([2, 5, 10, 20, 50, 100], colors_b):
        dh = thickness_excess(L_km * 1e3, tau_fixed, d_range)
        ax.plot(d_range, dh, color=col, lw=1.6, label=f'L={L_km} km')

    # Observed matched-pair mean (horizontal reference)
    ax.axhline(OBS_MEAN_DH, color='black', lw=1.2, ls='--',
               label=f'Obs. matched-pair mean ({OBS_MEAN_DH:.0f} m)')
    ax.scatter([OBS_MEAN_D], [OBS_MEAN_DH], color='black', s=60, zorder=5)

    ax.set_xlabel('Terminus water depth (m)')
    ax.set_ylabel('Mean thickness excess Δh (m)')
    ax.set_title(f'B  Δh vs depth for varying L\n(τ₀={tau_fixed/1e3:.0f} kPa fixed)')
    ax.legend(fontsize=7.5, ncol=2)

    # ------------------------------------------------------------------ #
    # Panel C: distribution of effective α across ensemble                #
    # ------------------------------------------------------------------ #
    ax = axes[1, 0]

    # per-geometry α from the grid (cleaner than 50k-point ensemble)
    alphas = df_geom['alpha'].values
    ax.hist(alphas, bins=40, color='steelblue', edgecolor='white',
            linewidth=0.4, density=True, alpha=0.85)

    # Observed CI and point estimate
    ax.axvline(OBS_ALPHA,  color='black', lw=2.0, ls='-',
               label=f'Obs. OLS α = {OBS_ALPHA:.3f}')
    ax.axvline(OBS_CI_LO,  color='black', lw=1.2, ls='--',
               label=f'95% CI [{OBS_CI_LO:.3f}, {OBS_CI_HI:.3f}]')
    ax.axvline(OBS_CI_HI,  color='black', lw=1.2, ls='--')
    ax.axvspan(OBS_CI_LO, OBS_CI_HI, color='black', alpha=0.08)

    frac_in_ci = ((alphas >= OBS_CI_LO) & (alphas <= OBS_CI_HI)).mean()
    ax.text(0.97, 0.95, f'{100*frac_in_ci:.0f}% of model α\ninside observed CI',
            transform=ax.transAxes, ha='right', va='top', fontsize=8.5,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Effective slope α = Δh / d (m m⁻¹)')
    ax.set_ylabel('Density')
    ax.set_title('C  Distribution of model-implied α\nacross (L, τ₀) ensemble')
    ax.legend(fontsize=8)

    # ------------------------------------------------------------------ #
    # Panel D: implied Lc/L vs glacier length                             #
    # ------------------------------------------------------------------ #
    ax = axes[1, 1]

    sc = ax.scatter(df_geom['L_km'], df_geom['lc_L'],
                    c=df_geom['tau_kPa'], cmap='plasma',
                    s=18, alpha=0.7, vmin=50, vmax=200)
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label('Yield stress τ₀ (kPa)', fontsize=8.5)

    ax.axhline(OBS_ALPHA, color='black', lw=1.5, ls='--',
               label=f'Obs. α = {OBS_ALPHA:.3f}  (≡ Lc/L if theory holds)')
    ax.set_xscale('log')
    ax.set_xlabel('Glacier length L (km)')
    ax.set_ylabel('Coupling-length fraction Lc / L')
    ax.set_title('D  Model Lc/L vs glacier length\n(Lc = distance where excess falls to 50% of terminus value)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = os.path.join(FIGURES_PATH, 'flowline_results.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {out}')
    return out


# ---------------------------------------------------------------------------
# Summary numbers
# ---------------------------------------------------------------------------

def print_and_save_summary(df_ens, df_geom):
    print('\n' + '='*65)
    print('FLOWLINE MODEL — KEY RESULTS')
    print('='*65)

    # 1. Spot checks at representative glaciers
    print('\n--- Δh at representative glaciers (τ₀=100 kPa) ---')
    for L_km in [5, 10, 20, 50, 100]:
        for d in [100, 200, 400]:
            dh = thickness_excess(L_km * 1e3, 100e3, d)
            print(f'  L={L_km:>4} km, d={d:>4} m → Δh = {dh:6.1f} m  '
                  f'(α_local = {dh/d:.3f})')

    # 2. Ensemble regression
    print('\n--- Ensemble-wide regression (n=50,000 samples) ---')
    print('  Fitting ...')
    s, intercept, r2, p = fit_slope(df_ens['d_m'].values, df_ens['dh_m'].values)
    print(f'  Slope α   = {s:.4f}')
    print(f'  Intercept = {intercept:.1f} m')
    print(f'  R²        = {r2:.3f}')
    print(f'  p         = {p:.2e}')

    # 3. Distribution of per-geometry α
    alphas = df_geom['alpha'].values
    print(f'\n--- Per-geometry α distribution (n={len(alphas)} grid points) ---')
    print(f'  Median α         = {np.median(alphas):.3f}')
    print(f'  Mean α           = {alphas.mean():.3f}')
    print(f'  10th–90th pctile = [{np.percentile(alphas,10):.3f}, '
          f'{np.percentile(alphas,90):.3f}]')
    frac_in_ci = ((alphas >= OBS_CI_LO) & (alphas <= OBS_CI_HI)).mean()
    print(f'  Fraction inside observed CI [{OBS_CI_LO}, {OBS_CI_HI}]:  '
          f'{100*frac_in_ci:.1f}%')

    # 4. At observed mean depth, what range of Δh does the model predict?
    print(f'\n--- Δh predicted at obs. mean depth (d={OBS_MEAN_D:.0f} m) ---')
    dh_at_obs = thickness_excess(
        np.logspace(np.log10(1e3), np.log10(200e3), 200),
        np.full(200, 100e3),
        OBS_MEAN_D,
    )
    print(f'  Range across L=1–200 km (τ₀=100 kPa):  '
          f'[{dh_at_obs.min():.1f}, {dh_at_obs.max():.1f}] m')
    print(f'  Observed matched-pair mean Δh: {OBS_MEAN_DH} m')

    # 5. Lc/L statistics
    lc_L = df_geom['lc_L'].values
    print(f'\n--- Coupling-length fraction Lc/L ---')
    print(f'  Median Lc/L = {np.median(lc_L):.3f}')
    print(f'  Mean Lc/L   = {lc_L.mean():.3f}')
    print(f'  10th–90th   = [{np.percentile(lc_L,10):.3f}, '
          f'{np.percentile(lc_L,90):.3f}]')
    print(f'  Observed α  = {OBS_ALPHA:.3f}  (would imply Lc/L ≈ {OBS_ALPHA:.3f} '
          f'if linear)')

    # 6. Verdict
    print(f'\n--- Interpretation ---')
    if frac_in_ci > 0.5:
        print('  ✓ Majority of model geometries produce α inside the observed CI.')
        print('    The Lc/L scaling is physically consistent with perfect plasticity.')
    elif frac_in_ci > 0.1:
        print('  ~ Substantial fraction of model geometries overlap the observed CI,')
        print('    but the central model prediction is above the observed estimate.')
        print('    Observed α could be under-estimated due to n=17 noise.')
    else:
        print('  ✗ Few model geometries produce α inside the observed CI.')
        print('    Either the mechanism is incorrect or mean-thickness attenuation')
        print('    is more severe than perfect plasticity predicts.')

    # Save CSV
    summary = pd.DataFrame([{
        'ensemble_slope_alpha': round(s, 4),
        'ensemble_r2':          round(r2, 3),
        'per_geom_median_alpha': round(float(np.median(alphas)), 3),
        'per_geom_mean_alpha':   round(float(alphas.mean()), 3),
        'frac_in_obs_ci':        round(float(frac_in_ci), 3),
        'median_lc_L':           round(float(np.median(lc_L)), 3),
    }])
    out = os.path.join(RESULTS_PATH, 'flowline_summary.csv')
    summary.to_csv(out, index=False)
    print(f'\nSaved {out}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('Building ensemble ...')
    df_ens  = build_ensemble(n=50_000)

    print('Computing per-geometry slopes ...')
    df_geom = per_glacier_slope()

    print('Making figures ...')
    make_figures(df_ens, df_geom)

    print_and_save_summary(df_ens, df_geom)
