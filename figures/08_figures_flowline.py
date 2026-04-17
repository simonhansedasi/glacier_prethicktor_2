"""
08_figures_flowline.py — Paper figure for the 1D flowline model.

Companion to ml/06_flowline_model.py. Self-contained: redefines the
physics inline so it can run without importing from ml/.

Two panels:

  Left  — Δh vs terminus depth for a range of glacier lengths
           (τ₀=100 kPa fixed), with the observed matched-pair mean
           (60.2 m at ~200 m depth) overlaid as a reference point.

  Right — Distribution of the effective slope α across the full
           (L, τ₀) ensemble grid, with the observed OLS estimate
           and bootstrap 95% CI marked.

Output: flowline_paper.png  (saved to FIGURES_PATH)

Run from the repo root:
    python figures/08_figures_flowline.py
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import FIGURES_PATH

# ---------------------------------------------------------------------------
# Physics constants
# ---------------------------------------------------------------------------
RHO_I  = 917.0
RHO_W  = 1025.0
GRAV   = 9.81

# Observed constraints
OBS_ALPHA  = 0.066
OBS_CI_LO  = -0.122
OBS_CI_HI  =  0.337
OBS_MEAN_DH =  60.2   # matched-pair mean excess (m)
OBS_MEAN_D  = 200.0   # representative depth for that excess (m)

# ---------------------------------------------------------------------------
# Analytical model  (exact; see ml/06_flowline_model.py for derivation)
# ---------------------------------------------------------------------------

def _B(L_m, tau_Pa):
    return np.sqrt(2.0 * tau_Pa * L_m / (RHO_I * GRAV))

def _A(d_m):
    return (RHO_W / RHO_I) * d_m

def dh_mean(L_m, tau_Pa, d_m):
    A = _A(d_m)
    B = _B(L_m, tau_Pa)
    water = (2.0 / (3.0 * B**2)) * ((A**2 + B**2)**1.5 - A**3)
    land  = (2.0 / 3.0) * B
    return water - land

# ---------------------------------------------------------------------------
# Build data
# ---------------------------------------------------------------------------

d_range = np.linspace(0, 600, 400)

# Panel A: Δh curves for several glacier lengths (τ₀ fixed)
tau_fixed = 100e3
L_vals_km = [5, 10, 20, 50, 100]
colors_A   = ['#c62828', '#ef6c00', '#2e7d32', '#1565c0', '#6a1b9a']

# Panel B: per-geometry α distribution
L_grid   = np.logspace(np.log10(1), np.log10(200), 20)
tau_grid = np.linspace(50, 200, 10)
d_fit    = np.linspace(10, 600, 100)

alphas = []
for L in L_grid:
    for tau in tau_grid:
        dh = dh_mean(L * 1e3, tau * 1e3, d_fit)
        if dh.max() < 1:
            continue
        # slope via linear regression on Δh vs d
        slope = np.polyfit(d_fit, dh, 1)[0]
        alphas.append(slope)
alphas = np.array(alphas)

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

# ------------------------------------------------------------------ #
# Left panel: Δh vs depth curves                                       #
# ------------------------------------------------------------------ #
ax = axes[0]

for L_km, col in zip(L_vals_km, colors_A):
    dh = dh_mean(L_km * 1e3, tau_fixed, d_range)
    ax.plot(d_range, dh, color=col, lw=1.8, label=f'$L = {L_km}$\,km')

# Reference line: HAB theory slope = 1 from origin
ax.plot(d_range, (RHO_W / RHO_I) * d_range, 'k--', lw=1.2, alpha=0.5,
        label=r'Flotation thickness $(\rho_w/\rho_i)\,d$')

# Observed matched-pair mean
ax.scatter([OBS_MEAN_D], [OBS_MEAN_DH], color='black', s=80, zorder=6,
           marker='*', label=f'Observed matched-pair mean\n'
                              r'($\overline{\Delta h}=60.2$\,m, $d\approx200$\,m)')
ax.axhline(OBS_MEAN_DH, color='black', lw=0.8, ls=':', alpha=0.5)

ax.set_xlabel('Terminus water depth $d$ (m)', fontsize=11)
ax.set_ylabel(r'Mean thickness excess $\overline{\Delta h}$ (m)', fontsize=11)
ax.set_title('A  Model prediction: $\\overline{\\Delta h}$ vs depth\n'
             r'($\tau_0 = 100$\,kPa)', fontsize=10)
ax.legend(fontsize=8.5, loc='upper left')
ax.set_xlim(0, 610)
ax.set_ylim(bottom=0)

# ------------------------------------------------------------------ #
# Right panel: α distribution                                          #
# ------------------------------------------------------------------ #
ax2 = axes[1]

frac_in_ci = ((alphas >= OBS_CI_LO) & (alphas <= OBS_CI_HI)).mean()

ax2.hist(alphas, bins=30, color='#1565c0', edgecolor='white',
         linewidth=0.5, density=True, alpha=0.8,
         label=f'Model geometries ($n={len(alphas)}$)')

# Shade CI region
ax2.axvspan(OBS_CI_LO, OBS_CI_HI, color='black', alpha=0.10,
            label=f'Observed 95% CI [{OBS_CI_LO:.3f}, {OBS_CI_HI:.3f}]')
ax2.axvline(OBS_ALPHA, color='black', lw=2.0,
            label=f'Observed OLS $\\alpha = {OBS_ALPHA:.3f}$')
ax2.axvline(OBS_CI_LO, color='black', lw=1.0, ls='--')
ax2.axvline(OBS_CI_HI, color='black', lw=1.0, ls='--')

# Model median
ax2.axvline(np.median(alphas), color='#1565c0', lw=1.5, ls='-.',
            label=f'Model median $\\alpha = {np.median(alphas):.2f}$')

ax2.text(0.97, 0.97,
         f'{100*frac_in_ci:.0f}\\% of model $\\alpha$\ninside observed CI',
         transform=ax2.transAxes, ha='right', va='top', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.85))

ax2.set_xlabel(r'Effective slope $\alpha = \overline{\Delta h}\,/\,d$', fontsize=11)
ax2.set_ylabel('Density', fontsize=11)
ax2.set_title('B  Distribution of model-implied $\\alpha$\n'
              '($L$ log-uniform 1--200\,km; '
              r'$\tau_0$ uniform 50--200\,kPa)', fontsize=10)
ax2.legend(fontsize=8.5, loc='upper right')

plt.suptitle(
    'Perfect-plasticity flowline model: expected depth-scaling of mean thickness excess',
    fontsize=10.5, y=1.01,
)
plt.tight_layout()

out = os.path.join(FIGURES_PATH, 'flowline_paper.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
print(f'\nModel α distribution:  median={np.median(alphas):.3f}, '
      f'10th–90th=[{np.percentile(alphas,10):.3f}, {np.percentile(alphas,90):.3f}]')
print(f'Fraction in observed CI [{OBS_CI_LO}, {OBS_CI_HI}]: {100*frac_in_ci:.1f}%')
