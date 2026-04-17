"""
07_figures_matched_pairs.py — Matched-pair figure for the observed analysis.

Companion to ml/05_robustness.py (Test 7).  Re-runs the k=5 nearest-neighbour
matching in standardised feature space and produces two panels:

  Left  — connected dot plot: each water-terminating glacier (coloured by
           terminus depth) shown alongside its 5 nearest land analogues (grey),
           both plotted on an absolute thickness axis.  Sorted by ascending
           terminus water depth so depth structure is visible.

  Right — dot plot of matched-pair differences (water − mean land),
           one point per water glacier, coloured by terminus depth.
           Horizontal reference line at zero.  Mean excess and t-test
           result annotated.

Output:
  matched_pairs.png   (saved to FIGURES_PATH)

Run from the repo root:
    python figures/07_figures_matched_pairs.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_PATH, RGI_PATH, RESULTS_PATH, FIGURES_PATH
from glacierml import load_RGI, coregister_data

SEED    = 378
KNN_K   = 5
LABEL   = 'Thickness'
FEATURES = ['CenLat', 'CenLon', 'Slope', 'Zmin', 'Zmed', 'Zmax', 'Area', 'Aspect', 'Lmax']

OUT_PATH = os.path.join(FIGURES_PATH, 'matched_pairs.png')


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_data():
    all_training = coregister_data(DATA_PATH, coregistration='4')
    rgi = load_RGI(RGI_PATH)
    all_training = all_training.merge(rgi[['RGIId', 'TermType']], on='RGIId', how='left')

    land_df  = all_training[all_training['TermType'].isin([0, 9])].copy().reset_index(drop=True)
    water_df = all_training[all_training['TermType'].isin([1, 2])].copy().reset_index(drop=True)

    depths = pd.read_pickle(os.path.join(RESULTS_PATH, 'glathida_water_depths.pkl'))
    water_df = water_df.merge(depths[['RGIId', 'water_depth_m']], on='RGIId', how='left')

    return land_df, water_df


# ---------------------------------------------------------------------------
# Run matching
# ---------------------------------------------------------------------------

def run_matching(land_df, water_df):
    scaler = StandardScaler()
    X_land_sc  = scaler.fit_transform(land_df[FEATURES].values)
    X_water_sc = scaler.transform(water_df[FEATURES].values)

    dists   = cdist(X_water_sc, X_land_sc, metric='euclidean')
    knn_idx = np.argsort(dists, axis=1)[:, :KNN_K]

    records = []
    for i, row_idx in enumerate(knn_idx):
        h_water  = water_df[LABEL].iloc[i]
        h_land   = land_df[LABEL].iloc[row_idx].values   # shape (KNN_K,)
        depth    = water_df['water_depth_m'].iloc[i]
        tt       = water_df['TermType'].iloc[i]
        records.append({
            'h_water':    h_water,
            'h_land_mean': h_land.mean(),
            'h_land_vals': h_land,
            'excess':     h_water - h_land.mean(),
            'water_depth': depth,
            'TermType':   tt,
        })

    return records


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_figure(records):
    # Sort by water depth (NaN last)
    records_sorted = sorted(
        records,
        key=lambda r: r['water_depth'] if not np.isnan(r['water_depth']) else np.inf,
    )

    n       = len(records_sorted)
    excesses = np.array([r['excess'] for r in records_sorted])
    depths   = np.array([r['water_depth'] for r in records_sorted])

    t_stat, p_val = stats.ttest_1samp(excesses, 0)

    # Colour map: terminus depth  (viridis, depth 0–500 m)
    d_valid = depths[~np.isnan(depths)]
    vmin, vmax = 0, max(d_valid.max(), 50)
    norm   = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap   = cm.viridis_r
    scalar_map = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={'width_ratios': [1.6, 1]})

    # ------------------------------------------------------------------ #
    # Left panel: connected dot plot (absolute thickness)                 #
    # ------------------------------------------------------------------ #
    ax = axes[0]
    x_ticks = []
    x_labels = []

    for i, rec in enumerate(records_sorted):
        xpos   = i + 1
        depth  = rec['water_depth']
        color  = scalar_map.to_rgba(depth) if not np.isnan(depth) else 'dimgray'

        # Draw land neighbours first (grey, small)
        for h_l in rec['h_land_vals']:
            ax.scatter(xpos, h_l, color='#aaaaaa', s=18, zorder=2, alpha=0.7)

        # Connect water point to land mean with a vertical line
        ax.plot([xpos, xpos], [rec['h_land_mean'], rec['h_water']],
                color='#cccccc', lw=1.0, zorder=1)

        # Water glacier (larger, coloured)
        ax.scatter(xpos, rec['h_water'], color=color, s=60, zorder=3,
                   edgecolors='white', linewidths=0.6)

        x_ticks.append(xpos)
        d_label = f'{int(depth)} m' if not np.isnan(depth) else '?'
        x_labels.append(d_label)

    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=7.5)
    ax.set_xlabel('Glacier (sorted by terminus depth)', fontsize=10)
    ax.set_ylabel('Mean ice thickness (m)', fontsize=10)
    ax.set_title('Water vs. matched land analogues\n(each water glacier and its 5 nearest morphometric neighbours)',
                 fontsize=9.5)

    # Legend: water point, land point
    ax.scatter([], [], color='steelblue', s=55, label='Water-terminating glacier', zorder=3)
    ax.scatter([], [], color='#aaaaaa',   s=18, label='Matched land analogue (k=5)', zorder=2)
    ax.legend(fontsize=8.5, loc='upper left')

    # Colourbar
    cbar = plt.colorbar(scalar_map, ax=ax, orientation='vertical', pad=0.01, fraction=0.04)
    cbar.set_label('Terminus depth (m)', fontsize=8.5)

    # ------------------------------------------------------------------ #
    # Right panel: matched-pair differences                               #
    # ------------------------------------------------------------------ #
    ax2 = axes[1]

    for i, rec in enumerate(records_sorted):
        xpos  = i + 1
        depth = rec['water_depth']
        color = scalar_map.to_rgba(depth) if not np.isnan(depth) else 'dimgray'
        ax2.scatter(xpos, rec['excess'], color=color, s=55, zorder=3,
                    edgecolors='white', linewidths=0.6)
        ax2.plot([xpos, xpos], [0, rec['excess']],
                 color=color, alpha=0.4, lw=1.0, zorder=2)

    ax2.axhline(0, color='black', lw=1.0, ls='--', alpha=0.7)
    ax2.axhline(excesses.mean(), color='steelblue', lw=1.5, ls='-',
                label=f'Mean = {excesses.mean():.1f} m')

    # Annotate t-test
    p_str = f'p = {p_val:.4f}' if p_val >= 0.0001 else f'p < 0.0001'
    ax2.text(0.97, 0.97,
             f'$t = {t_stat:.2f}$\n{p_str}\n$n = {n}$',
             transform=ax2.transAxes, fontsize=9,
             ha='right', va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax2.set_xticks(range(1, n + 1))
    ax2.set_xticklabels([f'{i+1}' for i in range(n)], fontsize=7)
    ax2.set_xlabel('Glacier index (sorted by terminus depth)', fontsize=10)
    ax2.set_ylabel('Matched-pair thickness excess (m)', fontsize=10)
    ax2.set_title('Water − mean land analogue\n(matched-pair differences)',
                  fontsize=9.5)
    ax2.legend(fontsize=8.5, loc='lower right')

    plt.suptitle(
        'Model-free evidence: water-terminating glaciers are thicker than their morphometric twins',
        fontsize=10.5, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT_PATH}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    land_df, water_df = load_data()
    records = run_matching(land_df, water_df)
    make_figure(records)

    excesses = np.array([r['excess'] for r in records])
    t_stat, p_val = stats.ttest_1samp(excesses, 0)
    print(f'\nMatched-pair summary:')
    print(f'  n        = {len(excesses)}')
    print(f'  mean Δh  = {excesses.mean():.1f} m')
    print(f'  std      = {excesses.std():.1f} m')
    print(f'  t        = {t_stat:.2f}')
    print(f'  p        = {p_val:.4f}')
