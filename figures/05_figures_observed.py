"""
03_figures_observed.py — Figures for the GlaThiDa observed analysis (Section 1).

Companion to ml/03_constraint_analysis.py. n=17 — establishes the question
and motivates the modeled analysis in Section 2.

Figures produced:
  scatter_excess_vs_depth.png       — thickness excess vs water depth with HAB line
  population_shift_by_depth_bin.png — violin plots + median trend per depth bin

Run from the repo root:
    python figures/03_figures_observed.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_PATH, FIGURES_PATH

BIN_LABELS        = ['0–50 m', '50–100 m', '100–200 m', '200–400 m', '>400 m']
BIN_CENTRES       = [25, 75, 150, 300, 500]
TERM_COLORS       = {1: '#1f77b4', 2: '#2ca02c'}
TERM_LABELS       = {1: 'Tidewater', 2: 'Lake'}

SCATTER_OUT       = os.path.join(FIGURES_PATH, 'scatter_excess_vs_depth.png')
VIOLIN_OUT        = os.path.join(FIGURES_PATH, 'population_shift_by_depth_bin.png')


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_analysis():
    path = os.path.join(RESULTS_PATH, 'analysis_df.pkl')
    df = pd.read_pickle(path)

    # Recompute HAB fit (attrs may not survive pickle round-trip)
    x = df['water_depth_m'].values
    y = df['thickness_excess_m'].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return df, slope, intercept, r ** 2


# ---------------------------------------------------------------------------
# Figure 1: scatter — excess vs depth
# ---------------------------------------------------------------------------

def fig_scatter(df, slope, intercept, r2):
    fig, ax = plt.subplots(figsize=(7, 5))

    for tt, label in TERM_LABELS.items():
        mask = df['TermType'] == tt
        ax.scatter(
            df.loc[mask, 'water_depth_m'],
            df.loc[mask, 'thickness_excess_m'],
            color=TERM_COLORS[tt], label=label,
            alpha=0.7, s=40, edgecolors='white', linewidths=0.5,
        )

    x_fit = np.linspace(0, df['water_depth_m'].max(), 200)
    ax.plot(x_fit, slope * x_fit + intercept, 'k-', lw=1.5,
            label=f'Fit: slope={slope:.2f}, R²={r2:.2f}')
    ax.plot(x_fit, x_fit, 'r--', lw=1.2, label='HAB theory (slope=1, intercept=0)')
    ax.axhline(0, color='gray', lw=0.8, ls=':')

    ax.set_xlabel('Water depth at terminus (m)')
    ax.set_ylabel('Thickness excess over land baseline (m)')
    ax.set_title('Constraint strength → population shift\n(HAB framework test)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(SCATTER_OUT, dpi=150)
    plt.close()
    print(f'Saved {SCATTER_OUT}')


# ---------------------------------------------------------------------------
# Figure 2: violin + median trend
# ---------------------------------------------------------------------------

def fig_violin(df):
    groups = [
        df.loc[df['depth_bin'] == b, 'thickness_excess_m'].dropna().values
        for b in BIN_LABELS
    ]
    nonempty = [(b, c, g) for b, c, g in zip(BIN_LABELS, BIN_CENTRES, groups) if len(g) > 1]
    if not nonempty:
        print('Not enough data for violin plot.')
        return

    labels_ne, centres_ne, data_ne = zip(*nonempty)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # --- Left: violin ---
    ax = axes[0]
    parts = ax.violinplot(data_ne, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.6)
    ax.set_xticks(range(1, len(labels_ne) + 1))
    ax.set_xticklabels(labels_ne, rotation=20)
    ax.axhline(0, color='red', ls='--', lw=1)
    ax.set_xlabel('Water depth bin')
    ax.set_ylabel('Thickness excess over land baseline (m)')
    ax.set_title('Distribution shift by constraint strength')

    # --- Right: median + IQR trend ---
    ax2 = axes[1]
    medians = [np.median(g) for g in data_ne]
    q25     = [np.percentile(g, 25) for g in data_ne]
    q75     = [np.percentile(g, 75) for g in data_ne]
    ax2.errorbar(
        centres_ne, medians,
        yerr=[np.array(medians) - np.array(q25), np.array(q75) - np.array(medians)],
        fmt='o', color='#1f77b4', capsize=4,
    )
    x_theory = np.array(centres_ne)
    ax2.plot(x_theory, x_theory, 'r--', label='HAB theory (slope=1)', lw=1.2)
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('Approximate depth bin centre (m)')
    ax2.set_ylabel('Median thickness excess (m)')
    ax2.set_title('Median excess vs constraint strength')
    ax2.legend()

    plt.suptitle('Terminus water depth as a continuous constraint on glacier thickness', fontsize=11)
    plt.tight_layout()
    plt.savefig(VIOLIN_OUT, dpi=150)
    plt.close()
    print(f'Saved {VIOLIN_OUT}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    df, slope, intercept, r2 = load_analysis()
    fig_scatter(df, slope, intercept, r2)
    fig_violin(df)
    print('All figures done.')
