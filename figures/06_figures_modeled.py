"""
04_figures_modeled.py — Figures for the Farinotti-based HAB analysis (Section 2).

Companion to ml/04_hab_modeled.py. ~3,500 water-terminating glaciers —
the main result panels of the paper.

Run from the repo root:
    python figures/04_figures_modeled.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_PATH, FIGURES_PATH

BIN_LABELS  = ['0–50 m', '50–100 m', '100–200 m', '200–400 m', '>400 m']
BIN_CENTRES = [25, 75, 150, 300, 500]
TERM_COLORS = {1: '#1f77b4', 2: '#2ca02c'}
TERM_LABELS = {1: 'Tidewater', 2: 'Lake'}

SCATTER_OUT = os.path.join(FIGURES_PATH, 'modeled_scatter_excess_vs_depth.png')
VIOLIN_OUT  = os.path.join(FIGURES_PATH, 'modeled_population_shift_by_depth_bin.png')


def load_analysis():
    path = os.path.join(RESULTS_PATH, 'modeled_analysis_df.pkl')
    df = pd.read_pickle(path)
    x = df['water_depth_m'].values
    y = df['thickness_excess_m'].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    return df, slope, intercept, r ** 2, p


def fig_scatter(df, slope, intercept, r2, p):
    fig, ax = plt.subplots(figsize=(8, 6))

    for tt, label in TERM_LABELS.items():
        mask = df['TermType'] == tt
        ax.scatter(
            df.loc[mask, 'water_depth_m'],
            df.loc[mask, 'thickness_excess_m'],
            color=TERM_COLORS[tt], label=label,
            alpha=0.35, s=15, edgecolors='none',
        )

    x_fit = np.linspace(0, df['water_depth_m'].max(), 300)
    ax.plot(x_fit, slope * x_fit + intercept, 'k-', lw=2,
            label=f'Fit: slope={slope:.2f}, R²={r2:.2f}, p={p:.1e}')
    ax.plot(x_fit, x_fit, 'r--', lw=1.5, label='HAB theory (slope=1, intercept=0)')
    ax.axhline(0, color='gray', lw=0.8, ls=':')

    ax.set_xlabel('Water depth at terminus (m)', fontsize=12)
    ax.set_ylabel('Thickness excess over land baseline (m)', fontsize=12)
    ax.set_title(
        'HAB constraint test — Farinotti consensus thickness\n'
        f'n={len(df):,} water-terminating glaciers',
        fontsize=12,
    )
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(SCATTER_OUT, dpi=150)
    plt.close()
    print(f'Saved {SCATTER_OUT}')


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
    counts = [len(g) for g in data_ne]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    # --- Left: violin ---
    ax = axes[0]
    parts = ax.violinplot(data_ne, showmedians=True, showextrema=False)
    for pc in parts['bodies']:
        pc.set_facecolor('#1f77b4')
        pc.set_alpha(0.55)
    parts['cmedians'].set_color('navy')
    parts['cmedians'].set_linewidth(2)

    ax.set_xticks(range(1, len(labels_ne) + 1))
    ax.set_xticklabels(
        [f'{b}\n(n={n:,})' for b, n in zip(labels_ne, counts)],
        fontsize=9,
    )
    ax.axhline(0, color='red', ls='--', lw=1.2)
    ax.set_xlabel('Water depth bin', fontsize=11)
    ax.set_ylabel('Thickness excess over land baseline (m)', fontsize=11)
    ax.set_title('Population shift by constraint strength\n(Farinotti consensus)', fontsize=11)

    # --- Right: median + IQR trend ---
    ax2 = axes[1]
    medians = [np.median(g) for g in data_ne]
    q25     = [np.percentile(g, 25) for g in data_ne]
    q75     = [np.percentile(g, 75) for g in data_ne]

    ax2.errorbar(
        centres_ne, medians,
        yerr=[np.array(medians) - np.array(q25), np.array(q75) - np.array(medians)],
        fmt='o', color='#1f77b4', capsize=5, markersize=8, lw=1.5,
    )

    x_theory = np.linspace(0, max(centres_ne) * 1.2, 200)
    ax2.plot(x_theory, x_theory, 'r--', label='HAB theory (slope=1)', lw=1.5)
    ax2.axhline(0, color='gray', ls=':', lw=0.8)
    ax2.set_xlabel('Approximate depth bin centre (m)', fontsize=11)
    ax2.set_ylabel('Median thickness excess (m)', fontsize=11)
    ax2.set_title('Median excess vs constraint strength', fontsize=11)
    ax2.legend(fontsize=10)

    plt.suptitle(
        'Terminus water depth as a continuous constraint on glacier thickness\n'
        '(Farinotti et al. 2019 consensus — full RGI water-terminating population)',
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig(VIOLIN_OUT, dpi=150)
    plt.close()
    print(f'Saved {VIOLIN_OUT}')


if __name__ == '__main__':
    df, slope, intercept, r2, p = load_analysis()
    fig_scatter(df, slope, intercept, r2, p)
    fig_violin(df)
    print('All figures done.')
