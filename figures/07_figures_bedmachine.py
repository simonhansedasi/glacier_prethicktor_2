"""
07_figures_bedmachine.py — BedMachine vs Farinotti depth-scaling comparison.

Two-panel figure: same 9 Greenland peripheral glaciers, same RF land baseline.
Left: BedMachine Δh vs terminus depth (shows HAB signal).
Right: Farinotti Δh vs terminus depth (signal absent).

Run from repo root:
    conda run -n glacierml python figures/07_figures_bedmachine.py
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import RESULTS_PATH, FIGURES_PATH

OUT_PATH = os.path.join(FIGURES_PATH, 'bedmachine_vs_farinotti.png')


def load_data():
    df = pd.read_pickle(os.path.join(RESULTS_PATH, 'bedmachine_region05.pkl'))
    df = df.dropna(subset=['bm_excess_m', 'farinotti_excess_m', 'water_depth_m'])
    print(f'Glaciers in figure: {len(df)}')
    return df


def regression(x, y):
    sl, ic, r, p, _ = stats.linregress(x, y)
    return sl, ic, r**2, p


def make_figure(df):
    d = df['water_depth_m'].values

    sl_bm,  ic_bm,  r2_bm,  p_bm  = regression(d, df['bm_excess_m'].values)
    sl_far, ic_far, r2_far, p_far = regression(d, df['farinotti_excess_m'].values)

    d_line = np.linspace(0, max(d) * 1.1, 300)

    # shared y-axis limits centred on the data
    all_vals = np.concatenate([df['bm_excess_m'].values, df['farinotti_excess_m'].values])
    ypad = 60
    ylim = (min(all_vals) - ypad, max(all_vals) + ypad)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

    panel_cfg = [
        {
            'ax':      axes[0],
            'y':       df['bm_excess_m'].values,
            'slope':   sl_bm,
            'icept':   ic_bm,
            'r2':      r2_bm,
            'p':       p_bm,
            'color':   '#1f77b4',
            'title':   'BedMachine v6\n(measurement-based)',
            'label':   'BedMachine',
        },
        {
            'ax':      axes[1],
            'y':       df['farinotti_excess_m'].values,
            'slope':   sl_far,
            'icept':   ic_far,
            'r2':      r2_far,
            'p':       p_far,
            'color':   '#d62728',
            'title':   'Farinotti consensus\n(no flotation boundary condition)',
            'label':   'Farinotti',
        },
    ]

    for cfg in panel_cfg:
        ax = cfg['ax']

        # HAB theory line (slope=1, intercept=0) — dominant visual
        ax.plot(d_line, d_line, 'k-', lw=2.5, zorder=1,
                label='HAB theory (slope = 1)')

        # OLS fit
        ax.plot(d_line, cfg['slope'] * d_line + cfg['icept'],
                color=cfg['color'], lw=2, ls='--', zorder=2,
                label=(f"Fit: slope={cfg['slope']:.2f}, "
                       f"R\u00b2={cfg['r2']:.2f}, p={cfg['p']:.3f}"))

        # Data points
        ax.scatter(d, cfg['y'], color=cfg['color'], s=60,
                   edgecolors='k', linewidths=0.6, zorder=3,
                   label=cfg['label'])

        ax.axhline(0, color='gray', lw=0.8, ls=':')

        ax.set_xlabel('Terminus water depth (m)', fontsize=12)
        ax.set_title(cfg['title'], fontsize=12)
        ax.set_ylim(ylim)
        ax.legend(fontsize=9, loc='upper left')

    axes[0].set_ylabel('Thickness excess over RF land baseline (m)', fontsize=12)

    n = len(df)
    fig.suptitle(
        f'Depth-dependent thickness signal: BedMachine vs Farinotti\n'
        f'Greenland peripheral glaciers with BedMachine coverage  (n = {n})',
        fontsize=12,
    )

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved {OUT_PATH}')


if __name__ == '__main__':
    df = load_data()
    make_figure(df)
