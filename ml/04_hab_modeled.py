"""
04_hab_modeled.py — HAB constraint analysis, Farinotti consensus thickness (Section 2).

Uses Farinotti et al. 2019 modeled mean thickness for all RGI glaciers,
giving ~3,500 water-terminating glaciers for the HAB test vs. 17 in Section 1.

Land baseline models (both fit and reported):
  1. Linear Regression   — interpretable baseline
  2. Random Forest + GridSearchCV — main model for excess calculation

Pipeline:
  1. Load Farinotti FMT + RGI features; split by TermType
  2. Fit Linear Regression on land-terminating glaciers (~180,000)
  3. Fit Random Forest with GridSearchCV
  4. Report both validation MAEs and RF feature importances
  5. Predict FMT for water-terminating glaciers using RF
  6. Thickness excess = Farinotti FMT − RF prediction
  7. Join with terminus water depths from terminus_depths.pkl
  8. HAB linear fit + depth-bin statistics
  9. Save modeled_analysis_df.pkl and modeled_constraint_summary.csv

Run from the repo root:
    python ml/04_hab_modeled.py
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_PATH, RGI_PATH, REF_PATH, RESULTS_PATH
from glacierml import load_RGI

SEED = 378
np.random.seed(SEED)

FEATURES = ['CenLat', 'CenLon', 'Slope', 'Zmin', 'Zmed', 'Zmax', 'Area', 'Aspect', 'Lmax']
LABEL    = 'FMT'

ANALYSIS_OUT = os.path.join(RESULTS_PATH, 'modeled_analysis_df.pkl')
SUMMARY_OUT  = os.path.join(RESULTS_PATH, 'modeled_constraint_summary.csv')

BIN_EDGES  = [0, 50, 100, 200, 400, np.inf]
BIN_LABELS = ['0–50 m', '50–100 m', '100–200 m', '200–400 m', '>400 m']


# ---------------------------------------------------------------------------
# 1. Load and merge
# ---------------------------------------------------------------------------

def load_and_merge():
    refs = pd.read_pickle(os.path.join(REF_PATH, 'refs.pkl'))
    print(f'Farinotti FMT: {len(refs):,} glaciers')

    rgi = load_RGI(RGI_PATH)[['RGIId', 'TermType'] + FEATURES].dropna()
    df  = refs.merge(rgi, on='RGIId', how='inner')
    df  = df[df['FMT'] > 0].copy()
    print(f'After merge with RGI features: {len(df):,} glaciers')

    land_df  = df[df['TermType'].isin([0, 9])].copy().reset_index(drop=True)
    water_df = df[df['TermType'].isin([1, 2])].copy().reset_index(drop=True)
    print(f'  Land-terminating  (train): {len(land_df):,}')
    print(f'  Water-terminating (test):  {len(water_df):,}')
    return land_df, water_df


# ---------------------------------------------------------------------------
# 2. Linear Regression
# ---------------------------------------------------------------------------

def fit_linear(land_df):
    print('\n--- Linear Regression ---')
    X = land_df[FEATURES].values
    y = land_df[LABEL].values

    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])

    # 3-fold CV (5-fold is fine but 3 is faster at 180k samples)
    cv_scores = -cross_val_score(lr, X, y, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    print(f'  3-fold CV MAE: {cv_scores.mean():.1f} ± {cv_scores.std():.1f} m')

    lr.fit(X, y)
    return lr


# ---------------------------------------------------------------------------
# 3. Random Forest + GridSearchCV
# ---------------------------------------------------------------------------

# Fixed config matching 03_constraint_analysis best params (cross-validated on GlaThiDa land set).
# Using the same configuration for both analyses ensures the land baselines are comparable.
RF_PARAMS = dict(
    n_estimators=500, max_depth=10, min_samples_leaf=5,
    max_features='log2', random_state=SEED, n_jobs=-1,
)

def fit_random_forest(land_df):
    print('\n--- Random Forest (fixed config, matching 03_constraint_analysis) ---')
    X = land_df[FEATURES].values
    y = land_df[LABEL].values

    rf = RandomForestRegressor(**RF_PARAMS)
    cv_mae = -cross_val_score(rf, X, y, cv=5,
                              scoring='neg_mean_absolute_error', n_jobs=-1).mean()
    print(f'  Params: n_estimators=500, max_depth=10, min_samples_leaf=5, max_features=log2')
    print(f'  5-fold CV MAE: {cv_mae:.1f} m')
    rf.fit(X, y)
    return rf


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

def report_importance(rf_model):
    importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
    print('\nFeature importances (RF):')
    print(importances.sort_values(ascending=False).round(3).to_string())


# ---------------------------------------------------------------------------
# 5 & 6. Predict + compute excess
# ---------------------------------------------------------------------------

def compute_excess(rf_model, water_df):
    X_water = water_df[FEATURES].values
    water_df = water_df.copy()
    water_df['land_prediction_m'] = rf_model.predict(X_water)
    water_df['thickness_excess_m'] = water_df['FMT'] - water_df['land_prediction_m']

    print('\nThickness excess statistics (m):')
    print(water_df['thickness_excess_m'].describe().round(1))
    return water_df


# ---------------------------------------------------------------------------
# 7. Join terminus depths
# ---------------------------------------------------------------------------

def join_depths(water_df):
    depths = pd.read_pickle(os.path.join(RESULTS_PATH, 'terminus_depths.pkl'))
    df = water_df.merge(depths[['RGIId', 'water_depth_m']], on='RGIId', how='left')
    df = df[df['water_depth_m'].notna()].copy()
    print(f'\nGlaciers with Farinotti thickness AND terminus depth: {len(df):,}')
    print(f'  Tidewater: {(df["TermType"]==1).sum():,}')
    print(f'  Lake:      {(df["TermType"]==2).sum():,}')
    return df


# ---------------------------------------------------------------------------
# 8. HAB fit + depth bins
# ---------------------------------------------------------------------------

def fit_hab(df):
    x = df['water_depth_m'].values
    y = df['thickness_excess_m'].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    print(f'\nHAB linear fit (OLS):')
    print(f'  slope     = {slope:.4f}  (theory ~1.0)')
    print(f'  intercept = {intercept:.1f} m')
    print(f'  R²        = {r**2:.3f}')
    print(f'  p         = {p:.2e}')
    print(f'  n         = {len(df):,}')
    return slope, intercept, r, p


def fit_hab_clustered(df):
    """Cluster-robust OLS clustered by RGI 2-digit region."""
    df = df.copy()
    df['rgi_region'] = df['RGIId'].str[6:8]
    n_clusters = df['rgi_region'].nunique()

    X = sm.add_constant(df['water_depth_m'].values)
    y = df['thickness_excess_m'].values
    ols = sm.OLS(y, X).fit()
    robust = ols.get_robustcov_results(cov_type='cluster',
                                       groups=df['rgi_region'].values)

    print(f'\n--- Cluster-robust regression (k={n_clusters} RGI regions) ---')
    print(f'  OLS:            slope={ols.params[1]:.4f}, SE={ols.bse[1]:.4f}, '
          f'p={ols.pvalues[1]:.4e}')
    print(f'  Cluster-robust: slope={robust.params[1]:.4f}, SE={robust.bse[1]:.4f}, '
          f'p={robust.pvalues[1]:.4e}')
    print(f'  Cluster-robust 95% CI: [{robust.conf_int()[1,0]:.4f}, '
          f'{robust.conf_int()[1,1]:.4f}]')
    print(f'  Regions present: {sorted(df["rgi_region"].unique())}')
    return robust


def bin_by_depth(df):
    df = df.copy()
    df['depth_bin'] = pd.cut(df['water_depth_m'], bins=BIN_EDGES, labels=BIN_LABELS)
    summary = df.groupby('depth_bin')['thickness_excess_m'].agg(
        n='count', mean='mean', median='median', std='std',
        q25=lambda x: x.quantile(0.25),
        q75=lambda x: x.quantile(0.75),
    ).round(1)
    print('\nThickness excess by depth bin:')
    print(summary)
    return df, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.path.exists(ANALYSIS_OUT):
        print(f'{ANALYSIS_OUT} already exists. Delete to re-run.')
        sys.exit(0)

    land_df, water_df = load_and_merge()

    lr_model = fit_linear(land_df)
    rf_model = fit_random_forest(land_df)
    report_importance(rf_model)

    water_df    = compute_excess(rf_model, water_df)
    df          = join_depths(water_df)
    slope, intercept, r, p = fit_hab(df)
    robust = fit_hab_clustered(df)
    df, summary = bin_by_depth(df)

    df.attrs.update({
        'hab_slope': slope, 'hab_intercept': intercept,
        'hab_r2': r**2, 'hab_p': p,
        'hab_slope_clustered': robust.params[1],
        'hab_p_clustered': robust.pvalues[1],
        'hab_se_clustered': robust.bse[1],
    })

    df.to_pickle(ANALYSIS_OUT)
    summary.to_csv(SUMMARY_OUT)
    print(f'\nSaved modeled_analysis_df.pkl and modeled_constraint_summary.csv')
