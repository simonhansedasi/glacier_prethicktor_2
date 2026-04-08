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

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
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

# Coarser grid than script 03 — 180k samples makes exhaustive search slow
RF_GRID = {
    'n_estimators':     [100, 300],
    'max_depth':        [None, 20],
    'min_samples_leaf': [2, 5],
    'max_features':     ['sqrt', 'log2'],
}

def fit_random_forest(land_df):
    print('\n--- Random Forest + GridSearchCV ---')
    X = land_df[FEATURES].values
    y = land_df[LABEL].values

    gs = GridSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        RF_GRID,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X, y)

    best_mae = -gs.best_score_
    print(f'  Best params: {gs.best_params_}')
    print(f'  Best CV MAE: {best_mae:.1f} m')
    return gs.best_estimator_


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
    print(f'\nHAB linear fit:')
    print(f'  slope     = {slope:.3f}  (theory ~1.0)')
    print(f'  intercept = {intercept:.1f} m')
    print(f'  R²        = {r**2:.3f}')
    print(f'  p         = {p:.2e}')
    print(f'  n         = {len(df):,}')
    return slope, intercept, r, p


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
    df, summary = bin_by_depth(df)

    df.attrs.update({
        'hab_slope': slope, 'hab_intercept': intercept,
        'hab_r2': r**2, 'hab_p': p,
    })

    df.to_pickle(ANALYSIS_OUT)
    summary.to_csv(SUMMARY_OUT)
    print(f'\nSaved modeled_analysis_df.pkl and modeled_constraint_summary.csv')
