"""
03_constraint_analysis.py — HAB constraint analysis, GlaThiDa observed data (Section 1).

Uses observed GlaThiDa thickness labels (n≈273 co-registered glaciers).
Establishes the question; motivates the modeled analysis in Section 2.

Land baseline models (both fit and reported):
  1. Linear Regression   — interpretable ceiling; shows whether the relationship is linear
  2. Random Forest + GridSearchCV — handles non-linearities; main model for excess calculation

Pipeline:
  1. Load co-registered GlaThiDa training data; split by terminus type
  2. Fit Linear Regression on land-terminating glaciers
  3. Fit Random Forest with GridSearchCV on land-terminating glaciers
  4. Report both model validation MAEs
  5. Predict thickness for water-terminating glaciers using RF
  6. Thickness excess = observed − RF prediction
  7. Join with terminus water depths
  8. HAB linear fit + depth-bin statistics
  9. Save analysis_df.pkl and constraint_summary.csv

Run from the repo root:
    python ml/03_constraint_analysis.py
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
from sklearn.metrics import mean_absolute_error
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_PATH, RGI_PATH, RESULTS_PATH
from glacierml import load_RGI, coregister_data

SEED = 378
np.random.seed(SEED)

FEATURES = ['CenLat', 'CenLon', 'Slope', 'Zmin', 'Zmed', 'Zmax', 'Area', 'Aspect', 'Lmax']
LABEL    = 'Thickness'

ANALYSIS_OUT = os.path.join(RESULTS_PATH, 'analysis_df.pkl')
SUMMARY_OUT  = os.path.join(RESULTS_PATH, 'constraint_summary.csv')

BIN_EDGES  = [0, 50, 100, 200, 400, np.inf]
BIN_LABELS = ['0–50 m', '50–100 m', '100–200 m', '200–400 m', '>400 m']


# ---------------------------------------------------------------------------
# 1. Load and split
# ---------------------------------------------------------------------------

def load_and_split():
    all_training = coregister_data(DATA_PATH, coregistration='4')

    rgi = load_RGI(RGI_PATH)
    all_training = all_training.merge(rgi[['RGIId', 'TermType']], on='RGIId', how='left')

    land_train  = all_training[all_training['TermType'].isin([0, 9])].copy()
    water_train = all_training[all_training['TermType'].isin([1, 2])].copy()

    print(f'Land-terminating training glaciers:  {len(land_train)}')
    print(f'Water-terminating (held out):        {len(water_train)}')
    return land_train, water_train


# ---------------------------------------------------------------------------
# 2. Linear Regression
# ---------------------------------------------------------------------------

def fit_linear(land_train):
    print('\n--- Linear Regression ---')
    X = land_train[FEATURES].values
    y = land_train[LABEL].values

    lr = Pipeline([
        ('scaler', StandardScaler()),
        ('model',  LinearRegression()),
    ])

    cv_scores = -cross_val_score(lr, X, y, cv=5, scoring='neg_mean_absolute_error')
    print(f'  5-fold CV MAE: {cv_scores.mean():.1f} ± {cv_scores.std():.1f} m')

    lr.fit(X, y)
    return lr


# ---------------------------------------------------------------------------
# 3. Random Forest + GridSearchCV
# ---------------------------------------------------------------------------

RF_GRID = {
    'n_estimators': [100, 300, 500],
    'max_depth':    [None, 10, 20],
    'min_samples_leaf': [1, 2, 5],
    'max_features': ['sqrt', 'log2'],
}

def fit_random_forest(land_train):
    print('\n--- Random Forest + GridSearchCV ---')
    X = land_train[FEATURES].values
    y = land_train[LABEL].values

    gs = GridSearchCV(
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        RF_GRID,
        cv=5,
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
# 4. Report feature importance
# ---------------------------------------------------------------------------

def report_importance(rf_model):
    importances = pd.Series(rf_model.feature_importances_, index=FEATURES)
    print('\nFeature importances (RF):')
    print(importances.sort_values(ascending=False).round(3).to_string())


# ---------------------------------------------------------------------------
# 5 & 6. Predict and compute excess using RF
# ---------------------------------------------------------------------------

def compute_excess(rf_model, water_train):
    X_water = water_train[FEATURES].values
    water_train = water_train.copy()
    water_train['land_prediction_m'] = rf_model.predict(X_water)
    water_train['thickness_excess_m'] = water_train[LABEL] - water_train['land_prediction_m']

    print('\nThickness excess statistics (m):')
    print(water_train['thickness_excess_m'].describe().round(1))
    return water_train


# ---------------------------------------------------------------------------
# 7. Join water depths
# ---------------------------------------------------------------------------

def join_depths(water_train):
    depths = pd.read_pickle(os.path.join(RESULTS_PATH, 'glathida_water_depths.pkl'))
    df = water_train.merge(depths[['RGIId', 'water_depth_m']], on='RGIId', how='left')
    df = df[df['water_depth_m'].notna()].copy()
    print(f'\nGlaciers with observed thickness AND terminus depth: {len(df)}')
    return df


# ---------------------------------------------------------------------------
# 8. HAB fit + depth bins
# ---------------------------------------------------------------------------

def fit_hab(df):
    x = df['water_depth_m'].values
    y = df['thickness_excess_m'].values
    slope, intercept, r, p, se = stats.linregress(x, y)
    print(f'\nHAB linear fit: slope={slope:.3f} (theory ~1.0), R²={r**2:.3f}, p={p:.4f}, n={len(df)}')
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

    land_train, water_train = load_and_split()

    lr_model = fit_linear(land_train)
    rf_model = fit_random_forest(land_train)
    report_importance(rf_model)

    water_train  = compute_excess(rf_model, water_train)
    analysis_df  = join_depths(water_train)
    slope, intercept, r, p = fit_hab(analysis_df)
    analysis_df, summary   = bin_by_depth(analysis_df)

    analysis_df.attrs.update({
        'hab_slope': slope, 'hab_intercept': intercept,
        'hab_r2': r**2, 'hab_p': p,
    })

    analysis_df.to_pickle(ANALYSIS_OUT)
    summary.to_csv(SUMMARY_OUT)
    print(f'\nSaved analysis_df.pkl and constraint_summary.csv to {RESULTS_PATH}')
