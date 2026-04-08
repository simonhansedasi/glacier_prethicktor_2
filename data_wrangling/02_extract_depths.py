"""
02_extract_depths.py — Extract terminus water depth for every water-terminating glacier.

Strategy:
  TermType 1 (tidewater) → sample GEBCO 2023 within 1.5×Lmax of centroid; take deepest ocean cell
  TermType 2 (lake)      → spatial join centroid to HydroLAKES polygon; use Depth_avg

Outputs (written to RESULTS_PATH):
  terminus_depths.pkl        — all water-terminating RGI glaciers + water_depth_m
  glathida_water_depths.pkl  — subset with observed GlaThiDa thickness (for HAB validation)

Run from the repo root:
    python data_wrangling/02_extract_depths.py
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (
    RGI_PATH, GLATHIDA_PATH, DATA_PATH,
    GEBCO_PATH, HYDROLAKES_PATH, RESULTS_PATH,
)
from glacierml import load_RGI, coregister_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
METERS_PER_DEGREE = 111_000
SEARCH_FACTOR     = 1.5   # search SEARCH_FACTOR × Lmax around each centroid
MIN_RADIUS_DEG    = 0.05  # floor: ~5 km, catches very small glaciers

TERMINUS_DEPTHS_OUT     = os.path.join(RESULTS_PATH, 'terminus_depths.pkl')
GLATHIDA_DEPTHS_OUT     = os.path.join(RESULTS_PATH, 'glathida_water_depths.pkl')


# ---------------------------------------------------------------------------
# Load RGI
# ---------------------------------------------------------------------------

def load_water_terminating():
    rgi = load_RGI(RGI_PATH)
    print(f'Total RGI glaciers: {len(rgi):,}')
    tidewater = rgi[rgi['TermType'] == 1].copy().reset_index(drop=True)
    lake_term  = rgi[rgi['TermType'] == 2].copy().reset_index(drop=True)
    print(f'  Tidewater  (TermType=1): {len(tidewater):,}')
    print(f'  Lake-term  (TermType=2): {len(lake_term):,}')
    return rgi, tidewater, lake_term


# ---------------------------------------------------------------------------
# Tidewater — sample GEBCO
# ---------------------------------------------------------------------------

def _build_gebco_sampler():
    nc_path = os.path.join(GEBCO_PATH, 'GEBCO_2023.nc')
    ds   = xr.open_dataset(nc_path)
    elev = ds['elevation']
    lat_ascending = bool(elev['lat'].values[0] < elev['lat'].values[-1])

    def sample(lat, lon, lmax_m):
        r = max((lmax_m * SEARCH_FACTOR) / METERS_PER_DEGREE, MIN_RADIUS_DEG)
        if lat_ascending:
            sub = elev.sel(lat=slice(lat - r, lat + r), lon=slice(lon - r, lon + r))
        else:
            sub = elev.sel(lat=slice(lat + r, lat - r), lon=slice(lon - r, lon + r))
        vals = sub.values.ravel()
        ocean = vals[vals < 0]
        return float(np.min(ocean)) if len(ocean) > 0 else np.nan

    return sample


def extract_tidewater_depths(tidewater):
    print('\n[1/2] Sampling GEBCO for tidewater termini...')
    sample_gebco = _build_gebco_sampler()
    depths = [
        sample_gebco(row['CenLat'], row['CenLon'], row['Lmax'])
        for _, row in tqdm(tidewater.iterrows(), total=len(tidewater), desc='  Tidewater')
    ]
    tidewater = tidewater.copy()
    tidewater['water_depth_m'] = np.abs(depths)
    found = tidewater['water_depth_m'].notna().sum()
    print(f'  Depth found: {found}/{len(tidewater)} ({100*found/len(tidewater):.1f}%)')
    return tidewater


# ---------------------------------------------------------------------------
# Lake-terminating — match HydroLAKES
# ---------------------------------------------------------------------------

def extract_lake_depths(lake_term):
    print('\n[2/2] Matching lake-terminating glaciers to HydroLAKES...')
    shp_files = glob.glob(os.path.join(HYDROLAKES_PATH, '**', '*.shp'), recursive=True)
    if not shp_files:
        raise FileNotFoundError(
            f'No HydroLAKES .shp found in {HYDROLAKES_PATH}. '
            'Run 01_acquire_data.py first.'
        )

    hl = gpd.read_file(shp_files[0], engine='pyogrio')
    print(f'  HydroLAKES loaded: {len(hl):,} lakes')

    lake_gdf = gpd.GeoDataFrame(
        lake_term,
        geometry=gpd.points_from_xy(lake_term['CenLon'], lake_term['CenLat']),
        crs='EPSG:4326',
    )

    # Primary match: centroid inside a lake polygon
    joined = gpd.sjoin(
        lake_gdf,
        hl[['Hylak_id', 'Depth_avg', 'geometry']],
        how='left',
        predicate='within',
    )
    print(f'  Matched within polygon: {joined["Depth_avg"].notna().sum()}/{len(lake_term)}')

    # Fallback: nearest lake within 0.5°
    unmatched = joined[joined['Depth_avg'].isna()].index
    if len(unmatched) > 0:
        nearest = gpd.sjoin_nearest(
            lake_gdf.loc[unmatched],
            hl[['Hylak_id', 'Depth_avg', 'geometry']],
            how='left',
            max_distance=0.5,
        )
        joined.loc[unmatched, 'Depth_avg'] = nearest['Depth_avg'].values

    lake_term = lake_term.copy()
    lake_term['water_depth_m'] = joined['Depth_avg'].values
    found = lake_term['water_depth_m'].notna().sum()
    print(f'  Lake depths found: {found}/{len(lake_term)} ({100*found/len(lake_term):.1f}%)')
    return lake_term


# ---------------------------------------------------------------------------
# Combine and save
# ---------------------------------------------------------------------------

KEEP_COLS = ['RGIId', 'CenLat', 'CenLon', 'Area', 'Lmax', 'Slope',
             'Zmin', 'Zmed', 'Zmax', 'Aspect', 'TermType', 'region', 'water_depth_m']


def save_terminus_depths(tidewater, lake_term):
    combined = pd.concat(
        [tidewater[KEEP_COLS], lake_term[KEEP_COLS]],
        ignore_index=True,
    )
    combined.to_pickle(TERMINUS_DEPTHS_OUT)
    n = combined['water_depth_m'].notna().sum()
    print(f'\nSaved terminus_depths.pkl — {n:,} glaciers with depth ({len(combined):,} total)')
    return combined


def save_glathida_depths(combined):
    """Cross-match with GlaThiDa co-registered training set; save for HAB validation."""
    training = coregister_data(DATA_PATH, coregistration='4')

    rgi = load_RGI(RGI_PATH)
    training = training.merge(rgi[['RGIId', 'TermType']], on='RGIId', how='left')
    water_train = training[training['TermType'].isin([1, 2])].copy()

    glathida_depths = water_train.merge(
        combined[['RGIId', 'water_depth_m']], on='RGIId', how='left'
    )
    found = glathida_depths['water_depth_m'].notna().sum()
    print(f'GlaThiDa water-terminating with depth: {found}/{len(water_train)}')

    glathida_depths.to_pickle(GLATHIDA_DEPTHS_OUT)
    print(f'Saved glathida_water_depths.pkl')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.path.exists(TERMINUS_DEPTHS_OUT) and os.path.exists(GLATHIDA_DEPTHS_OUT):
        print('Both output files already exist. Delete them to re-run.')
        sys.exit(0)

    rgi, tidewater, lake_term = load_water_terminating()
    tidewater  = extract_tidewater_depths(tidewater)
    lake_term  = extract_lake_depths(lake_term)
    combined   = save_terminus_depths(tidewater, lake_term)
    save_glathida_depths(combined)
    print('\nDone.')
