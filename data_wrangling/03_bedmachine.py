"""
03_bedmachine.py — Download BedMachine Greenland v5 and compare to Farinotti
                   for RGI region 05 (Greenland periphery) tidewater glaciers.

Tests whether BedMachine (measurement-based) shows the depth-dependent thickness
signal that is absent from the Farinotti consensus.

Requires Earthdata login: https://urs.earthdata.nasa.gov/
Store credentials in ~/.netrc or answer the interactive prompt.

Run from repo root:
    conda run -n glacierml python data_wrangling/03_bedmachine.py
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import earthaccess
from pyproj import Transformer
from tqdm import tqdm
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA_PATH, RGI_PATH, REF_PATH, RESULTS_PATH, P2_DATA
from glacierml import load_RGI

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BEDMACHINE_DIR  = os.path.join(P2_DATA, 'bedmachine')
BEDMACHINE_FILE = os.path.join(BEDMACHINE_DIR, 'BedMachineGreenland-v6.nc')
OUTPUT_PKL      = os.path.join(RESULTS_PATH, 'bedmachine_region05.pkl')

os.makedirs(BEDMACHINE_DIR, exist_ok=True)

# BedMachine Greenland uses EPSG:3413 (Polar Stereographic North)
EPSG_BM = 'EPSG:3413'
EPSG_WGS84 = 'EPSG:4326'
to_bm = Transformer.from_crs(EPSG_WGS84, EPSG_BM, always_xy=True)

FEATURES = ['CenLat', 'CenLon', 'Slope', 'Zmin', 'Zmed', 'Zmax', 'Area', 'Aspect', 'Lmax']
SEED = 378


# ---------------------------------------------------------------------------
# 1. Download
# ---------------------------------------------------------------------------

def download_bedmachine():
    if os.path.exists(BEDMACHINE_FILE) and os.path.getsize(BEDMACHINE_FILE) > 1e8:
        print(f'BedMachine already present: {BEDMACHINE_FILE}')
        return

    print('Logging in to Earthdata (NASA)...')
    auth = earthaccess.login(strategy='netrc')
    if not auth:
        earthaccess.login(strategy='interactive')

    print('Searching for BedMachine Greenland v6...')
    results = earthaccess.search_data(short_name='IDBMG4', version='6')
    if not results:
        raise RuntimeError(
            'BedMachine Greenland v6 not found via earthaccess.\n'
            'Download manually from https://nsidc.org/data/idbmg4\n'
            f'and place the .nc file at:\n  {BEDMACHINE_FILE}'
        )

    # Filter to only the full NetCDF (not the bed-only tif, ~800 MB)
    nc_granules = [g for g in results
                   if any('BedMachineGreenland-v6.nc' in str(url)
                          for url in g.data_links())]
    if not nc_granules:
        nc_granules = results   # fallback: download all and rename

    url = 'https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/ICEBRIDGE/IDBMG4/6/1993/01/01/BedMachineGreenland-v6.nc'
    print(f'Downloading BedMachineGreenland-v6.nc (~2.8 GB)...')
    session = earthaccess.get_requests_https_session()
    r = session.get(url, stream=True)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    tmp = BEDMACHINE_FILE + '.part'
    with open(tmp, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, unit_divisor=1024,
        desc='BedMachine v6', ncols=80
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 512):
            f.write(chunk)
            bar.update(len(chunk))
    os.rename(tmp, BEDMACHINE_FILE)

    sz = os.path.getsize(BEDMACHINE_FILE) if os.path.exists(BEDMACHINE_FILE) else 0
    print(f'Saved: {BEDMACHINE_FILE}  ({sz/1e9:.2f} GB)')


# ---------------------------------------------------------------------------
# 2. Extract mean BedMachine thickness for region 05 glaciers
# ---------------------------------------------------------------------------

def extract_bm_thickness(region05_df):
    """
    For each glacier centroid, sample BedMachine thickness within a search
    radius (1.5 × Lmax). Uses polar stereographic x/y for fast slicing.
    Excludes main Greenland Ice Sheet pixels (mask == 2).
    """
    print(f'\nLoading BedMachine from {BEDMACHINE_FILE}...')
    ds = xr.open_dataset(BEDMACHINE_FILE, engine='netcdf4')

    if 'thickness' not in ds:
        raise KeyError(f"'thickness' not in BedMachine vars: {list(ds.data_vars)}")

    thickness = ds['thickness']   # (y, x), metres
    mask      = ds['mask']        # 0=ocean,1=land,2=ice sheet,3=float,4=periph glacier

    x_coords = thickness.coords['x'].values   # 1D, metres (EPSG:3413)
    y_coords = thickness.coords['y'].values   # 1D, metres

    dx = abs(float(x_coords[1] - x_coords[0]))   # ~150 m
    x_asc = x_coords[0] < x_coords[-1]   # True if ascending
    y_asc = y_coords[0] < y_coords[-1]   # BedMachine y is typically descending

    print(f'Grid: {thickness.shape}, resolution ~{dx:.0f} m')
    print(f'x range: {x_coords.min()/1e3:.0f} to {x_coords.max()/1e3:.0f} km  ({"asc" if x_asc else "desc"})')
    print(f'y range: {y_coords.min()/1e3:.0f} to {y_coords.max()/1e3:.0f} km  ({"asc" if y_asc else "desc"})')

    SEARCH_FACTOR = 1.5
    MIN_RADIUS_M  = 5_000.0   # 5 km floor

    bm_thick   = []
    bm_n_cells = []

    for _, row in tqdm(region05_df.iterrows(), total=len(region05_df),
                       desc='Extracting BedMachine'):
        clat, clon = row['CenLat'], row['CenLon']
        lmax_m = row['Lmax'] if pd.notna(row['Lmax']) else 5_000.0
        r = max(lmax_m * SEARCH_FACTOR, MIN_RADIUS_M)

        # Convert centroid to BedMachine projection
        cx, cy = to_bm.transform(clon, clat)

        # Fast 1D slice — order must match coordinate direction
        x_lo, x_hi = cx - r, cx + r
        y_lo, y_hi = cy - r, cy + r

        xs = slice(x_lo, x_hi) if x_asc else slice(x_hi, x_lo)
        ys = slice(y_lo, y_hi) if y_asc else slice(y_hi, y_lo)

        sub_thick = thickness.sel(x=xs, y=ys)
        sub_mask  = mask.sel(x=xs, y=ys)

        t = sub_thick.values.ravel()
        m = sub_mask.values.ravel()

        # Keep non-zero ice that is NOT the main Greenland Ice Sheet (mask != 2)
        valid = (t > 0) & (m != 2) & (m != 0)

        if valid.sum() >= 3:
            bm_thick.append(float(np.mean(t[valid])))
            bm_n_cells.append(int(valid.sum()))
        else:
            bm_thick.append(np.nan)
            bm_n_cells.append(0)

    ds.close()

    out = region05_df.copy()
    out['bm_thickness_m'] = bm_thick
    out['bm_n_cells']     = bm_n_cells

    found = out['bm_thickness_m'].notna().sum()
    print(f'BedMachine coverage: {found}/{len(out)} glaciers ({100*found/len(out):.1f}%)')
    return out


# ---------------------------------------------------------------------------
# 3. Land-baseline RF (same approach as 04_hab_modeled.py)
# ---------------------------------------------------------------------------

def fit_land_rf():
    print('\nFitting RF land baseline...')
    rgi = load_RGI(RGI_PATH)[['RGIId', 'TermType'] + FEATURES].dropna()
    refs = pd.read_pickle(os.path.join(REF_PATH, 'refs.pkl'))

    df = refs.merge(rgi, on='RGIId', how='inner')
    df = df[df['FMT'] > 0]
    land = df[df['TermType'].isin([0, 9])].copy()
    water_r05 = df[(df['TermType'].isin([1, 2])) &
                   (df['RGIId'].str.startswith('RGI60-05'))].copy()

    X_land = land[FEATURES].values
    y_land = land['FMT'].values

    rf = RandomForestRegressor(
        n_estimators=100, max_depth=None, min_samples_leaf=2,
        max_features='sqrt', random_state=SEED, n_jobs=-1
    )
    cv_mae = -cross_val_score(rf, X_land, y_land, cv=3,
                              scoring='neg_mean_absolute_error', n_jobs=-1).mean()
    print(f'  Land RF 3-fold CV MAE: {cv_mae:.1f} m')

    rf.fit(X_land, y_land)

    # Predict land baseline for region 05 water-terminating glaciers
    if len(water_r05) > 0:
        water_r05 = water_r05.copy()
        water_r05['land_pred_m'] = rf.predict(water_r05[FEATURES].values)
        water_r05['farinotti_excess_m'] = water_r05['FMT'] - water_r05['land_pred_m']
    else:
        water_r05 = pd.DataFrame(columns=['RGIId', 'FMT', 'land_pred_m', 'farinotti_excess_m'])

    print(f'  Region 05 water-term with Farinotti FMT: {len(water_r05)}')
    return rf, water_r05


# ---------------------------------------------------------------------------
# 4. Compute BedMachine excess and compare to Farinotti
# ---------------------------------------------------------------------------

def compute_and_compare(bm_df, rf, farinotti_r05):
    """
    Compute BedMachine thickness excess over RF land baseline.
    Merge with Farinotti excess and terminus depth.
    """
    # Predict land baseline for glaciers with BedMachine coverage
    bm_valid = bm_df[bm_df['bm_thickness_m'].notna()].copy()
    X = bm_valid[FEATURES].values
    bm_valid['land_pred_m']  = rf.predict(X)
    bm_valid['bm_excess_m']  = bm_valid['bm_thickness_m'] - bm_valid['land_pred_m']

    # Merge Farinotti excess
    f_cols = ['RGIId', 'FMT', 'farinotti_excess_m']
    merged = bm_valid.merge(farinotti_r05[f_cols], on='RGIId', how='left')

    # Depth data already in bm_valid via terminus_depths
    merged = merged[merged['water_depth_m'].notna()].copy()
    print(f'\nGlaciers with BedMachine + Farinotti + depth: {len(merged)}')

    return merged


# ---------------------------------------------------------------------------
# 5. Depth-scaling test
# ---------------------------------------------------------------------------

def depth_scaling_test(merged):
    print('\n--- Depth-scaling: BedMachine excess vs terminus depth ---')
    bm_sub = merged.dropna(subset=['bm_excess_m', 'water_depth_m'])
    if len(bm_sub) < 5:
        print(f'  Too few glaciers ({len(bm_sub)}) for regression.')
        return

    sl, ic, r, p, se = stats.linregress(bm_sub['water_depth_m'], bm_sub['bm_excess_m'])
    print(f'  BedMachine Δh ~ depth:   slope={sl:.3f}, R²={r**2:.3f}, p={p:.4f}, n={len(bm_sub)}')

    # Farinotti for the same glaciers
    far_sub = merged.dropna(subset=['farinotti_excess_m', 'water_depth_m'])
    if len(far_sub) >= 5:
        sl_f, ic_f, r_f, p_f, _ = stats.linregress(
            far_sub['water_depth_m'], far_sub['farinotti_excess_m'])
        print(f'  Farinotti  Δh ~ depth:   slope={sl_f:.3f}, R²={r_f**2:.3f}, p={p_f:.4f}, n={len(far_sub)}')

    return bm_sub


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if os.path.exists(OUTPUT_PKL):
        print(f'Output already exists: {OUTPUT_PKL}')
        print('Delete it to re-run. Loading and re-running depth test only...')
        merged = pd.read_pickle(OUTPUT_PKL)
        depth_scaling_test(merged)
        sys.exit(0)

    # Step 1: download
    download_bedmachine()

    # Step 2: load region 05 tidewater glaciers with terminus depths
    # term_depths already contains all FEATURES columns — no extra merge needed
    term_depths = pd.read_pickle(os.path.join(RESULTS_PATH, 'terminus_depths.pkl'))
    region05 = term_depths[
        (term_depths['region'] == '05') & term_depths[FEATURES].notna().all(axis=1)
    ].copy()
    print(f'\nRegion 05 glaciers with RGI features: {len(region05)}')

    # Step 3: extract BedMachine thickness
    bm_df = extract_bm_thickness(region05)

    # Step 4: fit RF and get Farinotti excess
    rf, farinotti_r05 = fit_land_rf()

    # Step 5: merge and compute BedMachine excess
    merged = compute_and_compare(bm_df, rf, farinotti_r05)

    # Step 6: depth-scaling test
    depth_scaling_test(merged)

    # Save
    merged.to_pickle(OUTPUT_PKL)
    print(f'\nSaved: {OUTPUT_PKL}')
    print('Done.')
