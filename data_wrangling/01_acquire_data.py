"""
01_acquire_data.py — Download all datasets needed for the constraint analysis.

Datasets downloaded here:
  - GEBCO 2023       (~8 GB)  ocean/lake bathymetry
  - HydroLAKES v1.0  (~400 MB) lake polygons with depth estimates

RGI and GlaThiDa are assumed to already be present from project1.
If they are missing, this script will download them too.

Run from the repo root:
    python data_wrangling/01_acquire_data.py
"""
import os
import sys
import glob
import zipfile
import io
import requests
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import (
    RGI_PATH, GLATHIDA_PATH, GEBCO_PATH, HYDROLAKES_PATH
)

# pyogrio is required by 02_extract_depths.py — install if missing
try:
    import pyogrio  # noqa: F401
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'pyogrio', '-q'])
    print('  pyogrio installed.')

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
RGI_OGGM_URL    = 'https://cluster.klima.uni-bremen.de/~oggm/rgi/rgi62.zip'
GLATHIDA_URL    = 'https://www.gtn-g.ch/database/glathida-3.1.0.zip'
GEBCO_URL       = 'https://www.bodc.ac.uk/data/open_download/gebco/gebco_2023/zip/'
HYDROLAKES_URL  = 'https://figshare.com/ndownloader/files/9626776'

# Figshare (and some other servers) block Python's default user-agent
_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    )
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stream_download(url, dest_path, label):
    """Stream-download with progress bar. Skips if file already present and valid."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
        print(f'  {label}: already downloaded, skipping.')
        return
    print(f'  Downloading {label}...')
    r = requests.get(url, stream=True, headers=_HEADERS)
    r.raise_for_status()
    total = int(r.headers.get('content-length', 0))
    with open(dest_path, 'wb') as f, tqdm(
        total=total, unit='B', unit_scale=True, desc=label
    ) as bar:
        for chunk in r.iter_content(chunk_size=1024 * 256):
            f.write(chunk)
            bar.update(len(chunk))
    size = os.path.getsize(dest_path)
    if size == 0:
        os.remove(dest_path)
        raise RuntimeError(
            f'Download of {label} returned 0 bytes — URL may have moved or require auth.\n'
            f'Download manually from https://www.hydrosheds.org/products/hydrolakes\n'
            f'and place the zip at: {dest_path}'
        )
    print(f'  {label} saved ({size / 1e6:.1f} MB).')


def _valid_zip(path):
    try:
        with zipfile.ZipFile(path) as zf:
            zf.namelist()
        return True
    except (zipfile.BadZipFile, FileNotFoundError):
        return False


# ---------------------------------------------------------------------------
# RGI 6.2 (OGGM mirror — no authentication required)
# ---------------------------------------------------------------------------

def acquire_rgi():
    print('\n[1/4] RGI 6.2')
    csv_files = glob.glob(os.path.join(RGI_PATH, '*.csv'))
    if csv_files:
        print(f'  RGI CSVs already present ({len(csv_files)} regions), skipping.')
        return

    rgi_zip = os.path.join(RGI_PATH, 'RGI.zip')
    if not _valid_zip(rgi_zip):
        if os.path.exists(rgi_zip):
            os.remove(rgi_zip)
        _stream_download(RGI_OGGM_URL, rgi_zip, 'RGI 6.2')

    # Extract using dbfread — avoids fiona/geopandas dependency
    from dbfread import DBF
    import pandas as pd

    outer_dir = os.path.join(RGI_PATH, 'rgi62_extracted')
    os.makedirs(outer_dir, exist_ok=True)

    with zipfile.ZipFile(rgi_zip) as outer_zf:
        region_zips = [n for n in outer_zf.namelist()
                       if n.endswith('.zip') and 'rgi62' in n]
        print(f'  Extracting {len(region_zips)} regional zips...')
        for rz_name in tqdm(region_zips, desc='  Regions'):
            region_dir = os.path.join(outer_dir, rz_name.replace('.zip', ''))
            os.makedirs(region_dir, exist_ok=True)
            dbf_files = glob.glob(os.path.join(region_dir, '**', '*.dbf'), recursive=True)
            if not dbf_files:
                data = outer_zf.read(rz_name)
                with zipfile.ZipFile(io.BytesIO(data)) as inner_zf:
                    inner_zf.extractall(region_dir)
                dbf_files = glob.glob(os.path.join(region_dir, '**', '*.dbf'), recursive=True)
            if not dbf_files:
                continue
            table = DBF(dbf_files[0], encoding='latin-1', ignorecase=True)
            df = pd.DataFrame(iter(table))
            csv_name = rz_name[:2] + '_rgi62_attribs.csv'
            df.to_csv(os.path.join(RGI_PATH, csv_name), index=False)

    print(f'  RGI extraction complete.')


# ---------------------------------------------------------------------------
# GlaThiDa 3.1.0
# ---------------------------------------------------------------------------

def acquire_glathida():
    print('\n[2/4] GlaThiDa 3.1.0')
    csv_out = os.path.join(GLATHIDA_PATH, 'T.csv')
    if os.path.exists(csv_out):
        print('  T.csv already extracted, skipping.')
        return

    gz_path = os.path.join(GLATHIDA_PATH, 'glathida.zip')
    if not _valid_zip(gz_path):
        if os.path.exists(gz_path):
            os.remove(gz_path)
        _stream_download(GLATHIDA_URL, gz_path, 'GlaThiDa 3.1.0')

    target = 'glathida-3.1.0/data/T.csv'
    with zipfile.ZipFile(gz_path) as zf:
        file_size = zf.getinfo(target).file_size
        with zf.open(target) as src, open(csv_out, 'wb') as dst:
            with tqdm(total=file_size, unit='B', unit_scale=True, desc='  Extracting T.csv') as bar:
                for chunk in iter(lambda: src.read(1024), b''):
                    dst.write(chunk)
                    bar.update(len(chunk))
    print(f'  GlaThiDa T.csv saved to {csv_out}')


# ---------------------------------------------------------------------------
# GEBCO 2023
# ---------------------------------------------------------------------------

def acquire_gebco():
    print('\n[3/4] GEBCO 2023 (~8 GB download)')
    nc_out = os.path.join(GEBCO_PATH, 'GEBCO_2023.nc')
    if os.path.exists(nc_out):
        print('  GEBCO_2023.nc already present, skipping.')
        return

    gz_path = os.path.join(GEBCO_PATH, 'GEBCO_2023.zip')
    if not _valid_zip(gz_path):
        if os.path.exists(gz_path):
            os.remove(gz_path)
        _stream_download(GEBCO_URL, gz_path, 'GEBCO 2023')

    print('  Extracting GEBCO...')
    with zipfile.ZipFile(gz_path) as zf:
        zf.extractall(GEBCO_PATH)
    print('  GEBCO extracted.')


# ---------------------------------------------------------------------------
# HydroLAKES v1.0
# ---------------------------------------------------------------------------

def acquire_hydrolakes():
    print('\n[4/4] HydroLAKES v1.0')

    # Already extracted?
    shp_files = glob.glob(os.path.join(HYDROLAKES_PATH, '**', '*.shp'), recursive=True)
    if shp_files:
        print(f'  HydroLAKES already extracted: {shp_files[0]}')
        return

    # Find any zip in the hydrolakes directory (filename from manual download may vary)
    zip_candidates = glob.glob(os.path.join(HYDROLAKES_PATH, '*.zip'))
    valid_zips = [z for z in zip_candidates if _valid_zip(z)]

    if not valid_zips:
        # Try automated download (often blocked by figshare CDN)
        hl_zip = os.path.join(HYDROLAKES_PATH, 'HydroLAKES_polys_v10.zip')
        _stream_download(HYDROLAKES_URL, hl_zip, 'HydroLAKES')
        valid_zips = [hl_zip]

    hl_zip = valid_zips[0]
    print(f'  Extracting {os.path.basename(hl_zip)}...')
    with zipfile.ZipFile(hl_zip) as zf:
        zf.extractall(HYDROLAKES_PATH)
    shp_files = glob.glob(os.path.join(HYDROLAKES_PATH, '**', '*.shp'), recursive=True)
    print(f'  HydroLAKES extracted: {shp_files[0] if shp_files else "no .shp found"}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    acquire_rgi()
    acquire_glathida()
    acquire_gebco()
    acquire_hydrolakes()
    print('\nAll datasets ready.')
