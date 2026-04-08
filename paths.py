"""
paths.py — central path configuration for glacier_prethicktor_2.

Edit HOME_PATH if your data root is in a different location.
All other paths are derived from it.
"""
import os

# ---------------------------------------------------------------------------
# Root — shared with project1 (RGI and GlaThiDa already live here)
# ---------------------------------------------------------------------------
HOME_PATH = '/home/simonhans/glacierml'

# Shared datasets (downloaded by project1; reused here)
DATA_PATH       = os.path.join(HOME_PATH, 'data')
RGI_PATH        = os.path.join(DATA_PATH, 'RGI')
GLATHIDA_PATH   = os.path.join(DATA_PATH, 'glathida')
REF_PATH        = os.path.join(DATA_PATH, 'ref')

# Project2-specific data and outputs
P2_ROOT         = os.path.join(HOME_PATH, 'project2')
P2_DATA         = os.path.join(P2_ROOT,   'data')
GEBCO_PATH      = os.path.join(P2_DATA,   'gebco')
HYDROLAKES_PATH = os.path.join(P2_DATA,   'hydrolakes')
RESULTS_PATH    = os.path.join(P2_ROOT,   'results')
FIGURES_PATH    = os.path.join(RESULTS_PATH, 'figures')

# ---------------------------------------------------------------------------
# Create any missing directories at import time
# ---------------------------------------------------------------------------
for _p in [DATA_PATH, RGI_PATH, GLATHIDA_PATH, REF_PATH,
           P2_DATA, GEBCO_PATH, HYDROLAKES_PATH, RESULTS_PATH, FIGURES_PATH]:
    os.makedirs(_p, exist_ok=True)
