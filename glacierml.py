"""
glacierml.py — ML utilities for glacier_prethicktor_2.

Minimal subset of project1's glacierml.py — only what project2 needs.
"""
import os
import warnings
import numpy as np
import pandas as pd

np.random.seed(42)


# ---------------------------------------------------------------------------
# RGI loader
# ---------------------------------------------------------------------------

def load_RGI(rgi_path):
    """Read all regional RGI attribute CSVs into one DataFrame."""
    frames = []
    for fname in os.listdir(rgi_path):
        if fname.endswith('.csv'):
            df = pd.read_csv(
                os.path.join(rgi_path, fname),
                encoding_errors='replace',
                on_bad_lines='skip',
            )
            frames.append(df)
    rgi = pd.concat(frames, ignore_index=True)
    rgi = rgi.sort_values('RGIId').reset_index(drop=True)
    rgi['region'] = rgi['RGIId'].str[6:8]
    return rgi


# ---------------------------------------------------------------------------
# Training data loader
# ---------------------------------------------------------------------------

def load_training_data(data_path):
    """Load the matched GlaThiDa–RGI dataset and merge with RGI attributes."""
    df = pd.read_pickle(os.path.join(data_path, 'matched.pkl'))
    df = df[[
        'RGIId', 'RGI Centroid Distance', 'SURVEY_DATE',
        'AREA', 'MEAN_THICKNESS_UNCERTAINTY', 'GLACIER_NAME', 'MEAN_THICKNESS',
    ]].rename(columns={'MEAN_THICKNESS': 'Thickness'})

    rgi = load_RGI(os.path.join(data_path, 'RGI'))
    df = pd.merge(df, rgi, how='inner', on='RGIId')
    return df


# ---------------------------------------------------------------------------
# Co-registration filter (coregistration='4' thresholds hardcoded)
# ---------------------------------------------------------------------------
# Coregistration 4: distance_threshold=0.75, size_threshold=0.75, area_scrubber=on

def coregister_data(data_path, coregistration='4'):
    """
    Apply co-registration quality filters to the training data.
    Only coregistration='4' is used by project2 — other values raise an error.
    """
    PARAMS = {
        '1': dict(distance_threshold=9999,  size_threshold=0.00, area_scrubber=False),
        '2': dict(distance_threshold=0.25,  size_threshold=0.25, area_scrubber=True),
        '3': dict(distance_threshold=0.50,  size_threshold=0.50, area_scrubber=True),
        '4': dict(distance_threshold=0.75,  size_threshold=0.75, area_scrubber=True),
    }
    p = PARAMS[str(coregistration)]

    df = load_training_data(data_path)

    # Distance test: centroid distance / glacier radius
    df['RGI Centroid Distance'] = df['RGI Centroid Distance'] * 1e3
    radius = np.sqrt((df['Area'] * 1e6) / np.pi)
    df['distance test'] = df['RGI Centroid Distance'] / radius
    df = df[df['distance test'] < p['distance_threshold']]

    # Area scrubber
    if p['area_scrubber']:
        perc_diff = (df['Area'] - df['AREA']) / df['AREA']
        t = p['size_threshold']
        df = df[(perc_diff >= -t) & (perc_diff <= t)]

    # Drop duplicates
    df = df.drop(df[df['RGIId'].duplicated(keep=False)].index)

    df = df[['RGIId', 'CenLat', 'CenLon', 'Slope', 'Zmin', 'Zmed', 'Zmax',
             'Area', 'Aspect', 'Lmax', 'Thickness']]
    return df.sort_values('RGIId').reset_index(drop=True)


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_dnn_model(normalizer, learning_rate=0.01, layer_1=10, layer_2=5, loss='mae'):
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.filterwarnings('ignore', message='oneDNN custom operations are on.*')
    warnings.filterwarnings('ignore', message='Could not find cuda drivers.*')
    warnings.filterwarnings('ignore', message='This TensorFlow binary is optimized.*')
    warnings.filterwarnings('ignore', message='TF-TRT Warning.*')

    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        normalizer,
        layers.Dense(layer_1, activation='relu'),
        layers.Dropout(rate=0.1, seed=0),
        layers.Dense(layer_2, activation='relu'),
        layers.Dense(1),
    ])
    loss_fn = 'mean_absolute_error' if loss == 'mae' else 'mean_squared_error'
    model.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(learning_rate=learning_rate))
    return model
