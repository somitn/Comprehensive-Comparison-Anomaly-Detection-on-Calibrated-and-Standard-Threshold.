# ============================================================
# data_loader.py
# Fast cached loader for SWaT data
# ============================================================

import numpy as np
import os

CACHE_FILE = r"C:\Users\somit\Downloads\Industrial_Anomaly_Prototype\Anamoly test\cache\swat_cache.npz"

TRAIN_SIZE = 100000
VAL_SIZE   = 20000
TUNE_FRAC  = 0.20


def load_swat(cache_file=CACHE_FILE,
              train_size=TRAIN_SIZE,
              val_size=VAL_SIZE,
              tune_frac=TUNE_FRAC):
    """Load cached SWaT data and produce train/val/tune/test splits."""
    if not os.path.exists(cache_file):
        raise FileNotFoundError(
            f"Cache not found: {cache_file}\n"
            f"Run prepare_data.py first."
        )

    data = np.load(cache_file, allow_pickle=True)
    X_normal = data['X_normal']
    X_attack = data['X_attack']
    y_attack = data['y_attack']
    features = list(data['features'])

    X_train = X_normal[:train_size].copy()
    X_val   = X_normal[train_size:train_size + val_size].copy()

    n_tune = int(len(X_attack) * tune_frac)
    X_tune = X_attack[:n_tune].copy()
    y_tune = y_attack[:n_tune].copy()

    X_test = X_attack[n_tune:].copy()
    y_test = y_attack[n_tune:].copy()

    return {
        'X_train': X_train, 'X_val': X_val,
        'X_tune': X_tune,   'y_tune': y_tune,
        'X_test': X_test,   'y_test': y_test,
        'features': features,
    }


if __name__ == "__main__":
    import time
    t0 = time.time()
    d = load_swat()
    print(f"Loaded in {time.time()-t0:.2f}s")
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: {v.shape}  {v.dtype}")
        else:
            print(f"  {k}: {len(v)} items")