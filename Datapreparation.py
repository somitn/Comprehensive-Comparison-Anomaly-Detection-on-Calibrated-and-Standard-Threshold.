# ============================================================
# prepare_data.py
# Load SWaT Excel files ONCE, auto-detect headers, cache to .npz
# ============================================================
# Run: python prepare_data.py
# ============================================================

import pandas as pd
import numpy as np
import os
import time

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = r"C:\Users\somit\Downloads\Industrial_Anomaly_Prototype\Anamoly test\data"

NORMAL_FILE = "SWaT_Dataset_Normal_v1.xlsx"
ATTACK_FILE = "SWaT_Dataset_Attack_v0.xlsx"

CACHE_DIR  = r"C:\Users\somit\Downloads\Industrial_Anomaly_Prototype\Anamoly test\cache"
CACHE_FILE = os.path.join(CACHE_DIR, "swat_cache.npz")

# ============================================================
# AUTO-DETECT HEADER ROW
# ============================================================
def detect_header_row(path, sheet=0, max_check=15):
    """
    Find which row of the Excel file contains the real column headers.
    Heuristic: the header row has many non-null cells AND most are strings
    whose values look like names (not numbers, not dates).
    """
    raw = pd.read_excel(path, sheet_name=sheet, header=None,
                        nrows=max_check, engine='openpyxl')

    print(f"\n  Scanning first {max_check} rows for header...")
    for i in range(len(raw)):
        row = raw.iloc[i]
        non_null = row.notna().sum()

        # Count "string-like" header candidates
        # (a real header has values like 'FIT101', 'Timestamp', 'Normal/Attack')
        str_like = 0
        for v in row:
            if pd.isna(v):
                continue
            if isinstance(v, str):
                s = v.strip()
                # Names are short and don't look like timestamps
                if 1 <= len(s) <= 40 and '/' not in s and ':' not in s:
                    str_like += 1

        frac_filled = non_null / len(row) if len(row) else 0
        frac_str    = str_like / non_null if non_null else 0

        print(f"    Row {i:2d}: filled={frac_filled:.2f}  "
              f"string-like={frac_str:.2f}  "
              f"sample={[str(v)[:12] for v in row.tolist()[:4]]}")

        # Header row: at least 40% filled, and 70% of those are string-like
        if frac_filled >= 0.4 and frac_str >= 0.7:
            print(f"  → Chosen header row: {i}")
            return i

    print(f"  → No header detected, defaulting to row 0")
    return 0


# ============================================================
# ROBUST EXCEL LOADER
# ============================================================
def load_swat_excel(path, sheet=0):
    """Load a SWaT Excel file with automatic header detection."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    print(f"\nLoading: {os.path.basename(path)}")
    header_row = detect_header_row(path, sheet=sheet)

    df = pd.read_excel(path, sheet_name=sheet, header=header_row,
                       engine='openpyxl')

    # Clean column names
    df.columns = df.columns.astype(str).str.strip()

    # Drop only truly empty phantom columns
    phantom = [c for c in df.columns if c.startswith('Unnamed:')]
    if phantom:
        print(f"  Dropping {len(phantom)} phantom columns: {phantom[:3]}...")
        df = df.drop(columns=phantom)

    print(f"  Shape after load: {df.shape}")
    print(f"  First 5 columns: {df.columns[:5].tolist()}")
    print(f"  Last 3 columns:  {df.columns[-3:].tolist()}")

    # Sanity check
    if len(df.columns) < 10:
        raise ValueError(
            f"Only {len(df.columns)} columns detected after cleanup. "
            f"Header detection likely failed. Check the Excel file layout."
        )

    return df


# ============================================================
# CLEAN FUNCTION
# ============================================================
def clean_swat(df, is_attack=False):
    """
    Given a raw SWaT DataFrame:
      - Drops time columns
      - Returns (X_features, y_labels or None, feature_names)
    """
    df = df.copy()

    # Drop time columns
    time_cols = [c for c in df.columns if 'time' in c.lower()]
    if time_cols:
        print(f"  Dropping time columns: {time_cols}")
        df = df.drop(columns=time_cols, errors='ignore')

    if is_attack:
        # Last column = labels
        label_col = df.columns[-1]
        print(f"  Label column: '{label_col}'")
        raw_labels = df[label_col].astype(str).str.strip().str.lower()
        print(f"  Label values: {raw_labels.unique()[:5]}")

        # Accept 'attack', 'anomaly', '1' as positive; 'normal', '0' as negative
        y = raw_labels.isin(['attack', 'anomaly', '1', 'true', 'yes']).astype(np.int8).values
        X = df.drop(columns=[label_col])
        return X, y, None
    else:
        # Last column often has 'Normal/Attack' but all Normal — drop it
        if df.columns[-1].lower().startswith(('normal', 'label', 'class')):
            print(f"  Dropping unused label column: '{df.columns[-1]}'")
            df = df.drop(columns=[df.columns[-1]])
        return df, None, None


# ============================================================
# MAIN
# ============================================================
def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    if os.path.exists(CACHE_FILE):
        size_mb = os.path.getsize(CACHE_FILE) / 1e6
        print(f"Cache already exists: {CACHE_FILE} ({size_mb:.1f} MB)")
        print("Delete it to regenerate.")
        return

    t0 = time.time()
    print("=" * 60)
    print("PREPARING SWaT DATA CACHE")
    print("=" * 60)

    normal_path = os.path.join(DATA_DIR, NORMAL_FILE)
    attack_path = os.path.join(DATA_DIR, ATTACK_FILE)

    # ---- Load ----
    df_normal = load_swat_excel(normal_path)
    df_attack = load_swat_excel(attack_path)

    # ---- Clean ----
    print("\nCleaning normal...")
    df_normal, _, _ = clean_swat(df_normal, is_attack=False)

    print("\nCleaning attack...")
    X_attack_df, y_attack, _ = clean_swat(df_attack, is_attack=True)

    # ---- Align columns ----
    common_cols = [c for c in df_normal.columns if c in X_attack_df.columns]
    print(f"\nCommon feature columns: {len(common_cols)}")
    if len(common_cols) < 20:
        raise ValueError(
            f"Only {len(common_cols)} common columns. "
            f"Normal cols: {df_normal.columns[:5].tolist()}... | "
            f"Attack cols: {X_attack_df.columns[:5].tolist()}..."
        )

    # ---- Extract numeric arrays ----
    X_normal = (df_normal[common_cols]
                .apply(pd.to_numeric, errors='coerce')
                .ffill().fillna(0)
                .astype(np.float32)
                .values)

    X_attack = (X_attack_df[common_cols]
                .apply(pd.to_numeric, errors='coerce')
                .ffill().fillna(0)
                .astype(np.float32)
                .values)

    print(f"\nFinal arrays:")
    print(f"  X_normal: {X_normal.shape}  {X_normal.dtype}")
    print(f"  X_attack: {X_attack.shape}  {X_attack.dtype}")
    print(f"  y_attack: {y_attack.shape}  {y_attack.dtype}")
    print(f"  Positive labels: {y_attack.sum()} / {len(y_attack)} "
          f"({100*y_attack.mean():.2f}%)")

    # ---- Save ----
    print(f"\nSaving to {CACHE_FILE}...")
    np.savez_compressed(
        CACHE_FILE,
        X_normal=X_normal,
        X_attack=X_attack,
        y_attack=y_attack,
        features=np.array(common_cols, dtype=object),
    )

    size_mb = os.path.getsize(CACHE_FILE) / 1e6
    print(f"Done in {time.time()-t0:.1f}s | Cache size: {size_mb:.1f} MB")
    print("\n✓ Every algorithm can now load this in ~1 second.")


if __name__ == "__main__":
    main()