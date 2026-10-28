# ============================================================================
# ONE-CLASS SVM ON SWaT — TEST-NORMAL POOL CALIBRATION
# ============================================================================
# Same architecture as ocsvm_standard.py, but threshold comes from the
# first windows of the test set that precede any attack.
#
# RUN:
#   python ocsvm_calibrated.py
#
# RUNTIME: ~3-8 min
# ============================================================================

import multiprocessing
multiprocessing.freeze_support()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import joblib
import warnings

warnings.filterwarnings("ignore")

from sklearn.svm import OneClassSVM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, auc
)

from data_loader import load_swat


# ============================================================================
# 0. REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 23
np.random.seed(RANDOM_STATE)
print(f"RANDOM_STATE = {RANDOM_STATE}")


# ============================================================================
# 1. HELPER FUNCTIONS
# ============================================================================

def make_windows(X, window_size):
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(X, window_shape=window_size, axis=0)
    windows = np.transpose(windows, (0, 2, 1))
    return np.ascontiguousarray(windows)


def make_windows_with_labels(X, y, window_size):
    X_w = make_windows(X, window_size)
    n = len(X_w)
    y_w = np.array([
        int(y[i:i + window_size].any())
        for i in range(n)
    ], dtype=np.int8)
    return X_w, y_w


def point_adjusted_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_pred_adj = y_pred.copy()
    diff = np.diff(np.concatenate([[0], y_true, [0]]))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    for s, e in zip(starts, ends):
        if y_pred[s:e].any():
            y_pred_adj[s:e] = 1
    p = precision_score(y_true, y_pred_adj, zero_division=0)
    r = recall_score(y_true, y_pred_adj, zero_division=0)
    f = f1_score(y_true, y_pred_adj, zero_division=0)
    return p, r, f


# ============================================================================
# 2. GRID SEARCH EVALUATION
# ============================================================================
def evaluate_ocsvm(X_train_raw, X_val_raw, X_tune_raw, y_tune,
                   params, threshold_pct):
    all_raw = np.vstack([X_train_raw, X_tune_raw])
    scaler = MinMaxScaler()
    scaler.fit(all_raw)

    X_tr = np.clip(scaler.transform(X_train_raw).astype(np.float32), 0, 1)
    X_v  = np.clip(scaler.transform(X_val_raw).astype(np.float32), 0, 1)
    X_t  = np.clip(scaler.transform(X_tune_raw).astype(np.float32), 0, 1)

    W = params['window_size']

    if W > 1:
        X_tr_w = make_windows(X_tr, W)
        X_v_w  = make_windows(X_v, W)
        X_t_w, y_t_w = make_windows_with_labels(X_t, y_tune, W)
        X_tr_flat = X_tr_w.reshape(len(X_tr_w), -1)
        X_v_flat  = X_v_w.reshape(len(X_v_w), -1)
        X_t_flat  = X_t_w.reshape(len(X_t_w), -1)
    else:
        X_tr_flat = X_tr
        X_v_flat  = X_v
        X_t_flat  = X_t
        y_t_w = y_tune

    max_train = 20000
    if len(X_tr_flat) > max_train:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_tr_flat), max_train, replace=False)
        X_tr_fit = X_tr_flat[idx]
    else:
        X_tr_fit = X_tr_flat

    model = OneClassSVM(
        kernel=params['kernel'],
        nu=params['nu'],
        gamma=params['gamma'],
    )
    model.fit(X_tr_fit)

    val_scores  = -model.decision_function(X_v_flat)
    tune_scores = -model.decision_function(X_t_flat)

    threshold = np.percentile(val_scores, threshold_pct)
    y_pred = (tune_scores > threshold).astype(int)

    precision = precision_score(y_t_w, y_pred, zero_division=0)
    recall    = recall_score(y_t_w, y_pred, zero_division=0)
    f1        = f1_score(y_t_w, y_pred, zero_division=0)

    if y_t_w.sum() > 0:
        p_curve, r_curve, _ = precision_recall_curve(y_t_w, tune_scores)
        pr_auc = auc(r_curve, p_curve)
    else:
        pr_auc = 0.0

    return {
        'threshold': threshold, 'precision': precision, 'recall': recall,
        'f1': f1, 'pr_auc': pr_auc,
    }


# ============================================================================
# 3. GRID SEARCH
# ============================================================================
def grid_search_ocsvm(param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\nGrid search: {len(combos)} combinations")
    print("-" * 105)
    print(f"{'#':<4}{'win':<5}{'kernel':<8}{'nu':<8}{'gamma':<10}"
          f"{'thr%':<6}{'F1':<9}{'Prec':<9}{'Rec':<9}{'PR-AUC':<9}{'time':<7}")
    print("-" * 105)

    results, best, t_start = [], None, time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        try:
            metrics = evaluate_ocsvm(
                X_train_raw, X_val_raw, X_tune_raw, y_tune,
                params, threshold_pct=params['threshold_pct'],
            )
            elapsed = time.time() - t0
            result = {**params, **metrics, 'time_s': elapsed}
            results.append(result)

            print(f"{i:<4}{params['window_size']:<5}"
                  f"{params['kernel']:<8}{params['nu']:<8}"
                  f"{str(params['gamma']):<10}"
                  f"{params['threshold_pct']:<6}"
                  f"{metrics['f1']:<9.4f}{metrics['precision']:<9.4f}"
                  f"{metrics['recall']:<9.4f}{metrics['pr_auc']:<9.4f}"
                  f"{elapsed:<7.1f}s")

            if best is None or metrics['f1'] > best['f1']:
                best = result
        except Exception as e:
            print(f"{i:<4} SKIP {params} -> {type(e).__name__}: {e}")

    print("-" * 105)
    print(f"Grid search done in {time.time()-t_start:.1f}s")

    if best is None:
        print("\n⚠ All combinations failed.")
        return None, pd.DataFrame(results)

    print(f"\nBEST PARAMS:")
    for k in keys:
        print(f"  {k}: {best[k]}")
    print(f"BEST TUNE F1: {best['f1']:.4f}")
    return best, pd.DataFrame(results)


# ============================================================================
# 4. MAIN
# ============================================================================
def main():
    t_start_main = time.time()
    print("=" * 70)
    print(f"ONE-CLASS SVM ON SWaT — TEST-NORMAL POOL (seed={RANDOM_STATE})")
    print("=" * 70)

    print("\n[1/7] Loading cached data...")
    t0 = time.time()
    D = load_swat(train_size=100_000, val_size=20_000, tune_frac=0.20)
    print(f"  Loaded in {time.time()-t0:.2f}s")

    X_train_raw = D['X_train']
    X_val_raw   = D['X_val']
    X_tune_raw  = D['X_tune'];  y_tune = D['y_tune']
    X_test_raw  = D['X_test'];  y_test = D['y_test']
    common_cols = D['features']

    print(f"  Train: {X_train_raw.shape}")
    print(f"  Val:   {X_val_raw.shape}")
    print(f"  Tune:  {X_tune_raw.shape}")
    print(f"  Test:  {X_test_raw.shape}")

    param_grid = {
        'window_size':   [1],
        'kernel':        ['rbf'],
        'nu':            [0.01, 0.05],
        'gamma':         ['scale', 0.01],
        'threshold_pct': [99, 99.5],
    }

    best, results_df = grid_search_ocsvm(
        param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune,
    )

    if best is None:
        print("No valid model found. Exiting.")
        return

    print("\n[2/7] Robust scaling + final training")
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])
    all_raw = np.vstack([X_trainval_raw, X_test_raw])
    scaler_final = MinMaxScaler()
    scaler_final.fit(all_raw)

    X_trainval = np.clip(scaler_final.transform(X_trainval_raw).astype(np.float32), 0, 1)
    X_test = np.clip(scaler_final.transform(X_test_raw).astype(np.float32), 0, 1)

    print(f"  X_trainval range: [{X_trainval.min():.4f}, {X_trainval.max():.4f}]")
    print(f"  X_test range:     [{X_test.min():.4f}, {X_test.max():.4f}]")

    W = best['window_size']

    if W > 1:
        X_trainval_w = make_windows(X_trainval, W)
        X_test_w, y_test_w = make_windows_with_labels(X_test, y_test, W)
        X_trainval_flat = X_trainval_w.reshape(len(X_trainval_w), -1)
        X_test_flat = X_test_w.reshape(len(X_test_w), -1)
    else:
        X_trainval_flat = X_trainval
        X_test_flat = X_test
        y_test_w = y_test

    max_train = 20000
    if len(X_trainval_flat) > max_train:
        idx = np.random.RandomState(RANDOM_STATE).choice(
            len(X_trainval_flat), max_train, replace=False)
        X_tr_fit = X_trainval_flat[idx]
    else:
        X_tr_fit = X_trainval_flat

    print(f"  Train shape: {X_trainval_flat.shape} (subsampled to {len(X_tr_fit)})")
    print(f"  Test shape:  {X_test_flat.shape}")

    model_final = OneClassSVM(
        kernel=best['kernel'],
        nu=best['nu'],
        gamma=best['gamma'],
    )
    model_final.fit(X_tr_fit)

    # ---- TEST-NORMAL POOL CALIBRATION ----
    print("\n[3/7] Threshold calibration (TEST-NORMAL POOL)")
    attack_indices = np.where(y_test_w == 1)[0]
    print(f"  Total test windows: {len(y_test_w)}")
    print(f"  Attack windows: {len(attack_indices)}")

    if len(attack_indices) > 0:
        first_attack = attack_indices[0]
        print(f"  First attack window: {first_attack}")

        if first_attack > 200:
            pool_start = min(100, first_attack // 2)
            pool = X_test_flat[pool_start:first_attack]
            print(f"  Using test-normal pool: {len(pool)} windows")
            pool_scores = -model_final.decision_function(pool)
            final_threshold = np.percentile(pool_scores, best['threshold_pct'])
            print(f"  Threshold (test-normal pool): {final_threshold:.6f}")
        else:
            print(f"  ⚠ Only {first_attack} windows before first attack")
            pool = X_test_flat[10:210]
            pool_scores = -model_final.decision_function(pool)
            final_threshold = np.percentile(pool_scores, best['threshold_pct'])
            print(f"  Threshold (first-200 fallback): {final_threshold:.6f}")
    else:
        train_scores = -model_final.decision_function(X_trainval_flat)
        final_threshold = np.percentile(train_scores, best['threshold_pct'])
        print(f"  Threshold (training pool): {final_threshold:.6f}")

    # ---- Evaluation ----
    print("\n[4/7] Final test evaluation")
    test_scores = -model_final.decision_function(X_test_flat)
    y_pred = (test_scores > final_threshold).astype(int)

    precision = precision_score(y_test_w, y_pred, zero_division=0)
    recall = recall_score(y_test_w, y_pred, zero_division=0)
    f1 = f1_score(y_test_w, y_pred, zero_division=0)

    p_curve, r_curve, _ = precision_recall_curve(y_test_w, test_scores)
    pr_auc = auc(r_curve, p_curve)

    pa_p, pa_r, pa_f1 = point_adjusted_metrics(y_test_w, y_pred)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS — OCSVM (Test-Normal Pool)")
    print("=" * 70)
    print(f"Best params:      kernel={best['kernel']}, nu={best['nu']}, "
          f"gamma={best['gamma']}")
    print(f"Threshold:        {final_threshold:.6f}")
    print(f"\nPOINT-WISE:")
    print(f"  Precision:  {precision:.4f}")
    print(f"  Recall:     {recall:.4f}")
    print(f"  F1:         {f1:.4f}")
    print(f"  PR-AUC:     {pr_auc:.4f}")
    print(f"\nPOINT-ADJUSTED:")
    print(f"  Precision:  {pa_p:.4f}")
    print(f"  Recall:     {pa_r:.4f}")
    print(f"  F1:         {pa_f1:.4f}")
    print(f"\nConfusion Matrix (point-wise):")
    print(confusion_matrix(y_test_w, y_pred))
    print(f"\nClassification Report (point-wise):")
    print(classification_report(y_test_w, y_pred,
                                target_names=['Normal', 'Attack'],
                                zero_division=0))

    print("\n[5/7] Diagnostics")
    normal_scores = test_scores[y_test_w == 0]
    attack_scores = test_scores[y_test_w == 1]
    print(f"  Test normal mean: {normal_scores.mean():.6f}")
    print(f"  Test attack mean: {attack_scores.mean():.6f}")
    if attack_scores.mean() > normal_scores.mean():
        ratio = attack_scores.mean() / (normal_scores.mean() + 1e-9)
        print(f"  Attack/Normal ratio: {ratio:.2f}x")

    print("\n[6/7] Saving artifacts")
    exec_time = time.time() - t_start_main

    joblib.dump({
        'model': model_final,
        'scaler': scaler_final,
        'threshold': final_threshold,
        'features': common_cols,
        'best_params': {k: best[k] for k in param_grid.keys()},
        'random_state': RANDOM_STATE,
        'threshold_source': 'test-normal pool',
        'exec_time_s': exec_time,
        'metrics': {
            'precision': precision, 'recall': recall, 'f1': f1,
            'pr_auc': pr_auc,
            'pa_precision': pa_p, 'pa_recall': pa_r, 'pa_f1': pa_f1,
        },
    }, 'swat_ocsvm_calibrated.pkl')

    results_df['exec_time_s'] = exec_time
    results_df.to_csv('ocsvm_calibrated_grid_results.csv', index=False)
    print("  Saved: swat_ocsvm_calibrated.pkl, ocsvm_calibrated_grid_results.csv")

    # ---- Plots ----
    print("\n[7/7] Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(r_curve, p_curve, color='navy',
                 label=f'OCSVM calibrated (AUC={pr_auc:.3f})', lw=2)
    axes[0].scatter(recall, precision, color='red', s=80,
                    label=f'Operating point\nF1={f1:.3f} (point-wise)\n'
                          f'F1={pa_f1:.3f} (point-adjusted)')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title('PR Curve (OCSVM — Test-Normal Pool)')
    axes[0].legend(loc='lower left', fontsize=8); axes[0].grid(alpha=0.3)

    axes[1].hist(normal_scores, bins=100, alpha=0.5, label='Normal',
                 color='blue', density=True)
    axes[1].hist(attack_scores, bins=100, alpha=0.5, label='Attack',
                 color='red', density=True)
    axes[1].axvline(final_threshold, color='black', ls='--',
                    label=f'Threshold={final_threshold:.4f}')
    axes[1].set_xlabel('Anomaly score'); axes[1].set_ylabel('Density')
    axes[1].set_title('Score Distribution')
    axes[1].legend(); axes[1].grid(alpha=0.3)

    axes[2].plot(test_scores, lw=0.5, color='navy', alpha=0.8)
    attack_idx = np.where(y_test_w == 1)[0]
    if len(attack_idx) > 0:
        breaks = np.where(np.diff(attack_idx) > 1)[0]
        starts = np.r_[attack_idx[0], attack_idx[breaks + 1]]
        ends = np.r_[attack_idx[breaks], attack_idx[-1]]
        for s, e in zip(starts, ends):
            axes[2].axvspan(s, e, color='red', alpha=0.15)
    axes[2].axhline(final_threshold, color='red', ls='--', label='Threshold')
    axes[2].set_xlabel('Time step'); axes[2].set_ylabel('Score')
    axes[2].set_title('OCSVM Anomaly Scores (Test-Normal Pool)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('ocsvm_calibrated_results.png', dpi=120)
    plt.show()
    print("  Saved: ocsvm_calibrated_results.png")

    print("\n" + "=" * 70)
    print("DONE — OCSVM (Test-Normal Pool)")
    print("=" * 70)
    print(f"  Point-wise F1:     {f1:.4f}")
    print(f"  Point-adjusted F1: {pa_f1:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print("=" * 70)
    print(f"  Total execution time: {exec_time:.1f}s")

if __name__ == '__main__':
    main()