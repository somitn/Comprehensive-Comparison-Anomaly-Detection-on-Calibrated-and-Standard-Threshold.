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

from sklearn.decomposition import PCA
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
def reconstruction_error(pca, X):
    """Per-sample MSE between input and PCA reconstruction."""
    Z = pca.transform(X)
    X_rec = pca.inverse_transform(Z)
    return np.mean((X - X_rec) ** 2, axis=1)


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
def evaluate_pca(X_train_raw, X_val_raw, X_tune_raw, y_tune,
                 params, threshold_pct):
    all_raw = np.vstack([X_train_raw, X_tune_raw])
    scaler = MinMaxScaler()
    scaler.fit(all_raw)

    X_tr = np.clip(scaler.transform(X_train_raw).astype(np.float32), 0, 1)
    X_v  = np.clip(scaler.transform(X_val_raw).astype(np.float32), 0, 1)
    X_t  = np.clip(scaler.transform(X_tune_raw).astype(np.float32), 0, 1)

    pca = PCA(n_components=params['n_components'], random_state=RANDOM_STATE)
    pca.fit(X_tr)

    val_scores  = reconstruction_error(pca, X_v)
    tune_scores = reconstruction_error(pca, X_t)

    threshold = np.percentile(val_scores, threshold_pct)
    y_pred = (tune_scores > threshold).astype(int)

    precision = precision_score(y_tune, y_pred, zero_division=0)
    recall    = recall_score(y_tune, y_pred, zero_division=0)
    f1        = f1_score(y_tune, y_pred, zero_division=0)

    if y_tune.sum() > 0:
        p_curve, r_curve, _ = precision_recall_curve(y_tune, tune_scores)
        pr_auc = auc(r_curve, p_curve)
    else:
        pr_auc = 0.0

    return {
        'threshold': threshold, 'precision': precision, 'recall': recall,
        'f1': f1, 'pr_auc': pr_auc,
        'explained_var': float(pca.explained_variance_ratio_.sum()),
    }


def grid_search_pca(param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\nGrid search: {len(combos)} combinations")
    print("-" * 95)
    print(f"{'#':<4}{'n_comp':<8}{'thr%':<6}{'expl_var':<11}"
          f"{'F1':<9}{'Prec':<9}{'Rec':<9}{'PR-AUC':<9}{'time':<7}")
    print("-" * 95)

    results, best, t_start = [], None, time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        try:
            metrics = evaluate_pca(
                X_train_raw, X_val_raw, X_tune_raw, y_tune,
                params, threshold_pct=params['threshold_pct'],
            )
            elapsed = time.time() - t0
            result = {**params, **metrics, 'time_s': elapsed}
            results.append(result)

            print(f"{i:<4}{params['n_components']:<8}"
                  f"{params['threshold_pct']:<6}"
                  f"{metrics['explained_var']:<11.4f}"
                  f"{metrics['f1']:<9.4f}{metrics['precision']:<9.4f}"
                  f"{metrics['recall']:<9.4f}{metrics['pr_auc']:<9.4f}"
                  f"{elapsed:<7.2f}s")

            if best is None or metrics['f1'] > best['f1']:
                best = result
        except Exception as e:
            print(f"{i:<4} SKIP {params} -> {type(e).__name__}: {e}")

    print("-" * 95)
    print(f"Grid search done in {time.time()-t_start:.1f}s")

    if best is None:
        return None, pd.DataFrame(results)

    print(f"\nBEST PARAMS:")
    for k in keys:
        print(f"  {k}: {best[k]}")
    print(f"BEST TUNE F1: {best['f1']:.4f}")
    return best, pd.DataFrame(results)


# ============================================================================
# 3. MAIN
# ============================================================================
def main():
    t_start_main = time.time()
    print("=" * 70)
    print(f"PCA ON SWaT — STANDARD (seed={RANDOM_STATE})")
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
        'n_components':  [5, 10, 15, 20, 30],
        'threshold_pct': [95, 99, 99.5],
    }

    best, results_df = grid_search_pca(
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

    pca_final = PCA(n_components=best['n_components'], random_state=RANDOM_STATE)
    pca_final.fit(X_trainval)

    print(f"  Final PCA: {best['n_components']} components, "
          f"explained var = {pca_final.explained_variance_ratio_.sum():.4f}")

    # ============================================================
    # STANDARD THRESHOLD CALIBRATION
    # ============================================================
    print("\n[3/7] Threshold calibration (STANDARD — training pool)")
    n_holdout = min(5000, len(X_trainval) // 10)
    X_threshold_pool = X_trainval[-n_holdout:]
    print(f"  Threshold pool: last {n_holdout} timesteps of train (all normal)")

    threshold_scores = reconstruction_error(pca_final, X_threshold_pool)
    final_threshold = np.percentile(threshold_scores, best['threshold_pct'])
    print(f"  Threshold: {final_threshold:.6f}")

    # ---- Evaluation ----
    print("\n[4/7] Final test evaluation")
    test_scores = reconstruction_error(pca_final, X_test)
    y_pred = (test_scores > final_threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    p_curve, r_curve, _ = precision_recall_curve(y_test, test_scores)
    pr_auc = auc(r_curve, p_curve)

    pa_p, pa_r, pa_f1 = point_adjusted_metrics(y_test, y_pred)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS — PCA (Standard)")
    print("=" * 70)
    print(f"Best params:      n_components={best['n_components']}")
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
    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Attack'],
                                zero_division=0))

    # ---- Diagnostics ----
    print("\n[5/7] Diagnostics")
    normal_scores = test_scores[y_test == 0]
    attack_scores = test_scores[y_test == 1]
    print(f"  Test normal mean: {normal_scores.mean():.6f}")
    print(f"  Test attack mean: {attack_scores.mean():.6f}")
    if attack_scores.mean() > normal_scores.mean():
        ratio = attack_scores.mean() / (normal_scores.mean() + 1e-9)
        print(f"  Attack/Normal ratio: {ratio:.2f}x")

    print("\n[6/7] Saving artifacts")
    exec_time = time.time() - t_start_main

    joblib.dump({
        'pca': pca_final,
        'scaler': scaler_final,
        'threshold': final_threshold,
        'features': common_cols,
        'best_params': {k: best[k] for k in param_grid.keys()},
        'random_state': RANDOM_STATE,
        'threshold_source': 'validation (standard)',
        'exec_time_s': exec_time,
        'architecture': {
            'n_components': best['n_components'],
            'explained_var': float(pca_final.explained_variance_ratio_.sum()),
        },
        'test_metrics': {
            'precision': precision, 'recall': recall,
            'f1': f1, 'pr_auc': pr_auc,
            'pa_precision': pa_p, 'pa_recall': pa_r, 'pa_f1': pa_f1,
        },
    }, 'swat_pca_standard.pkl')

    results_df['exec_time_s'] = exec_time
    results_df.to_csv('pca_standard_grid_results.csv', index=False)
    print("  Saved: swat_pca_standard.pkl, pca_standard_grid_results.csv")

    # ---- Plots ----
    print("\n[7/7] Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(r_curve, p_curve, color='steelblue',
                 label=f'PCA standard (AUC={pr_auc:.3f})', lw=2)
    axes[0].scatter(recall, precision, color='red', s=80,
                    label=f'Operating point\nF1={f1:.3f} (point-wise)\n'
                          f'F1={pa_f1:.3f} (point-adjusted)')
    axes[0].set_xlabel('Recall'); axes[0].set_ylabel('Precision')
    axes[0].set_title('PR Curve (PCA — Standard)')
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

    axes[2].plot(test_scores, lw=0.5, color='steelblue', alpha=0.8)
    attack_idx = np.where(y_test == 1)[0]
    if len(attack_idx) > 0:
        breaks = np.where(np.diff(attack_idx) > 1)[0]
        starts = np.r_[attack_idx[0], attack_idx[breaks + 1]]
        ends = np.r_[attack_idx[breaks], attack_idx[-1]]
        for s, e in zip(starts, ends):
            axes[2].axvspan(s, e, color='red', alpha=0.15)
    axes[2].axhline(final_threshold, color='red', ls='--',
                    label=f'Threshold={final_threshold:.4f}')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('Reconstruction error')
    axes[2].set_title('PCA Anomaly Scores (Standard)')
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('pca_standard_results.png', dpi=120)
    plt.show()
    print("  Saved: pca_standard_results.png")

    print("\n" + "=" * 70)
    print("DONE — PCA (Standard)")
    print("=" * 70)
    print(f"  Point-wise F1:     {f1:.4f}")
    print(f"  Point-adjusted F1: {pa_f1:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print(f"  Total time:        {exec_time:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()