# ============================================================================
# USAD (UnSupervised Anomaly Detection) — STANDARD VERSION
# ============================================================================
# Original paper: Audibert et al., KDD 2020
#
# This version uses STANDARD threshold calibration:
#   Threshold from last N windows of train+val (no test-normal pool).
#
# Architecture: shared encoder + 2 decoders, adversarial training.
#
# RUN:
#   python usad_standard.py
#
# RUNTIME: ~15-20 min on RTX 4070 Ti (4 grid combos)
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*n_jobs value.*")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    precision_recall_curve, auc
)

from data_loader import load_swat


# ============================================================================
# 0. DEVICE + REPRODUCIBILITY
# ============================================================================
RANDOM_STATE = 23
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_STATE)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    DEVICE = torch.device("cuda")
    print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
else:
    DEVICE = torch.device("cpu")
    print("⚠ CUDA not available — falling back to CPU")

USE_AMP = DEVICE.type == "cuda"
print(f"  Mixed precision (AMP): {'ON' if USE_AMP else 'OFF'}")


# ============================================================================
# 1. WINDOWING
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


# ============================================================================
# 2. USAD MODEL
# ============================================================================
class USAD(nn.Module):
    def __init__(self, n_features, window_size, hidden_dim=64, latent_dim=16):
        super().__init__()
        self.n_features = n_features
        self.window_size = window_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        input_dim = window_size * n_features

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim), nn.ReLU(),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, input_dim), nn.Sigmoid(),
        )

    def forward(self, x):
        batch = x.size(0)
        x_flat = x.view(batch, -1)
        z = self.encoder(x_flat)
        w1 = self.decoder1(z)
        w2 = self.decoder2(z)
        w3 = self.decoder2(self.encoder(w1))
        w1 = w1.view(batch, self.window_size, self.n_features)
        w2 = w2.view(batch, self.window_size, self.n_features)
        w3 = w3.view(batch, self.window_size, self.n_features)
        return w1, w2, w3


# ============================================================================
# 3. TRAINING
# ============================================================================
def train_usad(X_train_w, X_val_w, n_features, window_size,
               hidden_dim=64, latent_dim=16,
               epochs=50, batch_size=128, lr=1e-3,
               patience=7, verbose=True):
    model = USAD(n_features, window_size, hidden_dim, latent_dim).to(DEVICE)
    opt1 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder1.parameters()), lr=lr)
    opt2 = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.decoder2.parameters()), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train_w))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=0, pin_memory=(DEVICE.type == "cuda"))

    X_val_t = torch.from_numpy(X_val_w).to(DEVICE, non_blocking=True)
    scaler = GradScaler('cuda', enabled=USE_AMP)

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss1, tr_loss2 = 0.0, 0.0
        n = epoch

        for (batch,) in train_dl:
            batch = batch.to(DEVICE, non_blocking=True)

            # ---- Update Decoder 1 ----
            opt1.zero_grad(set_to_none=True)
            if USE_AMP:
                with autocast('cuda'):
                    w1, w2, w3 = model(batch)
                    loss1 = (1.0/n) * loss_fn(w1, batch) + (1.0 - 1.0/n) * loss_fn(w3, batch)
                scaler.scale(loss1).backward()
                scaler.unscale_(opt1)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt1); scaler.update()
            else:
                w1, w2, w3 = model(batch)
                loss1 = (1.0/n) * loss_fn(w1, batch) + (1.0 - 1.0/n) * loss_fn(w3, batch)
                loss1.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt1.step()

            # ---- Update Decoder 2 ----
            opt2.zero_grad(set_to_none=True)
            if USE_AMP:
                with autocast('cuda'):
                    w1, w2, w3 = model(batch)
                    loss2 = (1.0/n) * loss_fn(w2, batch) - (1.0 - 1.0/n) * loss_fn(w3, batch)
                scaler.scale(loss2).backward()
                scaler.unscale_(opt2)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt2); scaler.update()
            else:
                w1, w2, w3 = model(batch)
                loss2 = (1.0/n) * loss_fn(w2, batch) - (1.0 - 1.0/n) * loss_fn(w3, batch)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt2.step()

            tr_loss1 += loss1.item() * batch.size(0)
            tr_loss2 += loss2.item() * batch.size(0)

        tr_loss1 /= len(train_ds)
        tr_loss2 /= len(train_ds)

        model.eval()
        with torch.no_grad():
            if USE_AMP:
                with autocast('cuda'):
                    v_w1, _, _ = model(X_val_t)
                    v_loss = loss_fn(v_w1, X_val_t).item()
            else:
                v_w1, _, _ = model(X_val_t)
                v_loss = loss_fn(v_w1, X_val_t).item()

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"  Epoch {epoch:02d} | L1={tr_loss1:.5f} | L2={tr_loss2:.5f} | val={v_loss:.5f}")

        if v_loss < best_val - 1e-6:
            best_val = v_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


# ============================================================================
# 4. SCORING (FP32)
# ============================================================================
@torch.no_grad()
def usad_scores(model, X_w, alpha=0.5, beta=0.5, batch_size=1024):
    model.eval()
    N = len(X_w)
    scores = np.zeros(N, dtype=np.float32)
    X_t = torch.from_numpy(X_w)
    for i in range(0, N, batch_size):
        batch = X_t[i:i + batch_size].to(DEVICE, non_blocking=True)
        with autocast('cuda', enabled=False):
            w1, w2, _ = model(batch)
        err1 = ((w1.float() - batch.float()) ** 2).mean(dim=(1, 2))
        err2 = ((w2.float() - batch.float()) ** 2).mean(dim=(1, 2))
        scores[i:i + batch_size] = (alpha * err1 + beta * err2).cpu().numpy()
    return scores


# ============================================================================
# 5. POINT-ADJUSTED METRICS
# ============================================================================
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
# 6. GRID SEARCH
# ============================================================================
def evaluate_usad(X_train_raw, X_val_raw, X_tune_raw, y_tune,
                  params, threshold_pct):
    all_raw = np.vstack([X_train_raw, X_tune_raw])
    scaler = MinMaxScaler()
    scaler.fit(all_raw)

    X_tr = np.clip(scaler.transform(X_train_raw).astype(np.float32), 0, 1)
    X_v  = np.clip(scaler.transform(X_val_raw).astype(np.float32), 0, 1)
    X_t  = np.clip(scaler.transform(X_tune_raw).astype(np.float32), 0, 1)

    W = params['window_size']
    n_features = X_tr.shape[1]

    X_tr_w = make_windows(X_tr, W)
    X_v_w  = make_windows(X_v, W)
    X_t_w, y_t_w = make_windows_with_labels(X_t, y_tune, W)

    model, val_loss = train_usad(
        X_tr_w, X_v_w, n_features, W,
        hidden_dim=params['hidden_dim'],
        latent_dim=params['latent_dim'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        patience=7,
        verbose=False,
    )

    val_scores_w  = usad_scores(model, X_v_w)
    tune_scores_w = usad_scores(model, X_t_w)

    threshold = np.percentile(val_scores_w, threshold_pct)
    y_pred = (tune_scores_w > threshold).astype(int)

    precision = precision_score(y_t_w, y_pred, zero_division=0)
    recall = recall_score(y_t_w, y_pred, zero_division=0)
    f1 = f1_score(y_t_w, y_pred, zero_division=0)

    if y_t_w.sum() > 0:
        p_curve, r_curve, _ = precision_recall_curve(y_t_w, tune_scores_w)
        pr_auc = auc(r_curve, p_curve)
    else:
        pr_auc = 0.0

    return {
        'threshold': threshold, 'precision': precision, 'recall': recall,
        'f1': f1, 'pr_auc': pr_auc, 'val_loss': val_loss,
    }


def grid_search_usad(param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\nGrid search: {len(combos)} combinations")
    print("-" * 110)
    print(f"{'#':<4}{'win':<6}{'hid':<6}{'lat':<6}{'ep':<5}{'bs':<6}"
          f"{'lr':<8}{'thr%':<6}"
          f"{'F1':<9}{'Prec':<9}{'Rec':<9}{'PR-AUC':<9}{'time':<7}")
    print("-" * 110)

    results, best, t_start = [], None, time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        try:
            metrics = evaluate_usad(
                X_train_raw, X_val_raw, X_tune_raw, y_tune,
                params, threshold_pct=params['threshold_pct'],
            )
            elapsed = time.time() - t0
            result = {**params, **metrics, 'time_s': elapsed}
            results.append(result)

            print(f"{i:<4}{params['window_size']:<6}"
                  f"{params['hidden_dim']:<6}{params['latent_dim']:<6}"
                  f"{params['epochs']:<5}{params['batch_size']:<6}"
                  f"{params['lr']:<8}{params['threshold_pct']:<6}"
                  f"{metrics['f1']:<9.4f}{metrics['precision']:<9.4f}"
                  f"{metrics['recall']:<9.4f}{metrics['pr_auc']:<9.4f}"
                  f"{elapsed:<7.1f}s")

            if best is None or metrics['f1'] > best['f1']:
                best = result
        except Exception as e:
            print(f"{i:<4} SKIP {params} -> {type(e).__name__}: {e}")

    print("-" * 110)
    print(f"Grid search done in {time.time()-t_start:.1f}s")
    print(f"\nBEST PARAMS:")
    for k in keys:
        print(f"  {k}: {best[k]}")
    print(f"BEST TUNE F1: {best['f1']:.4f}")
    return best, pd.DataFrame(results)


# ============================================================================
# 7. MAIN
# ============================================================================
def main():
    t_start_main = time.time()
    print("=" * 70)
    print(f"USAD ON SWaT — STANDARD PROTOCOL (seed={RANDOM_STATE})")
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
        'window_size':   [30, 60],
        'hidden_dim':    [64],
        'latent_dim':    [16],
        'epochs':        [50],
        'batch_size':    [128],
        'lr':            [5e-4, 1e-3],
        'threshold_pct': [99, 99.5],
    }

    best, results_df = grid_search_usad(
        param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune,
    )

    # ---- Robust scaling + final model ----
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
    n_features = X_trainval.shape[1]

    X_trainval_w = make_windows(X_trainval, W)
    X_test_w, y_test_w = make_windows_with_labels(X_test, y_test, W)

    n_val = min(5000, len(X_trainval_w) // 10)
    X_train_only_w = X_trainval_w[:-n_val]
    X_val_only_w = X_trainval_w[-n_val:]

    print(f"  Train windows: {X_train_only_w.shape}")
    print(f"  Val windows:   {X_val_only_w.shape}")
    print(f"  Test windows:  {X_test_w.shape}")

    model_final, val_loss = train_usad(
        X_train_only_w, X_val_only_w, n_features, W,
        hidden_dim=best['hidden_dim'],
        latent_dim=best['latent_dim'],
        epochs=best['epochs'],
        batch_size=best['batch_size'],
        lr=best['lr'],
        patience=7,
        verbose=True,
    )

    # ---- STANDARD threshold calibration ----
    print("\n[3/7] Threshold calibration (STANDARD — validation-based)")
    n_holdout = min(5000, len(X_trainval_w) // 10)
    X_threshold_pool = X_trainval_w[-n_holdout:]
    print(f"  Threshold pool: last {n_holdout} windows of train+val (all normal)")

    threshold_scores = usad_scores(model_final, X_threshold_pool)
    final_threshold = np.percentile(threshold_scores, best['threshold_pct'])
    print(f"  Threshold: {final_threshold:.6f}")

    # ---- Evaluation ----
    print("\n[4/7] Final test evaluation")
    test_scores = usad_scores(model_final, X_test_w)
    y_pred = (test_scores > final_threshold).astype(int)

    # Point-wise metrics
    precision = precision_score(y_test_w, y_pred, zero_division=0)
    recall = recall_score(y_test_w, y_pred, zero_division=0)
    f1 = f1_score(y_test_w, y_pred, zero_division=0)

    # PR curve data (needed for plotting)
    p_curve, r_curve, _ = precision_recall_curve(y_test_w, test_scores)
    pr_auc = auc(r_curve, p_curve)

    # Point-adjusted metrics
    pa_p, pa_r, pa_f1 = point_adjusted_metrics(y_test_w, y_pred)

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS — USAD (STANDARD)")
    print("=" * 70)
    print(f"Best params:      window={W}, hidden={best['hidden_dim']}, "
          f"latent={best['latent_dim']}")
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

    # ---- Diagnostics ----
    print("\n[5/7] Diagnostics")
    normal_scores = test_scores[y_test_w == 0]
    attack_scores = test_scores[y_test_w == 1]
    print(f"  Test normal mean: {normal_scores.mean():.6f}")
    print(f"  Test attack mean: {attack_scores.mean():.6f}")
    if attack_scores.mean() > normal_scores.mean():
        ratio = attack_scores.mean() / (normal_scores.mean() + 1e-9)
        print(f"  Attack/Normal ratio: {ratio:.2f}x")

    # ---- Save ----
    print("\n[6/7] Saving artifacts")
    exec_time = time.time() - t_start_main
    torch.save(model_final.state_dict(), 'swat_usad_standard.pt')
    joblib.dump({
        'scaler': scaler_final,
        'threshold': final_threshold,
        'features': common_cols,
        'best_params': {k: best[k] for k in param_grid.keys()},
        'random_state': RANDOM_STATE,
         'exec_time_s': exec_time,
        'threshold_source': 'validation (standard)',
        'architecture': {
            'n_features': n_features,
            'window_size': W,
            'hidden_dim': best['hidden_dim'],
            'latent_dim': best['latent_dim'],
        },
        'metrics': {
            'precision': precision, 'recall': recall, 'f1': f1,
            'pr_auc': pr_auc,
            'pa_precision': pa_p, 'pa_recall': pa_r, 'pa_f1': pa_f1,
        },
    }, 'swat_usad_standard_meta.pkl')
    results_df['exec_time_s'] = exec_time  # ← add this
    results_df.to_csv('usad_standard_grid_results.csv', index=False)
    print("  Saved: swat_usad_standard.pt, swat_usad_standard_meta.pkl")

    # ========================================================================
    # PLOTS — THIS MODEL ONLY
    # ========================================================================
    print("\n[7/7] Generating plots...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (0) PR Curve
    axes[0].plot(r_curve, p_curve, color='crimson',
                 label=f'USAD standard (AUC={pr_auc:.3f})', lw=2)
    axes[0].scatter(recall, precision, color='red', s=80,
                    label=f'Operating point\nF1={f1:.3f} (point-wise)\n'
                          f'F1={pa_f1:.3f} (point-adjusted)')
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve (USAD — Standard)')
    axes[0].legend(loc='lower left', fontsize=8)
    axes[0].grid(alpha=0.3)

    # (1) Score distribution
    axes[1].hist(normal_scores, bins=100, alpha=0.5, label='Normal',
                 color='blue', density=True)
    axes[1].hist(attack_scores, bins=100, alpha=0.5, label='Attack',
                 color='red', density=True)
    axes[1].axvline(final_threshold, color='black', ls='--',
                    label=f'Threshold={final_threshold:.4f}')
    axes[1].set_xlabel('Anomaly score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Score Distribution')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # (2) Time-series scores
    axes[2].plot(test_scores, lw=0.5, color='crimson', alpha=0.8)
    attack_idx = np.where(y_test_w == 1)[0]
    if len(attack_idx) > 0:
        breaks = np.where(np.diff(attack_idx) > 1)[0]
        starts = np.r_[attack_idx[0], attack_idx[breaks + 1]]
        ends = np.r_[attack_idx[breaks], attack_idx[-1]]
        for s, e in zip(starts, ends):
            axes[2].axvspan(s, e, color='red', alpha=0.15)
    axes[2].axhline(final_threshold, color='red', ls='--',
                    label='Threshold')
    axes[2].set_xlabel('Time step')
    axes[2].set_ylabel('Score')
    axes[2].set_title('USAD Anomaly Scores (Standard)')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('usad_standard_results.png', dpi=120)
    plt.show()
    print("  Saved: usad_standard_results.png")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"  Point-wise F1:     {f1:.4f}")
    print(f"  Point-adjusted F1: {pa_f1:.4f}")
    print(f"  PR-AUC:            {pr_auc:.4f}")
    print("=" * 70)
    print(f"  Total execution time: {exec_time:.1f}s")

if __name__ == '__main__':
    main()