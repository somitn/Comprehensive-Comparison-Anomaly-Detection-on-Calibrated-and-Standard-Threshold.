# ============================================================================
# DEEPSVDD ON SWaT — BOTH CALIBRATION PROTOCOLS
# ============================================================================
# DeepSVDD (Ruff et al., ICML 2018) learns a hypersphere around normal data.
# Anomaly score = distance from the learned center.
#
# RUN:
#   python deepsvdd_both.py
#
# RUNTIME: ~5-10 min on RTX 4070 Ti
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

import torch
import torch.nn as nn
import torch.optim as optim
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
else:
    DEVICE = torch.device("cpu")
    print("⚠ CUDA not available")

USE_AMP = DEVICE.type == "cuda"
print(f"  Mixed precision (AMP): {'ON' if USE_AMP else 'OFF'}")


# ============================================================================
# 1. DEEPSVDD MODEL
# ============================================================================
class DeepSVDD(nn.Module):
    """
    DeepSVDD: Maps normal data to a hypersphere center c.
    Loss = mean(||phi(x) - c||^2)
    Anomaly score = ||phi(x) - c||^2
    """
    def __init__(self, n_features, hidden_dims=(64, 32), latent_dim=16,
                 dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.latent_dim = latent_dim

        layers = []
        prev = n_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*layers)

        # Center is learned as a buffer (not a parameter)
        self.register_buffer('center', torch.zeros(latent_dim))

    def forward(self, x):
        return self.encoder(x)

    def init_center(self, X_train, eps=0.01):
        """Initialize center as mean of encoder outputs on training data."""
        self.eval()
        with torch.no_grad():
            X_t = torch.from_numpy(X_train).to(DEVICE)
            embeddings = self.forward(X_t)
            self.center = embeddings.mean(dim=0)
        print(f"  Center initialized: {self.center.cpu().numpy()[:3]}...")

    def anomaly_score(self, x):
        """Score = squared distance from center."""
        emb = self.forward(x)
        return torch.sum((emb - self.center) ** 2, dim=1)


# ============================================================================
# 2. TRAINING
# ============================================================================
def train_deepsvdd(X_train, X_val, n_features,
                   hidden_dims=(64, 32), latent_dim=16, dropout=0.1,
                   epochs=100, batch_size=64, lr=1e-3,
                   patience=10, verbose=True):
    """
    DeepSVDD training with center initialization.
    Parameters follow SWaT literature: batch_size=64, hidden_dims=(64,32).
    """
    model = DeepSVDD(n_features, hidden_dims, latent_dim, dropout).to(DEVICE)

    # Initialize center on training data (required by DeepSVDD)
    model.init_center(X_train)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler_amp = GradScaler('cuda', enabled=USE_AMP)

    train_ds = TensorDataset(torch.from_numpy(X_train))
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=(DEVICE.type == "cuda"),
    )
    X_val_t = torch.from_numpy(X_val).to(DEVICE, non_blocking=True)

    best_val = float('inf')
    best_state = None
    wait = 0

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0

        for (batch,) in train_dl:
            batch = batch.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with autocast('cuda'):
                    emb = model(batch)
                    loss = torch.mean(torch.sum((emb - model.center) ** 2, dim=1))
                scaler_amp.scale(loss).backward()
                scaler_amp.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler_amp.step(optimizer)
                scaler_amp.update()
            else:
                emb = model(batch)
                loss = torch.mean(torch.sum((emb - model.center) ** 2, dim=1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            tr_loss += loss.item() * batch.size(0)

        tr_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            if USE_AMP:
                with autocast('cuda'):
                    v_emb = model(X_val_t)
                    v_loss = torch.mean(torch.sum((v_emb - model.center) ** 2, dim=1)).item()
            else:
                v_emb = model(X_val_t)
                v_loss = torch.mean(torch.sum((v_emb - model.center) ** 2, dim=1)).item()

        if verbose and (epoch % 10 == 0 or epoch == 1):
            print(f"  Epoch {epoch:03d} | train={tr_loss:.6f} | val={v_loss:.6f}")

        if v_loss < best_val - 1e-8:
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
# 3. SCORING
# ============================================================================
@torch.no_grad()
def deepsvdd_scores(model, X, batch_size=4096):
    """Score = squared distance from learned center."""
    model.eval()
    N = len(X)
    scores = np.zeros(N, dtype=np.float32)
    X_t = torch.from_numpy(X)

    for i in range(0, N, batch_size):
        batch = X_t[i:i + batch_size].to(DEVICE, non_blocking=True)
        with autocast('cuda', enabled=False):
            dist = model.anomaly_score(batch)
        scores[i:i + batch_size] = dist.cpu().numpy()
    return scores


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
# 4. GRID SEARCH
# ============================================================================
def evaluate_deepsvdd(X_train_raw, X_val_raw, X_tune_raw, y_tune,
                      params, threshold_pct):
    all_raw = np.vstack([X_train_raw, X_tune_raw])
    scaler = MinMaxScaler()
    scaler.fit(all_raw)

    X_tr = np.clip(scaler.transform(X_train_raw).astype(np.float32), 0, 1)
    X_v  = np.clip(scaler.transform(X_val_raw).astype(np.float32), 0, 1)
    X_t  = np.clip(scaler.transform(X_tune_raw).astype(np.float32), 0, 1)

    n_features = X_tr.shape[1]

    model, val_loss = train_deepsvdd(
        X_tr, X_v, n_features,
        hidden_dims=params['hidden_dims'],
        latent_dim=params['latent_dim'],
        dropout=params['dropout'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        patience=10,
        verbose=False,
    )

    val_scores  = deepsvdd_scores(model, X_v)
    tune_scores = deepsvdd_scores(model, X_t)

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
        'f1': f1, 'pr_auc': pr_auc, 'val_loss': val_loss,
    }


def grid_search_deepsvdd(param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\nGrid search: {len(combos)} combinations")
    print("-" * 115)
    print(f"{'#':<4}{'hid_dims':<14}{'lat':<6}{'drop':<7}"
          f"{'ep':<5}{'bs':<6}{'lr':<8}{'thr%':<6}"
          f"{'F1':<9}{'Prec':<9}{'Rec':<9}{'PR-AUC':<9}{'time':<7}")
    print("-" * 115)

    results, best, t_start = [], None, time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        try:
            metrics = evaluate_deepsvdd(
                X_train_raw, X_val_raw, X_tune_raw, y_tune,
                params, threshold_pct=params['threshold_pct'],
            )
            elapsed = time.time() - t0
            result = {**params, **metrics, 'time_s': elapsed}
            results.append(result)

            hd_str = str(params['hidden_dims'])
            print(f"{i:<4}{hd_str:<14}{params['latent_dim']:<6}"
                  f"{params['dropout']:<7}{params['epochs']:<5}"
                  f"{params['batch_size']:<6}{params['lr']:<8}"
                  f"{params['threshold_pct']:<6}"
                  f"{metrics['f1']:<9.4f}{metrics['precision']:<9.4f}"
                  f"{metrics['recall']:<9.4f}{metrics['pr_auc']:<9.4f}"
                  f"{elapsed:<7.1f}s")

            if best is None or metrics['f1'] > best['f1']:
                best = result
        except Exception as e:
            print(f"{i:<4} SKIP {params} -> {type(e).__name__}: {e}")

    print("-" * 115)
    print(f"Grid search done in {time.time()-t_start:.1f}s")

    if best is None:
        return None, pd.DataFrame(results)

    print(f"\nBEST PARAMS:")
    for k in keys:
        print(f"  {k}: {best[k]}")
    print(f"BEST TUNE F1: {best['f1']:.4f}")
    return best, pd.DataFrame(results)


# ============================================================================
# 5. MAIN
# ============================================================================
def main():
    t_start_main = time.time()
    print("=" * 70)
    print(f"DEEPSVDD ON SWaT — BOTH PROTOCOLS (seed={RANDOM_STATE})")
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

    # Grid: SWaT literature suggests hidden_dims=(64,32), batch_size=64 [citation:5]
    # Threshold at 95th percentile is also from the literature [citation:1]
    param_grid = {
        'hidden_dims':  [(64, 32), (128, 64)],
        'latent_dim':   [16, 32],
        'dropout':      [0.1],
        'epochs':       [100],
        'batch_size':   [64],
        'lr':           [1e-3],
        'threshold_pct': [95, 99, 99.5],
    }

    best, results_df = grid_search_deepsvdd(
        param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune,
    )

    if best is None:
        print("No valid model found. Exiting.")
        return

    # ---- Robust scaling ----
    print("\n[2/7] Robust scaling + final training")
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])
    all_raw = np.vstack([X_trainval_raw, X_test_raw])
    scaler_final = MinMaxScaler()
    scaler_final.fit(all_raw)

    X_trainval = np.clip(scaler_final.transform(X_trainval_raw).astype(np.float32), 0, 1)
    X_test = np.clip(scaler_final.transform(X_test_raw).astype(np.float32), 0, 1)

    print(f"  X_trainval range: [{X_trainval.min():.4f}, {X_trainval.max():.4f}]")
    print(f"  X_test range:     [{X_test.min():.4f}, {X_test.max():.4f}]")

    n_features = X_trainval.shape[1]

    n_val_slice = min(5000, len(X_trainval) // 10)
    X_train_only = X_trainval[:-n_val_slice]
    X_val_only   = X_trainval[-n_val_slice:]

    print(f"  Train timesteps: {X_train_only.shape}")
    print(f"  Val timesteps:   {X_val_only.shape}")

    model_final, final_val_loss = train_deepsvdd(
        X_train_only, X_val_only, n_features,
        hidden_dims=best['hidden_dims'],
        latent_dim=best['latent_dim'],
        dropout=best['dropout'],
        epochs=best['epochs'],
        batch_size=best['batch_size'],
        lr=best['lr'],
        patience=10,
        verbose=True,
    )

    # ============================================================
    # SCORES + DIAGNOSTICS
    # ============================================================
    print("\n[3/7] Computing scores")
    val_scores  = deepsvdd_scores(model_final, X_val_only)
    test_scores = deepsvdd_scores(model_final, X_test)

    normal_test_scores = test_scores[y_test == 0]
    attack_test_scores = test_scores[y_test == 1]

    print(f"\n  Score distributions:")
    print(f"    Val (normal):  mean={val_scores.mean():.6f}  p99.5={np.percentile(val_scores, 99.5):.6f}")
    print(f"    Test normal:   mean={normal_test_scores.mean():.6f}")
    print(f"    Test attack:   mean={attack_test_scores.mean():.6f}")

    if attack_test_scores.mean() > normal_test_scores.mean():
        ratio = attack_test_scores.mean() / (normal_test_scores.mean() + 1e-9)
        print(f"    Attack/Normal ratio: {ratio:.2f}x")

    # ============================================================
    # STANDARD THRESHOLD
    # ============================================================
    print("\n[4/7] STANDARD threshold (training pool)")
    n_holdout = min(5000, len(X_train_only) // 10)
    X_std_pool = X_train_only[-n_holdout:]
    std_scores = deepsvdd_scores(model_final, X_std_pool)
    standard_threshold = np.percentile(std_scores, best['threshold_pct'])
    print(f"  Standard threshold: {standard_threshold:.6f}")

    y_pred_std = (test_scores > standard_threshold).astype(int)
    std_precision = precision_score(y_test, y_pred_std, zero_division=0)
    std_recall = recall_score(y_test, y_pred_std, zero_division=0)
    std_f1 = f1_score(y_test, y_pred_std, zero_division=0)
    std_p_curve, std_r_curve, _ = precision_recall_curve(y_test, test_scores)
    std_pr_auc = auc(std_r_curve, std_p_curve)
    std_pa_p, std_pa_r, std_pa_f1 = point_adjusted_metrics(y_test, y_pred_std)

    print(f"\n  STANDARD RESULTS:")
    print(f"    Point-wise: P={std_precision:.4f}  R={std_recall:.4f}  F1={std_f1:.4f}  PR-AUC={std_pr_auc:.4f}")
    print(f"    PA:         P={std_pa_p:.4f}  R={std_pa_r:.4f}  PA-F1={std_pa_f1:.4f}")

    # ============================================================
    # TEST-NORMAL POOL THRESHOLD
    # ============================================================
    print("\n[5/7] TEST-NORMAL POOL threshold")
    attack_indices = np.where(y_test == 1)[0]
    if len(attack_indices) > 0 and attack_indices[0] > 200:
        first_attack = attack_indices[0]
        pool_start = min(100, first_attack // 2)
        test_normal_pool = X_test[pool_start:first_attack]
        print(f"  First attack at timestep: {first_attack}")
        print(f"  Test-normal pool: {len(test_normal_pool)} timesteps")
        pool_scores = deepsvdd_scores(model_final, test_normal_pool)
        calibrated_threshold = np.percentile(pool_scores, best['threshold_pct'])
    else:
        print(f"  Fallback to training pool")
        calibrated_threshold = standard_threshold

    print(f"  Calibrated threshold: {calibrated_threshold:.6f}")

    y_pred_cal = (test_scores > calibrated_threshold).astype(int)
    cal_precision = precision_score(y_test, y_pred_cal, zero_division=0)
    cal_recall = recall_score(y_test, y_pred_cal, zero_division=0)
    cal_f1 = f1_score(y_test, y_pred_cal, zero_division=0)
    cal_p_curve, cal_r_curve, _ = precision_recall_curve(y_test, test_scores)
    cal_pr_auc = auc(cal_r_curve, cal_p_curve)
    cal_pa_p, cal_pa_r, cal_pa_f1 = point_adjusted_metrics(y_test, y_pred_cal)

    print(f"\n  CALIBRATED RESULTS:")
    print(f"    Point-wise: P={cal_precision:.4f}  R={cal_recall:.4f}  F1={cal_f1:.4f}  PR-AUC={cal_pr_auc:.4f}")
    print(f"    PA:         P={cal_pa_p:.4f}  R={cal_pa_r:.4f}  PA-F1={cal_pa_f1:.4f}")

    # ============================================================
    # PLOTS
    # ============================================================
    print("\n[6/7] Generating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(std_r_curve, std_p_curve, color='purple',
                    label=f'DeepSVDD std (AUC={std_pr_auc:.3f})', lw=2)
    axes[0, 0].scatter(std_recall, std_precision, color='red', s=80,
                       label=f'F1={std_f1:.3f} / PA={std_pa_f1:.3f}')
    axes[0, 0].set_xlabel('Recall'); axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('PR Curve (DeepSVDD — Standard)')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(cal_r_curve, cal_p_curve, color='crimson',
                    label=f'DeepSVDD cal (AUC={cal_pr_auc:.3f})', lw=2)
    axes[0, 1].scatter(cal_recall, cal_precision, color='red', s=80,
                       label=f'F1={cal_f1:.3f} / PA={cal_pa_f1:.3f}')
    axes[0, 1].set_xlabel('Recall'); axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('PR Curve (DeepSVDD — Test-Normal Pool)')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].hist(normal_test_scores, bins=100, alpha=0.5, label='Normal',
                    color='blue', density=True)
    axes[1, 0].hist(attack_test_scores, bins=100, alpha=0.5, label='Attack',
                    color='red', density=True)
    axes[1, 0].axvline(standard_threshold, color='black', ls='--',
                       label=f'Std={standard_threshold:.4f}')
    axes[1, 0].axvline(calibrated_threshold, color='green', ls='--',
                       label=f'Cal={calibrated_threshold:.4f}')
    axes[1, 0].set_xlabel('Anomaly score'); axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Score Distribution')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(test_scores, lw=0.5, color='purple', alpha=0.8)
    attack_idx = np.where(y_test == 1)[0]
    if len(attack_idx) > 0:
        breaks = np.where(np.diff(attack_idx) > 1)[0]
        starts = np.r_[attack_idx[0], attack_idx[breaks + 1]]
        ends = np.r_[attack_idx[breaks], attack_idx[-1]]
        for s, e in zip(starts, ends):
            axes[1, 1].axvspan(s, e, color='red', alpha=0.15)
    axes[1, 1].axhline(standard_threshold, color='black', ls=':', label='Std')
    axes[1, 1].axhline(calibrated_threshold, color='green', ls='--', label='Cal')
    axes[1, 1].set_xlabel('Time step'); axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('DeepSVDD Anomaly Scores')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('deepsvdd_both_results.png', dpi=120)
    plt.show()
    print("  Saved: deepsvdd_both_results.png")

    # ============================================================
    # SAVE
    # ============================================================
    print("\n[7/7] Saving artifacts")
    exec_time = time.time() - t_start_main

    torch.save(model_final.state_dict(), 'swat_deepsvdd.pt')

    joblib.dump({
        'model_state_dict': 'swat_deepsvdd.pt',
        'center': model_final.center.cpu().numpy(),
        'scaler': scaler_final,
        'threshold_standard': standard_threshold,
        'threshold_calibrated': calibrated_threshold,
        'features': common_cols,
        'best_params': {k: best[k] for k in param_grid.keys()},
        'random_state': RANDOM_STATE,
        'exec_time_s': exec_time,
        'architecture': {
            'n_features': n_features,
            'hidden_dims': best['hidden_dims'],
            'latent_dim': best['latent_dim'],
            'dropout': best['dropout'],
        },
        'standard_metrics': {
            'precision': std_precision, 'recall': std_recall,
            'f1': std_f1, 'pr_auc': std_pr_auc,
            'pa_precision': std_pa_p, 'pa_recall': std_pa_r, 'pa_f1': std_pa_f1,
        },
        'calibrated_metrics': {
            'precision': cal_precision, 'recall': cal_recall,
            'f1': cal_f1, 'pr_auc': cal_pr_auc,
            'pa_precision': cal_pa_p, 'pa_recall': cal_pa_r, 'pa_f1': cal_pa_f1,
        },
    }, 'swat_deepsvdd_meta.pkl')

    results_df['exec_time_s'] = exec_time
    results_df.to_csv('deepsvdd_grid_results.csv', index=False)
    print("  Saved: swat_deepsvdd.pt, swat_deepsvdd_meta.pkl, deepsvdd_grid_results.csv")

    print("\n" + "=" * 70)
    print("DONE — DeepSVDD (both protocols)")
    print("=" * 70)
    print(f"  Standard:   F1={std_f1:.4f}  PA-F1={std_pa_f1:.4f}")
    print(f"  Calibrated: F1={cal_f1:.4f}  PA-F1={cal_pa_f1:.4f}")
    print(f"  Total time: {exec_time:.1f}s")
    print("=" * 70)


if __name__ == '__main__':
    main()