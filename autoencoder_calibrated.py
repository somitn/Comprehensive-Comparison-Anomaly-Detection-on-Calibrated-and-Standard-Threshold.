# ============================================================
# VANILLA AUTOENCODER ANOMALY DETECTION ON SWaT
# GPU-optimized for RTX 4070 Ti | Windows-safe
# Direct comparison to LSTM-AE (no temporal modeling)
# ============================================================
# Install: pip install torch --index-url https://download.pytorch.org/whl/cu121
#          pip install pandas numpy matplotlib scikit-learn joblib
# Run after: python prepare_data.py
# ============================================================

import multiprocessing
multiprocessing.freeze_support()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import time
import joblib
import os

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

# ============================================================
# 0. DEVICE + REPRODUCIBILITY
# ============================================================
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


# ============================================================
# 1. MODEL
# ============================================================
class Autoencoder(nn.Module):
    def __init__(self, n_features, hidden_dims=(64, 32), latent_dim=16,
                 dropout=0.1):
        super().__init__()
        self.n_features = n_features
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim

        # Encoder
        enc_layers = []
        prev = n_features
        for h in hidden_dims:
            enc_layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        enc_layers.append(nn.Linear(prev, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder (mirror)
        dec_layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [
                nn.Linear(prev, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        dec_layers.append(nn.Linear(prev, n_features))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


# ============================================================
# 2. TRAINING
# ============================================================
def train_autoencoder(X_train, X_val, n_features,
                      hidden_dims=(64, 32), latent_dim=16, dropout=0.1,
                      epochs=50, batch_size=512, lr=1e-3,
                      patience=5, verbose=True):
    model = Autoencoder(
        n_features=n_features,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = TensorDataset(torch.from_numpy(X_train))
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    X_val_t = torch.from_numpy(X_val).to(DEVICE, non_blocking=True)
    scaler = GradScaler('cuda', enabled=USE_AMP)

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
                    out = model(batch)
                    loss = loss_fn(out, batch)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            tr_loss += loss.item() * batch.size(0)

        tr_loss /= len(train_ds)

        model.eval()
        with torch.no_grad():
            if USE_AMP:
                with autocast('cuda'):
                    v_out = model(X_val_t)
                    v_loss = loss_fn(v_out, X_val_t).item()
            else:
                v_out = model(X_val_t)
                v_loss = loss_fn(v_out, X_val_t).item()

        if verbose and (epoch % 3 == 0 or epoch == 1):
            print(f"  Epoch {epoch:02d} | train={tr_loss:.6f} | val={v_loss:.6f}")

        if v_loss < best_val - 1e-7:
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


# ============================================================
# 3. SCORING — per-timestep MSE, FP32
# ============================================================
@torch.no_grad()
def ae_scores(model, X, batch_size=4096):
    """
    Compute per-timestep reconstruction error (MSE).
    X: (N, F) — no windowing, each row is one timestep.
    Returns: (N,) array of MSE per sample.
    """
    model.eval()
    N = len(X)
    scores = np.zeros(N, dtype=np.float32)
    X_t = torch.from_numpy(X)

    for i in range(0, N, batch_size):
        batch = X_t[i:i + batch_size].to(DEVICE, non_blocking=True)
        with autocast('cuda', enabled=False):
            out = model(batch)
        err = ((out.float() - batch.float()) ** 2).mean(dim=1)
        scores[i:i + batch_size] = err.cpu().numpy()
    return scores


# ============================================================
# 4. GRID SEARCH
# ============================================================
def evaluate_ae(X_train_raw, X_val_raw, X_tune_raw, y_tune,
                params, threshold_pct):
    """
    Grid search evaluation with robust scaling:
    fit scaler on train + tune, clip to [0, 1].
    """
    all_raw = np.vstack([X_train_raw, X_tune_raw])
    scaler = MinMaxScaler()
    scaler.fit(all_raw)

    X_tr = scaler.transform(X_train_raw).astype(np.float32)
    X_v  = scaler.transform(X_val_raw).astype(np.float32)
    X_t  = scaler.transform(X_tune_raw).astype(np.float32)

    X_tr = np.clip(X_tr, 0.0, 1.0)
    X_v  = np.clip(X_v,  0.0, 1.0)
    X_t  = np.clip(X_t,  0.0, 1.0)

    n_features = X_tr.shape[1]

    model, val_loss = train_autoencoder(
        X_tr, X_v, n_features,
        hidden_dims=params['hidden_dims'],
        latent_dim=params['latent_dim'],
        dropout=params['dropout'],
        epochs=params['epochs'],
        batch_size=params['batch_size'],
        lr=params['lr'],
        patience=5,
        verbose=False,
    )

    val_scores  = ae_scores(model, X_v)
    tune_scores = ae_scores(model, X_t)

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
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'val_loss': val_loss,
    }


def grid_search_ae(param_grid, X_train_raw, X_val_raw, X_tune_raw,
                   y_tune, verbose=True):
    keys = list(param_grid.keys())
    combos = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"\n[2/8] Grid search: {len(combos)} combinations (GPU-accelerated)")
    print("-" * 120)
    print(f"{'#':<4}{'hid_dims':<14}{'lat':<6}{'drop':<7}"
          f"{'ep':<5}{'bs':<6}{'lr':<8}{'thr%':<6}"
          f"{'F1':<9}{'Prec':<9}{'Rec':<9}{'PR-AUC':<9}{'time':<7}")
    print("-" * 120)

    results, best, t_start = [], None, time.time()

    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        t0 = time.time()
        try:
            metrics = evaluate_ae(
                X_train_raw, X_val_raw, X_tune_raw, y_tune,
                params, threshold_pct=params['threshold_pct'],
            )
            elapsed = time.time() - t0
            result = {**params, **metrics, 'time_s': elapsed}
            results.append(result)

            if verbose:
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
            if verbose:
                print(f"{i:<4} SKIP {params} -> {e}")

    print("-" * 120)
    print(f"Grid search done in {time.time()-t_start:.1f}s")
    print(f"\nBEST PARAMS:")
    for k in keys:
        print(f"  {k}: {best[k]}")
    print(f"BEST METRICS:")
    print(f"  F1:        {best['f1']:.4f}")
    print(f"  Precision: {best['precision']:.4f}")
    print(f"  Recall:    {best['recall']:.4f}")
    print(f"  PR-AUC:    {best['pr_auc']:.4f}")
    return best, pd.DataFrame(results)


# ============================================================
# 5. MAIN
# ============================================================
def main():
    t_start_main = time.time()
    print("\n" + "=" * 60)
    print(f"AUTOENCODER ANOMALY DETECTION ON SWaT (seed={RANDOM_STATE})")
    print("=" * 60)

    # ---- Load cached data ----
    print("\n[1/8] Loading cached data...")
    t0 = time.time()
    D = load_swat(train_size=300000, val_size=60000, tune_frac=0.25)
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
    print(f"  Features: {len(common_cols)}")

    # ---- Grid search ----
    param_grid = {
        'hidden_dims':  [(64, 32), (128, 64)],
        'latent_dim':   [16, 32],
        'dropout':      [0.1],
        'epochs':       [50],
        'batch_size':   [512],
        'lr':           [1e-3],
        'threshold_pct': [95, 99, 99.5],
    }

    best, results_df = grid_search_ae(param_grid, X_train_raw, X_val_raw, X_tune_raw, y_tune,
        verbose=True,
    )

    # ---- Retrain best model on Train + Val ----
    print("\n[3/8] Retraining best model on Train + Val...")
    X_trainval_raw = np.vstack([X_train_raw, X_val_raw])

    # ============================================================
    # ROBUST SCALING
    # ============================================================
    print("\n" + "=" * 60)
    print("SCALING (robust: fit on all raw, clip to [0, 1])")
    print("=" * 60)

    all_raw = np.vstack([X_trainval_raw, X_test_raw])
    print(f"  Fitting scaler on {all_raw.shape[0]} rows")

    scaler_final = MinMaxScaler()
    scaler_final.fit(all_raw)

    X_trainval = scaler_final.transform(X_trainval_raw).astype(np.float32)
    X_test_final = scaler_final.transform(X_test_raw).astype(np.float32)

    X_trainval = np.clip(X_trainval, 0.0, 1.0)
    X_test_final = np.clip(X_test_final, 0.0, 1.0)

    print(f"  X_trainval range: [{X_trainval.min():.4f}, {X_trainval.max():.4f}]")
    print(f"  X_test_final range: [{X_test_final.min():.4f}, {X_test_final.max():.4f}]")
    print("=" * 60)

    n_features = X_trainval.shape[1]

    # Hold out last 10% of trainval as val for early stopping
    n_val_slice = min(5000, len(X_trainval) // 10)
    X_train_only = X_trainval[:-n_val_slice]
    X_val_only   = X_trainval[-n_val_slice:]

    print(f"\n  Train timesteps: {X_train_only.shape}")
    print(f"  Val timesteps:   {X_val_only.shape}")

    model_final, final_val_loss = train_autoencoder(
        X_train_only, X_val_only, n_features,
        hidden_dims=best['hidden_dims'],
        latent_dim=best['latent_dim'],
        dropout=best['dropout'],
        epochs=best['epochs'],
        batch_size=best['batch_size'],
        lr=best['lr'],
        patience=5,
        verbose=True,
    )

    # ============================================================
    # SCORES
    # ============================================================
    print("\n" + "=" * 60)
    print("DIAGNOSTIC: RECONSTRUCTION + SCORES")
    print("=" * 60)

    # Reconstruction on normal training
    model_final.eval()
    with torch.no_grad():
        sample = torch.from_numpy(X_train_only[:500]).to(DEVICE)
        with autocast('cuda', enabled=False):
            recon = model_final(sample)
        err_normal = ((recon.float() - sample.float()) ** 2).mean().item()
    print(f"\n[1] Reconstruction on 500 NORMAL training samples:")
    print(f"    Mean error = {err_normal:.6f}  (expected < 0.001)")

    val_scores  = ae_scores(model_final, X_val_only)
    test_scores = ae_scores(model_final, X_test_final)

    normal_test_scores = test_scores[y_test == 0]
    attack_test_scores = test_scores[y_test == 1]

    print(f"\n[2] Score distributions:")
    print(f"    Val (normal):    mean={val_scores.mean():.6f}  "
          f"std={val_scores.std():.6f}  "
          f"p99.5={np.percentile(val_scores, 99.5):.6f}")
    print(f"    Test normal:     mean={normal_test_scores.mean():.6f}  "
          f"std={normal_test_scores.std():.6f}")
    print(f"    Test attack:     mean={attack_test_scores.mean():.6f}  "
          f"std={attack_test_scores.std():.6f}")

    print(f"\n[3] Overlap check:")
    print(f"    Val max:         {val_scores.max():.6f}")
    print(f"    Attack min:      {attack_test_scores.min():.6f}")
    print(f"    Attack 25th:     {np.percentile(attack_test_scores, 25):.6f}")
    print(f"    Attack median:   {np.percentile(attack_test_scores, 50):.6f}")
    print(f"    Normal median:   {np.percentile(normal_test_scores, 50):.6f}")

    if attack_test_scores.mean() > normal_test_scores.mean():
        ratio = attack_test_scores.mean() / (normal_test_scores.mean() + 1e-9)
        print(f"    Attack/Normal mean ratio: {ratio:.2f}x")
        if ratio < 2:
            print(f"    ⚠ Scores overlap")
        else:
            print(f"    ✓ Good separation")
    else:
        print(f"    ⚠⚠⚠ Attack scores LOWER than normal")
    print("=" * 60)

    # ============================================================
    # THRESHOLD CALIBRATION (test-normal pool if available)
    # ============================================================
    print("\n" + "=" * 60)
    print("THRESHOLD CALIBRATION")
    print("=" * 60)

    attack_indices = np.where(y_test == 1)[0]
    if len(attack_indices) > 0:
        first_attack_idx = attack_indices[0]
        print(f"  First attack at timestep: {first_attack_idx}")

        if first_attack_idx > 2000:
            pool_start = 100
            test_normal_pool = X_test_final[pool_start:first_attack_idx]
            print(f"  Using test-normal pool: {len(test_normal_pool)} timesteps")
            pool_scores = ae_scores(model_final, test_normal_pool)
            final_threshold = np.percentile(pool_scores, best['threshold_pct'])
            print(f"  Threshold (from test-normal pool): {final_threshold:.6f}")
        else:
            print(f"  Too few normal timesteps, using training pool")
            train_scores = ae_scores(model_final, X_train_only)
            final_threshold = np.percentile(train_scores, best['threshold_pct'])
            print(f"  Threshold (from training pool): {final_threshold:.6f}")
    else:
        train_scores = ae_scores(model_final, X_train_only)
        final_threshold = np.percentile(train_scores, best['threshold_pct'])
        print(f"  Threshold (from training pool): {final_threshold:.6f}")
    print("=" * 60)

    # ---- Final test ----
    print("\n[4/8] Evaluating on held-out test set...")
    y_pred = (test_scores > final_threshold).astype(int)

    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)

    p_curve, r_curve, _ = precision_recall_curve(y_test, test_scores)
    pr_auc = auc(r_curve, p_curve)

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS — AUTOENCODER ON SWaT")
    print("=" * 60)
    print(f"Best params:      hidden_dims={best['hidden_dims']}, "
          f"latent={best['latent_dim']}, lr={best['lr']}")
    print(f"Threshold:        {final_threshold:.6f}")
    print(f"Precision:        {precision:.4f}")
    print(f"Recall:           {recall:.4f}")
    print(f"F1-Score:         {f1:.4f}")
    print(f"PR-AUC:           {pr_auc:.4f}")
    print("\nConfusion Matrix (timestep-level):")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=['Normal', 'Attack'],
                                zero_division=0))

    # ---- Plot ----
    print("\n[5/8] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    axes[0, 0].plot(r_curve, p_curve, color='darkorange',
                    label=f'AE (AUC={pr_auc:.3f})', lw=2)
    axes[0, 0].scatter(recall, precision, color='red', zorder=5, s=80,
                       label=f'Operating point (F1={f1:.3f})')
    axes[0, 0].set_xlabel('Recall'); axes[0, 0].set_ylabel('Precision')
    axes[0, 0].set_title('Precision-Recall Curve (Test Set)')
    axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

    thresholds = np.percentile(test_scores, np.arange(50, 99.9, 0.5))
    precs, recs, f1s = [], [], []
    for t in thresholds:
        yp = (test_scores > t).astype(int)
        precs.append(precision_score(y_test, yp, zero_division=0))
        recs.append(recall_score(y_test, yp, zero_division=0))
        f1s.append(f1_score(y_test, yp, zero_division=0))

    axes[0, 1].plot(thresholds, precs, label='Precision', color='green')
    axes[0, 1].plot(thresholds, recs, label='Recall', color='blue')
    axes[0, 1].plot(thresholds, f1s, label='F1', color='red', lw=2)
    axes[0, 1].axvline(final_threshold, color='black', ls='--',
                       label=f'Chosen={final_threshold:.4f}')
    axes[0, 1].set_xlabel('Threshold'); axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Metric vs Threshold')
    axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(test_scores, lw=0.5, color='darkorange', alpha=0.8)
    attack_idx = np.where(y_test == 1)[0]
    if len(attack_idx) > 0:
        breaks = np.where(np.diff(attack_idx) > 1)[0]
        starts = np.r_[attack_idx[0], attack_idx[breaks + 1]]
        ends   = np.r_[attack_idx[breaks], attack_idx[-1]]
        for s, e in zip(starts, ends):
            axes[1, 0].axvspan(s, e, color='red', alpha=0.15)
    axes[1, 0].axhline(final_threshold, color='red', ls='--',
                       label=f'Threshold={final_threshold:.4f}')
    axes[1, 0].set_xlabel('Time step')
    axes[1, 0].set_ylabel('Reconstruction error')
    axes[1, 0].set_title('AE Anomaly Scores (red = true attacks)')
    axes[1, 0].legend(); axes[1, 0].grid(alpha=0.3)

    for lat in results_df['latent_dim'].unique():
        sub = results_df[results_df['latent_dim'] == lat]
        agg = sub.groupby(results_df['hidden_dims'].apply(str))['f1'].max()
        axes[1, 1].plot(range(len(agg)), agg.values, marker='o',
                        label=f'latent={lat}', lw=2)
    axes[1, 1].set_xlabel('hidden_dims config')
    axes[1, 1].set_ylabel('Best F1')
    axes[1, 1].set_title('Grid Search: F1 by config')
    axes[1, 1].legend(); axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('autoencoder_swat_results.png', dpi=120)
    plt.show()
    print("  Saved: autoencoder_swat_results.png")

    # ---- Save ----
    print("\n[6/8] Saving artifacts...")
    torch.save(model_final.state_dict(), 'swat_autoencoder.pt')
    exec_time = time.time() - t_start_main
    joblib.dump({
        'scaler': scaler_final,
        'threshold': final_threshold,
        'features': common_cols,
        'best_params': {k: best[k] for k in param_grid.keys()},
        'random_state': RANDOM_STATE,
        'exec_time_s': exec_time,
        'device_used': str(DEVICE),
        'architecture': {
            'n_features': n_features,
            'hidden_dims': best['hidden_dims'],
            'latent_dim': best['latent_dim'],
            'dropout': best['dropout'],
        },
        'test_metrics': {
            'precision': precision, 'recall': recall,
            'f1': f1, 'pr_auc': pr_auc,
        },
    }, 'swat_autoencoder_meta.pkl')
    print("  Saved: swat_autoencoder.pt, swat_autoencoder_meta.pkl")
    results_df['exec_time_s'] = exec_time 
    results_df.to_csv('autoencoder_grid_results.csv', index=False)
    print("  Saved: autoencoder_grid_results.csv")

    # ---- Summary ----
    print("\n[7/8] DONE")
    print("=" * 60)
    print(f"  Model:           Autoencoder (vanilla)")
    print(f"  Device:          {DEVICE}")
    print(f"  Hidden dims:     {best['hidden_dims']}")
    print(f"  Latent dim:      {best['latent_dim']}")
    print(f"  Test F1:         {f1:.4f}")
    print(f"  Test PR-AUC:     {pr_auc:.4f}")
    print("=" * 60)

    print("\n[8/8] COMPARISON PREVIEW")
    print("=" * 60)
    print(f"  AE (this run):    F1 = {f1:.4f},  P = {precision:.4f},  R = {recall:.4f}")
    print("=" * 60)
    print(f"  Total execution time: {exec_time:.1f}s")

if __name__ == '__main__':
    main()