"""
train.py — Production training engine.

Features:
  • Warmup + CosineAnnealingWarmRestarts LR schedule
  • Stratified K-Fold cross-validation
  • Optuna hyperparameter search (optional)
  • Early stopping with best-model checkpointing per fold
  • Focal loss option for class imbalance
  • Full metrics tracked per fold
"""

import os, time, logging, copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from typing import Dict, Tuple, List, Optional
import warnings

from config import Config, get_config
from model import QuantumMultiOmicsFusion
from quantum import pretrain_encoder
from data import OmicsDataset

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Losses
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    """Focal loss for class imbalance (Lin et al., 2017)."""
    def __init__(self, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.ls    = label_smoothing

    def forward(self, logits, targets):
        ce   = nn.functional.cross_entropy(logits, targets,
                                            label_smoothing=self.ls,
                                            reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ──────────────────────────────────────────────
# LR schedule with linear warmup
# ──────────────────────────────────────────────
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, min_lr=1e-5):
        self.opt    = optimizer
        self.warmup = warmup_epochs
        self.total  = total_epochs
        self.min_lr = min_lr
        self._base_lrs = [pg["lr"] for pg in optimizer.param_groups]

    def step(self, epoch: int):
        if epoch < self.warmup:
            scale = (epoch + 1) / max(1, self.warmup)
        else:
            progress = (epoch - self.warmup) / max(1, self.total - self.warmup)
            scale = self.min_lr / self._base_lrs[0] + \
                    0.5 * (1 - self.min_lr / self._base_lrs[0]) * \
                    (1 + np.cos(np.pi * progress))
        for pg, base in zip(self.opt.param_groups, self._base_lrs):
            pg["lr"] = base * scale


# ──────────────────────────────────────────────
# Early stopping
# ──────────────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=15, delta=1e-4):
        self.patience = patience; self.delta = delta
        self.best = None; self.counter = 0; self.stop = False
        self.best_state = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best is None or val_loss < self.best - self.delta:
            self.best = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

    def restore(self, model: nn.Module):
        model.load_state_dict(self.best_state)


# ──────────────────────────────────────────────
# DataLoader builder
# ──────────────────────────────────────────────
def make_loader(G, T, P, y, batch_size=32, shuffle=True) -> DataLoader:
    ds = TensorDataset(
        torch.tensor(G, dtype=torch.float32),
        torch.tensor(T, dtype=torch.float32),
        torch.tensor(P, dtype=torch.float32),
        torch.tensor(y, dtype=torch.long)
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=DEVICE.type == "cuda")


# ──────────────────────────────────────────────
# Single epoch
# ──────────────────────────────────────────────
def _train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = correct = n = 0
    for g, t, p, y in loader:
        g, t, p, y = g.to(DEVICE).float(), t.to(DEVICE).float(), p.to(DEVICE).float(), y.to(DEVICE).long()
        optimizer.zero_grad()
        logits = model(g, t, p)
        loss   = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += y.size(0)
    return total_loss / n, correct / n


@torch.no_grad()
def _eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = correct = n = 0
    all_probs, all_labels = [], []
    for g, t, p, y in loader:
        g, t, p, y = g.to(DEVICE).float(), t.to(DEVICE).float(), p.to(DEVICE).float(), y.to(DEVICE).long()
        logits = model(g, t, p)
        loss   = criterion(logits, y)
        probs  = torch.softmax(logits, dim=-1)
        total_loss += loss.item() * y.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        n          += y.size(0)
        all_probs.append(probs.cpu());  all_labels.append(y.cpu())
    probs_np  = torch.cat(all_probs).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return total_loss / n, correct / n, probs_np, labels_np


# ──────────────────────────────────────────────
# Single fold training
# ──────────────────────────────────────────────
def train_fold(cfg: Config, G_tr, T_tr, P_tr, y_tr,
               G_val, T_val, P_val, y_val,
               fold: int = 0) -> Tuple[nn.Module, Dict]:
    tc = cfg.train; qc = cfg.quantum; fc = cfg.fusion
    n_classes = len(np.unique(y_tr))

    model     = QuantumMultiOmicsFusion(qc, fc, n_classes=n_classes).to(DEVICE)

    # ── Optional contrastive pre-training ──────
    log.info(f"  [Fold {fold}] Pre-training quantum encoders...")
    for enc, X in [(model.enc_g, G_tr), (model.enc_t, T_tr), (model.enc_p, P_tr)]:
        Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
        pretrain_encoder(enc, Xt, epochs=15, lr=1e-2)

    # ── Fine-tune full model ───────────────────
    train_loader = make_loader(G_tr, T_tr, P_tr, y_tr, tc.batch_size, shuffle=True)
    val_loader   = make_loader(G_val, T_val, P_val, y_val, tc.batch_size, shuffle=False)

    # Class weights for imbalance
    classes, counts = np.unique(y_tr, return_counts=True)
    weights = torch.tensor(len(y_tr) / (len(classes) * counts),
                           dtype=torch.float32).to(DEVICE)

    criterion = FocalLoss(gamma=2.0, label_smoothing=tc.label_smoothing)
    optimizer = optim.AdamW(model.parameters(), lr=tc.lr,
                            weight_decay=tc.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=tc.warmup_epochs,
                                      total_epochs=tc.epochs)
    stopper   = EarlyStopping(patience=tc.patience)

    history = {k: [] for k in ["train_loss","val_loss","train_acc","val_acc"]}

    log.info(f"  [Fold {fold}] Fine-tuning for up to {tc.epochs} epochs...")
    for epoch in range(tc.epochs):
        scheduler.step(epoch)
        tl, ta = _train_epoch(model, train_loader, criterion, optimizer)
        vl, va, vp, vl_np = _eval_epoch(model, val_loader, criterion)

        history["train_loss"].append(tl);  history["val_loss"].append(vl)
        history["train_acc"].append(ta);   history["val_acc"].append(va)

        
        # Save best model
        if stopper.best is None or vl < stopper.best:
            torch.save(model.state_dict(), "outputs/best_model_fold{fold}.pt")

        # Early stopping check
        stopper(vl, model)

        if (epoch + 1) % 10 == 0:
            log.info(f"    ep {epoch+1:3d}  tl={tl:.4f} vl={vl:.4f} "
                 f"ta={ta:.3f} va={va:.3f}")

        if stopper.stop:
            log.info(f"    Early stopping at epoch {epoch+1}.")
            break

    stopper.restore(model)

    # Compute val metrics
    _, _, vprobs, vlabels = _eval_epoch(model, val_loader, criterion)
    nc = len(np.unique(vlabels))
    val_auc = roc_auc_score(vlabels,
                             vprobs[:, 1] if nc == 2 else vprobs,
                             multi_class="ovr" if nc > 2 else "raise")
    val_f1  = f1_score(vlabels, vprobs.argmax(1), average="weighted")
    log.info(f"  [Fold {fold}] Val AUC={val_auc:.4f}  F1={val_f1:.4f}")

    return model, {"history": history, "val_auc": val_auc, "val_f1": val_f1}


# ──────────────────────────────────────────────
# K-Fold cross-validation
# ──────────────────────────────────────────────
def cross_validate(cfg: Config, dataset: OmicsDataset) -> Tuple[List, Dict]:
    """
    Run stratified K-fold CV on train+val data.
    Returns list of fold models and aggregated metrics.
    """
    G  = np.concatenate([dataset.G_train, dataset.G_val])
    T  = np.concatenate([dataset.T_train, dataset.T_val])
    P  = np.concatenate([dataset.P_train, dataset.P_val])
    y  = np.concatenate([dataset.y_train, dataset.y_val])

    skf    = StratifiedKFold(n_splits=cfg.train.n_folds,
                              shuffle=True, random_state=cfg.data.random_seed)
    models, fold_metrics = [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(G, y)):
        log.info("\n"+ "-"*50)
        log.info(f"  Fold {fold+1} / {cfg.train.n_folds}")
        log.info("─"*50)
        model, metrics = train_fold(
            cfg,
            G[tr_idx], T[tr_idx], P[tr_idx], y[tr_idx],
            G[val_idx], T[val_idx], P[val_idx], y[val_idx],
            fold=fold+1
        )
        models.append(model)
        fold_metrics.append(metrics)

    aucs = [m["val_auc"] for m in fold_metrics]
    f1s  = [m["val_f1"]  for m in fold_metrics]
    agg  = {
        "auc_mean": np.mean(aucs), "auc_std": np.std(aucs),
        "f1_mean":  np.mean(f1s),  "f1_std":  np.std(f1s),
        "fold_aucs": aucs, "fold_f1s": f1s
    }

    log.info(f"\n{'='*50}")
    log.info(f"  CV Results: AUC = {agg['auc_mean']:.4f} ± {agg['auc_std']:.4f}")
    log.info(f"              F1  = {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    log.info(f"{'='*50}")

    return models, agg


# ──────────────────────────────────────────────
# Ensemble inference
# ──────────────────────────────────────────────
@torch.no_grad()
def ensemble_predict(models: List[nn.Module],
                     G, T, P, batch_size=64) -> np.ndarray:
    """Average softmax probabilities over all fold models."""
    all_probs = []
    loader = make_loader(G, T, P, np.zeros(len(G)), batch_size, shuffle=False)
    for model in models:
        model.eval(); fold_probs = []
        for g, t, p, _ in loader:
            g, t, p = g.to(DEVICE), t.to(DEVICE), p.to(DEVICE)
            probs = torch.softmax(model(g, t, p), dim=-1).cpu().numpy()
            fold_probs.append(probs)
        all_probs.append(np.concatenate(fold_probs))
    return np.mean(all_probs, axis=0)   # ensemble average


# ──────────────────────────────────────────────
# Optuna hyperparameter search
# ──────────────────────────────────────────────
def optuna_search(dataset: OmicsDataset, n_trials=20) -> Dict:
    """
    Optional: search over lr, n_layers, dropout, n_heads.
    Returns best params dict.
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.warning("Optuna not installed — skipping hyperparameter search.")
        return {}

    G_tv = np.concatenate([dataset.G_train, dataset.G_val])
    T_tv = np.concatenate([dataset.T_train, dataset.T_val])
    P_tv = np.concatenate([dataset.P_train, dataset.P_val])
    y_tv = np.concatenate([dataset.y_train, dataset.y_val])

    from sklearn.model_selection import train_test_split as tts
    G_tr, G_v, T_tr, T_v, P_tr, P_v, y_tr, y_v = tts(
        G_tv, T_tv, P_tv, y_tv, test_size=0.2,
        stratify=y_tv, random_state=42
    )

    def objective(trial):
        cfg = get_config()
        cfg.train.lr         = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        cfg.quantum.n_layers = trial.suggest_int("n_layers", 1, 4)
        cfg.fusion.dropout   = trial.suggest_float("dropout", 0.05, 0.4)
        cfg.fusion.n_heads   = trial.suggest_categorical("n_heads", [1, 2])
        cfg.train.epochs     = 20
        cfg.train.patience   = 8

        model, metrics = train_fold(cfg, G_tr, T_tr, P_tr, y_tr,
                                         G_v,  T_v,  P_v,  y_v)
        return metrics["val_auc"]

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    log.info(f"[Optuna] Best AUC: {study.best_value:.4f}")
    log.info(f"[Optuna] Best params: {study.best_params}")
    return study.best_params
