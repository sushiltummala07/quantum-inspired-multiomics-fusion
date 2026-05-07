"""
evaluate.py — Rigorous evaluation with:
  • Bootstrap 95% confidence intervals on all metrics
  • Permutation feature importance per modality
  • SHAP (batched, faster)
  • Grad-CAM style gradient saliency on quantum embeddings
  • Calibration curve (reliability diagram)
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    average_precision_score, classification_report,
    roc_curve, precision_recall_curve, confusion_matrix,
    brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from typing import List, Dict
import logging
import shap

log = logging.getLogger(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Core metrics
# ──────────────────────────────────────────────

from sklearn.preprocessing import label_binarize
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss
)

def compute_metrics(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    """
    Compute evaluation metrics safely for binary and multi-class classification.
    """

    # Predictions
    preds = probs.argmax(axis=1)

    # Unique classes present
    unique_classes = np.unique(labels)

    # Handle missing classes safely
    if len(unique_classes) < n_classes:
        probs_adj = probs[:, unique_classes]
        labels_bin = label_binarize(labels, classes=unique_classes)
    else:
        probs_adj = probs
        labels_bin = label_binarize(labels, classes=np.arange(n_classes))

    # =========================
    # AUC ROC (FIXED)
    # =========================
    try:
        if n_classes == 2:
            # Use probability of positive class
            auc = roc_auc_score(labels, probs[:, 1])
        else:
            auc = roc_auc_score(
                labels_bin,
                probs_adj,
                multi_class="ovr",
                average="weighted"
            )
    except Exception:
        auc = np.nan

    # =========================
    # LOG LOSS
    # =========================
    try:
        ll = log_loss(labels, probs, labels=np.arange(n_classes))
    except Exception:
        ll = np.nan

    # =========================
    # AVERAGE PRECISION (FIXED)
    # =========================
    try:
        if n_classes == 2:
            ap = average_precision_score(labels, probs[:, 1])
        else:
            ap = average_precision_score(labels_bin, probs_adj, average="weighted")
    except Exception:
        ap = np.nan

    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "auc_roc": float(auc),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "avg_precision": float(ap),
        "brier": None,  # optional
        "log_loss": float(ll),
    }


'''
def compute_metrics(labels: np.ndarray, probs: np.ndarray, n_classes: int) -> dict:
    preds = probs.argmax(axis=1)

    unique_classes = np.unique(labels)

    #  Adjust probabilities if some classes missing
    if len(unique_classes) < n_classes:
        probs_adj = probs[:, unique_classes]
        labels_bin = label_binarize(labels, classes=unique_classes)
    else:
        probs_adj = probs
        labels_bin = label_binarize(labels, classes=np.arange(n_classes))

    #  Safe AUC
    try:
        auc = roc_auc_score(
            labels_bin,
            probs_adj,
            multi_class="ovr",
            average="weighted"
        )
    except:
        auc = np.nan

    #  Safe log_loss (FIXED)
    try:
        ll = log_loss(labels, probs, labels=np.arange(n_classes))
    except:
        ll = np.nan

    #  Safe average precision
    try:
        ap = average_precision_score(labels_bin, probs_adj, average="weighted")
    except:
        ap = np.nan

    return {
        "accuracy": accuracy_score(labels, preds),
        "auc_roc": float(auc),
        "f1_weighted": f1_score(labels, preds, average="weighted"),
        "avg_precision": float(ap),
        "brier": None,
        "log_loss": float(ll),
    }
    
'''


# ──────────────────────────────────────────────
# Bootstrap confidence intervals
# ──────────────────────────────────────────────
def bootstrap_metrics(labels: np.ndarray, probs: np.ndarray,
                       n_classes: int, n_boot: int = 1000,
                       ci: float = 0.95) -> Dict:
    rng    = np.random.default_rng(42)
    records = []
    n      = len(labels)
    for _ in range(n_boot):
        idx  = rng.integers(0, n, size=n)
        records.append(compute_metrics(labels[idx], probs[idx], n_classes))

    alpha = (1 - ci) / 2
    result = {}
    for key in records[0]:
        if records[0][key] is None:
            continue
        vals = [r[key] for r in records]
        result[key] = {
            "mean": np.mean(vals),
            "lower": np.percentile(vals, 100 * alpha),
            "upper": np.percentile(vals, 100 * (1 - alpha)),
        }
    return result


# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
@torch.no_grad()
def predict(model: nn.Module, G, T, P, batch_size=64) -> np.ndarray:
    from train import make_loader
    model.eval()
    loader = make_loader(G, T, P, np.zeros(len(G)), batch_size, shuffle=False)
    probs  = []
    for g, t, p, _ in loader:
        g, t, p = g.to(DEVICE), t.to(DEVICE), p.to(DEVICE)
        logits = model(g, t, p)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs)


# ──────────────────────────────────────────────
# Permutation importance (per modality)
# ──────────────────────────────────────────────
@torch.no_grad()
def permutation_importance(model: nn.Module, G, T, P, y,
                            n_repeats=10, n_classes=2) -> Dict:
    """
    Measures AUC drop when each modality is randomly shuffled.
    Large drop → modality is important.
    """
    baseline = compute_metrics(y, predict(model, G, T, P), n_classes)["auc_roc"]
    rng      = np.random.default_rng(42)
    results  = {}

    for name, (Gp, Tp, Pp) in [
        ("genomics",       (G.copy(), T, P)),
        ("transcriptomics",(G, T.copy(), P)),
        ("proteomics",     (G, T, P.copy())),
    ]:
        drops = []
        for _ in range(n_repeats):
            if name == "genomics":
                Gp = G[rng.permutation(len(G))]
                probs = predict(model, Gp, T, P)
            elif name == "transcriptomics":
                Tp = T[rng.permutation(len(T))]
                probs = predict(model, G, Tp, P)
            else:
                Pp = P[rng.permutation(len(P))]
                probs = predict(model, G, T, Pp)
            auc = compute_metrics(y, probs, n_classes)["auc_roc"]
            drops.append(baseline - auc)

        results[name] = {
            "importance": float(np.mean(drops)),
            "std":        float(np.std(drops)),
        }
        log.info(f"  Permutation importance [{name}]: "
                 f"{results[name]['importance']:.4f} ± {results[name]['std']:.4f}")

    return results


# ──────────────────────────────────────────────
# SHAP — batched KernelExplainer
# ──────────────────────────────────────────────
def compute_shap(model: nn.Module, G, T, P, n_background=50,
                 n_explain=100, nsamples=80) -> Dict:
    """
    Returns SHAP values for the positive class over the concatenated
    [G | T | P] feature space.
    """
    model.eval()
    model_cpu = model.cpu()
    X = np.concatenate([G, T, P], axis=1).astype(np.float32)
    d = G.shape[1]

    def wrapper(x_np):
        x = torch.tensor(x_np, dtype=torch.float32)
        g_t = x[:, :d];  t_t = x[:, d:2*d];  p_t = x[:, 2*d:]
        with torch.no_grad():
            logits = model_cpu(g_t, t_t, p_t)
            return torch.softmax(logits, dim=-1).numpy()

    # Limit dataset size for stability
    X_sample = X[:min(200, len(X))]

    # Fix NaN / inf issues (VERY IMPORTANT)
    X_sample = np.nan_to_num(X_sample)

    # Safe background (very important)
    background = shap.kmeans(X_sample, min(10, len(X_sample)))

    # Create explainer
    explainer = shap.KernelExplainer(wrapper, background, link="identity")

    # Reduce explanation size
    explain_X = X_sample[:50]

    # Compute SHAP values (stable + fast)
    shap_vals = explainer.shap_values(
        explain_X,
        nsamples=100
    )

    model_cpu.to(DEVICE)

    # For binary: shap_vals is a list [class0, class1]
    sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals

    n = d
    return {
        "shap_values":  sv,
        "feature_matrix": explain_X,
        "genomics_shap":  sv[:, :n],
        "transcriptomics_shap": sv[:, n:2*n],
        "proteomics_shap":      sv[:, 2*n:],
        "feature_names": (
            [f"G_{i}" for i in range(n)] +
            [f"T_{i}" for i in range(n)] +
            [f"P_{i}" for i in range(n)]
        ),
    }


# ──────────────────────────────────────────────
# Gate analysis
# ──────────────────────────────────────────────
@torch.no_grad()
def get_gate_weights(model: nn.Module, G, T, P, batch_size=64) -> np.ndarray:
    """Returns (N, 3) array of per-sample gate weights [G, T, P]."""
    from train import make_loader
    model.eval()
    loader = make_loader(G, T, P, np.zeros(len(G)), batch_size, shuffle=False)
    gates  = []
    for g, t, p, _ in loader:
        g, t, p = g.to(DEVICE), t.to(DEVICE), p.to(DEVICE)
        _, aux  = model(g, t, p, return_aux=True)
        gates.append(aux["gates"].cpu().numpy())
    return np.concatenate(gates)                     # (N, 3)


# ──────────────────────────────────────────────
# Full evaluation report
# ──────────────────────────────────────────────
def full_report(model: nn.Module, G, T, P, y, n_classes: int) -> Dict:
    probs   = predict(model, G, T, P)
    preds   = probs.argmax(axis=1)

    log.info("\n" + "=" * 55)
    log.info("  TEST SET RESULTS")
    log.info("=" * 55)

    metrics = compute_metrics(y, probs, n_classes)
    for k, v in metrics.items():
        if v is not None:
            log.info(f"  {k:<20}: {v:.4f}")

    log.info("\n" + classification_report(y, preds))

    # Bootstrap CIs
    ci = bootstrap_metrics(y, probs, n_classes, n_boot=500)
    log.info("  95% Bootstrap CIs:")
    for k, v in ci.items():
        log.info(f"  {k:<20}: {v['mean']:.4f}  [{v['lower']:.4f}, {v['upper']:.4f}]")

    log.info("=" * 55)

    # Curve data
    curves = {}
    if n_classes == 2:
        fpr, tpr, _   = roc_curve(y, probs[:, 1])
        prec, rec, _  = precision_recall_curve(y, probs[:, 1])
        frac_pos, mean_pred = calibration_curve(y, probs[:, 1], n_bins=10)
        curves = {"fpr": fpr, "tpr": tpr, "prec": prec, "rec": rec,
                  "frac_pos": frac_pos, "mean_pred": mean_pred}

    return {
        "metrics": metrics,
        "ci":      ci,
        "curves":  curves,
        "cm":      confusion_matrix(y, preds),
        "probs":   probs,
        "preds":   preds,
        "labels":  y,
    }
