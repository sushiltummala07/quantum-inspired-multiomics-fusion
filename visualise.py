"""
visualise.py — Publication-quality visualisation suite.

Plots produced:
  1. training_curves.png      — loss + accuracy per epoch (all folds)
  2. cv_metrics.png           — K-fold AUC / F1 violin plots with CIs
  3. roc_pr.png               — ROC + Precision-Recall + calibration
  4. confusion_matrix.png     — normalised confusion matrix
  5. attention_gates.png      — per-sample gate distributions (violin)
  6. permutation_importance.png — modality importance bar chart
  7. shap_summary.png         — beeswarm + modality SHAP bar
  8. umap_embeddings.png      — 2D UMAP of fused quantum embedding
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import logging

log = logging.getLogger(__name__)

# ── Theme ─────────────────────────────────────
PALETTE  = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B2","#937860"]
FIGDPI   = 160
sns.set_theme(style="whitegrid", palette=PALETTE, font_scale=1.05)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})


def _save(fig, path):
    fig.savefig(path, dpi=FIGDPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"[SAVED] {path}")


# ──────────────────────────────────────────────
# 1. Training curves (all folds)
# ──────────────────────────────────────────────
def plot_training(fold_metrics, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    titles = [("train_loss","val_loss","Loss"), ("train_acc","val_acc","Accuracy")]

    for ax, (tr_key, val_key, title) in zip(axes, titles):
        for i, fm in enumerate(fold_metrics):
            h = fm["history"]
            e = range(1, len(h[tr_key]) + 1)
            ax.plot(e, h[tr_key], alpha=0.35, color=PALETTE[i], lw=1.2)
            ax.plot(e, h[val_key], alpha=0.9,  color=PALETTE[i], lw=1.8,
                    linestyle="--", label=f"Fold {i+1}")
        ax.set_title(f"Validation {title} — All Folds", fontweight="bold")
        ax.set_xlabel("Epoch")
        if i == 0:
            ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "training_curves.png"))


# ──────────────────────────────────────────────
# 2. CV metric distributions
# ──────────────────────────────────────────────
def plot_cv_metrics(agg: dict, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    metrics   = [("fold_aucs","AUC-ROC","auc_mean","auc_std"),
                 ("fold_f1s", "F1 (Weighted)","f1_mean","f1_std")]

    for ax, (key, label, mean_k, std_k) in zip(axes, metrics):
        vals = agg[key]
        ax.bar(range(1, len(vals)+1), vals, color=PALETTE[0],
               alpha=0.7, edgecolor="white")
        ax.axhline(agg[mean_k], color=PALETTE[1], lw=2,
                   linestyle="--", label=f"Mean = {agg[mean_k]:.3f}")
        ax.fill_between(
            [0.5, len(vals)+0.5],
            agg[mean_k] - agg[std_k],
            agg[mean_k] + agg[std_k],
            alpha=0.15, color=PALETTE[1], label=f"±1σ = {agg[std_k]:.3f}"
        )
        ax.set(title=f"K-Fold {label}", xlabel="Fold", ylabel=label,
               xticks=range(1, len(vals)+1))
        ax.legend(fontsize=9)
        ax.set_ylim(max(0, min(vals)*0.95), min(1, max(vals)*1.05))

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "cv_metrics.png"))


# ──────────────────────────────────────────────
# 3. ROC + PR + Calibration
# ──────────────────────────────────────────────
def plot_roc_pr_cal(report: dict, out_dir):
    c  = report["curves"]
    m  = report["metrics"]
    ci = report["ci"]

    if not c:
        log.warning("No binary curves to plot (multi-class).")
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ROC
    axes[0].plot(c["fpr"], c["tpr"], color=PALETTE[0], lw=2.5,
                 label=f"AUC = {m['auc_roc']:.3f} "
                       f"[{ci['auc_roc']['lower']:.3f}, "
                       f"{ci['auc_roc']['upper']:.3f}]")
    axes[0].fill_between(c["fpr"], c["tpr"], alpha=0.1, color=PALETTE[0])
    axes[0].plot([0,1],[0,1],"k--",lw=1,alpha=0.5)
    axes[0].set(title="ROC Curve", xlabel="False Positive Rate",
                ylabel="True Positive Rate")
    axes[0].legend(fontsize=9)

    # PR
    axes[1].plot(c["rec"], c["prec"], color=PALETTE[1], lw=2.5,
                 label=f"AP = {m['avg_precision']:.3f}")
    axes[1].fill_between(c["rec"], c["prec"], alpha=0.1, color=PALETTE[1])
    axes[1].set(title="Precision-Recall Curve",
                xlabel="Recall", ylabel="Precision")
    axes[1].legend(fontsize=9)

    # Calibration
    axes[2].plot(c["mean_pred"], c["frac_pos"],
                 "s-", color=PALETTE[2], lw=2, ms=7, label="Model")
    axes[2].plot([0,1],[0,1],"k--",lw=1,alpha=0.5, label="Perfect")
    axes[2].set(title="Calibration Curve (Reliability Diagram)",
                xlabel="Mean Predicted Probability",
                ylabel="Fraction of Positives")
    brier = m.get("brier"); brier_str = f"Brier = {brier:.4f}" if brier else ""
    axes[2].legend(fontsize=9, title=brier_str)

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "roc_pr_calibration.png"))


# ──────────────────────────────────────────────
# 4. Confusion matrix
# ──────────────────────────────────────────────
def plot_confusion(report: dict, out_dir):
    cm    = report["cm"].astype(float)
    norm  = cm / cm.sum(axis=1, keepdims=True)
    n     = cm.shape[0]
    labels_str = [f"Class {i}" for i in range(n)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, title, fmt in zip(axes,
                                     [cm, norm],
                                     ["Counts", "Normalised"],
                                     [".0f", ".2f"]):
        cmap = LinearSegmentedColormap.from_list("", ["#f7fbff","#2171b5"])
        sns.heatmap(data, annot=True, fmt=fmt, cmap=cmap,
                    linewidths=0.5, linecolor="white",
                    xticklabels=labels_str, yticklabels=labels_str, ax=ax)
        ax.set(title=f"Confusion Matrix ({title})",
               xlabel="Predicted", ylabel="Actual")
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "confusion_matrix.png"))


# ──────────────────────────────────────────────
# 5. Attention gate distributions
# ──────────────────────────────────────────────
def plot_gates(gates: np.ndarray, labels: np.ndarray, out_dir):
    """gates: (N, 3), labels: (N,)"""
    modalities = ["Genomics", "Transcriptomics", "Proteomics"]
    n_classes  = len(np.unique(labels))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Per-sample violin by class
    import pandas as pd
    rows = []
    for i, name in enumerate(modalities):
        for j, (g, lbl) in enumerate(zip(gates[:, i], labels)):
            rows.append({"Modality": name, "Gate Weight": g, "Class": f"C{lbl}"})
    df = pd.DataFrame(rows)
    sns.violinplot(data=df, x="Modality", y="Gate Weight", hue="Class",
                   split=(n_classes == 2), inner="quart",
                   palette=PALETTE[:n_classes], ax=axes[0])
    axes[0].set_title("Cross-Modal Gate Distribution by Class", fontweight="bold")
    axes[0].legend(title="Class", fontsize=9)

    # Mean bar
    means = gates.mean(axis=0)
    bars  = axes[1].bar(modalities, means, color=PALETTE[:3], edgecolor="white", lw=1.5)
    for bar, v in zip(bars, means):
        axes[1].text(bar.get_x() + bar.get_width()/2, v + 0.003,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=11)
    axes[1].set(title="Mean Gate Weights", ylabel="Gate Weight",
                ylim=(0, max(means) * 1.25))
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "attention_gates.png"))


# ──────────────────────────────────────────────
# 6. Permutation importance
# ──────────────────────────────────────────────
def plot_permutation_importance(perm: dict, out_dir):
    names  = list(perm.keys())
    imps   = [perm[n]["importance"] for n in names]
    stds   = [perm[n]["std"]        for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(names, imps, xerr=stds, color=PALETTE[:3],
                   edgecolor="white", lw=1.5, capsize=5)
    for bar, v in zip(bars, imps):
        ax.text(v + max(stds)*0.1, bar.get_y() + bar.get_height()/2,
                f"{v:.4f}", va="center", fontsize=10)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set(title="Permutation Feature Importance\n(AUC drop when modality shuffled)",
           xlabel="Mean AUC Drop")
    ax.invert_yaxis()
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "permutation_importance.png"))


# ──────────────────────────────────────────────
# 7. SHAP summary
# ──────────────────────────────────────────────
def plot_shap(shap_data: dict, out_dir):
    sv    = shap_data["shap_values"]    # (N, 3D)
    # Handle multi-dimensional SHAP output
    if len(sv.shape) == 3:
        sv = sv[:, :, 1]      
    X     = shap_data["feature_matrix"]
    names = shap_data["feature_names"]
    d     = len(names) // 3

    mean_abs = np.abs(sv).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:15]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Beeswarm-style
    ax = axes[0]
    y_pos  = np.arange(len(order))
    colors = [
        PALETTE[0] if idx < d
        else PALETTE[1] if idx < 2 * d
        else PALETTE[2]
        for idx in range(len(order))
    ]
    #colors = [PALETTE[0] if i < d else PALETTE[1] if i < 2*d else PALETTE[2]
     #         for i in order]
    for rank, (feat_i, col) in enumerate(zip(order[::-1], colors[::-1])):
        sv_col  = sv[:, feat_i]
        x_col   = X[:, feat_i]
        x_norm  = (x_col - x_col.min()) / (np.ptp(x_col) + 1e-8)
        scatter = ax.scatter(sv_col,
                             np.full_like(sv_col, rank) +
                             np.random.uniform(-0.3, 0.3, len(sv_col)),
                             c=x_norm, cmap="coolwarm", s=12, alpha=0.6)
    ax.set_yticks(y_pos); ax.set_yticklabels([names[i] for i in order[::-1]], fontsize=8)
    ax.axvline(0, color="black", lw=0.8, linestyle="--")
    ax.set(title="SHAP Beeswarm (Top 15 Features)", xlabel="SHAP value")
    cb = plt.colorbar(scatter, ax=ax, fraction=0.03)
    cb.set_label("Feature value", fontsize=8)

    # Modality-level bar
    ax2 = axes[1]
    mod_imp = {
        "Genomics":         np.abs(shap_data["genomics_shap"]).mean(),
        "Transcriptomics":  np.abs(shap_data["transcriptomics_shap"]).mean(),
        "Proteomics":       np.abs(shap_data["proteomics_shap"]).mean(),
    }
    bars = ax2.bar(mod_imp.keys(), mod_imp.values(),
                   color=PALETTE[:3], edgecolor="white", lw=1.5)
    for bar, v in zip(bars, mod_imp.values()):
        ax2.text(bar.get_x() + bar.get_width()/2, v + max(mod_imp.values())*0.01,
                 f"{v:.4f}", ha="center", va="bottom", fontsize=11)
    ax2.set(title="Mean |SHAP| by Modality", ylabel="Mean |SHAP value|")

    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "shap_summary.png"))


# ──────────────────────────────────────────────
# 8. UMAP embeddings
# ──────────────────────────────────────────────
def plot_umap(model, G, T, P, y, out_dir):
    try:
        import umap
    except ImportError:
        log.warning("[SKIP] umap-learn not installed.")
        return

    import torch
    from train import make_loader

    model.eval()
    embs, ys = [], []
    loader = make_loader(G, T, P, y, batch_size=64, shuffle=False)
    with torch.no_grad():
        for g, t, p, yi in loader:
            import os; dev = next(model.parameters()).device
            g, t, p = g.to(dev), t.to(dev), p.to(dev)
            emb = model.encode(g, t, p).cpu().numpy()
            embs.append(emb); ys.append(yi.numpy())

    emb_np = np.concatenate(embs)
    y_np   = np.concatenate(ys)

    reducer  = umap.UMAP(n_components=2, n_neighbors=15,
                          min_dist=0.1, random_state=42)
    coords   = reducer.fit_transform(emb_np)

    n_classes = len(np.unique(y_np))
    fig, ax   = plt.subplots(figsize=(7, 6))
    for c in range(n_classes):
        mask = y_np == c
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=22, alpha=0.75, label=f"Class {c}",
                   color=PALETTE[c], edgecolors="none")

    ax.set(title="UMAP — Fused Quantum Embedding Space",
           xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(title="Class", fontsize=9, markerscale=1.5)
    plt.tight_layout()
    _save(fig, os.path.join(out_dir, "umap_embeddings.png"))
