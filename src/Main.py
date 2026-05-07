
"""
main.py — End-to-end orchestration.

Usage:
  python main.py                    # run with defaults
  python main.py --tune             # run Optuna search first
  python main.py --config cfg.json  # load saved config
  python main.py --help
"""

import os, sys, time, logging, argparse, json
import numpy as np
import torch

# ── project imports ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config   import get_config, Config
from data     import load_data
from train    import cross_validate, ensemble_predict, optuna_search, train_fold
from evaluate import full_report, permutation_importance, compute_shap, get_gate_weights
from visualise import (plot_training, plot_cv_metrics, plot_roc_pr_cal,
                        plot_confusion, plot_gates, plot_permutation_importance,
                        plot_shap, plot_umap)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                             datefmt="%H:%M:%S")
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file, mode="w")]
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Quantum Multi-Omics Fusion")
    p.add_argument("--tune",   action="store_true",
                   help="Run Optuna hyperparameter search before training")
    p.add_argument("--trials", type=int, default=20,
                   help="Number of Optuna trials (default 20)")
    p.add_argument("--config", type=str, default=None,
                   help="Path to a JSON config file")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--folds",  type=int, default=None)
    return p.parse_args()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    args = parse_args()
    cfg  = Config.from_json(args.config) if args.config else get_config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    if args.epochs: cfg.train.epochs = args.epochs
    if args.folds:  cfg.train.n_folds = args.folds
    
    setup_logging(cfg.log_file)
    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info("  Quantum Multi-Omics Fusion — v2.0")
    log.info(f"  Device : {DEVICE}")
    log.info("=" * 60)

    # ── Save config ─────────────────────────────
    cfg.to_json(os.path.join(cfg.output_dir, "config.json"))

    # ── Load / generate data ────────────────────
    t0 = time.time()
    dataset = load_data(cfg)
    log.info(f"Data loaded in {time.time()-t0:.1f}s  "
             f"| train={len(dataset.y_train)}  "
             f"val={len(dataset.y_val)}  "
             f"test={len(dataset.y_test)}  "
             f"classes={dataset.n_classes}")

    # ── Optional Optuna search ──────────────────
    if args.tune:
        log.info("\n[Optuna] Starting hyperparameter search...")
        best_params = optuna_search(dataset, n_trials=args.trials)
        if best_params:
            if "lr"       in best_params: cfg.train.lr           = best_params["lr"]
            if "n_layers" in best_params: cfg.quantum.n_layers   = best_params["n_layers"]
            if "dropout"  in best_params: cfg.fusion.dropout     = best_params["dropout"]
            if "n_heads"  in best_params: cfg.fusion.n_heads     = best_params["n_heads"]
            log.info(f"[Optuna] Applied best params: {best_params}")
            cfg.to_json(os.path.join(cfg.output_dir, "config_tuned.json"))

    # ── K-Fold Cross-Validation ─────────────────
    log.info(f"\n[CV] {cfg.train.n_folds}-fold cross-validation...")
    models, agg = cross_validate(cfg, dataset)

    # ── Ensemble prediction on test set ─────────
    log.info("\n[Test] Evaluating ensemble on held-out test set...")
    ens_probs = ensemble_predict(
        models,
        dataset.G_test, dataset.T_test, dataset.P_test
    )

    # Build a pseudo-report with ensemble probs
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, average_precision_score
    from sklearn.calibration import calibration_curve
    from evaluate import compute_metrics, bootstrap_metrics
    import sklearn.metrics as skm

    y_test    = dataset.y_test
    n_classes = dataset.n_classes
    metrics   = compute_metrics(y_test, ens_probs, n_classes)
    ci        = bootstrap_metrics(y_test, ens_probs, n_classes, n_boot=500)
    preds     = ens_probs.argmax(axis=1)
    curves    = {}
    if n_classes == 2:
        fpr, tpr, _   = roc_curve(y_test, ens_probs[:, 1])
        prec, rec, _  = precision_recall_curve(y_test, ens_probs[:, 1])
        frac_pos, mean_pred = calibration_curve(y_test, ens_probs[:, 1], n_bins=10)
        curves = {"fpr":fpr,"tpr":tpr,"prec":prec,"rec":rec,
                  "frac_pos":frac_pos,"mean_pred":mean_pred}

    report = {
        "metrics": metrics, "ci": ci, "curves": curves,
        "cm":     confusion_matrix(y_test, preds),
        "probs":  ens_probs, "preds": preds, "labels": y_test
    }

    log.info("  Ensemble Test Metrics:")
    for k, v in metrics.items():
        if v is not None:
            log.info(f"    {k:<22}: {v:.4f}")
    log.info("  Bootstrap 95% CIs:")
    for k, v in ci.items():
        log.info(f"    {k:<22}: {v['mean']:.4f}  [{v['lower']:.4f}, {v['upper']:.4f}]")

    # Pick best fold model for interpretability
    best_fold = int(np.argmax([m["val_auc"] for m in
                               [{"val_auc": agg["fold_aucs"][i]}
                                for i in range(len(models))]]))
    best_model = models[best_fold]
    log.info(f"\n[Interpret] Using fold {best_fold+1} model for explanations.")

    # ── Gate weights ────────────────────────────
    log.info("[Gates] Computing cross-modal attention gates...")
    gates = get_gate_weights(best_model,
                              dataset.G_test, dataset.T_test, dataset.P_test)

    # ── Permutation importance ──────────────────
    log.info("[Perm] Computing permutation importance...")
    perm = permutation_importance(best_model,
                                   dataset.G_test, dataset.T_test, dataset.P_test,
                                   y_test, n_repeats=8, n_classes=n_classes)

    # ── SHAP ─────────────────────────────────────
    log.info("[SHAP] Computing SHAP values (this may take a few minutes)...")
    shap_data = compute_shap(best_model,
                              dataset.G_test, dataset.T_test, dataset.P_test,
                              n_background=40, n_explain=80, nsamples=60)

    # ── Visualisations ───────────────────────────
    log.info("\n[Plots] Generating visualisations...")
    od = cfg.output_dir

    fold_metrics = [{"history": m["history"]} for m in
                    [{"history": {"train_loss": [], "val_loss": [],
                                  "train_acc": [], "val_acc": []}}
                     for _ in models]]   # placeholder — real histories in agg

    # Re-run cross_validate with history capture isn't feasible post-hoc,
    # so we plot what we have from agg
    plot_cv_metrics(agg,od)
    plot_roc_pr_cal(report,od)
    plot_confusion(report,           od)
    plot_gates(gates, y_test,        od)
    plot_permutation_importance(perm, od)
    plot_shap(shap_data,             od)
    plot_umap(best_model,
              dataset.G_test, dataset.T_test, dataset.P_test, y_test, od)

    # ── Save artefacts ───────────────────────────
    torch.save(best_model.state_dict(),
               os.path.join(od, "best_model.pt"))
    np.save(os.path.join(od, "ensemble_probs.npy"), ens_probs)
    np.save(os.path.join(od, "shap_values.npy"),    shap_data["shap_values"])

    # Final JSON summary
    summary = {
        "cv": {k: float(v) for k, v in agg.items()
               if not isinstance(v, list)},
        "test_ensemble": {k: float(v) for k, v in metrics.items()
                          if v is not None},
        "test_ci": {k: {kk: float(vv) for kk, vv in v.items()}
                    for k, v in ci.items()},
        "permutation_importance": {k: {kk: float(vv) for kk, vv in v.items()}
                                   for k, v in perm.items()},
    }
    with open(os.path.join(od, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"  DONE in {elapsed:.1f}s")
    log.info(f"  Ensemble AUC : {metrics['auc_roc']:.4f}  "
             f"[{ci['auc_roc']['lower']:.4f}, {ci['auc_roc']['upper']:.4f}]")
    log.info(f"  Ensemble F1  : {metrics['f1_weighted']:.4f}")
    log.info(f"  All outputs  : {od}/")
    log.info(f"{'='*60}")


if __name__ == "__main__":
    main()




"""
main.py — SHAP-only visualization using existing trained model
"""

'''
import os
import sys
import time
import logging
import json
import numpy as np
import torch

# ── project imports ──────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from config import get_config, Config
from data import load_data
from evaluate import (
    compute_shap,
    permutation_importance,
    get_gate_weights
)

from visualise import (
    plot_gates,
    plot_permutation_importance,
    plot_shap
)

from model import QuantumMultiOmicsFusion

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ──────────────────────────────────────────────
# Logging setup
# ──────────────────────────────────────────────
def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, mode="w")
    ]

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    for h in handlers:
        h.setFormatter(fmt)
        root.addHandler(h)


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():

    cfg = get_config()

    os.makedirs(cfg.output_dir, exist_ok=True)

    setup_logging(cfg.log_file)

    log = logging.getLogger(__name__)

    log.info("=" * 60)
    log.info(" SHAP Visualisation Mode")
    log.info(f" Device : {DEVICE}")
    log.info("=" * 60)

    # ── Load dataset ──────────────────────────
    log.info("[Data] Loading dataset...")

    dataset = load_data(cfg)

    log.info(
        f"Loaded | "
        f"test={len(dataset.y_test)} | "
        f"classes={dataset.n_classes}"
    )

    # ── Load trained model ────────────────────
    log.info("[Model] Loading trained model...")

    model_path = os.path.join(cfg.output_dir, "best_model.pt")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Could not find trained model:\n{model_path}"
        )

    model = QuantumMultiOmicsFusion(
        qcfg=cfg.quantum,
        fcfg=cfg.fusion,
        n_classes=dataset.n_classes
    ).to(DEVICE)

    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE)
    )

    model.eval()

    log.info("[Model] Model loaded successfully.")

    # ── Gate weights ──────────────────────────
    try:
        log.info("[Gates] Computing attention gates...")

        gates = get_gate_weights(
            model,
            dataset.G_test,
            dataset.T_test,
            dataset.P_test
        )

        plot_gates(
            gates,
            dataset.y_test,
            cfg.output_dir
        )

        log.info("[DONE] attention_gates.png")

    except Exception as e:
        log.error(f"Gates failed: {e}")

    # ── Permutation importance ────────────────
    try:
        log.info("[Perm] Computing permutation importance...")

        perm = permutation_importance(
            model,
            dataset.G_test,
            dataset.T_test,
            dataset.P_test,
            dataset.y_test,
            n_repeats=8,
            n_classes=dataset.n_classes
        )

        plot_permutation_importance(
            perm,
            cfg.output_dir
        )

        log.info("[DONE] permutation_importance.png")

    except Exception as e:
        log.error(f"Permutation importance failed: {e}")

    # ── SHAP ──────────────────────────────────
    try:

        log.info("[SHAP] Computing SHAP values...")

        shap_data = compute_shap(
            model,
            dataset.G_test,
            dataset.T_test,
            dataset.P_test,
            n_background=40,
            n_explain=80,
            nsamples=60
        )

        log.info("[SHAP] Generating SHAP plot...")

        plot_shap(
            shap_data,
            cfg.output_dir
        )

        np.save(
            os.path.join(cfg.output_dir, "shap_values.npy"),
            shap_data["shap_values"]
        )

        log.info("[DONE] shap_summary.png")

    except Exception as e:
        log.error(f"SHAP failed: {e}")

    log.info("=" * 60)
    log.info(" Finished successfully.")
    log.info("=" * 60)


if __name__ == "__main__":
    main()

    '''
