'''
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

X = pd.read_csv("data/transcriptomics.csv", index_col=0)
y = pd.read_csv("data/labels.csv", index_col=0).values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

models = {
    "RandomForest": RandomForestClassifier(),
    "MLP": MLPClassifier(max_iter=500),
    "XGBoost": XGBClassifier()
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)

    print(f"\n{name}")
    print("F1:", f1_score(y_test, preds, average="weighted"))

    try:
        print("AUC:", roc_auc_score(y_test, probs, multi_class="ovr"))
    except:
        print("AUC: error")

'''        

# baseline_models.py
# Strong baseline comparison for multi-omics classification

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from xgboost import XGBClassifier

# ============================================================
# LOAD DATA
# ============================================================
# ============================================================
# LOAD PREPROCESSED DATA
# ============================================================

import numpy as np

G = np.load("data/G.npy")
T = np.load("data/T.npy")
P = np.load("data/P.npy")
y = np.load("data/y.npy")

print(f"Genomics shape       : {G.shape}")
print(f"Transcriptomics shape: {T.shape}")
print(f"Proteomics shape     : {P.shape}")
print(f"Labels shape         : {y.shape}")

# Convert to DataFrames for concatenation
G = pd.DataFrame(G)
T = pd.DataFrame(T)
P = pd.DataFrame(P)
# Labels
#y = pd.read_csv("data/labels.csv", index_col=0).values.ravel()

# Labels


print(G.shape)
print(T.shape)
print(P.shape)
print(len(y))





print(f"Genomics shape       : {G.shape}")
print(f"Transcriptomics shape: {T.shape}")
print(f"Proteomics shape     : {P.shape}")
print(f"Labels shape         : {y.shape}")

# ============================================================
# CREATE DATASET CONFIGURATIONS
# ============================================================

datasets = {
    "Genomics": G,
    "Transcriptomics": T,
    "Proteomics": P,
    "G+T": pd.concat([G, T], axis=1),
    "T+P": pd.concat([T, P], axis=1),
    "G+P": pd.concat([G, P], axis=1),
    "All_Modalities": pd.concat([G, T, P], axis=1),
}

# ============================================================
# DEFINE MODELS
# ============================================================

models = {

    "LogisticRegression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000))
    ]),

    "SVM": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ]),

    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "MLP": Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=1000,
            random_state=42
        ))
    ]),

    "XGBoost": XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42
    )
}

# ============================================================
# CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

# ============================================================
# RUN EXPERIMENTS
# ============================================================

for dataset_name, X in datasets.items():

    print("\n" + "=" * 70)
    print(f" DATASET: {dataset_name}")
    print("=" * 70)

    X = X.values

    for model_name, model in models.items():

        aucs = []
        f1s = []
        accs = []
        precs = []
        recalls = []

        print(f"\nRunning {model_name}...")

        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):

            X_train = X[train_idx]
            X_test = X[test_idx]

            y_train = y[train_idx]
            y_test = y[test_idx]

            # Train
            model.fit(X_train, y_train)

            # Predict
            preds = model.predict(X_test)

            # Probabilities
            probs = model.predict_proba(X_test)[:, 1]

            # Metrics
            auc = roc_auc_score(y_test, probs)
            f1 = f1_score(y_test, preds, average="weighted")
            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, average="weighted")
            rec = recall_score(y_test, preds, average="weighted")

            aucs.append(auc)
            f1s.append(f1)
            accs.append(acc)
            precs.append(prec)
            recalls.append(rec)

            print(
                f" Fold {fold} | "
                f"AUC={auc:.4f} | "
                f"F1={f1:.4f}"
            )

        # Aggregate results
        row = {
            "Dataset": dataset_name,
            "Model": model_name,

            "AUC Mean": np.mean(aucs),
            "AUC Std": np.std(aucs),

            "F1 Mean": np.mean(f1s),
            "F1 Std": np.std(f1s),

            "Accuracy Mean": np.mean(accs),
            "Precision Mean": np.mean(precs),
            "Recall Mean": np.mean(recalls),
        }

        results.append(row)

        print("\nSUMMARY")
        print(
            f"AUC : {np.mean(aucs):.4f} ± {np.std(aucs):.4f}"
        )
        print(
            f"F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}"
        )

# ============================================================
# SAVE RESULTS
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="AUC Mean",
    ascending=False
)

print("\n" + "=" * 70)
print(" FINAL RESULTS")
print("=" * 70)

print(results_df)

results_df.to_csv(
    "outputs/baseline_model_comparison.csv",
    index=False
)

print("\n[SAVED] outputs/baseline_model_comparison.csv")

# ============================================================
# BEST MODEL PER DATASET
# ============================================================

print("\n" + "=" * 70)
print(" BEST MODEL PER DATASET")
print("=" * 70)

for dataset_name in results_df["Dataset"].unique():

    subset = results_df[
        results_df["Dataset"] == dataset_name
    ]

    best = subset.iloc[0]

    print(
        f"\n{dataset_name}"
        f"\n  Model : {best['Model']}"
        f"\n  AUC   : {best['AUC Mean']:.4f}"
        f"\n  F1    : {best['F1 Mean']:.4f}"
    )

print("\nFinished successfully.")