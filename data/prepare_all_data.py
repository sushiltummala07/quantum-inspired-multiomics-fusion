import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# =========================
# 📁 FILE PATHS
# =========================
G_PATH = r"C:\Users\Sushil\Desktop\quantum\data\genomics.csv"
T_PATH = r"C:\Users\Sushil\Desktop\quantum\data\transcriptomics.csv"
P_PATH = r"C:\Users\Sushil\Desktop\quantum\data\proteomics.csv"
Y_PATH = r"C:\Users\Sushil\Desktop\quantum\data\labels.csv"

# =========================
# 🟢 LOAD DATA (FIXED)
# =========================
print("Loading data...")

# 🔥 IMPORTANT: sep="\t" + transpose
G = pd.read_csv(G_PATH, sep="\t", index_col=0).T
T = pd.read_csv(T_PATH, sep="\t", index_col=0).T
P = pd.read_csv(P_PATH, sep="\t", index_col=0).T
Y = pd.read_csv(Y_PATH, sep="\t", index_col=0)

print("\nRaw shapes:")
print("G:", G.shape)
print("T:", T.shape)
print("P:", P.shape)
print("Y:", Y.shape)

# =========================
# 🟢 CLEAN LABELS
# =========================
print("\nCleaning labels...")

labels = Y[["ER_Status_nature2012"]]

# Remove invalid labels
labels = labels[labels["ER_Status_nature2012"].isin(["Positive", "Negative"])]

# Convert to numeric
labels["label"] = labels["ER_Status_nature2012"].map({
    "Positive": 1,
    "Negative": 0
})

labels = labels[["label"]]

print("\nLabel distribution:")
print(labels["label"].value_counts())

# =========================
# 🟢 DEBUG CHECK (IMPORTANT)
# =========================
print("\nSample IDs check:")
print("G:", G.index[:3])
print("T:", T.index[:3])
print("P:", P.index[:3])
print("Y:", labels.index[:3])

# =========================
# 🟢 ALIGN SAMPLES
# =========================
print("\nAligning samples...")

common = G.index.intersection(T.index).intersection(P.index).intersection(labels.index)

G = G.loc[common]
T = T.loc[common]
P = P.loc[common]
labels = labels.loc[common]

y = labels.values.ravel()

print("\nAligned shapes:")
print("G:", G.shape)
print("T:", T.shape)
print("P:", P.shape)

# =========================
# 🟢 PREPROCESS FUNCTION
# =========================
def preprocess(df, name):
    print(f"\nProcessing {name}...")

    # Keep numeric only
    df = df.select_dtypes(include=[np.number])

    print(f"{name} numeric shape:", df.shape)

    # Impute
    imputer = KNNImputer(n_neighbors=5)
    X = imputer.fit_transform(df.values)

    # PCA (reduce but keep info)
    pca = PCA(n_components=64)
    X = pca.fit_transform(X)

    # Normalize
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"{name} final shape:", X.shape)

    return X

# =========================
# 🟢 APPLY PREPROCESSING
# =========================
G = preprocess(G, "Genomics")
T = preprocess(T, "Transcriptomics")
P = preprocess(P, "Proteomics")

# =========================
# 🟢 SAVE DATA
# =========================
print("\nSaving processed data...")

np.save(r"C:\Users\Sushil\Desktop\quantum\data\G.npy", G)
np.save(r"C:\Users\Sushil\Desktop\quantum\data\T.npy", T)
np.save(r"C:\Users\Sushil\Desktop\quantum\data\P.npy", P)
np.save(r"C:\Users\Sushil\Desktop\quantum\data\y.npy", y)

print("\n✅ DONE — Data ready for model!")