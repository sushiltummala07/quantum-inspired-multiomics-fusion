import pandas as pd

df = pd.read_csv(r"C:\Users\Sushil\Desktop\quantum\data\labels.csv",
                 sep="\t", index_col=0)

# Keep only ER status
labels = df[["ER_Status_nature2012"]]

# Keep only valid values
labels = labels[labels["ER_Status_nature2012"].isin(["Positive", "Negative"])]

# Convert to numeric
labels["label"] = labels["ER_Status_nature2012"].map({
    "Positive": 1,
    "Negative": 0
})

labels = labels[["label"]]

# Save clean file
labels.to_csv(r"C:\Users\Sushil\Desktop\quantum\data\labels_clean.csv")

print(labels["label"].value_counts())