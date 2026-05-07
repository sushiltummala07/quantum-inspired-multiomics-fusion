import pandas as pd

df = pd.read_csv("desktop\quantum\data\labels.csv", sep="\t", index_col=0)

print(df.columns)