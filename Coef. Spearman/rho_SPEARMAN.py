import pandas as pd
df = pd.read_csv("label_features.csv", encoding="utf-8-sig")
print(df.drop(columns="stl_path").corr(method="spearman").round(2))