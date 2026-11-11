import numpy as np, pandas as pd
from dataset import GingivalSTLDataset

m = pd.read_csv(r".\Entrada_datos\manifest_cm_inc.csv", encoding="utf-8-sig")
ds = GingivalSTLDataset([], cache_dir="cache_2_5d", view="incisors_vestibular", up_world="z+")
rows = []
for p in m.stl_path:
    raw, r = ds._load_raw(p), {}
    for c, n in enumerate(["depth", "curv", "render"]):
        lo, hi = np.percentile(raw[..., c], [2, 98])
        r[f"{n}_span"] = hi - lo
    rows.append(r)
d = pd.DataFrame(rows)
d["label"] = m.label.values
print((d.std() / d.mean().abs()).round(3))                 # coef. de variación de cada escala
print(d.corr(method="spearman")["label"].round(2))         # relación de la escala con la etiqueta