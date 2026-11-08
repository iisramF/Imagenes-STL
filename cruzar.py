import pandas as pd
q = pd.read_csv( ".\Prueba 3DTS\qc_orientation.csv", encoding="utf-8-sig")
m = pd.read_csv(".\Entrada_datos\manifest_3DTeethSeg.csv", encoding="utf-8-sig")
print(pd.crosstab(q.vert_cue, m.label))

kept = set(pd.read_csv(".\Entrada_datos\manifest_3DTeethSeg.csv", encoding="utf-8-sig").stl_path)


q["kept"] = q.stl_path.isin(kept)
print(q.groupby("kept")[["coverage", "n_front_vertices", "arch_width_mm"]].describe().round(2).T)