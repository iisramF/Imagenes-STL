"""
Diagnóstico de etiquetas: ¿qué mide realmente cada candidato a etiqueta?
==========================================================================
Para cada STL del manifest calcula, sobre la malla normalizada:

  Propiedades "nuisance" (dependen del escáner/recorte, no de la forma):
    n_vertices   -> densidad de la malla
    mean_edge    -> longitud media de arista (resolución del escaneo)
    footprint_xy -> tamaño del recorte en el plano XY tras alinear por PCA
  (No uso el área de la malla: mezcla el tamaño del recorte con la altura
  del relieve, así que una correlación con ella sería ambigua.)
  Candidatos a etiqueta de control:
    height_range, curvature_energy (suma), curvature_mean (media)

y muestra la correlación de Spearman entre ellos. Si un candidato
correlaciona fuerte con alguna de ellas, esa etiqueta está midiendo
densidad/tamaño de malla y el pipeline 2.5D (que remuestrea y normaliza
por imagen) no puede recuperarla.

Si el manifest trae una columna "label", se incluye también como
columna de comparación: una correlación alta de TUS etiquetas
fino/grueso con cualquiera de ellas sería una pista espuria que los
modelos podrían explotar (o que delata un sesgo en el etiquetado).

Uso:
  python3 diagnose_label_features.py --manifest manifest.csv
Guarda además un CSV con los valores por caso (--out, por defecto
label_features.csv) para poder inspeccionarlos.
"""

import argparse
import csv
import numpy as np
import trimesh
from scipy.stats import spearmanr

from dataset import GingivalSTLDataset

FEATURES = ["height_range", "curvature_energy", "curvature_mean"]
NUISANCE = ["n_vertices", "mean_edge", "footprint_xy"]


def case_features(ds: GingivalSTLDataset, stl_path: str) -> dict:
    mesh = ds._normalize_mesh(ds._load_stl_as_trimesh(stl_path))
    z = mesh.vertices[:, 2]
    curv = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, mesh.vertices, radius=ds.curvature_radius
    )
    ext = mesh.vertices[:, :2].max(axis=0) - mesh.vertices[:, :2].min(axis=0)
    return {
        "n_vertices": int(len(mesh.vertices)),
        "mean_edge": float(mesh.edges_unique_length.mean()),
        "footprint_xy": float(ext[0] * ext[1]),
        "height_range": float(z.max() - z.min()),
        "curvature_energy": float(np.sum(np.abs(curv))),
        "curvature_mean": float(np.mean(np.abs(curv))),
    }


def read_manifest(path: str):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main(manifest: str, out_csv: str):
    rows = read_manifest(manifest)
    has_label = "label" in rows[0]
    ds = GingivalSTLDataset(samples=[])

    records = []
    for i, row in enumerate(rows, 1):
        feats = case_features(ds, row["stl_path"])
        if has_label:
            feats["label"] = int(row["label"])
        records.append((row["stl_path"], feats))
        print(f"  [{i}/{len(rows)}] {row['stl_path']}")

    cols = NUISANCE + FEATURES + (["label"] if has_label else [])
    data = {c: np.array([r[1][c] for r in records], dtype=float) for c in cols}

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stl_path"] + cols)
        for path, feats in records:
            w.writerow([path] + [feats[c] for c in cols])

    print("\n=== Correlación de Spearman con las propiedades de malla ===")
    print(f"{'':20s}" + "".join(f"{c:>14s}" for c in NUISANCE))
    for c in FEATURES + (["label"] if has_label else []):
        vals = []
        for n in NUISANCE:
            if np.std(data[n]) == 0 or np.std(data[c]) == 0:
                vals.append("   (constante)")
            else:
                rho, _ = spearmanr(data[c], data[n])
                vals.append(f"{rho:14.2f}")
        print(f"{c:20s}" + "".join(vals))

    print("\n=== Variación de las propiedades de malla en tu conjunto ===")
    for n in NUISANCE:
        v = data[n]
        print(f"{n:13s} min={v.min():.3g}  mediana={np.median(v):.3g}  "
              f"max={v.max():.3g}  (max/min = {v.max()/max(v.min(),1e-9):.1f}x)")

    print("\nLectura: |rho| > ~0.7 con n_vertices o mean_edge => la etiqueta mide "
          "resolución de malla;\ncon footprint_xy => mide el tamaño del recorte. "
          "En ambos casos el pipeline 2.5D no puede recuperarla.")
    print(f"Valores por caso guardados en: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="label_features.csv")
    args = ap.parse_args()
    main(args.manifest, args.out)
