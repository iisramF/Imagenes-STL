"""
¿Está la información de la etiqueta en los mapas 2.5D?  (prueba SIN entrenar CNN)
===================================================================================
Cuando un control positivo se aprende solo a medias (p. ej. ~0.7 con
curvature_mean), hay que saber si es porque la información de la etiqueta
YA NO ESTÁ en las imágenes que reciben los backbones (pérdida en la
proyección/interpolación) o porque está pero la red no la aprovecha.

Para cada caso se calculan, sobre los mapas SIN normalizar (los .npy que
guarda GingivalSTLDataset en su caché), resúmenes sencillos de cada canal:
rango, desviación, media del valor absoluto, pendiente media... y se mide:

  1. Correlación de Spearman de cada resumen con el valor continuo de la
     característica con la que se creó la etiqueta (--mode).
  2. Concordancia entre dividir cada resumen por su mediana y la etiqueta.
     (Optimista: se elige el mejor entre varios resúmenes sobre los mismos
     datos. Sirve como referencia orientativa, no como estimación honesta.)
  3. Accuracy con validación cruzada (mismos folds que train.py) de una
     regresión logística sobre todos los resúmenes a la vez. Esta SÍ es una
     estimación honesta de lo que se puede sacar de esos resúmenes.

Uso:
  python3 check_label_in_maps.py --manifest manifest_cm.csv --mode curvature_mean \\
          --cache_dir cache_2_5d
(usa la misma --cache_dir y resolución que en train.py para reutilizar la caché)
"""

import argparse
import csv
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dataset import GingivalSTLDataset, add_view_args, view_kwargs
from generate_positive_control import compute_feature


def map_summaries(raw: np.ndarray) -> dict:
    depth, curv, render = raw[..., 0], raw[..., 1], raw[..., 2]
    gy, gx = np.gradient(depth)
    slope = np.sqrt(gx**2 + gy**2)
    p5, p95 = np.percentile(depth, [5, 95])
    c5, c95 = np.percentile(curv, [5, 95])
    return {
        "depth_range": float(depth.max() - depth.min()),
        "depth_p95_p5": float(p95 - p5),
        "depth_std": float(depth.std()),
        "slope_mean": float(slope.mean()),
        "curv_mean_abs": float(np.abs(curv).mean()),
        "curv_std": float(curv.std()),
        "curv_p95_p5": float(c95 - c5),
        "render_mean": float(render.mean()),
        "render_std": float(render.std()),
    }


def main(manifest: str, mode: str, cache_dir: str, resolution: int, out_csv: str,
         ds_kwargs: dict | None = None):
    with open(manifest, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    samples = [(r["stl_path"], int(r["label"])) for r in rows]
    labels = np.array([l for _, l in samples])

    ds = GingivalSTLDataset(samples, resolution=resolution, cache_dir=cache_dir,
                            **(ds_kwargs or {}))

    feats, summaries = [], []
    for i, (path, _) in enumerate(samples, 1):
        summaries.append(map_summaries(ds._load_raw(path)))
        feats.append(compute_feature(ds, path, mode))
        print(f"  [{i}/{len(samples)}] {path}")
    feats = np.array(feats)
    names = list(summaries[0].keys())
    X = np.array([[s[n] for n in names] for s in summaries])

    # Comprobación de coherencia: ¿la etiqueta del manifest es la división por mediana de --mode?
    expected = (feats > np.median(feats)).astype(int)
    mismatch = int((expected != labels).sum())
    if mismatch:
        print(f"\n[AVISO] {mismatch} etiquetas del manifest no coinciden con la división por "
              f"mediana de '{mode}'. ¿Es el --mode correcto para este manifest?")

    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stl_path", "label", f"feature_{mode}"] + names)
        for (path, lab), ft, row in zip(samples, feats, X):
            w.writerow([path, lab, ft] + list(row))

    print(f"\n=== Resúmenes de los mapas frente a la característica '{mode}' ===")
    print(f"{'resumen':16s} {'Spearman':>9s} {'concordancia':>13s}")
    for j, n in enumerate(names):
        col = X[:, j]
        if np.std(col) == 0:
            print(f"{n:16s} {'(constante)':>9s}")
            continue
        rho, _ = spearmanr(col, feats)
        pred = (col > np.median(col)).astype(int)
        agree = max((pred == labels).mean(), (1 - pred == labels).mean())
        print(f"{n:16s} {rho:9.2f} {agree:13.2f}")

    n_splits = int(min(5, np.bincount(labels).min()))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X, labels, cv=cv, scoring="accuracy")
    print(f"\nRegresión logística sobre todos los resúmenes (CV {n_splits} folds): "
          f"accuracy = {scores.mean():.3f} ± {scores.std():.3f}")

    print("\nLectura (compara con el accuracy de los backbones en este mismo control):")
    print("  - CV de los resúmenes ~0.9 o más -> la información SÍ está en los mapas;")
    print("    si los backbones se quedan en ~0.7, la limitación está en el entrenamiento/arquitectura.")
    print("  - CV de los resúmenes ~0.7        -> los mapas ya limitan la etiqueta a ese nivel;")
    print("    los backbones están en el techo y el cuello de botella es la proyección.")
    print(f"\nResúmenes por caso guardados en: {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV del control positivo (stl_path,label)")
    ap.add_argument("--mode", required=True,
                    choices=["height_range", "curvature_energy", "curvature_mean"],
                    help="Característica con la que se generó la etiqueta del manifest")
    ap.add_argument("--cache_dir", default="cache_2_5d")
    ap.add_argument("--resolution", type=int, default=224)
    ap.add_argument("--out", default="map_summaries.csv")
    add_view_args(ap)
    args = ap.parse_args()
    main(args.manifest, args.mode, args.cache_dir, args.resolution, args.out, view_kwargs(args))
