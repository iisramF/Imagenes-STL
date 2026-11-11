"""
svm_baseline.py — Prueba aislada con Máquinas de Soporte Vectorial (SVM)
==========================================================================
Reutiliza TODO lo desarrollado hasta ahora (vista vestibular, orientación,
mapas 2.5D, caché, normalización) y sustituye solo el clasificador: en lugar
de ajustar una red, se extraen características de cada caso y se entrena una SVM.

Tres tipos de características (--features admite varios a la vez):

  summaries   9 resúmenes estadísticos de los mapas (rango/desviación de
              profundidad, pendiente, curvatura media absoluta, etc.), calculados
              solo sobre los píxeles cubiertos. Es el enfoque "clásico": pocas
              variables interpretables.
  pixels      Mapas normalizados, reducidos a --pixel_size x --pixel_size y
              aplanados (3 canales). Se reducen con PCA (95 % de la varianza)
              antes de la SVM. Sin transfer learning ni redes.
  embeddings  Vector de características de un backbone preentrenado en ImageNet
              (congelado, sin ajuste fino) + SVM como cabeza. Es la variante
              "transfer learning sin entrenar la red".

Validación: MISMAS particiones que train.py (StratifiedKFold, k=5,
shuffle, random_state=42), de modo que los resultados son comparables.
Sin fuga de información:
  - el escalado (y el PCA) se ajustan solo con el fold de entrenamiento;
  - los hiperparámetros de la SVM (kernel, C, gamma) se eligen con una
    validación cruzada INTERNA (3 folds) dentro del fold de entrenamiento;
  - con --norm global, los límites de normalización se calculan solo con el
    fold de entrenamiento.

Kernel (--kernel): 'linear' o 'rbf' fijan el kernel (C y gamma se siguen
ajustando con la CV interna); 'auto' deja que la CV interna elija entre ambos en cada
fold; 'all' (por defecto) ejecuta las tres configuraciones para compararlas.

Salida: tabla por fold, resumen media ± desviación (con el baseline mayoritario
medio y, si se pasan con --ref, cifras de referencia de otros métodos), matriz de
confusión acumulada y un CSV por configuración con la predicción de cada caso
(svm_predictions_<características>_<kernel>.csv), útil para pruebas pareadas (McNemar).

Uso:
  python3 svm_baseline.py --manifest manifest_cm_inc.csv --cache_dir cache_2_5d \
      --norm global --features summaries pixels embeddings \
      --view incisors_vestibular --up_world z+
"""

import argparse
import csv
import time
import warnings

import numpy as np
from scipy import ndimage
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from check_label_in_maps import map_summaries
from dataset import GingivalSTLDataset, add_view_args, view_kwargs

FEATURE_SETS = ["summaries", "pixels", "embeddings"]


def read_manifest(csv_path):
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            samples.append((row["stl_path"], int(row["label"])))
    return samples, np.array([lab for _, lab in samples])


# ---------------------------------------------------------------------------
# Extracción de características
# ---------------------------------------------------------------------------

def _downsample(img: np.ndarray, size: int) -> np.ndarray:
    """(H, W, 3) -> (size, size, 3) por promedio de bloques (o zoom si no divide)."""
    h = img.shape[0]
    if h == size:
        return img
    if h % size == 0:
        b = h // size
        return img.reshape(size, b, size, b, img.shape[2]).mean(axis=(1, 3))
    return ndimage.zoom(img, (size / h, size / h, 1), order=1)


def features_summaries(ds, samples, depth_mm):
    rows = []
    for path, _ in samples:
        s = map_summaries(ds._load_raw(path), depth_mm=depth_mm)
        rows.append(list(s.values()))
    return np.array(rows, dtype=np.float64)


def features_pixels(ds, samples, pixel_size):
    out = []
    for path, _ in samples:
        img = ds._normalize_channels(ds._load_raw(path))     # (H, W, 3) en [0, 1]
        out.append(_downsample(img, pixel_size).ravel())
    return np.array(out, dtype=np.float64)


def _embedding_model(name):
    """Backbone preentrenado con la cabeza sustituida por identidad -> vector."""
    import torch.nn as nn
    from backbones import build_backbone
    model = build_backbone(name, freeze_backbone=True, pretrained=True)
    if name == "resnet50":
        model.fc = nn.Identity()
    elif name == "densenet121":
        model.classifier = nn.Identity()
    else:                                    # efficientnet_b3, convnext_tiny
        model.classifier[-1] = nn.Identity()
    return model.eval()


def features_embeddings(ds, samples, model, device):
    import torch
    out = []
    with torch.no_grad():
        for i in range(len(samples)):
            x, _ = ds[i]                     # tensor (3, H, W) ya con normalización ImageNet
            out.append(model(x.unsqueeze(0).to(device)).cpu().numpy().ravel())
    return np.array(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# SVM con búsqueda de hiperparámetros anidada
# ---------------------------------------------------------------------------

def build_search(feature_name, kernel="auto", seed=0):
    steps = [("scaler", StandardScaler())]
    if feature_name == "pixels":
        steps.append(("pca", PCA(n_components=0.95, svd_solver="full", random_state=seed)))
    steps.append(("svc", SVC(class_weight="balanced")))
    gammas = ["scale", 0.01, 0.03, 0.1, 0.3, 1.0] if feature_name == "summaries" \
        else ["scale", 1e-4, 1e-3, 1e-2]
    linear = {"svc__kernel": ["linear"], "svc__C": [0.01, 0.1, 1, 10]}
    rbf = {"svc__kernel": ["rbf"], "svc__C": [0.1, 1, 10, 100], "svc__gamma": gammas}
    grid = {"linear": [linear], "rbf": [rbf], "auto": [linear, rbf]}[kernel]
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    return GridSearchCV(Pipeline(steps), grid, cv=inner, scoring="accuracy", n_jobs=-1)


def run(args):
    samples, labels = read_manifest(args.manifest)
    kw = view_kwargs(args)
    skf = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)
    depth_mm = kw.get("depth_mm", 15.0) if kw.get("view", "occlusal") != "occlusal" else None
    kernels = ["linear", "rbf", "auto"] if args.kernel == "all" else [args.kernel]

    emb_model, device = None, "cpu"
    if "embeddings" in args.features:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        emb_model = _embedding_model(args.embed_backbone).to(device)

    print(f"Vista: {kw.get('view', 'occlusal')} | normalización: {args.norm} | "
          f"casos: {len(samples)} (fino={int((labels == 0).sum())}, grueso={int((labels == 1).sum())})")

    summary, baselines = {}, []
    for feat in args.features:
        name = feat + (f" ({args.embed_backbone})" if feat == "embeddings" else "")
        # Las características se extraen UNA vez por fold y se reutilizan en todos los kernels
        folds = []
        for tr, va in skf.split(samples, labels):
            tr_s = [samples[i] for i in tr]
            va_s = [samples[i] for i in va]
            tr_ds = GingivalSTLDataset(tr_s, cache_dir=args.cache_dir, augment=False,
                                       norm_mode=args.norm, **kw)
            bounds = tr_ds.compute_norm_bounds() if args.norm == "global" else None
            va_ds = GingivalSTLDataset(va_s, cache_dir=args.cache_dir, augment=False,
                                       norm_mode=args.norm, norm_bounds=bounds, **kw)
            if feat == "summaries":
                Xtr, Xva = features_summaries(tr_ds, tr_s, depth_mm), features_summaries(va_ds, va_s, depth_mm)
            elif feat == "pixels":
                Xtr, Xva = (features_pixels(tr_ds, tr_s, args.pixel_size),
                            features_pixels(va_ds, va_s, args.pixel_size))
            else:
                Xtr = features_embeddings(tr_ds, tr_s, emb_model, device)
                Xva = features_embeddings(va_ds, va_s, emb_model, device)
            folds.append((Xtr, labels[tr], Xva, labels[va], va_s))

        for kern in kernels:
            print(f"\n=== SVM | características: {name} | kernel: {kern} ===")
            accs, f1s, bases, cm_total, rows = [], [], [], np.zeros((2, 2), dtype=int), []
            for fold, (Xtr, ytr, Xva, yva, va_s) in enumerate(folds, 1):
                search = build_search(feat, kern, seed=args.seed)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    search.fit(Xtr, ytr)
                pred = search.predict(Xva)
                acc, f1 = accuracy_score(yva, pred), f1_score(yva, pred, zero_division=0)
                base = np.mean(yva == np.bincount(yva).argmax())
                accs.append(acc); f1s.append(f1); bases.append(base)
                cm_total += confusion_matrix(yva, pred, labels=[0, 1])
                bp = {k.replace("svc__", ""): v for k, v in search.best_params_.items()}
                print(f"  Fold {fold}/{args.k} -> accuracy={acc:.3f}  f1={f1:.3f}  "
                      f"(baseline mayoritario={base:.3f}, preds={np.bincount(pred, minlength=2).tolist()}, "
                      f"CV interna={search.best_score_:.3f}, SVM={bp})")
                for (path, lab), p in zip(va_s, pred):
                    rows.append((path, lab, int(p), fold))
            print("Matriz de confusión acumulada (filas: real [fino, grueso]; "
                  "columnas: predicción [fino, grueso]):")
            print(cm_total)
            summary[f"{feat}/{kern}"] = (np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s))
            baselines.append(np.mean(bases))
            out = f"svm_predictions_{feat}_{kern}.csv"
            with open(out, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["stl_path", "label", "pred", "fold"])
                w.writerows(rows)
            print(f"Predicciones por caso: {out}")

    print("\n=== Resumen comparativo (media ± desviación típica entre folds) ===")
    print(f"{'Configuración':28s} {'Accuracy':>16s} {'F1':>16s}")
    print(f"{'baseline mayoritario':28s} {np.mean(baselines):>10.3f}")
    for key, (a, sa, f, sf) in summary.items():
        print(f"{key:28s} {a:.3f} ± {sa:.3f}    {f:.3f} ± {sf:.3f}")
    for ref in args.ref or []:
        label, _, val = ref.partition("=")
        print(f"{'ref: ' + label:28s} {float(val):>10.3f}   (cifra aportada, no calculada aquí)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="CSV con columnas stl_path,label")
    ap.add_argument("--cache_dir", default="cache_2_5d")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--norm", choices=["per_image", "global"], default="global")
    ap.add_argument("--features", nargs="+", choices=FEATURE_SETS, default=FEATURE_SETS)
    ap.add_argument("--kernel", choices=["linear", "rbf", "auto", "all"], default="all",
                    help="linear/rbf: kernel fijo; auto: lo elige la CV interna; "
                         "all: ejecuta linear, rbf y auto para compararlos")
    ap.add_argument("--ref", nargs="*", default=None, metavar="NOMBRE=ACC",
                    help="Cifras de referencia para el resumen, p. ej. --ref resnet50=0.751 convnext=0.704")
    ap.add_argument("--pixel_size", type=int, default=32,
                    help="Lado al que se reducen los mapas en 'pixels' (3*lado^2 variables)")
    ap.add_argument("--embed_backbone", default="resnet50",
                    choices=["resnet50", "efficientnet_b3", "densenet121", "convnext_tiny"])
    ap.add_argument("--seed", type=int, default=0, help="Semilla de la CV interna y del PCA")
    add_view_args(ap)
    run(ap.parse_args())
