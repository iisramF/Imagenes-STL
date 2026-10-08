"""
Control positivo: etiqueta determinista calculada directamente de la malla
=============================================================================
Objetivo: aislar si el "techo bajo" de accuracy que estás viendo se debe al
etiquetado (ruido en fino/grueso) o a una limitación real del pipeline/
backbones. Para ello generamos una etiqueta SIN significado clínico, pero
100% determinista y calculable desde la propia geometría del STL, y
reentrenamos exactamente el mismo pipeline sobre ella.

Si con esta etiqueta el modelo alcanza accuracy alta (~0.9+), queda
demostrado que el pipeline y los backbones SÍ son capaces de aprender
cuando existe una relación real que aprender -- y el problema de fondo en
tus resultados anteriores es el etiquetado, no el código ni la arquitectura.

Dos variantes de dificultad:
  - "height_range": rango de altura (Z) de la malla normalizada. Es la
    variante MÁS FÁCIL, porque el depth_map prácticamente "copia" esta
    información directamente -> sirve para confirmar que no hay fugas/bugs
    básicos en el pipeline.
  - "curvature_energy": energía total de curvatura (suma de |curvatura|
    sobre la malla). Requiere integrar información espacial, no es una
    copia directa de un canal -> control algo más exigente.

En ambos casos, el umbral es la MEDIANA de la muestra, así la etiqueta
queda balanceada 50/50 por construcción (elimina el factor de desbalance
de clases de la ecuación).
"""

import argparse
import csv
import numpy as np

from dataset import GingivalSTLDataset


def compute_feature(ds: GingivalSTLDataset, stl_path: str, mode: str) -> float:
    mesh = ds._normalize_mesh(ds._load_stl_as_trimesh(stl_path))

    if mode == "height_range":
        z = mesh.vertices[:, 2]
        return float(z.max() - z.min())

    elif mode == "curvature_energy":
        curvature = mesh.vertices  # placeholder shape check
        import trimesh
        curv = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=ds.curvature_radius
        )
        return float(np.sum(np.abs(curv)))

    else:
        raise ValueError(f"Modo no reconocido: {mode}")


def generate_manifest(input_manifest: str, output_manifest: str, mode: str = "height_range"):
    with open(input_manifest, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        stl_paths = [row["stl_path"] for row in reader]

    ds = GingivalSTLDataset(samples=[])  # instancia solo para reutilizar los métodos privados

    print(f"Calculando '{mode}' para {len(stl_paths)} casos...")
    features = [compute_feature(ds, path, mode) for path in stl_paths]
    threshold = float(np.median(features))

    rows = []
    for path, feat in zip(stl_paths, features):
        label = 1 if feat > threshold else 0
        rows.append((path, label, feat))

    with open(output_manifest, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["stl_path", "label"])
        for path, label, _ in rows:
            writer.writerow([path, label])

    n_pos = sum(label for _, label, _ in rows)
    print(f"Umbral (mediana de '{mode}'): {threshold:.4f}")
    print(f"Manifest de control positivo guardado en: {output_manifest}")
    print(f"Balance de clases: {n_pos} / {len(rows)} en clase 1 (balanceado por construcción)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True,
                         help="CSV original con columna stl_path (la columna label se ignora)")
    parser.add_argument("--output_manifest", default="manifest_positive_control.csv")
    parser.add_argument("--mode", choices=["height_range", "curvature_energy"],
                         default="height_range")
    args = parser.parse_args()

    generate_manifest(args.input_manifest, args.output_manifest, args.mode)
