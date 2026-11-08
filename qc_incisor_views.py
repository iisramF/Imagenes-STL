"""
Control de calidad de la vista vestibular del sector incisivo
================================================================
La orientación anatómica se calcula automáticamente (sin segmentar dientes) y
puede fallar con escaneos parciales, muy asimétricos o con un sistema de
coordenadas atípico. Este script genera, para TODOS los casos del manifest:

  1. Montajes (qc_montage_1.png, qc_montage_2.png...) con la vista de cada caso.
     Los casos con avisos salen con el título en rojo.
  2. qc_orientation.csv con los diagnósticos numéricos por caso (para ajustar
     los umbrales de aviso mirando la distribución en TUS datos).
  3. orientation_overrides_template.csv con solo los casos marcados y las
     columnas vacías flip_up, flip_front, rot_deg, roll_deg, para corregirlos a
     mano. (Los casos NO marcados también pueden necesitar corrección: revisa
     todos los montajes, el aviso es solo una ayuda.)

Qué buscar en el montaje (el sector incisivo bien orientado):
  - Borde incisal arriba, corona y encía hacia abajo, simétrico respecto al centro.
  - Si sale boca abajo           -> flip_up = 1
  - Si se ve la cara lingual     -> flip_front = 1
  - Si el sector se ve oblicuo (dientes de un lado mucho más estrechos)
                                 -> rot_deg = grados (prueba +/- 15)
  - Si la hilera de dientes sale ladeada (inclinada en la propia imagen)
                                 -> roll_deg = grados (prueba +/- 10, 20...).
    La columna roll_est_deg del CSV es una estimación de ese ángulo a partir
    del contorno incisal; si coincide con lo que ves, úsala como punto de partida.
Después de corregir, vuelve a ejecutar con --overrides orientation_overrides.csv.

Uso:
  python3 qc_incisor_views.py --manifest manifest.csv --resolution 128
(Consejo: --resolution 128 acelera mucho la revisión; la vista no se cachea aquí.)
"""

import argparse
import csv
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import GingivalSTLDataset, add_view_args, view_kwargs

CHANNELS = {"depth": (0, "viridis"), "curvature": (1, "coolwarm"), "render": (2, "gray")}


def main(args):
    with open(args.manifest, newline="", encoding="utf-8-sig") as f:
        paths = [row["stl_path"] for row in csv.DictReader(f)]

    kwargs = view_kwargs(args)
    kwargs["view"] = "incisors_vestibular"
    ds = GingivalSTLDataset([], resolution=args.resolution, **kwargs)

    ch, cmap = CHANNELS[args.channel]
    records, images = [], []
    for i, path in enumerate(paths, 1):
        try:
            raw, diag = ds.incisor_view(path)
        except Exception as e:                      # un STL corrupto no debe parar el lote
            raw, diag = None, dict(reasons=[f"error: {type(e).__name__}"])
        images.append(raw[..., ch] if raw is not None else None)
        records.append((path, diag))
        print(f"  [{i}/{len(paths)}] {os.path.basename(path)}  "
              f"{'AVISO: ' + ', '.join(diag['reasons']) if diag['reasons'] else 'ok'}")

    fields = ["arch_width_mm", "arch_depth_mm", "vert_conf", "vert_cue", "foot_ratio",
              "rough_ratio", "v_world_z", "front_ratio", "sym_score", "sym_margin",
              "roll_est_deg", "coverage", "dropped_fragments", "n_front_vertices"]
    with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["index", "stl_path", "flagged", "reasons"] + fields)
        for i, (path, d) in enumerate(records, 1):
            w.writerow([i, path, int(bool(d["reasons"])), "; ".join(d["reasons"])]
                       + [("" if k not in d else d[k] if isinstance(d[k], str) else round(d[k], 3))
                          for k in fields])

    flagged = [(p, d) for p, d in records if d["reasons"]]
    with open(args.overrides_template, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["stl_path", "flip_up", "flip_front", "rot_deg", "roll_deg"])
        for p, _ in flagged:
            w.writerow([p, "", "", "", ""])

    # Montajes paginados
    n_pages = int(np.ceil(len(paths) / args.per_page))
    for page in range(n_pages):
        sl = slice(page * args.per_page, (page + 1) * args.per_page)
        items = list(zip(range(1, len(paths) + 1), images, records))[sl]
        rows = int(np.ceil(len(items) / args.cols))
        fig, axes = plt.subplots(rows, args.cols, figsize=(args.cols * 2.1, rows * 2.35), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")
        for ax, (idx, img, (path, d)) in zip(axes.ravel(), items):
            bad = bool(d["reasons"])
            if img is not None:
                lo, hi = np.percentile(img, [2, 98]) if args.channel == "curvature" else (None, None)
                ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi)
            ax.set_title(f"{idx}: {os.path.basename(path)[:16]}", fontsize=7,
                         color="red" if bad else "black")
            if bad:
                ax.text(0.5, -0.04, d["reasons"][0][:30], transform=ax.transAxes,
                        ha="center", va="top", fontsize=6, color="red")
        fig.tight_layout()
        out = f"{args.out}_{page + 1}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print(f"Montaje guardado: {out}")

    cues = [d.get("vert_cue") for _, d in records if "vert_cue" in d]
    print(f"\nSentido vertical decidido por: normales {cues.count('normales')} casos | "
          f"eje Z del fichero {cues.count('eje_z')} | huella (poco fiable) {cues.count('huella')}.")
    print(f"{len(flagged)} de {len(paths)} casos con avisos.")
    print(f"Diagnósticos: {args.csv}")
    print(f"Plantilla de correcciones (solo casos marcados): {args.overrides_template}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--channel", choices=list(CHANNELS), default="render")
    ap.add_argument("--resolution", type=int, default=128)
    ap.add_argument("--per_page", type=int, default=30)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--out", default="qc_montage")
    ap.add_argument("--csv", default="qc_orientation.csv")
    ap.add_argument("--overrides_template", default="orientation_overrides_template.csv")
    add_view_args(ap)
    main(ap.parse_args())
