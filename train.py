"""
train.py — Entrenamiento y comparación de los 4 backbones con K-Fold estratificado
=====================================================================================
Para cada backbone (ResNet-50, EfficientNet-B3, DenseNet-121, ConvNeXt-Tiny):
  - Se repite un K-Fold estratificado (por defecto k=5) sobre el mismo conjunto
    de casos, garantizando que todos los backbones se evalúan con las mismas
    particiones (comparación justa).
  - Se hace fine-tuning parcial (backbone congelado salvo última etapa + cabeza).
  - Se registran métricas por fold y se agregan media ± desviación típica.

Al final se imprime una tabla comparativa que sirve como resultado del
apartado 2.2 (arquitectura y transfer learning) del trabajo.
"""

import argparse
import csv
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from backbones import build_backbone
from dataset import GingivalSTLDataset, add_view_args, view_kwargs
from project_logger import Logger


def set_seed(seed: int):
    """Fija todas las fuentes de aleatoriedad del entrenamiento (inicialización de
    la cabeza, orden de los lotes, aumentado de datos)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


BACKBONES = ["resnet50", "efficientnet_b3", "densenet121", "convnext_tiny"]


def read_manifest(csv_path: str):
    """
    Lee un CSV con columnas: stl_path,label   (label: 0=fino, 1=grueso)
    Devuelve lista de (stl_path, label) y el array de labels para estratificar.
    """
    samples = []
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append((row["stl_path"], int(row["label"])))
    labels = np.array([label for _, label in samples])
    return samples, labels


def train_one_fold(backbone_name: str, train_samples, val_samples,
                    cache_dir: str, epochs: int, lr_head: float, lr_backbone: float,
                    device: str, pretrained: bool = True,
                    norm_mode: str = "per_image", seed: int | None = None,
                    ds_kwargs: dict | None = None) -> dict:
    ds_kwargs = ds_kwargs or {}
    if seed is not None:
        set_seed(seed)   # antes de crear el modelo: la cabeza nueva usa el RNG
    train_ds = GingivalSTLDataset(train_samples, cache_dir=cache_dir, augment=True,
                                  norm_mode=norm_mode, **ds_kwargs)
    # Con normalización global, los límites se calculan SOLO con el fold de
    # entrenamiento y se reutilizan en validación (sin fuga de información).
    bounds = train_ds.compute_norm_bounds() if norm_mode == "global" else None
    val_ds = GingivalSTLDataset(val_samples, cache_dir=cache_dir, augment=False,
                                norm_mode=norm_mode, norm_bounds=bounds, **ds_kwargs)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False)

    model = build_backbone(backbone_name, freeze_backbone=True, pretrained=pretrained).to(device)

    # Learning rate diferenciada: más baja para las capas del backbone que sí
    # se entrenan (ajuste fino), más alta para la cabeza nueva (entrenada desde cero).
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name for key in ["fc", "classifier"]):
            head_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": lr_backbone},
        {"params": head_params, "lr": lr_head},
    ])

    # Pesos de clase inversamente proporcionales a su frecuencia en ESTE fold
    # de entrenamiento: sin esto, con un desbalance ~65/35 el optimizador
    # encuentra un mínimo "barato" prediciendo siempre la clase mayoritaria,
    # que es exactamente el colapso que estamos observando.
    train_labels = np.array([label for _, label in train_samples])
    class_counts = np.bincount(train_labels, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / (2 * np.clip(class_counts, 1, None))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    epoch_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_losses.append(running_loss / len(train_ds))

    # Validación del fold
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    # Baseline ingenuo: accuracy de predecir siempre la clase mayoritaria del
    # propio fold de validación. Si el modelo no la supera con holgura, no
    # está aprendiendo nada más que el desbalance de clases.
    majority_class = np.bincount(all_labels).argmax()
    baseline_acc = np.mean(np.array(all_labels) == majority_class)

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "baseline_acc": baseline_acc,
        "confusion_matrix": confusion_matrix(all_labels, all_preds, labels=[0, 1]),
        "pred_distribution": np.bincount(all_preds, minlength=2),
        "loss_start": epoch_losses[0],
        "loss_end": epoch_losses[-1],
    }


def run_comparison(manifest_csv: str, cache_dir: str = "cache_2_5d",
                    k: int = 5, epochs: int = 10,
                    lr_head: float = 1e-3, lr_backbone: float = 1e-5,
                    pretrained: bool = True, norm_mode: str = "per_image",
                    backbones: list[str] | None = None, seed: int = 0,
                    ds_kwargs: dict | None = None):
    backbones = backbones or BACKBONES
    device = "cuda" if torch.cuda.is_available() else "cpu"
    samples, labels = read_manifest(manifest_csv)

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    results = {name: [] for name in backbones}

    view = (ds_kwargs or {}).get("view", "occlusal")
    print(f"Vista: {view} | normalización de canales: {norm_mode} | semilla de entrenamiento: {seed}")
    for backbone_name in backbones:
        print(f"\n=== Backbone: {backbone_name} ===")
        Logger.traza("info", f"\n=== Backbone: {backbone_name} ===")
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]

            metrics = train_one_fold(
                backbone_name, train_samples, val_samples, cache_dir,
                epochs, lr_head, lr_backbone, device, pretrained, norm_mode,
                seed=seed + fold_idx,   # misma semilla por fold para todos los backbones
                ds_kwargs=ds_kwargs,
            )
            results[backbone_name].append(metrics)
            degenerate = " <-- predicción casi constante" if metrics["pred_distribution"].max() >= len(val_samples) - 1 else ""
            print(f"  Fold {fold_idx+1}/{k} -> "
                  f"accuracy={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
                  f"(baseline mayoritario={metrics['baseline_acc']:.3f}, "
                  f"preds={metrics['pred_distribution'].tolist()}, "
                  f"loss {metrics['loss_start']:.3f}->{metrics['loss_end']:.3f}){degenerate}")
            Logger.traza("info", f"  Fold {fold_idx+1}/{k} -> "
                         f"accuracy={metrics['accuracy']:.3f}  f1={metrics['f1']:.3f}  "
                         f"(baseline mayoritario={metrics['baseline_acc']:.3f}, "
                         f"preds={metrics['pred_distribution'].tolist()}, "
                         f"loss {metrics['loss_start']:.3f}->{metrics['loss_end']:.3f}){degenerate}")  

        summed_confusion_matrix = np.sum(
            [metrics["confusion_matrix"] for metrics in results[backbone_name]],
            axis=0,
        )
        confusion_matrix_output = (
            f"\nMatriz de confusión acumulada — {backbone_name} "
            "(filas: real [fino, grueso]; columnas: predicción [fino, grueso]):\n"
            f"{summed_confusion_matrix}"
        )
        print(confusion_matrix_output)
        Logger.traza("info", confusion_matrix_output)
                         
    print("\n=== Resumen comparativo (media ± desviación típica) ===")
    Logger.traza("info", "\n=== Resumen comparativo (media ± desviación típica) ===")
    print(f"{'Backbone':18s} {'Accuracy':>18s} {'F1':>18s}")
    Logger.traza("info", f"{'Backbone':18s} {'Accuracy':>18s} {'F1':>18s}")
    for name in backbones:
        accs = [m["accuracy"] for m in results[name]]
        f1s = [m["f1"] for m in results[name]]
        print(f"{name:18s} {np.mean(accs):.3f} ± {np.std(accs):.3f}    "
              f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
        Logger.traza("info", f"{name:18s} {np.mean(accs):.3f} ± {np.std(accs):.3f}    "
                     f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}")    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True,
                         help="CSV con columnas stl_path,label")
    parser.add_argument("--cache_dir", default="cache_2_5d")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no_pretrained", action="store_true",
                         help="Solo para pruebas offline sin descarga de pesos ImageNet")
    parser.add_argument("--norm", choices=["per_image", "global"], default="per_image",
                         help="per_image: cada canal de cada caso a [0,1] (original). "
                              "global: mismos límites para todos los casos (calculados en train)")
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=None,
                         help="Subconjunto de backbones a evaluar (por defecto, los 4)")
    parser.add_argument("--seed", type=int, default=0,
                         help="Semilla del entrenamiento (los folds NO cambian: siempre 42). "
                              "Repite con --seed 1, 2... para estimar la variabilidad entre ejecuciones")
    add_view_args(parser)
    args = parser.parse_args()
    Logger.traza("info", f"Ejecutando comparación de backbones con manifest={args.manifest}, "
                       f"cache_dir={args.cache_dir}, k={args.k}, epochs={args.epochs}, no_pretrained={args.no_pretrained}") 
    run_comparison(
        manifest_csv=args.manifest,
        cache_dir=args.cache_dir,
        k=args.k,
        epochs=args.epochs,
        pretrained=not args.no_pretrained,
        norm_mode=args.norm,
        backbones=args.backbones,
        seed=args.seed,
        ds_kwargs=view_kwargs(args),
    )
