"""
Arquitectura y transfer learning: 4 backbones preentrenados en ImageNet
=========================================================================
Construye cada backbone con sus pesos de ImageNet, sustituye la cabeza de
clasificación original (1000 clases) por una cabeza binaria (fenotipo
fino/grueso), y congela las primeras etapas para fine-tuning parcial.

Entrada esperada: imagen de 3 canales, 224x224, generada apilando
[depth_map, curvature_map, render] del pipeline STL -> 2.5D.
"""

import torch
import torch.nn as nn
from project_logger import Logger
from torchvision.models import (
    resnet50, ResNet50_Weights,
    efficientnet_b3, EfficientNet_B3_Weights,
    densenet121, DenseNet121_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
)

NUM_CLASSES = 2  # fino / grueso


def build_backbone(name: str, freeze_backbone: bool = True, pretrained: bool = True) -> nn.Module:
    """
    Devuelve el modelo completo (backbone + cabeza nueva) listo para
    fine-tuning, según el nombre del backbone solicitado.

    freeze_backbone=True congela todas las capas salvo la última etapa
    convolucional y la cabeza de clasificación (fine-tuning parcial,
    recomendado dado el tamaño muestral pequeño esperado).

    pretrained=False permite construir la arquitectura sin descargar pesos
    de ImageNet (útil solo para pruebas offline; en producción usar True).
    """
    name = name.lower()
    Logger.traza("info", f"Construyendo backbone={name}, pretrained={pretrained}, freeze_backbone={freeze_backbone}")

    if name == "resnet50":
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = resnet50(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.layer4.parameters():   # última etapa
                param.requires_grad = True
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, NUM_CLASSES)

    elif name == "efficientnet_b3":
        weights = EfficientNet_B3_Weights.IMAGENET1K_V1 if pretrained else None
        model = efficientnet_b3(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features[-1].parameters():  # último bloque
                param.requires_grad = True
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    elif name == "densenet121":
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = densenet121(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features.denseblock4.parameters():  # último bloque denso
                param.requires_grad = True
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, NUM_CLASSES)

    elif name == "convnext_tiny":
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        model = convnext_tiny(weights=weights)
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.features[-1].parameters():  # última etapa
                param.requires_grad = True
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    else:
        Logger.traza("error", f"Backbone no reconocido: {name}")
        raise ValueError(f"Backbone no reconocido: {name}")

    return model


def count_trainable_params(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


if __name__ == "__main__":
    backbones = ["resnet50", "efficientnet_b3", "densenet121", "convnext_tiny"]

    # Imagen de prueba: simula un batch de 2 imágenes 3x224x224
    # (el canal 0 sería depth_map, canal 1 curvature_map, canal 2 render)
    dummy_batch = torch.randn(2, 3, 224, 224)

    for name in backbones:
        model = build_backbone(name, freeze_backbone=True)
        model.eval()
        with torch.no_grad():
            output = model(dummy_batch)
        trainable, total = count_trainable_params(model)
        pct = 100 * trainable / total
        Logger.traza("info", f"Backbone {name}: salida {tuple(output.shape)}, "
                            f"parámetros entrenables {trainable:,} / {total:,} ({pct:.1f}%)")
        print(f"{name:18s} -> salida: {tuple(output.shape)} | "
              f"parámetros entrenables: {trainable:,} / {total:,} ({pct:.1f}%)")
