"""
Dataset: STL -> imagen apilada de 3 canales (depth, curvature, render)
=========================================================================
Versión compacta: el pipeline completo de conversión STL -> 2.5D
(normalización de malla, depth map, curvature map, render ortográfico)
vive como métodos privados de GingivalSTLDataset, en vez de depender de
funciones sueltas en stl_to_2_5d.py.

Se cachean en disco los mapas SIN normalizar (valores físicos), y la
normalización de cada canal se aplica después, al servir cada muestra:
  - norm_mode="per_image": cada canal de cada caso se reescala a [0, 1]
    con sus propios percentiles 2-98 (comportamiento original; descarta las
    magnitudes absolutas entre casos).
  - norm_mode="global": todos los casos comparten los mismos límites por
    canal (percentiles 2-98 agrupados sobre el conjunto de ENTRENAMIENTO),
    de modo que las diferencias de magnitud entre casos se conservan.
"""

import os
import numpy as np
import trimesh
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from stl import mesh as stlmesh
from project_logger import Logger                                 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class GingivalSTLDataset(Dataset):
    """
    samples: lista de tuplas (ruta_stl, label) con label en {0: fino, 1: grueso}.
    cache_dir: si se indica, guarda/lee los mapas sin normalizar como .npy para
    no recalcular la conversión STL->2.5D en cada época (el paso más costoso).
    norm_mode: "per_image" (por defecto) o "global" (ver docstring del módulo).
    norm_bounds: array (3, 2) con (límite_inf, límite_sup) por canal. Solo se
    usa con norm_mode="global"; se obtiene con compute_norm_bounds() sobre el
    dataset de entrenamiento y se pasa al de validación.
    augment: aplica aumentado geométrico 2D ligero (flip/rotación) sobre la
    imagen ya generada -- una aproximación pragmática al aumentado a nivel
    de malla 3D mencionado en las limitaciones metodológicas.
    """

    def __init__(self, samples, resolution: int = 224,
                 cache_dir: str | None = None, augment: bool = False,
                 curvature_radius: float = 1.0,
                 norm_mode: str = "per_image", norm_bounds=None):
        if norm_mode not in ("per_image", "global"):
            raise ValueError(f"norm_mode no válido: {norm_mode}")
        self.norm_mode = norm_mode
        self.norm_bounds = (None if norm_bounds is None
                            else np.asarray(norm_bounds, dtype=np.float32))
        self.samples = samples
        self.resolution = resolution
        self.cache_dir = cache_dir
        self.curvature_radius = curvature_radius
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        aug_transforms = []
        if augment:
            aug_transforms = [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=10),
            ]

        self.transform = T.Compose([
            *aug_transforms,
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    # -----------------------------------------------------------------
    # Pipeline STL -> 2.5D (métodos privados, antes en stl_to_2_5d.py)
    # -----------------------------------------------------------------

    def _load_stl_as_trimesh(self, path: str) -> trimesh.Trimesh:
        """Carga con numpy-stl y fusiona vértices duplicados -> trimesh.Trimesh."""
        raw = stlmesh.Mesh.from_file(path)
        vertices = raw.vectors.reshape(-1, 3)
        unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)
        faces = inverse.reshape(-1, 3)
        return trimesh.Trimesh(vertices=unique_vertices, faces=faces, process=True)

    def _normalize_mesh(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """Centra la malla y la orienta por PCA (normal dominante -> eje Z)."""
        mesh = mesh.copy()
        mesh.vertices -= mesh.vertices.mean(axis=0)

        cov = np.cov(mesh.vertices.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        rotation = eigvecs[:, [1, 2, 0]]
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1

        mesh.vertices = mesh.vertices @ rotation
        return mesh

    def _generate_depth_map(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Proyecta los vértices sobre XY e interpola Z -> mapa de profundidad."""
        xy = mesh.vertices[:, :2]
        z = mesh.vertices[:, 2]
        grid_x, grid_y = np.meshgrid(
            np.linspace(xy[:, 0].min(), xy[:, 0].max(), self.resolution),
            np.linspace(xy[:, 1].min(), xy[:, 1].max(), self.resolution),
        )
        return griddata(xy, z, (grid_x, grid_y), method="linear", fill_value=z.min())

    def _generate_curvature_map(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Curvatura media discreta por vértice, interpolada sobre la rejilla XY."""
        curvature = trimesh.curvature.discrete_mean_curvature_measure(
            mesh, mesh.vertices, radius=self.curvature_radius
        )
        xy = mesh.vertices[:, :2]
        grid_x, grid_y = np.meshgrid(
            np.linspace(xy[:, 0].min(), xy[:, 0].max(), self.resolution),
            np.linspace(xy[:, 1].min(), xy[:, 1].max(), self.resolution),
        )
        return griddata(xy, curvature, (grid_x, grid_y), method="linear", fill_value=0.0)

    def _generate_orthographic_render(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """Render ortográfico con sombreado Lambert, sin motor gráfico externo."""
        light_dir = np.array([0.3, 0.3, 1.0])
        light_dir /= np.linalg.norm(light_dir)
        intensity = np.clip(mesh.face_normals @ light_dir, 0, 1)

        centroids = mesh.triangles_center[:, :2]
        grid_x, grid_y = np.meshgrid(
            np.linspace(centroids[:, 0].min(), centroids[:, 0].max(), self.resolution),
            np.linspace(centroids[:, 1].min(), centroids[:, 1].max(), self.resolution),
        )
        return griddata(centroids, intensity, (grid_x, grid_y),
                         method="nearest", fill_value=0.0)

    @staticmethod
    def _minmax(array: np.ndarray) -> np.ndarray:
        """Normaliza un mapa 2D a [0, 1], robusto a outliers (percentiles 2-98)."""
        lo, hi = np.percentile(array, [2, 98])
        array = np.clip(array, lo, hi)
        if hi - lo < 1e-8:
            return np.zeros_like(array)
        return (array - lo) / (hi - lo)

    def _stl_to_raw_maps(self, stl_path: str) -> np.ndarray:
        """Pipeline completo SIN normalizar: STL -> [depth, curvature, render] (H, W, 3)."""
        mesh = self._normalize_mesh(self._load_stl_as_trimesh(stl_path))

        depth = self._generate_depth_map(mesh)
        curvature = self._generate_curvature_map(mesh)
        render = self._generate_orthographic_render(mesh)

        return np.stack([depth, curvature, render], axis=-1).astype(np.float32)

    # -----------------------------------------------------------------
    # Normalización de canales
    # -----------------------------------------------------------------

    def compute_norm_bounds(self, low: float = 2, high: float = 98) -> np.ndarray:
        """
        Percentiles (low, high) por canal, agrupando todos los píxeles de todos
        los casos de ESTE dataset. Llamarlo sobre el dataset de entrenamiento
        (nunca sobre validación) y pasar el resultado como norm_bounds al de
        validación. Guarda los límites en self.norm_bounds y los devuelve.
        """
        raws = [self._load_raw(path) for path, _ in self.samples]
        bounds = np.zeros((3, 2), dtype=np.float32)
        for c in range(3):
            pooled = np.concatenate([r[..., c].ravel() for r in raws])
            bounds[c] = np.percentile(pooled, [low, high])
        self.norm_bounds = bounds
        return bounds

    def _normalize_channels(self, raw: np.ndarray) -> np.ndarray:
        if self.norm_mode == "per_image":
            return np.stack([self._minmax(raw[..., c]) for c in range(3)],
                            axis=-1).astype(np.float32)

        if self.norm_bounds is None:
            raise RuntimeError(
                "norm_mode='global' requiere norm_bounds: llama a "
                "compute_norm_bounds() en el dataset de entrenamiento y pásalos "
                "al de validación."
            )
        out = np.empty_like(raw)
        for c in range(3):
            lo, hi = self.norm_bounds[c]
            ch = np.clip(raw[..., c], lo, hi)
            out[..., c] = 0.0 if hi - lo < 1e-8 else (ch - lo) / (hi - lo)
        return out

    # -----------------------------------------------------------------
    # Interfaz Dataset de PyTorch
    # -----------------------------------------------------------------

    def _load_raw(self, stl_path: str) -> np.ndarray:
        if self.cache_dir:
            name = (f"{os.path.basename(stl_path)}_raw_r{self.curvature_radius}"
                    f"_{self.resolution}.npy")
            cache_path = os.path.join(self.cache_dir, name)
            if os.path.exists(cache_path):
                return np.load(cache_path)
            raw = self._stl_to_raw_maps(stl_path)
            Logger.traza("info", f"Cacheando imagen apilada en {cache_path}")                                                                             
            np.save(cache_path, raw)
            return raw
        return self._stl_to_raw_maps(stl_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        stl_path, label = self.samples[idx]
        stacked = self._normalize_channels(self._load_raw(stl_path))  # (H, W, 3) en [0,1]
        tensor = torch.from_numpy(stacked).permute(2, 0, 1)            # (3, H, W)
        tensor = self.transform(tensor)
        return tensor, torch.tensor(label, dtype=torch.long)
