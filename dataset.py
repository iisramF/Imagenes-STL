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

import csv
import os
import numpy as np
import trimesh
from scipy import ndimage
from scipy.interpolate import griddata
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from stl import mesh as stlmesh
from project_logger import Logger   

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

VIEWS = ("occlusal", "incisors_vestibular")


def load_orientation_overrides(csv_path: str) -> dict:
    """
    CSV con columnas: stl_path, flip_up (0/1), flip_front (0/1), rot_deg (grados;
    vacío = 0). Sirve para corregir a mano los casos en que la orientación
    automática falla (ver qc_incisor_views.py): flip_up invierte arriba/abajo,
    flip_front invierte delante/detrás y rot_deg gira la dirección frontal en el
    plano oclusal. Se indexa por ruta completa y por nombre de fichero.
    """
    overrides = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            flags = {"flip_up": int(row.get("flip_up") or 0),
                     "flip_front": int(row.get("flip_front") or 0),
                     "rot_deg": float(row.get("rot_deg") or 0)}
            overrides[row["stl_path"]] = flags
            overrides.setdefault(os.path.basename(row["stl_path"]), flags)
    return overrides


def add_view_args(parser):
    """Argumentos de línea de comandos comunes a los scripts que construyen el Dataset."""
    parser.add_argument("--view", choices=VIEWS, default="occlusal",
                        help="occlusal: vista en planta de toda la arcada (original). "
                             "incisors_vestibular: sector incisivo visto desde vestibular")
    parser.add_argument("--window_mm", type=float, default=36.0,
                        help="Lado (mm) de la ventana cuadrada de la vista vestibular")
    parser.add_argument("--depth_mm", type=float, default=15.0,
                        help="Profundidad (mm) hacia atrás desde el ápice incisivo")
    parser.add_argument("--overrides", default=None,
                        help="CSV de correcciones de orientación (stl_path,flip_up,flip_front,rot_deg)")


def view_kwargs(args) -> dict:
    """Convierte los argumentos de add_view_args en kwargs de GingivalSTLDataset."""
    return dict(
        view=args.view, window_mm=args.window_mm, depth_mm=args.depth_mm,
        orientation_overrides=(load_orientation_overrides(args.overrides)
                               if args.overrides else None),
    )



class GingivalSTLDataset(Dataset):
    """
    samples: lista de tuplas (ruta_stl, label) con label en {0: fino, 1: grueso}.
    cache_dir: si se indica, guarda/lee los mapas sin normalizar como .npy para
    no recalcular la conversión STL->2.5D en cada época (el paso más costoso).
    norm_mode: "per_image" (por defecto) o "global" (ver docstring del módulo).
    norm_bounds: array (3, 2) con (límite_inf, límite_sup) por canal. Solo se
    usa con norm_mode="global"; se obtiene con compute_norm_bounds() sobre el
    dataset de entrenamiento y se pasa al de validación.
    view: "occlusal" (vista en planta de toda la arcada, orientación por PCA) o
    "incisors_vestibular" (sector incisivo visto desde vestibular, orientación
    anatómica, ventana cuadrada de window_mm milímetros: misma escala en todos
    los casos). Ver incisor_view().
    augment: aplica aumentado geométrico 2D ligero (flip/rotación) sobre la
    imagen ya generada -- una aproximación pragmática al aumentado a nivel
    de malla 3D mencionado en las limitaciones metodológicas.
    """

    def __init__(self, samples, resolution: int = 224,
                 cache_dir: str | None = None, augment: bool = False,
                 curvature_radius: float = 1.0,
                 norm_mode: str = "per_image", norm_bounds=None,
                 view: str = "occlusal", window_mm: float = 36.0,
                 depth_mm: float = 15.0, top_margin_mm: float = 3.0,
                 orientation_overrides: dict | None = None):
        if view not in VIEWS:
            raise ValueError(f"view no válida: {view}")
        self.view = view
        self.window_mm = float(window_mm)
        self.depth_mm = float(depth_mm)
        self.top_margin_mm = float(top_margin_mm)
        self.orientation_overrides = orientation_overrides
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
        if self.view == "incisors_vestibular":
            return self.incisor_view(stl_path)[0]
        mesh = self._normalize_mesh(self._load_stl_as_trimesh(stl_path))

        depth = self._generate_depth_map(mesh)
        curvature = self._generate_curvature_map(mesh)
        render = self._generate_orthographic_render(mesh)

        return np.stack([depth, curvature, render], axis=-1).astype(np.float32)

    # -----------------------------------------------------------------
    # Vista vestibular del sector incisivo
    # -----------------------------------------------------------------
    # Marco anatómico (coordenadas en mm, origen en el centroide de la malla):
    #   X = derecha de la imagen (u = v x a), Y = hacia delante (a), Z = hacia
    #   oclusal (v). La cámara está delante de los incisivos mirando hacia atrás.

    def _flags_for(self, stl_path: str) -> dict:
        flags = {"flip_up": 0, "flip_front": 0, "rot_deg": 0.0}
        overrides = self.orientation_overrides or {}
        found = overrides.get(stl_path) or overrides.get(os.path.basename(stl_path))
        if found:
            flags.update(found)
        return flags

    def _incisor_geometry(self, stl_path: str) -> dict:
        """Orienta la malla con criterios anatómicos y define la ventana del sector incisivo."""
        mesh = self._load_stl_as_trimesh(stl_path)
        flags = self._flags_for(stl_path)

        # Momentos ponderados por área de cara (robustos a la densidad de malla)
        C, A = mesh.triangles_center, mesh.area_faces
        wsum = A.sum()
        m = (A[:, None] * C).sum(axis=0) / wsum
        D = C - m
        cov = (A[:, None] * D).T @ D / wsum
        _, evecs = np.linalg.eigh(cov)

        # Eje vertical: menor varianza. Signo: hacia donde apunta la suma de
        # normales ponderadas por área (en un escaneo abierto, hacia oclusal).
        v = evecs[:, 0]
        vert_conf = float((A * (mesh.face_normals @ v)).sum() / wsum)
        if vert_conf < 0:
            v, vert_conf = -v, -vert_conf
        if flags["flip_up"]:
            v = -v

        # Eje anteroposterior = eje de simetría especular de la arcada en el plano
        # oclusal (plano medio sagital). No se usan los ejes del PCA en el plano
        # porque, si las dos varianzas son parecidas, quedan indeterminados.
        b1, b2 = evecs[:, 1], evecs[:, 2]
        P2 = np.column_stack([D @ b1, D @ b2])
        if len(A) > 60000:                     # submuestreo proporcional al área
            sel = np.random.default_rng(0).choice(len(A), 60000, replace=False, p=A / wsum)
            P2s, ws = P2[sel], np.ones(60000)
        else:
            P2s, ws = P2, A
        radius = np.percentile(np.linalg.norm(P2s, axis=1), 99) * 1.05
        edges = np.linspace(-radius, radius, 33)
        H0, _, _ = np.histogram2d(P2s[:, 0], P2s[:, 1], bins=[edges, edges], weights=ws)

        def asymmetry(theta: float) -> float:
            d = np.array([np.cos(theta), np.sin(theta)])
            refl = 2 * (P2s @ d)[:, None] * d[None, :] - P2s
            Hr, _, _ = np.histogram2d(refl[:, 0], refl[:, 1], bins=[edges, edges], weights=ws)
            return float(np.abs(H0 - Hr).sum() / (H0.sum() + Hr.sum()))

        coarse = np.deg2rad(np.arange(0, 180, 4.0))
        theta0 = coarse[int(np.argmin([asymmetry(t) for t in coarse]))]
        fine = theta0 + np.deg2rad(np.arange(-4, 4.01, 0.5))
        fine_scores = [asymmetry(t) for t in fine]
        theta = fine[int(np.argmin(fine_scores))]
        sym_score = float(np.min(fine_scores))
        sym_margin = float(sym_score / max(asymmetry(theta + np.pi / 2), 1e-9))  # pequeño = eje claro

        e_ap = np.cos(theta) * b1 + np.sin(theta) * b2
        e_perp = np.cross(v, e_ap)
        p_ap, q = D @ e_ap, D @ e_perp

        # Sentido: el extremo anterior de la arcada es el más estrecho en
        # transversal (los molares son la zona más ancha).
        lo, hi = np.percentile(p_ap, [2, 98])
        span = hi - lo

        def extent(mask):
            if mask.sum() < 10:
                return np.nan
            ql, qh = np.percentile(q[mask], [2, 98])
            return qh - ql

        ext_lo = extent(p_ap <= lo + 0.15 * span)
        ext_hi = extent(p_ap >= hi - 0.15 * span)
        anterior_is_high = bool(ext_hi < ext_lo)
        a = e_ap if anterior_is_high else -e_ap
        if flags["flip_front"]:
            a = -a
        if flags["rot_deg"]:                       # giro manual alrededor del eje vertical
            phi = np.deg2rad(flags["rot_deg"])
            a = np.cos(phi) * a + np.sin(phi) * np.cross(v, a)
        u = np.cross(v, a)
        T = np.column_stack([u, a, v])

        Fv = (mesh.vertices - m) @ T      # vértices en el marco anatómico
        Fc = D @ T                        # centroides de cara en el marco anatómico

        # Ventana del sector incisivo
        W = self.window_mm
        u_lo, u_hi = np.percentile(Fc[:, 0], [1, 99])
        u_mid = 0.5 * (u_lo + u_hi)                      # línea media de la arcada
        band = np.abs(Fv[:, 0] - u_mid) < 6.0
        a_apex = (float(np.percentile(Fv[band, 1], 99.5)) if band.sum() > 20
                  else float(Fv[:, 1].max()))            # punto más anterior en la línea media
        sector = (np.abs(Fv[:, 0] - u_mid) <= W / 2) & (Fv[:, 1] >= a_apex - 8.0)
        top_v = (float(np.percentile(Fv[sector, 2], 98)) if sector.sum() > 20
                 else float(Fv[:, 2].max()))             # bordes incisales
        win = dict(u_left=u_mid - W / 2, v_top=top_v + self.top_margin_mm,
                   a_apex=a_apex, a_min=a_apex - self.depth_mm)
        inbox = ((Fv[:, 0] >= win["u_left"]) & (Fv[:, 0] < win["u_left"] + W) &
                 (Fv[:, 2] <= win["v_top"]) & (Fv[:, 2] > win["v_top"] - W) &
                 (Fv[:, 1] >= win["a_min"]) & (Fv[:, 1] <= a_apex + 1.0))

        a_lo, a_hi = np.percentile(Fc[:, 1], [1, 99])
        diag = dict(
            arch_width_mm=float(u_hi - u_lo), arch_depth_mm=float(a_hi - a_lo),
            vert_conf=vert_conf,
            front_ratio=float(max(ext_lo, ext_hi) / max(min(ext_lo, ext_hi), 1e-6)),
            sym_score=sym_score, sym_margin=sym_margin,
        )
        return dict(mesh=mesh, m=m, T=T, Fv=Fv, inbox=inbox, win=win, diag=diag)

    def incisor_view(self, stl_path: str):
        """
        Vista vestibular del sector incisivo. Devuelve (raw, diag):
          raw  (N, N, 3) float32 = [profundidad, curvatura, render] SIN normalizar.
               profundidad = distancia en mm al ápice incisivo (<= ~1; fondo = -depth_mm)
          diag diccionario de calidad de la orientación (para qc_incisor_views.py).
        Solo se usan los vértices que miran a la cámara (normal con componente
        anterior > 0) y se queda el más cercano en cada píxel (z-buffer), de modo
        que la superficie lingual, situada detrás, no se mezcla con la vestibular.
        """
        g = self._incisor_geometry(stl_path)
        mesh, Fv, win, diag = g["mesh"], g["Fv"], g["win"], g["diag"]
        N, W = self.resolution, self.window_mm
        pix = W / N
        vn = mesh.vertex_normals @ g["T"]
        idx = np.where(g["inbox"] & (vn[:, 1] > 0.0))[0]

        depth = np.full((N, N), np.nan, dtype=np.float32)
        curv = np.full((N, N), np.nan, dtype=np.float32)
        rend = np.full((N, N), np.nan, dtype=np.float32)

        if len(idx) > 0:
            r = np.clip(((win["v_top"] - Fv[idx, 2]) / pix).astype(int), 0, N - 1)
            c = np.clip(((Fv[idx, 0] - win["u_left"]) / pix).astype(int), 0, N - 1)
            d = Fv[idx, 1] - win["a_apex"]
            buf = np.full((N, N), -np.inf)
            np.maximum.at(buf, (r, c), d)
            winner = d >= buf[r, c] - 1e-6
            iw, rw, cw = idx[winner], r[winner], c[winner]
            depth[rw, cw] = d[winner]
            curv[rw, cw] = trimesh.curvature.discrete_mean_curvature_measure(
                mesh, mesh.vertices[iw], radius=self.curvature_radius)
            light = np.array([-0.3, 1.0, 0.4])
            light /= np.linalg.norm(light)
            rend[rw, cw] = np.clip(vn[iw] @ light, 0, 1)

        valid = np.isfinite(depth)
        coverage, closed = 0.0, valid
        if valid.sum() >= 4:
            # Cerrar huecos por falta de vértices (no el fondo vacío) y rellenarlos
            iters = int(np.clip(np.ceil(mesh.edges_unique_length.mean() / pix), 1, 12))
            padded = np.pad(valid, iters, mode="edge")
            closed = ndimage.binary_closing(padded, structure=np.ones((3, 3), bool),
                                            iterations=iters)[iters:-iters, iters:-iters]
            closed = ndimage.binary_fill_holes(closed)
            holes = closed & ~valid
            if holes.any():
                pts = np.column_stack(np.where(valid))
                hp = np.column_stack(np.where(holes))
                for ch in (depth, curv, rend):
                    vals = ch[valid]
                    filled = griddata(pts, vals, hp, method="linear")
                    nan = np.isnan(filled)
                    if nan.any():
                        filled[nan] = griddata(pts, vals, hp[nan], method="nearest")
                    ch[holes] = filled
            coverage = float(closed.mean())
        depth[~closed] = -self.depth_mm
        curv[~closed] = 0.0
        rend[~closed] = 0.0

        reasons = []
        if not diag["vert_conf"] >= 0.15:
            reasons.append("vertical dudosa")
        if not diag["front_ratio"] >= 1.25:
            reasons.append("frente dudoso")
        if diag["sym_margin"] > 0.7 or diag["sym_score"] > 0.25:
            reasons.append("ejes ambiguos o arcada asimétrica")
        if diag["arch_width_mm"] < 45 or diag["arch_depth_mm"] < 25:
            reasons.append("arcada pequeña (¿escaneo parcial?)")
        if coverage < 0.15:
            reasons.append("ventana casi vacía")
        diag.update(coverage=coverage, reasons=reasons, n_front_vertices=int(len(idx)))

        return np.stack([depth, curv, rend], axis=-1).astype(np.float32), diag

    def incisor_sector_vertices(self, stl_path: str):
        """
        Vértices dentro de la ventana del sector (coordenadas del marco anatómico,
        en mm) y su curvatura media discreta. Sirve para calcular las etiquetas de
        control SOBRE EL MISMO RECORTE que ve el modelo.
        """
        g = self._incisor_geometry(stl_path)
        idx = np.where(g["inbox"])[0]
        curv = trimesh.curvature.discrete_mean_curvature_measure(
            g["mesh"], g["mesh"].vertices[idx], radius=self.curvature_radius)
        return g["Fv"][idx], curv

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
            base = os.path.basename(stl_path)
            if self.view == "occlusal":
                name = f"{base}_raw_r{self.curvature_radius}_{self.resolution}.npy"
            else:
                fl = self._flags_for(stl_path)
                tag = f"{fl['flip_up']}{fl['flip_front']}{fl['rot_deg']:g}"
                name = (f"{base}_incv_w{self.window_mm}_d{self.depth_mm}_m{self.top_margin_mm}"
                        f"_o{tag}_r{self.curvature_radius}_{self.resolution}.npy")
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
