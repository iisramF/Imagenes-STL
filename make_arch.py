import numpy as np
from scipy.spatial.transform import Rotation
from stl import mesh as stlmesh

def make_arch(path, X=28.0, depth=42.0, seed=0, rotate=True, jitter=0.0, left=1.0, right=1.0, closed=False, zbase=-26.0, rot=None):
    """Arcada inferior sintética. Frente (incisivos) = +y, oclusal = +z antes de rotar.
    Devuelve (R, t): rotación y traslación aplicadas, para conocer la verdad."""
    rng = np.random.default_rng(seed)
    t = np.linspace(-X * left, X * right, 420)
    cy = depth * (1 - (t / X) ** 2)                       # ápice en y=depth, extremos y=0
    dy = -2 * depth * t / X**2
    tang = np.column_stack([np.ones_like(t), dy]); tang /= np.linalg.norm(tang, axis=1)[:, None]
    nrm = np.column_stack([-tang[:, 1], tang[:, 0]])      # normal hacia fuera (vestibular), +y en el ápice
    s = np.unique(np.concatenate([np.linspace(-9, 10, 70), np.linspace(4.3, 5.2, 18), np.linspace(-5.2, -4.3, 18)]))
    zk = np.array([-18, -14, -8, 0, 0, -9, -18, -22.0]); sk = np.array([-9, -7, -5, -4.5, 4.5, 5.0, 7.5, 10.0])
    zs = np.interp(s, sk, zk)
    T_, S_ = np.meshgrid(np.arange(len(t)), np.arange(len(s)), indexing="ij")
    P = np.zeros((len(t), len(s), 3))
    P[..., 0] = cy[:, None] * 0 + t[:, None] + s[None, :] * nrm[:, 0][:, None]
    P[..., 1] = cy[:, None] + s[None, :] * nrm[:, 1][:, None]
    P[..., 2] = zs[None, :] + 0.35 * np.sin(2 * np.pi * t / 6.5)[:, None] * (np.abs(s)[None, :] < 4.5)
    if jitter: P += rng.normal(0, jitter, P.shape)
    V = P.reshape(-1, 3)
    idx = lambda i, j: i * len(s) + j
    F = []
    for i in range(len(t) - 1):
        for j in range(len(s) - 1):
            a, b, c, d = idx(i, j), idx(i + 1, j), idx(i, j + 1), idx(i + 1, j + 1)
            F += [[a, b, c], [b, d, c]]          # normales hacia +z en la zona oclusal
    F = np.array(F)
    if closed:
        # Malla cerrada: superficie superior + base plana + 4 paredes laterales
        nt, ns = len(t), len(s)
        Vb = V.copy(); Vb[:, 2] = zbase
        off = len(V); V = np.vstack([V, Vb])
        Fb = [[a + off, c + off, b + off] for a, b, c in F[::2]] + [[b + off, c + off, d + off] for b, d, c in [(f[0], f[1], f[2]) for f in F[1::2]]]
        # base: mismos cuadrados con orientación invertida (normales hacia abajo)
        Fb = []
        for i in range(nt - 1):
            for j in range(ns - 1):
                a, b, c, d = idx(i, j) + off, idx(i + 1, j) + off, idx(i, j + 1) + off, idx(i + 1, j + 1) + off
                Fb += [[a, c, b], [b, c, d]]
        walls = []
        def wall(p, q):          # arista superior p->q y su inferior; normal hacia fuera
            walls.extend([[p, q + off, q], [p, p + off, q + off]])
        for i in range(nt - 1):
            wall(idx(i + 1, 0), idx(i, 0))                    # borde s mínimo
            wall(idx(i, ns - 1), idx(i + 1, ns - 1))          # borde s máximo
        for j in range(ns - 1):
            wall(idx(0, j), idx(0, j + 1))                    # extremo t mínimo
            wall(idx(nt - 1, j + 1), idx(nt - 1, j))          # extremo t máximo
        F = np.vstack([F, Fb, walls])
        import trimesh
        _m = trimesh.Trimesh(V, F, process=True)
        trimesh.repair.fix_normals(_m)          # winding coherente y normales hacia fuera
        V, F = np.asarray(_m.vertices), np.asarray(_m.faces)
    R = rot if rot is not None else (Rotation.random(random_state=seed) if rotate else Rotation.identity())
    tr = rng.uniform(-50, 50, 3)
    V = R.apply(V) + tr
    d = np.zeros(F.shape[0], dtype=stlmesh.Mesh.dtype)
    d['vectors'] = V[F]
    stlmesh.Mesh(d).save(path)
    return R, tr
