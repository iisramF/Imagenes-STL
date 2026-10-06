#Libreriás necesarias
#pip install numpy-stl trimesh scipy matplotlib rtree

"""
Pipeline STL -> representaciones 2.5D
=======================================
Convierte un archivo STL (malla 3D de un escaneo intraoral) en tres tipos
de imagen 2.5D utilizables como entrada de una CNN 2D (ResNet, EfficientNet, etc.):

  1. Mapa de profundidad / altura (depth map)
  2. Mapa de curvatura
  3. Render ortográfico (vista "fotográfica" de la superficie, con sombreado)

Librerías: numpy-stl (carga), trimesh (curvatura y utilidades de malla),
scipy (interpolación a rejilla regular), matplotlib (guardado/visualización).
"""
from pathlib import Path
import numpy as np
import trimesh
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from stl import mesh as stlmesh
from project_logger import Logger


# ---------------------------------------------------------------------------
# 1. Carga y normalización
# ---------------------------------------------------------------------------

def load_stl_as_trimesh(path: str) -> trimesh.Trimesh:
    """
    Carga el STL con numpy-stl (tal como ya lo haces) y lo convierte
    a trimesh.Trimesh, que ofrece las operaciones de curvatura y
    normales que numpy-stl no trae de serie.
    """
    raw = stlmesh.Mesh.from_file(path)
    Logger.traza("info", f"STL cargado: {path}")

    # numpy-stl no fusiona vértices duplicados (cada triángulo lleva sus propios
    # 3 puntos), así que reconstruimos vértices únicos + índices de caras.
    #El problema de este formato: En el formato STL tradicional, 
    # cada cara/triángulo almacena de forma independiente las coordenadas 3D de sus tres vértices. 
    # Esto significa que si dos o más triángulos comparten un vértice en el espacio, 
    # las coordenadas de ese vértice se guardan duplicadas tantas veces como triángulos lo compartan.

    #Extraer y aplanar las coordenadas de los vértices.
    #raw.vectors es un array con dimensiones (N_triangulos, 3, 3).
    #Al aplicar .reshape(-1, 3), se aplanan las primeras dos dimensiones para obtener un array (N_triangulos * 3, 3) 
    # donde cada fila es un punto en 3D ([x, y, z]).
    vertices = raw.vectors.reshape(-1, 3)

    #Eliminar duplicados y generar el mapa de índices (np.unique). 
    # np.unique: Encuentra los vértices únicos en el array de vértices 3D.
    # axis=0: Compara filas completas (cada fila es un vértice [x, y, z]). Busca filas duplicadas exactas (mismos puntos [x, y, z]).
    # return_inverse=True: Devuelve un array que indica, para cada vértice original, el índice del vértice único correspondiente.
    #Este es el formato estándar de malla de polígonos indexada.
    unique_vertices, inverse = np.unique(vertices, axis=0, return_inverse=True)

    #Reconstrucción de la matriz de caras (Faces)
    # Cada triángulo original tenía 3 vértices, y ahora cada vértice ha sido reemplazado por su índice único.
    # inverse: Array de índices que mapea cada vértice original al índice del vértice único correspondiente.
    # reshape(-1, 3): Reorganiza el array de índices para que cada fila represente un triángulo (3 vértices).
    faces = inverse.reshape(-1, 3)

    mesh = trimesh.Trimesh(vertices=unique_vertices, faces=faces, process=True)
    return mesh


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """
    Normalización estándar antes de proyectar a 2.5D:
      - Centrado en el origen.
      - Orientación mediante PCA: el eje de menor varianza (normal
        "media" de la superficie) se alinea con Z, para que la
        proyección a un plano XY tenga sentido como vista bucal/oclusal.
    """
    mesh = mesh.copy()
    mesh.vertices -= mesh.vertices.mean(axis=0)

    # Análisis de Componentes Principales (PCA)
    # PCA sobre los vértices: el eje asociado al menor autovalor
    # es la normal dominante de la superficie -> lo alineamos con Z.
    cov = np.cov(mesh.vertices.T)
    # Calculamos autovalores y autovectores
    eigvals, eigvecs = np.linalg.eigh(cov)
    # eigh devuelve autovalores ascendentes; el primero (más pequeño) es la normal.
    rotation = eigvecs[:, [1, 2, 0]]  # reordenar: X, Y, Z(normal)
    if np.linalg.det(rotation) < 0:   # evitar reflexión (mantener orientación)
        rotation[:, -1] *= -1

    mesh.vertices = mesh.vertices @ rotation
    # Multiplica espacialmente las coordenadas de los vértices por la matriz de rotación procesada.  
    # Resultado final: La malla devuelta está perfectamente centrada y alineada, 
    #  lista para ser proyectada en el plano X Y para generar los mapas 2.5D 
    #  (profundidad, curvatura y render) sin importar cómo viniera orientada originalmente en el STL. 

    return mesh


# ---------------------------------------------------------------------------
# 2. Generación de representaciones 2.5D
# ---------------------------------------------------------------------------

def generate_depth_map(mesh: trimesh.Trimesh, resolution: int = 256) -> np.ndarray:
    """
    Proyecta los vértices sobre el plano XY e interpola su coordenada Z
    en una rejilla regular -> mapa de profundidad/altura.
    """
    # xy: Extrae únicamente las columnas de las coordenadas X e Y de todos los vértices.
    #  Esto equivale a "aplastar" o proyectar la malla 3D sobre la vista superior XY.
    xy = mesh.vertices[:, :2]
    # z: Extrae la coordenada Z asociada a cada punto.
    #  El valor de Z representa la "altura" o elevación de la superficie en dicho punto XY.  
    z = mesh.vertices[:, 2]

    # np.linspace: Crea un vector de puntos distribuidos de forma uniforme desde el valor mínimo hasta el máximo en las dimensiones X e Y.
    #  El número de puntos equivale al parámetro resolution (por defecto, 256 píxeles). 
    # np.meshgrid: Genera una matriz bidimensional con las coordenadas de todos los píxeles de la futura imagen de tamaño (resolution, resolution).  
    grid_x, grid_y = np.meshgrid(
        np.linspace(xy[:, 0].min(), xy[:, 0].max(), resolution),
        np.linspace(xy[:, 1].min(), xy[:, 1].max(), resolution),
    )

    # griddata: Interpola los valores de Z en la rejilla regular definida por grid_x y grid_y.
    #  - xy: coordenadas originales de los vértices (puntos conocidos).
    #  - z: valores de profundidad asociados a cada vértice.
    #  - (grid_x, grid_y): coordenadas de la rejilla donde queremos estimar los valores de Z.
    #  - method="linear": interpolación lineal entre los puntos conocidos.
    #  - fill_value=z.min(): para los píxeles fuera del área cubierta por la malla, se asigna el valor mínimo de Z (profundidad mínima
    depth_map = griddata(xy, z, (grid_x, grid_y), method="linear", fill_value=z.min())

    # Devuelve una matriz 2D de 256x256 (o la resolución indicada) donde cada píxel almacena un valor numérico de profundidad (Z).
    #  Esta matriz se guarda directamente como una imagen en escala de grises o mapa de calor, 
    # siendo una entrada idónea para redes neuronales convolucionales 2D (CNNs).  
    return depth_map


def generate_curvature_map(mesh: trimesh.Trimesh, resolution: int = 256,
                            radius: float = 1.0) -> np.ndarray:
    """
    Calcula la curvatura geométrica de la superficie 3D en cada vértice del modelo 
    y proyecta esos datos sobre la misma rejilla 2D usada en el mapa de profundidad.
    Esto permite generar una imagen 2.5D que resalta las zonas de cambio drástico 
    de forma (como bordes, valles o crestas dentales).  

    Calcula la curvatura media discreta por vértice (trimesh) y la
    interpola sobre la misma rejilla XY que el depth map, para que
    ambos canales estén espacialmente alineados (útil si luego los
    combinas como canales de una misma imagen multicanal).
    """
    #1. Cálculo de la curvatura discreta por vértice
    # discrete_mean_curvature_measure: Función de trimesh que evalúa la curvatura media discreta sobre la malla triangular.  
    # radius=radius: Define la esfera local alrededor de cada vértice en la que se analiza el comportamiento de la superficie.
    #  Un radio mayor suaviza el cálculo, mientras que un radio pequeño detecta detalles muy finos.
    # Resultado: Un vector 1D donde cada posición contiene el valor numérico de la curvatura calculada exactamente para ese vértice.
    curvature = trimesh.curvature.discrete_mean_curvature_measure(
        mesh, mesh.vertices, radius=radius
    )

    #2. Interpolación a rejilla regular XY
    # xy: coordenadas X e Y de los vértices del modelo.
    xy = mesh.vertices[:, :2]

    # grid_x, grid_y: Genera una malla 2D de coordenadas XY que cubre el área de la malla, con resolución definida por el parámetro resolution.
    #  Esto crea una cuadrícula de píxeles donde se proyectará la curvatura.
    # np.meshgrid: Crea matrices 2D de coordenadas X e Y para cada píxel de la imagen final.
    # np.linspace: Distribuye uniformemente los valores de X e Y desde el mínimo hasta el máximo de la malla, 
    #  generando un rango completo de coordenadas.
    
    # Crea la misma matriz bidimensional de resolution x resolution píxeles que en generate_depth_map.
    #  Esto garantiza un alineamiento espacial perfecto píxel a píxel entre el mapa de profundidad y el mapa de curvatura.
    grid_x, grid_y = np.meshgrid(
        np.linspace(xy[:, 0].min(), xy[:, 0].max(), resolution),
        np.linspace(xy[:, 1].min(), xy[:, 1].max(), resolution),
    )

    #3. Interpolación con scipy.interpolate.griddata
    # griddata: Interpola los valores de curvatura calculados en los vértices sobre la rejilla regular definida por grid_x y grid_y.    
    #  - xy: coordenadas originales de los vértices (puntos conocidos).
    #  - curvature: valores de curvatura asociados a cada vértice.
    #  - (grid_x, grid_y): coordenadas de la rejilla donde queremos estimar los valores de curvatura.
    #  - method="linear": interpolación lineal entre los puntos conocidos.
    #                     Convierte la información puntual de curvatura asignada a vértices 3D dispersos en una imagen matricial continua mediante interpolación lineal (method="linear")
    #  - fill_value=0.0: Asigna un valor de 0.0 (curvatura plana o neutra) a los píxeles del fondo que caen fuera de los límites del objeto 3D.
    curvature_map = griddata(xy, curvature, (grid_x, grid_y),
                              method="linear", fill_value=0.0)

    # Devuelve una matriz 2D de 256x256 (o la resolución indicada) donde cada píxel almacena un valor numérico de curvatura.
    #  Esta matriz se guarda directamente como una imagen en escala de grises o mapa de calor,
    #  siendo una entrada idónea para redes neuronales convolucionales 2D (CNNs).
    return curvature_map


def generate_orthographic_render(mesh: trimesh.Trimesh, resolution: int = 256) -> np.ndarray:
    """
    Simula una iluminación virtual sobre la superficie 3D del objeto para generar 
    una representación en 2D similar a una fotografía en escala de grises (con sombras y relieve). 
    A diferencia de los visores 3D convencionales, esta función implementa 
    un motor de sombreado directo en Python sin depender de librerías gráficas pesadas como OpenGL, PyGame o PyRender,
    lo que permite ejecutar el script en servidores sin pantalla o sin GPU (headless servers).  

    Render ortográfico simple con sombreado tipo "Lambert" (ángulo entre
    normal y una luz fija), sin necesitar un motor gráfico externo
    (pyrender/OpenGL), lo cual facilita ejecutarlo en un servidor sin GPU/pantalla.
    """
    #1. Definición de la fuente de luz
    # light_dir: Vector de dirección de la luz en coordenadas 3D.
    #  Dirección de la luz: Crea un vector 3D [0.3, 0.3, 1.0] que indica de dónde proviene la luz virtual 
    #  (ligeramente desplazada a la derecha y arriba, apuntando hacia abajo)
    # np.linalg.norm: Calcula la magnitud del vector light_dir para normalizarlo.
    #  Esto asegura que la dirección de la luz tenga una longitud de 1,
    #  lo cual es importante para el cálculo del ángulo entre la luz y las normales de la superficie.      
    light_dir = np.array([0.3, 0.3, 1.0])
    light_dir /= np.linalg.norm(light_dir)

    #2. Sombreado Lambertiano. Sombreado por cara (flat shading): intensidad = cos(ángulo normal-luz)
    # Obtiene el vector normal de cada cara/triángulo de la malla (dirección hacia la que apunta el triángulo).
    face_normals = mesh.face_normals
    # Calcula la intensidad de luz en cada cara mediante el producto punto entre la normal de la cara y la dirección de la luz.
    #  - face_normals @ light_dir: Producto escalar entre cada normal de cara y la dirección de la luz, obteniendo un valor escalar que representa el coseno del ángulo entre la normal y la luz.
    #  - np.clip(..., 0, 1): Limita los valores de intensidad entre 0 y 1, asegurando que no haya valores negativos (caras orientadas hacia atrás) ni valores superiores a 1 (caras perfectamente alineadas con
    intensity = np.clip(face_normals @ light_dir, 0, 1)

    # Proyección: asignamos a cada punto de la rejilla la intensidad
    # de la cara más cercana (vía el centroide de cada triángulo).
    centroids = mesh.triangles_center[:, :2]
    grid_x, grid_y = np.meshgrid(
        np.linspace(centroids[:, 0].min(), centroids[:, 0].max(), resolution),
        np.linspace(centroids[:, 1].min(), centroids[:, 1].max(), resolution),
    )

    render = griddata(centroids, intensity, (grid_x, grid_y),
                       method="nearest", fill_value=0.0)
    return render


# ---------------------------------------------------------------------------
# 3. Guardado
# ---------------------------------------------------------------------------
#def save_map_as_image(array_2d: np.ndarray, out_path: str, cmap: str = "viridis"):
#    plt.imsave(out_path, array_2d, cmap=cmap, origin="lower")
def save_map_as_image(array_2d: np.ndarray, out_path: str, cmap: str = "viridis",
                       robust: bool = False, percentile: float = 2.0):
    """
    robust=True aplica recorte por percentiles antes de mapear a color.
    Es imprescindible para curvatura: unos pocos vértices en aristas muy
    afiladas (quilla de un barco, borde de un diente) pueden tener valores
    órdenes de magnitud mayores que el resto de la superficie, y sin recorte
    esos outliers acaparan todo el rango de color dejando el resto plano.
    """
    # Copia del array para no modificar el original
    data = array_2d.copy()
    # Recorte por percentiles si se solicita
    if robust:
        lo, hi = np.percentile(data, [percentile, 100 - percentile])
        data = np.clip(data, lo, hi)

    # Guardado de la imagen con el colormap especificado.
    plt.imsave(out_path, data, cmap=cmap, origin="lower")


# ---------------------------------------------------------------------------
# Uso
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    stl_path = "Y:\\Máster\\Imágenes STL\\3DBenchy.stl"  # sustituye por tu STL real
    #stl_path = "Y:\\Máster\\Imágenes STL\\Calibration Cube.stl"  # sustituye por tu STL real
    stl_path = r"Y:\Máster\TFM\Imágenes STL\Walter\Prueba\POST\315079438_lprofile_occlusion_l.stl"
    stl_path = r"Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower\OSZV3Q38\OSZV3Q38_lower.obj"

    #Carga imagen stl y fusiona vértices duplicados para convertirlo en un trimesh.Trimesh, 
    # necesario porque numpy-stl no comparte vértices entre triángulos y eso rompe los cálculos de curvatura/normales.
    mesh = load_stl_as_trimesh(stl_path)
    #Normalización de la malla: centrado y alineación PCA para que la proyección a XY tenga sentido.
    mesh = normalize_mesh(mesh)

    #Generación de las tres representaciones 2.5D: mapa de profundidad, mapa de curvatura y render ortográfico.
    depth_map = generate_depth_map(mesh, resolution=512)
    curvature_map = generate_curvature_map(mesh, resolution=512, radius=1.0)
    render = generate_orthographic_render(mesh, resolution=512)

    out_dir = Path(r"Y:\Máster\Imágenes STL")
    out_dir.mkdir(parents=True, exist_ok=True)

    save_map_as_image(depth_map, str(out_dir / "depth_map_512.png"), cmap="viridis")
    save_map_as_image(curvature_map, str(out_dir / "curvature_map_512.png"), cmap="coolwarm", robust=True)
    save_map_as_image(render, str(out_dir / "render_512.png"), cmap="gray")

    print("Depth map:", depth_map.shape, depth_map.min(), depth_map.max())
    print("Curvature map:", curvature_map.shape, curvature_map.min(), curvature_map.max())
    print("Render:", render.shape, render.min(), render.max())
