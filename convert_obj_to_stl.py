"""
    Prormpta para construir un modelo de clasificación de imágenes 2D a partir de un backbone preentrenado en ImageNet y una cabeza de clasificación personalizada.
    Crea un nuevo módulo python que busque subdirectorios en Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower
    en cada subdirectorios contiene un archivo json con este formato:
    {
        "id_patient": "OSZV3Q38",
        "jaw": "lower",
        "labels": [46, 0, 33, 2]
    }
    Desprecia los subdirectorios cuyo json no contenga el valor lower en la etiqueta jaw.
    Si contiene lower, montamos la siguiente ruta:
    Ruta subdirectorio + '\' + valor de etiqueta id_patient + '_lower.obj'
    Ejemplo:
    Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower\OSZV3Q38\OSZV3Q38_lower.obj

    Convertimos la imagen obj a stl con el siguiente código

    import trimesh
    obj_path = r"Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower\OSZV3Q38\OSZV3Q38_lower.obj"
    stl_path = r"Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower\OSZV3Q38\OSZV3Q38_lower.stl"
    mesh = trimesh.load(obj_path)
    mesh.export(stl_path)  # Lo guarda como STL

    Por cada imagen STL generada, guardamos su ruta completa en el fichero RutasStl.txt
    
"""

import json
from pathlib import Path

import trimesh

from project_logger import Logger


LOWER_DATASET_DIR = Path(
    r"Y:\Máster\TFM\Imágenes STL\3DTeethSeg\Imágenes\lower"
)


def convert_lower_meshes(
    root_dir: str | Path = LOWER_DATASET_DIR,
    routes_file: str | Path | None = None,
) -> list[Path]:
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"No existe el directorio de datos: {root}")

    output_file = Path(routes_file) if routes_file else root / "RutasStl.txt"
    converted_paths = []

    for subdirectory in sorted(root.iterdir()):
        if not subdirectory.is_dir():
            continue

        for json_path in sorted(subdirectory.glob("*.json")):
            try:
                with json_path.open(encoding="utf-8-sig") as json_file:
                    metadata = json.load(json_file)
            except (OSError, json.JSONDecodeError) as error:
                Logger.traza("warn", f"No se pudo leer {json_path}: {error}")
                continue

            if not isinstance(metadata, dict) or metadata.get("jaw") != "lower":
                continue

            patient_id = metadata.get("id_patient")
            if not isinstance(patient_id, str) or not patient_id.strip():
                Logger.traza("warn", f"id_patient no válido en {json_path}")
                continue

            obj_path = subdirectory / f"{patient_id}_lower.obj"
            stl_path = subdirectory / f"{patient_id}_lower.stl"
            if not obj_path.is_file():
                Logger.traza("warn", f"No se encontró el OBJ: {obj_path}")
                continue

            try:
                mesh = trimesh.load(obj_path, force="mesh")
                if not isinstance(mesh, trimesh.Trimesh) or mesh.is_empty:
                    Logger.traza("warn", f"El OBJ no contiene una malla válida: {obj_path}")
                    continue
                mesh.export(str(stl_path), file_type="stl")
            except Exception as error:
                Logger.traza("error", f"Error convirtiendo {obj_path}: {error}")
                continue

            converted_paths.append(stl_path.resolve())
            Logger.traza("info", f"STL generado: {stl_path}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(
        "".join(f"{path}\n" for path in converted_paths),
        encoding="utf-8",
    )
    Logger.traza(
        "info",
        f"Conversión finalizada: {len(converted_paths)} STL; rutas guardadas en {output_file}",
    )
    return converted_paths


if __name__ == "__main__":
    convert_lower_meshes()
