import os
import sys
from pathlib import Path


logger_utils_dir = Path(
    os.environ.get(
        "APP_MONITOREO_LOGGER_DIR",
        Path(__file__).resolve().parent.parent / "AppMonitoreo" / "logger" / "utils",
    )
).resolve()

if not (logger_utils_dir / "Logger.py").is_file():
    raise ModuleNotFoundError(
        f"No se encontró Logger.py en {logger_utils_dir}. "
        "Configura APP_MONITOREO_LOGGER_DIR con la ruta de logger/utils."
    )

sys.path.insert(0, str(logger_utils_dir))
from Logger import Logger