# Imagenes-STL
Clasificación del fenotipo gingival mediante IA aplicada a imágenes STL
<img width="1389" height="935" alt="image" src="https://github.com/user-attachments/assets/221a0977-8fb4-4a4b-a9d9-2e11f1e3e4f8" />


Diagrama del proceso de Transfer Learning para clasificar imágenes STL
He revisado los tres módulos que has adjuntado: train.py, dataset.py y backbones.py.
Tu implementación utiliza Transfer Learning con Fine-Tuning parcial para clasificar mallas STL de escaneos intraorales en dos categorías:
* 0: Fenotipo gingival fino.
* 1: Fenotipo gingival grueso.

El aspecto fundamental es que no entrenas una red neuronal desde cero. Utilizas 
cuatro arquitecturas previamente entrenadas con ImageNet, conservas gran parte de sus conocimientos y adaptas sus últimas capas para distinguir los dos fenotipos.

## Descripción de los módulos python

### Bloque principal

* **train.py:** Entrenamiento y comparación de los 4 backbones con K-Fold estratificado.
* **dataset.py:** Genera la imagen apilada de 3 canales (depth, curvature, render).
* **backbones.py:** Arquitectura y transfer learning: 4 backbones preentrenados en ImageNet.

### Resto de módulos
* **str_to_2_5d:** Convierte un archivo STL (malla 3D de un escaneo intraoral) en tres tipos de imagen 2.5D utilizables como entrada de una CNN 2D (ResNet, EfficientNet, etc.). Esta funcionalidad está integrada en dataset.py.
* **convert_obj_to_stl.py:** Proceso que convierte una lista de archivos OBJ en sus equivalentes STL.
* **\Coef. Spearman:** Diagnóstico de etiquetas: ¿qué mide realmente cada candidato a etiqueta?. 
* **\Entrada_datos:**  Archivos CSV utilizados como parámetros en la iteraciones de prueba.

## Diagrama general del entrenamiento
El siguiente diagrama representa el flujo real de tu código, desde MANIFEST.CSV hasta la comparación final de los cuatro modelos.

<img width="619" height="1158" alt="image" src="https://github.com/user-attachments/assets/87a62b52-eee7-4d0d-91f1-43c42de4659e" />
