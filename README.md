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
* **svm_baseline:**. Prueba aislada con Máquinas de Soporte Vectorial (SVM). Reutiliza TODO lo desarrollado hasta ahora (vista vestibular, orientación, mapas 2.5D, caché, normalización) y sustituye solo el clasificador: en lugar de ajustar una red, se extraen características de cada caso y se entrena una SVM.
* **dataset.py:** Genera la imagen apilada de 3 canales (depth, curvature, render).
* **backbones.py:** Arquitectura y transfer learning: 4 backbones preentrenados en ImageNet.


## Diagrama general del entrenamiento - SMV
El siguiente diagrama representa el flujo real de código, desde MANIFEST.CSV hasta la comparación final de los cuatro modelos.
<img width="1656" height="3436" alt="diagrama_SVM" src="https://github.com/user-attachments/assets/4b38b18a-3181-4633-89ef-2fd311b997e7" />

