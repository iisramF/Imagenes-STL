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

**Diagrama general del entrenamiento**  
El siguiente diagrama representa el flujo real de tu código, desde MANIFEST.CSV hasta la comparación final de los cuatro modelos.

<img width="619" height="1158" alt="image" src="https://github.com/user-attachments/assets/87a62b52-eee7-4d0d-91f1-43c42de4659e" />
