# Car Classification Model (CCM)

Descarga el dataset [Car Model Image Classification](https://www.kaggle.com/datasets/lightangle/car-model-image-classification)
en `data/raw/` y guarda una vista previa de hasta seis imágenes para comprobar la descarga.

## Requisitos

- Python 3.10 o posterior con pip
- Git para clonar el repositorio, conexión a internet y espacio libre para el dataset completo y su extracción.
- Acceso al dataset en Kaggle. Si exige iniciar sesión o aceptar condiciones, hazlo primero en su página.

## Preparación en Windows (PowerShell)

1. Abre una terminal en la carpeta donde quieras guardar el proyecto y clona el repositorio:

   ```powershell
   git clone https://github.com/XMaster417/CCM.git
   cd CCM
   ```

2. Crea un entorno virtual e instala las dependencias:

   ```powershell
   python -m venv env
   .\env\Scripts\python.exe -m pip install --upgrade pip
   .\env\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Ejecuta el script:

   ```powershell
   .\env\Scripts\python.exe src/main.py
   ```
