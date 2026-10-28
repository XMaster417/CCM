# Car Classification Model (CCM)

Descarga el dataset [Car Model Image Classification](https://www.kaggle.com/datasets/lightangle/car-model-image-classification)
en `data/raw/`, realiza el análisis y preprocesamiento, divide los datos y entrena
una CNN para clasificar las 10 clases de automóviles.

## Requisitos

- Python 3.12 de 64 bits con pip, recomendado para Windows y CPU.
- TensorFlow CPU 2.21.0, fijado en `requirements.txt`, y Keras mediante `tf.keras`.
  Esta versión dispone de paquetes para Python 3.10–3.13, no para Python 3.14.
  Consulta la [distribución oficial](https://pypi.org/project/tensorflow-cpu/2.21.0/).
- Microsoft Visual C++ Redistributable x64, requerido por TensorFlow en Windows.
- Git para clonar el repositorio, conexión a internet y espacio libre para el dataset completo y su extracción.
- Acceso al dataset en Kaggle. Si exige iniciar sesión o aceptar condiciones, hazlo primero en su página.

## Preparación en Windows (PowerShell)

1. Abre una terminal en la carpeta donde quieras guardar el proyecto y clona el repositorio:

   ```powershell
   git clone https://github.com/XMaster417/CCM.git
   cd CCM
   ```

2. Instala Python 3.12 de 64 bits y crea un entorno virtual con sus dependencias.
   Si ya tienes `env` con Python 3.14, utiliza un entorno `.venv` nuevo:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. Ejecuta el script:

   ```powershell
   .\.venv\Scripts\python.exe src/main.py
   ```

## GPU NVIDIA en Windows mediante WSL2

Se conserva el código de TensorFlow/Keras. Ambas CNN usan automáticamente GPU
si TensorFlow dispone de ella, o CPU en caso contrario, y muestran el dispositivo
en consola. `main.py` ejecuta actualmente `train_model_v2()`.

### Preparación (una sola vez)

1. En PowerShell instala Ubuntu, si todavía no está disponible:

   ```powershell
   wsl --install -d Ubuntu-24.04
   ```

   Si la descarga falla con `WININET_E_TIMEOUT`, descarga la imagen WSL
   directamente desde Ubuntu e instálala desde el archivo local en PowerShell:

   ```powershell
   curl.exe --fail --location --retry 2 --output "$env:TEMP\ccm-ubuntu-24.04.5-amd64.wsl" https://releases.ubuntu.com/24.04/ubuntu-24.04.5-wsl-amd64.wsl
   Get-FileHash -Algorithm SHA256 "$env:TEMP\ccm-ubuntu-24.04.5-amd64.wsl"
   ```

   Compara el hash con la entrada de esa imagen en los
   [SHA256 oficiales](https://releases.ubuntu.com/24.04/SHA256SUMS).
   Si coincide, instala con:

   ```powershell
   wsl --install --from-file "$env:TEMP\ccm-ubuntu-24.04.5-amd64.wsl" --name Ubuntu-24.04 --no-launch
   ```

   Reinicia Windows si el instalador lo solicita. Abre Ubuntu con
   `wsl -d Ubuntu-24.04` y completa la creación del usuario y contraseña Linux
   si aparece el asistente. Comprueba con `wsl --list --verbose` en PowerShell
   que Ubuntu utiliza la versión 2.

2. En la terminal de Ubuntu instala Python y verifica el acceso a NVIDIA:

   ```bash
   sudo apt-get update
   sudo apt-get install -y python3.12-venv python3-pip
   nvidia-smi
   ```

   WSL utiliza el controlador NVIDIA instalado en Windows. No instales un
   controlador de pantalla NVIDIA de Linux dentro de WSL.

3. Desde Ubuntu entra en la carpeta del proyecto y crea un entorno independiente:

   ```bash
   cd '/mnt/c/Users/yaelc/Documents/CLASES_TEC/7MO SEMESTRE/MODULOS_AVANZADOS/MOD7_DP/CCM'
   python3.12 -m venv ~/.venvs/ccm
   source ~/.venvs/ccm/bin/activate
   python -m pip install --upgrade pip
   python -m pip install -r requirements-gpu.txt
   python -m pip check
   export LD_LIBRARY_PATH="$(printf '%s:' "$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}"
   ```

   Ajusta la ruta si clonaste el proyecto en otra ubicación. El entorno se crea
   en el sistema de archivos de Linux (`/home/yaelc/.venvs/ccm`), fuera del
   repositorio y de su versionamiento. `.venv-wsl/` también está excluido de Git
   para cubrir intentos anteriores; las listas de dependencias sí se versionan. No reutilices
   `.venv/Scripts/python.exe`, pues pertenece a Windows. Tampoco instales
   `requirements.txt` en este entorno: contiene TensorFlow exclusivo de CPU.
   `requirements-common.txt` reúne las dependencias compartidas y
   `requirements-gpu.txt` instala TensorFlow 2.21.0 con sus bibliotecas CUDA.

4. Confirma que TensorFlow detecta una GPU:

   ```bash
   python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
   ```

   Debe aparecer al menos un `PhysicalDevice` de tipo `GPU`. Una lista vacía
   indica que la aceleración no está disponible. La detección por sí sola no
   garantiza que todos los kernels sean compatibles con la tarjeta; confirma
   también una operación de convolución antes de un entrenamiento largo.

### Activar el entorno y ejecutar main

En PowerShell, abre Ubuntu:

```powershell
wsl -d Ubuntu-24.04
```

Después, en Ubuntu:

```bash
cd '/mnt/c/Users/yaelc/Documents/CLASES_TEC/7MO SEMESTRE/MODULOS_AVANZADOS/MOD7_DP/CCM'
source ~/.venvs/ccm/bin/activate
export LD_LIBRARY_PATH="$(printf '%s:' "$VIRTUAL_ENV"/lib/python3.12/site-packages/nvidia/*/lib)${LD_LIBRARY_PATH:-}"
python src/main.py
```

Para ejecutar únicamente el entrenamiento con los datos ya preparados:

```bash
python src/models/cnn_v2.py
# O la primera versión:
python src/models/cnn_v1.py
```

Para salir del entorno: `deactivate`. Los reportes mantienen sus ubicaciones
actuales y ejecutar el entrenamiento puede sobrescribir los PNG existentes.
En VS Code abre el proyecto conectado a WSL y selecciona
`/home/yaelc/.venvs/ccm/bin/python` como intérprete.

### Repetibilidad del entrenamiento

Ambas CNN activan `tf.config.experimental.enable_op_determinism()` y fijan
la semilla con `tf.keras.utils.set_random_seed(seed)` (42 por defecto).
La CNN v2 reinicia la semilla antes de entrenar cada candidato. Esto permite
repetir los resultados con los mismos datos, hiperparámetros, dispositivo y
entorno de software. No garantiza resultados idénticos entre CPU y GPU ni
entre versiones distintas de las dependencias.

El determinismo puede reducir la velocidad de entrenamiento. Si una operación
no dispone de implementación determinista, TensorFlow puede lanzar
`UnimplementedError`. La configuración se aplica tanto desde `main.py` como
al ejecutar cada CNN directamente.

### Diagnóstico

- Si `venv` falla en `ensurepip` o muestra `Operation not permitted` dentro de
  `/mnt/c`, crea el entorno en Linux con `python3.12 -m venv ~/.venvs/ccm` y
  actívalo con `source ~/.venvs/ccm/bin/activate`. El proyecto puede permanecer
  en su carpeta Windows. La carpeta `.venv-wsl` incompleta no se utiliza.
- Si aparece `CPU`, revisa el intérprete activo con `which python`, ejecuta
  `nvidia-smi` en Ubuntu y repite la consulta de dispositivos TensorFlow.
- Si TensorFlow informa `Cannot dlopen some GPU libraries`, ejecuta el comando
  `export LD_LIBRARY_PATH=...` de la sección de activación en esa terminal.
  Añade al buscador de bibliotecas las rutas CUDA instaladas por pip en el entorno.

## Análisis de datos

`src/main.py` ejecuta el análisis después de preparar el dataset. Para analizar
únicamente los datos ya descargados, sin conexión a Kaggle:

```powershell
.\.venv\Scripts\python.exe src/scripts/data_analysis.py
```

Todas las salidas del análisis se guardan en `public/images/` con el prefijo
`analisis_`: gráficas y tablas como PNG y tablas completas como CSV. Una nueva ejecución
sobrescribe las salidas con el mismo nombre. Las imágenes originales no se modifican.

El análisis incluye conteos y balance por clase, dimensiones y proporciones,
peso y formatos, integridad y correspondencia con el CSV, duplicados exactos
de archivo y de píxeles RGB, candidatos visuales por dHash, brillo, contraste,
nitidez aproximada, muestras por clase y concentración de fuentes y creadores.
Los mosaicos por clase usan semilla 42; las métricas se calculan sobre todas las
imágenes que se pueden decodificar. Las incidencias detectadas se cuentan en consola.

Los candidatos dHash (64 bits, distancia de Hamming de hasta 5) requieren
revisión visual y no equivalen a duplicados confirmados. Se comparan todos los
pares, por lo que esa etapa tiene costo cuadrático. Los indicadores de calidad
se calculan en gris sobre miniaturas de hasta 256 píxeles por lado; dependen
también del fondo y las texturas y no justifican descartar imágenes automáticamente.

## Preprocesamiento a 320×320 y augmentación

El tamaño se define en `src/scripts/transform_data.py` con `IMAGE_SIZE = 320`.
Cada imagen se lee desde `data/raw/` y se ajusta directamente a 320×320 sin
desenfoque gaussiano, conservando corrección EXIF, RGB, proporciones,
interpolación LANCZOS y relleno negro centrado. Se mantiene JPEG con calidad 95
y sin submuestreo de color en `data/processed/`.

La augmentación conserva las dimensiones recibidas y añade variantes solo a
train. Las imágenes y los metadatos necesarios para entrenar se guardan en
`data/augmented/images/` y `data/augmented/sets/`. Los conteos se muestran
únicamente en consola; la augmentación no genera reportes en `public/`.

```text
data/raw/ → ajuste directo a 320×320, sin desenfoque → data/processed/
         → augmentación a 320×320 → data/augmented/ → CNN v2
```

Ambas CNN toman su entrada del tamaño del preprocesamiento: `(320, 320, 3)`.
La normalización sigue siendo `float32` entre 0 y 1. Con la arquitectura actual,
`Flatten` tiene 51,200 valores y el modelo completo tiene 6,652,458 parámetros.

Para regenerar desde los originales, conservando los splits existentes y sin
entrenar ni generar nuevos reportes en `public/`:

```powershell
.\env\Scripts\python.exe -c "import sys; sys.path.insert(0, 'src'); from scripts.transform_data import transform_dataset; from scripts.data_augmentation import augment_dataset; transform_dataset(); augment_dataset()"
```

Ajusta el intérprete a tu entorno (`env`, `.venv` o `python` dentro de WSL).
La regeneración sustituye las imágenes procesadas anteriores a partir de los
originales de `data/raw/`; no amplía las versiones ya reducidas a 224×224.
En una primera ejecución, `main.py` prepara los splits antes de la augmentación
y después inicia el entrenamiento con las rutas del dataset aumentado.

## División del dataset

Después de la transformación, `src/main.py` divide `data/processed` en train
(70%), validation (15%) y test (15%), estratificando por clase con semilla 42.
Utiliza dos llamadas a `train_test_split` de scikit-learn con `stratify`: primero
separa train del resto y después divide el resto en validation y test.
Para ejecutar únicamente la división:

```powershell
.\.venv\Scripts\python.exe src/scripts/split_data.py
```

Los metadatos se guardan en `data/sets/train/metadata.csv`,
`data/sets/val/metadata.csv` y `data/sets/test/metadata.csv`. La columna `path`
contiene rutas relativas a la carpeta de entrada; las imágenes permanecen allí.
Para cargar una imagen, combina esa carpeta con el valor de `path`.

Las gráficas por conjunto y la tabla visual se guardan en `public/images/split/`.
Los conteos por clase se exportan a `public/documents/split_conteos_clases.csv`.
Una nueva ejecución actualiza los CSV y PNG generados.

La entrada debe contener `metadata.csv` con el mismo esquema que el procesado
y las imágenes referenciadas. Configura las rutas y la semilla en las constantes
al inicio de `src/scripts/split_data.py`:

```python
INPUT_DIR = ROOT / "data" / "processed"
SETS_DIR = ROOT / "data" / "sets"
SEED = 42
```

Los tamaños se configuran con `TRAIN_PERCENT = 70`, `VAL_PERCENT = 15` y
`TEST_PERCENT = 15`; deben ser enteros positivos y sumar 100.
