# Car Classification Model (CCM)

Descarga el dataset [Car Model Image Classification](https://www.kaggle.com/datasets/lightangle/car-model-image-classification)
en `data/raw/`, realiza el análisis y preprocesamiento, divide los datos y entrena
una línea base de EfficientNet-B0 con pesos ImageNet congelados para clasificar
las 10 clases de automóviles.

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

Se conserva el código de TensorFlow/Keras. Los modelos usan automáticamente GPU
si TensorFlow dispone de ella, o CPU en caso contrario, y muestran el dispositivo
en consola. `main.py` ejecuta actualmente `train_efficientnet_b0()` sobre
`data/augmented/images`, con augmentación fija generada antes de entrenar.

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
# O la nueva línea base preentrenada (también ejecutada por main.py):
python src/models/efficientnet_b0.py
```

Para salir del entorno: `deactivate`. Los reportes mantienen sus ubicaciones
actuales y ejecutar el entrenamiento puede sobrescribir los PNG existentes.
En VS Code abre el proyecto conectado a WSL y selecciona
`/home/yaelc/.venvs/ccm/bin/python` como intérprete.

### Repetibilidad del entrenamiento

Ambas CNN fijan la semilla con `tf.keras.utils.set_random_seed(seed)`
(42 por defecto). La CNN v2 reinicia la semilla antes de entrenar cada
candidato para controlar la inicialización y la mezcla de datos.

### Regularización L2 en CNN v2

La CNN v2 compara `L2_RATES = (0.0, 1e-5, 1e-4, 1e-3)` y elige la
combinación con mayor F1 macro de validación. `0` es la referencia sin L2 y
puede ganar; ante un empate se conserva el primer candidato evaluado. Test
se evalúa después de seleccionar al ganador y no participa en la selección.
Con learning rate `0.001`, dropout `0.5` y pooling `3×3`, la configuración
actual ejecuta cuatro entrenamientos en lugar de uno; el tiempo depende
de las épocas que alcance cada candidato antes de la detención temprana.

El artículo disponible en `docs/Data Augmentation-Driven Improvements in
Malignant Lymphoma Image Classification.pdf`, sección 2.2 (página 6), usa L2
de `0.01` con VGG16 preentrenada y learning rate de `1e-6`
([publicación](https://www.mdpi.com/2073-431X/14/7/252)). Ese valor no demuestra
ser óptimo para automóviles. El rango inicial de `1e-5` a `1e-3` es una
adaptación conservadora para esta CNN entrenada desde cero, con aproximadamente
6.65 millones de parámetros, unas 1,446 imágenes originales de entrenamiento,
augmentación y dropout de `0.5`. Su beneficio debe verificarse en validación.

L2 se aplica a los kernels de las cuatro convoluciones y ambas capas densas,
sin penalizar sesgos ni cambiar el número de parámetros. Según
[Keras](https://keras.io/api/layers/regularizers/), añade
`l2_rate * sum(square(kernel))` por cada kernel regularizado a la entropía
cruzada. Por tanto, `loss`, `val_loss` y `test_loss` incluyen esa penalización;
la selección entre coeficientes sigue usando F1 macro. CNN v1 mantiene L2
desactivada por defecto.

Para ajustar la búsqueda, cambia `L2_RATES` en `src/models/cnn_v2.py` o pasa
los valores explícitamente:

```python
from models.cnn_v2 import train_model_v2

result = train_model_v2(l2_rates=(0.0, 1e-5, 1e-4, 1e-3))
print(result["best_l2_rate"])
# Para evaluar un único valor: train_model_v2(l2_rates=(1e-4,))
```

La colección no puede estar vacía y los coeficientes deben ser finitos y no
negativos. `search_results` incluye `l2_rate` para cada candidato y el modelo
devuelto conserva el coeficiente ganador. Cuando se comparan varios valores,
se genera `public/images/cnn_v2/cnn_v2_l2_comparison.png`; todas las etiquetas
de comparación incluyen L2. Las comparaciones de parámetros fijos conservan
sus imágenes existentes.

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
         → augmentación fija a 320×320 → data/augmented/ → EfficientNet-B0
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
`main.py` prepara los splits, genera la augmentación fija y entrena EfficientNet-B0
con `data/augmented`. Las CNN anteriores pueden ejecutarse por separado.

## Línea base de EfficientNet-B0

`src/models/efficientnet_b0.py` utiliza `tf.keras.applications.EfficientNetB0`
con pesos de ImageNet, `include_top=False` y la base congelada en la primera etapa.
También llama la base con `training=False` para conservar las estadísticas de
BatchNormalization y desactivar la aleatoriedad interna de la base. Primero aprende
el clasificador: `GlobalAveragePooling2D → Dropout(0.5) → Dense(n_clases, softmax)`.
Las clases se infieren de train en orden alfabético; actualmente son diez y el
clasificador tiene 12,810 parámetros entrenables. Después ajusta `block7a_*` y
`top_conv` durante un máximo de 10 épocas con learning rate `1e-5`, manteniendo
BatchNormalization congelada. Conserva los mejores pesos entre ambas etapas
según F1 macro de validación; `fine_tune_epochs=0` omite el ajuste parcial.

La entrada por defecto es `data/augmented/images`, con los CSV existentes en
`data/augmented/sets/{train,val,test}/metadata.csv`. Train contiene originales
y variantes fijas; val y test conservan solo originales. No se vuelven a dividir los datos al
ejecutar únicamente este modelo. Los CSV deben contener `class` y `path`, con
rutas relativas a la carpeta de imágenes, las mismas clases en los tres
conjuntos y ninguna ruta repetida dentro de un conjunto o entre conjuntos.

Aunque las imágenes procesadas son de 320×320, el cargador las ajusta a
**224×224** con interpolación bilineal, antialias y relleno negro para conservar
proporciones. No modifica los archivos ni aplica augmentación. A diferencia de
las CNN anteriores, entrega RGB `float32` en **[0, 255]**, sin dividir por 255:
EfficientNet ya incorpora su normalización. Tampoco se usa
`load_normalized_image` ni se añade otra capa de reescalado.
Consulta la [documentación de EfficientNet de Keras](https://keras.io/api/applications/efficientnet/)
y la [guía de transferencia de aprendizaje](https://keras.io/guides/transfer_learning/).

La configuración inicial es Adam con learning rate `0.001`, batch `32`, máximo
`30` épocas y semilla `42`. Usa `SparseCategoricalCrossentropy` y accuracy;
calcula F1 macro de toda la validación al terminar cada época. EarlyStopping
restaura los pesos de mayor F1 macro de validación después de cinco épocas sin
mejora; si hay empate conserva la primera. Al terminar cada época también se
evalúa test para registrar `test_loss` y `test_accuracy` y mostrarlos junto con
train y val en las curvas. Estas métricas quedan en el historial devuelto por
la función; la selección de pesos sigue usando únicamente F1 macro de validación.
Los reportes finales evalúan los pesos seleccionados. Una pérdida o predicción de validación no finita
interrumpe la ejecución sin exportar un modelo de esa ejecución.

### Ejecutar y cambiar el conjunto

Para entrenar solo EfficientNet sobre los datos aumentados ya preparados:

```powershell
.\env\Scripts\python.exe src/models/efficientnet_b0.py
```

Dentro de WSL, con el entorno activado: `python src/models/efficientnet_b0.py`.
La primera ejecución necesita conexión para descargar los pesos oficiales de
ImageNet si todavía no están en la caché de Keras. No hay fallback a pesos
aleatorios. `python src/main.py` ejecuta antes toda la preparación de datos.

Para otro conjunto, cambia las constantes al inicio del archivo o pasa las
rutas a la función desde Python, teniendo `src` en la ruta de importación:

```python
from models.efficientnet_b0 import train_efficientnet_b0

result = train_efficientnet_b0(
    input_dir="data/mi_dataset/images",
    sets_dir="data/mi_dataset/sets",
    output_dir="public/images/efficientnet_b0_mi_dataset",
    artifacts_dir="public/models",
    epochs=30,
    batch_size=32,
    learning_rate=0.001,
    dropout_rate=0.5,
    patience=5,
    seed=42,
    image_size=224,
)
```

El modelo se guarda directamente en `public/models/efficientnet.keras`, sin
crear una subcarpeta del modelo ni generar un archivo JSON. Una nueva ejecución
en la misma carpeta sustituye ese archivo. El historial, las clases, la
configuración y las métricas siguen disponibles en el diccionario `result`.

El conjunto aumentado existente se usa por defecto. Para generarlo a partir de
los datos procesados y splits existentes, ejecuta `python src/scripts/data_augmentation.py`.
EfficientNet reutiliza esas imágenes sin aplicar augmentación dinámica.
Cambiar el dataset o sus splits no ejecuta ninguna
búsqueda automática de hiperparámetros.

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
