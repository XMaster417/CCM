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

## Flujo del programa

Al ejecutar `python src/main.py`, el programa realiza estas etapas en orden:

1. **Descarga o reutilización:** obtiene el dataset de Kaggle en `data/raw/`
   o reutiliza una descarga local completa.
2. **Exclusiones visuales:** retira de los originales y sus metadatos las imágenes
   registradas en `src/scripts/excluded_images.txt`.
3. **Vista previa:** genera un mosaico de imágenes en `public/images/eda/preview.png`.
4. **Análisis exploratorio (EDA):** revisa distribución de clases, características
   de las imágenes, integridad, duplicados y similitud visual. Guarda gráficas
   en `public/images/eda/` y tablas CSV en `public/documents/`.
5. **Limpieza y preprocesamiento:** usa los resultados del análisis para excluir
   conflictos de etiquetas y copias exactas o similares; convierte las imágenes
   a RGB y 320×320, conservando proporciones, en `data/processed/`.
6. **División estratificada:** asigna 70% a train, 15% a validación y 15% a test,
   con semilla 42. Guarda sus metadatos en `data/sets/` sin mover las imágenes.
7. **Augmentación fija:** copia los conjuntos y añade una variante por original
   de train en `data/augmented/`; validación y test conservan solo originales.
8. **Entrenamiento y evaluación:** entrena EfficientNet-B0 en dos etapas y
   selecciona los pesos con mayor F1 macro de validación.
9. **Salidas:** guarda curvas, métricas visuales y matrices de confusión en
   `public/images/efficientnet_b0/`, y el modelo en `public/models/efficientnet.keras`.

## Limpieza visual del dataset

Se revisaron visualmente algunas imagenes originales de BMW M3, BMW M5, Chevrolet Camaro y
Ford Mustang, en sus conjuntos de entrenamiento y validación. A partir de esa revisión
se eliminaron **101 originales**: 20 fotografías de interiores, 25 imágenes de
LEGO (incluidas piezas) y 56 anuncios. La limpieza se limita a los casos
identificados en esas cuatro clases, otras miniaturas, renders, detalles y
etiquetas dudosas siguen pendientes de revisión.

| Conjunto | Originales antes | Originales después | Entradas después en `data/augmented` |
|---|---:|---:|---:|
| Train | 1,446 | 1,360 | 2,720 |
| Validación | 310 | 295 | 295 |
| Test | 310 | 310 | 310 |
| Total | 2,066 | 1,965 | 3,325 |

## Preprocesamiento a 320×320

El tamaño se define en `src/scripts/transform_data.py` con `IMAGE_SIZE = 320`.
Cada imagen se lee desde `data/raw/` y se ajusta directamente a 320×320 sin
desenfoque gaussiano, conservando corrección EXIF, RGB, proporciones,
interpolación LANCZOS y relleno negro centrado. Se mantiene JPEG con calidad 95
y sin submuestreo de color en `data/processed/`.

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

## CNN v1

`src/models/cnn_v1.py` implementa una CNN que aprende desde cero. Tiene cuatro
bloques de convolución 3×3 con 16, 32, 64 y 128 filtros, activación ReLU y
max pooling. Después aplica `Flatten → Dense(128, ReLU) → Dense(10, softmax)`
para obtener las probabilidades de las diez clases. Usa `data/processed/` y
los metadatos de `data/sets/`, sin augmentación.

Hiperparámetros y configuración predeterminados:

- **Entrada:** 320×320×3, RGB `float32` normalizado a [0, 1].
- **Max pooling:** ventana 4×4, stride 2 y padding `same` en los cuatro bloques.
- **Optimizador:** Adam con learning rate `0.001`.
- **Batch:** 32; **máximo de épocas:** 30; **semilla:** 42.
- **Regularización:** dropout 0 (sin capa de dropout) y L2 0 (desactivada).
- **Pérdida:** `SparseCategoricalCrossentropy`; métrica de entrenamiento: accuracy.
- **Selección y EarlyStopping:** mayor F1 macro de validación, paciencia de
  5 épocas sin mejora y restauración de los mejores pesos.

Para entrenarla con los datos preparados:

```bash
python src/models/cnn_v1.py
```

Exporta los reportes PNG en `public/images/cnn_v1/`. Registra también
loss y accuracy de test por época, sin usarlos para seleccionar los pesos.

## CNN v2

`src/models/cnn_v2.py` aprende desde cero con cuatro convoluciones 3×3 de
16, 32, 64 y 128 filtros ReLU, max pooling y el clasificador
`Flatten → Dense(128, ReLU) → Dropout → Dense(10, softmax)`.
Reutiliza el constructor de CNN v1 e incorpora augmentación fija, dropout y
regularización L2 sobre los kernels de las convoluciones y capas densas.
Usa las imágenes de `data/augmented/images/` y sus CSV en `data/augmented/sets/`.

Hiperparámetros y configuración predeterminados:

- **Entrada:** 320×320×3, RGB `float32` normalizado a [0, 1].
- **Max pooling:** ventana 3×3, stride 2 y padding `same` en los cuatro bloques.
- **Optimizador:** Adam con learning rate `0.001`.
- **Batch:** 32; **máximo de épocas:** 30 por candidato; **semilla:** 42.
- **Regularización:** dropout `0.5` después de la capa densa de 128 unidades
  y L2 `1e-4`, sin penalizar sesgos.
- **Pérdida:** `SparseCategoricalCrossentropy` más la penalización L2;
  métrica de entrenamiento: accuracy.
- **Selección y EarlyStopping:** mayor F1 macro de validación, paciencia de
  5 épocas sin mejora y restauración de los mejores pesos de cada candidato.

Admite búsqueda mediante `ParameterGrid`, pero las listas actuales contienen
un solo valor de learning rate, dropout, pooling y L2: se evalúa **una sola
combinación**. Si se amplían, gana la de mayor F1 macro de validación; ante
un empate se conserva la primera. Test se evalúa después de elegir al ganador,
con sus pesos de cada época, y no interviene en la selección.

Para entrenarla con los datos aumentados preparados:

```bash
python src/models/cnn_v2.py
```

Exporta los reportes PNG en `public/images/cnn_v2/`. Las gráficas de
comparación se generan solo para los hiperparámetros que tengan varios valores.

## EfficientNet-B0

`src/models/efficientnet_b0.py` aplica transfer learning con
`tf.keras.applications.EfficientNetB0`, pesos de ImageNet, `include_top=False`
y la base congelada en la primera etapa.
También llama la base con `training=False` para conservar las estadísticas de
BatchNormalization y desactivar la aleatoriedad interna de la base. Primero aprende
el clasificador: `GlobalAveragePooling2D → Dropout(0.5) → Dense(n_clases, softmax)`.
Las clases se infieren de train en orden alfabético; actualmente son diez y el
clasificador tiene 12,810 parámetros entrenables. Después ajusta los bloques
`block6*`, `block7a_*` y `top_conv`, manteniendo BatchNormalization congelada.
Conserva los mejores pesos entre ambas etapas según F1 macro de validación;
`fine_tune_epochs=0` omite el ajuste parcial.

Hiperparámetros y configuración predeterminados:

- **Entrada:** 320×320×3, RGB `float32` en [0, 255]; la normalización está
  incorporada en EfficientNet.
- **Optimizador:** Adam; **dropout del clasificador:** `0.5`.
- **Batch:** 32; **semilla:** 42.
- **Primera etapa:** base ImageNet congelada, learning rate `0.001` y máximo
  30 épocas para entrenar el clasificador.
- **Fine-tuning:** bloques 6 y 7 y `top_conv`, con BatchNormalization congelada,
  learning rate inicial `1e-5` y máximo 15 épocas adicionales.
- **Reducción del learning rate:** `ReduceLROnPlateau` sobre `val_loss`, modo
  `min`, factor `0.5`, paciencia 2 y mínimo `1e-7`, en cada etapa.
- **Pérdida:** `SparseCategoricalCrossentropy`; métrica de entrenamiento: accuracy.
- **Selección y EarlyStopping:** mayor F1 macro de validación, paciencia de
  5 épocas sin mejora y restauración de los mejores pesos de cada etapa.
  Al terminar se conservan los mejores pesos entre ambas etapas.

La entrada por defecto es `data/augmented/images`, con los CSV existentes en
`data/augmented/sets/{train,val,test}/metadata.csv`. Train contiene originales
y variantes fijas; val y test conservan solo originales. No se vuelven a dividir los datos al
ejecutar únicamente este modelo. Los CSV deben contener `class` y `path`, con
rutas relativas a la carpeta de imágenes, las mismas clases en los tres
conjuntos y ninguna ruta repetida dentro de un conjunto o entre conjuntos.

El cargador entrega imágenes de **320×320** por defecto y ajusta cualquier
otro tamaño con interpolación bilineal, antialias y relleno negro para conservar
proporciones. No modifica los archivos ni aplica augmentación. A diferencia de
las CNN anteriores, entrega RGB `float32` en **[0, 255]**, sin dividir por 255:
