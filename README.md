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
al inicio de `src/scripts/split_data.py`. Por ejemplo, para usar `data/augmented`:

```python
INPUT_DIR = ROOT / "data" / "augmented"
SETS_DIR = ROOT / "data" / "sets"
SEED = 42
```

Los tamaños se configuran con `TRAIN_PERCENT = 70`, `VAL_PERCENT = 15` y
`TEST_PERCENT = 15`; deben ser enteros positivos y sumar 100.
