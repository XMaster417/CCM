"""Descarga el dataset de autos y guarda una muestra de imágenes locales."""

import csv
import json
import os
import random
from pathlib import Path, PurePosixPath
from tempfile import NamedTemporaryFile

import kagglehub
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PREVIEW_PATH = ROOT / "public" / "images" / "eda" / "preview.png"
MARKER = ROOT / "data" / ".car_dataset_complete.json"
DATASET = "lightangle/car-model-image-classification"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
EXCLUSIONS_FILE = Path(__file__).with_name("excluded_images.txt")


def remove_excluded_images(raw_dir=RAW_DIR, exclusions_file=EXCLUSIONS_FILE):
    """Retira de raw las imágenes revisadas y sus filas de metadata.csv.

    Valida toda la lista antes de modificar archivos. Las rutas son relativas
    a raw (clase/archivo); los archivos ya ausentes se omiten para permitir
    repetir la limpieza. Devuelve los conteos de archivos y filas eliminados.
    """
    raw_dir = Path(raw_dir).resolve()
    excluded = set()
    targets = []
    for line in Path(exclusions_file).read_text(encoding="utf-8-sig").splitlines():
        relative = line.strip()
        if not relative or relative.startswith("#"):
            continue
        parts = PurePosixPath(relative).parts
        if (len(parts) != 2 or any(part in {".", ".."} for part in parts)
                or "\\" in relative or ":" in relative
                or PurePosixPath(relative).is_absolute()
                or PurePosixPath(relative).as_posix() != relative):
            raise ValueError(f"Ruta de exclusión inválida: {relative}")
        target = raw_dir.joinpath(*parts)
        excluded.add(relative)
        targets.append(target)

    metadata_path = raw_dir / "metadata.csv"
    with metadata_path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        if not columns or not {"class", "filename"}.issubset(columns):
            raise ValueError("metadata.csv requiere las columnas class y filename")
        rows = list(reader)
    kept = [row for row in rows
            if f"{row['class']}/{row['filename']}" not in excluded]
    removed_rows = len(rows) - len(kept)

    temporary = None
    removed_files = 0
    try:
        if removed_rows:
            with NamedTemporaryFile(mode="w", encoding="utf-8-sig", newline="",
                                    dir=raw_dir, prefix=".metadata_", suffix=".tmp",
                                    delete=False) as destination:
                temporary = Path(destination.name)
                writer = csv.DictWriter(destination, fieldnames=columns)
                writer.writeheader()
                writer.writerows(kept)
        for target in targets:
            if target.is_file():
                target.unlink()
                removed_files += 1
        if temporary is not None:
            os.replace(temporary, metadata_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"Limpieza de raw: {removed_files} imágenes y "
          f"{removed_rows} filas de metadatos eliminadas.", flush=True)
    return {"removed_files": removed_files, "removed_rows": removed_rows}


def find_images():
    """
    Encuentra las rutas de una imagen, recorre la carpeta de data/raw y sus 
    subcarpetas, con rglob("*") menciona que es valido cualquier patron. 
    Verifica si es un archivo y si tiene una extensión valida, para retornar una 
    lista de rutas ordenada.
    
    Returns: 
        images (list): lista de rutas a las imagenes en la carpeta data/raw.
    """
    images = []

    for path in RAW_DIR.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(path)

    return sorted(images)

def download_data():
    """
    Reutiliza o descarga el dataset y aplica las exclusiones visuales de raw.

    Crea la carpeta destino, si es que no existe, verifica si existe no existe
    el contenido a descargar y lo descarga, si existe no hace nada.

    Returns:
        images (list): lista de rutas a las imagenes en la carpeta data/raw.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Verificar que exista el archivo .json
    complete = False
    if MARKER.exists():
        try:
            complete = json.loads(MARKER.read_text(encoding="utf-8")) \
                            .get("dataset") == DATASET

        except (OSError, ValueError):
            pass
    
    images = find_images()
    metadata = list(RAW_DIR.rglob("metadata.csv"))

    # Verifica que no haya una descarga, imagenes o metadata ya descargados
    if not (complete and images and metadata):
        print(f"Descargando {DATASET} en {RAW_DIR}", flush=True)
        kagglehub.dataset_download(DATASET, output_dir=str(RAW_DIR))

        # Verifica nuevamente la descarga de imagenes o metadata
        images = find_images()
        metadata = list(RAW_DIR.rglob("metadata.csv"))
        if not images or not metadata:
            raise RuntimeError("La descarga no contiene metadata.csv e imágenes")
        MARKER.write_text(json.dumps({"dataset": DATASET}), encoding="utf-8")
    else:
        print("Reutilizando la descarga local completada.")
    remove_excluded_images()
    images = find_images()
    print(f"Datos: {RAW_DIR}\nImágenes encontradas: {len(images)}")
    return images


def preview_images(images):
    """
    Guarda hasta seis imágenes que se puedan abrir en public/images/eda/preview.png.

    Args: 
        images (list): lista de rutas a las imagenes en la carpeta data/raw.
    """
    candidates = list(images)
    random.Random(42).shuffle(candidates)
    samples = []

    for path in candidates:
        try:
            with Image.open(path) as source:
                sample = source.convert("RGB")
                sample.thumbnail((640, 480))
            samples.append((path, sample))
        except (OSError, UnidentifiedImageError) as error:
            print(f"No se pudo abrir {path.name}: {error}")
        if len(samples) == 6:
            break

    # Crea la cuadricula con imagenes aleatorias.
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for axis in axes.flat:
        axis.axis("off")
    for axis, (path, sample) in zip(axes.flat, samples):
        axis.imshow(sample)
        axis.set_title(path.name, fontsize=9, wrap=True)

    fig.suptitle("Muestra del dataset descargado")
    fig.tight_layout()
    PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(PREVIEW_PATH, dpi=150)
    print(f"Vista previa guardada en: {PREVIEW_PATH}", flush=True)
    plt.close(fig)
