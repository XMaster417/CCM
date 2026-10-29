"""Guarda variantes de train sin volver a dividir ni modificar los originales.

Adapta las transformaciones geométricas simples del paper de Šegota et al.
(2025) a fotografías de automóviles. No utiliza parches ni giros de 90 grados.
"""

import hashlib
import math
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed"
SETS_DIR = ROOT / "data" / "sets"
OUTPUT_DIR = ROOT / "data" / "augmented"
SPLITS = ("train", "val", "test")
SEED = 42
VARIANTS_PER_IMAGE = 1
FLIP_PROBABILITY = 0.5
MAX_ROTATION_DEGREES = 5.0
MAX_SHIFT_FRACTION = 0.03
MAX_ZOOM_OUT = 0.10
GAMMA_RANGE = (0.9, 1.1)


def read_splits(sets_dir):
    """
    Lee los metadatos de los conjuntos previamente separados.

    Args:
        sets_dir (str | Path): Carpeta con train/metadata.csv, val/metadata.csv
            y test/metadata.csv.

    Returns:
        dict[str, pd.DataFrame]: Metadatos de train, val y test, conservando
            todas las columnas como texto y las celdas vacías como "".
    """
    return {
        name: pd.read_csv(Path(sets_dir) / name / "metadata.csv",
                          dtype=str, keep_default_na=False)
        for name in SPLITS
    }


def augment_image(image, seed):
    """
    Genera una variante mediante reflejo horizontal, giro, traslación, zoom y gamma.

    Combina las transformaciones geométricas en una sola interpolación bilineal
    y conserva el relleno negro. No modifica la imagen recibida.

    Args:
        image (Image.Image): Imagen RGB procesada; por defecto de 320 × 320 píxeles.
        seed (int): Semilla para reproducir las transformaciones.

    Returns:
        Image.Image: Variante RGB con las mismas dimensiones que la entrada.
    """
    rng = np.random.default_rng(seed)
    if rng.random() < FLIP_PROBABILITY:
        image = ImageOps.mirror(image)
    angle = math.radians(rng.uniform(-MAX_ROTATION_DEGREES, MAX_ROTATION_DEGREES))
    tx, ty = rng.uniform(-MAX_SHIFT_FRACTION, MAX_SHIFT_FRACTION, size=2) * image.size
    inverse_scale = 1 + rng.uniform(0, MAX_ZOOM_OUT)
    gamma = float(rng.uniform(*GAMMA_RANGE))

    # Combinar giro, traslación y escala evita interpolar tres veces la imagen.
    a, b = math.cos(angle) * inverse_scale, math.sin(angle) * inverse_scale
    d, e = -b, a
    cx, cy = image.width / 2, image.height / 2
    matrix = (a, b, cx - a * (cx + tx) - b * (cy + ty),
              d, e, cy - d * (cx + tx) - e * (cy + ty))
    image = image.transform(
        image.size, Image.Transform.AFFINE, matrix,
        resample=Image.Resampling.BILINEAR, fillcolor=(0, 0, 0),
    )
    # Usar la misma curva gamma en RGB conserva el negro del relleno.
    pixels = np.asarray(image, dtype=np.float32) / np.float32(255)
    pixels = np.clip(np.rint(np.power(pixels, gamma) * 255), 0, 255)
    return Image.fromarray(pixels.astype(np.uint8))


def augment_dataset(input_dir=INPUT_DIR, sets_dir=SETS_DIR, output_dir=OUTPUT_DIR,
                    seed=SEED, variants_per_image=VARIANTS_PER_IMAGE):
    """
    Copia los conjuntos originales y agrega variantes únicamente a train.

    Conserva las columnas de origen y guarda únicamente las imágenes y los
    metadatos necesarios para entrenar. Cada ejecución sobrescribe sus archivos;
    solo los CSV vigentes definen el dataset activo. Los conteos se muestran
    en consola y se devuelven en memoria, sin generar reportes en public/.

    Args:
        input_dir (str | Path): Carpeta de imágenes RGB procesadas, por defecto 320 × 320.
        sets_dir (str | Path): Carpeta de los metadatos de train, val y test.
        output_dir (str | Path): Carpeta donde se guardan images/ y sets/.
        seed (int): Semilla base para generar variantes reproducibles.
        variants_per_image (int): Número positivo de variantes por imagen de train.

    Returns:
        dict: input_dir (Path de las imágenes generadas), sets_dir (Path de los
            metadatos), counts (pd.DataFrame con los conteos por clase) y totals
            (dict con el número de imágenes por conjunto).
    """
    input_dir, sets_dir, output_dir = (
        Path(path).resolve() for path in (input_dir, sets_dir, output_dir)
    )

    frames = read_splits(sets_dir)
    target_images, target_sets = output_dir / "images", output_dir / "sets"
    output_frames = {}

    for name, frame in frames.items():
        rows = []
        for index, original in enumerate(frame.to_dict("records"), start=1):
            relative = Path(original["path"])
            source = input_dir / relative
            destination = (target_images / relative).resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            rows.append(original)

            if name == "train":
                with Image.open(source) as image:
                    for variant in range(1, variants_per_image + 1):
                        # La semilla depende de la ruta, no del orden de las filas.
                        identity = f"{seed}:{relative.as_posix()}:{variant}".encode("utf-8")
                        image_seed = int.from_bytes(hashlib.sha256(identity).digest()[:8], "big")
                        augmented = augment_image(image, image_seed)
                        variant_path = Path("variants") / relative.parent / (
                            f"{relative.name}__aug_{variant:02d}.png"
                        )

                        target = (target_images / variant_path).resolve()
                        target.parent.mkdir(parents=True, exist_ok=True)
                        augmented.save(target, format="PNG")
                        rows.append({**original, "path": variant_path.as_posix(),
                                     "filename": variant_path.name})

            if index % 250 == 0 or index == len(frame):
                print(f"Augmentación {name}: {index}/{len(frame)} originales", flush=True)
        output_frames[name] = pd.DataFrame(rows, columns=frame.columns)

    # Publicar los tres metadatos cuando todas las imágenes están disponibles.
    for name, frame in output_frames.items():
        destination = target_sets / name
        destination.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination / "metadata.csv", index=False, encoding="utf-8-sig")

    counts = pd.DataFrame({
        "original_train": frames["train"].groupby("class").size(),
        "augmented_train": frames["train"].groupby("class").size() * variants_per_image,
        **{name: frame.groupby("class").size() for name, frame in output_frames.items()},
    }).rename_axis("class").reset_index()
    counts["total"] = counts[list(SPLITS)].sum(axis=1)
    
    totals = {name: len(frame) for name, frame in output_frames.items()}
    total_images = sum(totals.values())
    print(f"\nData augmentation\nEntrada: {input_dir}\nSalida: {target_images}")
    print(counts.to_string(index=False))
    for name, count in totals.items():
        print(f"{name}: {count} imágenes ({count / total_images:.2%})")
    print(f"Variantes añadidas: {int(counts['augmented_train'].sum())}")
    print(f"Total: {total_images} imágenes", flush=True)
    return {"input_dir": target_images, "sets_dir": target_sets,
            "counts": counts, "totals": totals}

if __name__ == "__main__":
    augment_dataset()
