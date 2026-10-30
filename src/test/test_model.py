"""Imprime predicciones manuales sobre images y guarda una gráfica del top 3."""

from pathlib import Path
import sys

import matplotlib

# Genera el PNG sin abrir ventanas, también al ejecutar desde una terminal.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import UnidentifiedImageError

# Hace accesibles los scripts de src al ejecutar este archivo directamente.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.transform_data import IMAGE_SIZE, preprocess_image


ROOT = Path(__file__).resolve().parents[2]
IMAGES_DIR = Path(__file__).resolve().parent / "images"
MODEL_PATH = ROOT / "public" / "models" / "efficientnet.keras"
CLASSES_PATH = ROOT / "data" / "augmented" / "sets" / "train" / "metadata.csv"
PREDICTIONS_PATH = ROOT / "public" / "images" / "predictions" / "predictions_top3.png"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _class_key(name):
    """Unifica nombres de archivo y de clase para comparar sus etiquetas.

    Args:
        name (str): Nombre sin extensión, como nissan_gtr o Nissan_GT_R.

    Returns:
        str: Nombre sin guiones bajos y con mayúsculas y minúsculas unificadas.
    """
    return name.replace("_", "").casefold()


def _save_prediction_figure(results, images, class_names):
    """Guarda cada muestra junto a sus tres probabilidades en una sola figura.

    Args:
        results (list[dict]): Resultados no vacíos, en el mismo orden que images.
            Cada registro incluye path (Path), expected (str o None), correct
            (bool o None) y top_three (lista de pares clase-probabilidad).
        images (list[np.ndarray]): Imágenes RGB float32 de forma (320, 320, 3),
            con valores en [0, 255], alineadas con results.
        class_names (list[str]): Clases del modelo, utilizadas para asignar
            un color consistente a cada clase entre las distintas muestras.
    """
    # Distribuye las muestras en dos columnas y adapta las filas a su cantidad.
    columns = min(2, len(results))
    rows = (len(results) + columns - 1) // columns
    colors = {name: plt.get_cmap("tab10")(index % 10)
              for index, name in enumerate(class_names)}
    fig = plt.figure(figsize=(20, 4 * rows), layout="constrained")
    try:
        fig.suptitle(
            "Predicciones de EfficientNet-B0: tres clases más probables por imagen\n"
            "Clase esperada según el nombre del archivo",
            fontsize=17,
        )
        panels = fig.subfigures(rows, columns, squeeze=False)
        for panel, result, pixels in zip(panels.flat, results, images):
            expected = result["expected"] or "desconocida"
            correct = result["correct"]
            status = "SIN ETIQUETA" if correct is None else ("ACIERTO" if correct else "ERROR")
            panel.suptitle(
                f"{result['path'].name}\nEsperada: {expected} | {status}", fontsize=11,
            )
            image_axis, probability_axis = panel.subplots(
                1, 2, gridspec_kw={"width_ratios": (1, 1.35)},
            )
            image_axis.imshow(pixels.astype(np.uint8))
            image_axis.axis("off")
            names, probabilities = zip(*result["top_three"])
            labels = [f"{rank}. {name.replace('_', ' ')}"
                      for rank, name in enumerate(names, start=1)]
            # Las salidas softmax están en [0, 1]; las barras muestran porcentajes.
            bars = probability_axis.barh(
                labels, np.asarray(probabilities) * 100,
                color=[colors[name] for name in names],
            )
            probability_axis.invert_yaxis()
            # Deja espacio a la derecha para las etiquetas de las barras.
            probability_axis.set(
                xlim=(0, 118), xlabel="Probabilidad (%)", xticks=[0, 25, 50, 75, 100],
            )
            probability_axis.set_axisbelow(True)
            probability_axis.grid(axis="x", alpha=0.25)
            probability_axis.bar_label(
                bars, labels=[f"{value:.2%}" for value in probabilities],
                padding=3, fontsize=9,
            )
        # Si hay un número impar de muestras, oculta el último panel vacío.
        for panel in list(panels.flat)[len(results):]:
            panel.set_visible(False)
        PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREDICTIONS_PATH, dpi=150, facecolor="white")
    finally:
        # Libera la figura incluso si ocurre un error al guardar el archivo.
        plt.close(fig)
    print(f"Gráfica guardada en: {PREDICTIONS_PATH}", flush=True)


def run_predictions(model=None, class_names=None):
    """Imprime imagen, clase esperada, top 3 y aciertos, y guarda una gráfica.

    Sin argumentos carga el modelo guardado y las clases alfabéticas de train.
    Los nombres de archivo representan las clases esperadas de estas muestras.
    Las imágenes sin etiqueta reconocible se predicen pero no cuentan al medir
    aciertos. Guarda todas las muestras y sus probabilidades en predictions_top3.png.
    No devuelve resultados; se puede ejecutar solo o después del entrenamiento.

    Args:
        model (tf.keras.Model | None): Modelo entrenado que recibe imágenes RGB
            de 320x320 en [0, 255] y devuelve probabilidades por clase. Si es None,
            carga MODEL_PATH sin compilarlo para inferencia.
        class_names (list[str] | None): Nombres en el orden de las columnas de
            salida del modelo.
    """
    # Reutiliza el modelo recibido desde main o carga los pesos ya guardados.
    if model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    if class_names is None:
        if not CLASSES_PATH.is_file():
            raise FileNotFoundError(f"No se encontró el CSV de clases: {CLASSES_PATH}")
        frame = pd.read_csv(CLASSES_PATH, dtype=str, keep_default_na=False)
        if "class" not in frame or frame.empty or frame["class"].str.strip().eq("").any():
            raise ValueError(f"El CSV requiere una columna class con clases válidas: {CLASSES_PATH}")
        # Este orden debe coincidir con los índices usados al entrenar.
        class_names = sorted(frame["class"].unique())
    else:
        class_names = list(class_names)

    expected_classes = {_class_key(name): name for name in class_names}

    paths, images = [], []
    # Mantiene el mismo orden de archivos, imágenes y predicciones en la gráfica.
    for path in sorted(IMAGES_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            with preprocess_image(path) as image:
                # EfficientNet incorpora la normalización; no se divide por 255.
                pixels = np.asarray(image, dtype=np.float32)
        except (OSError, UnidentifiedImageError, ValueError) as error:
            print(f"Imagen omitida: {path.name}: {error}", flush=True)
            continue
        paths.append(path)
        images.append(pixels)
    if not images:
        raise ValueError(f"No hay imágenes válidas para predecir en: {IMAGES_DIR}")

    # Forma un lote (N, 320, 320, 3); cada fila de salida corresponde a una imagen.
    probabilities = model.predict(np.stack(images), verbose=0)
    print(f"\nPredicciones sobre {len(paths)} imágenes de {IMAGES_DIR}", flush=True)
    results = []
    for path, scores in zip(paths, probabilities):
        # Selecciona tres índices distintos de mayor a menor probabilidad.
        # En empates, el orden estable conserva el orden de las clases del modelo.
        top_indices = np.argsort(-scores, kind="stable")[:3]
        top_three = [(class_names[index], float(scores[index])) for index in top_indices]
        expected = expected_classes.get(_class_key(path.stem))
        predicted = top_three[0][0]
        correct = predicted == expected if expected is not None else None
        status = "SIN ETIQUETA" if correct is None else ("ACIERTO" if correct else "ERROR")
        print(f"\n{path.name} | Esperada: {expected or 'desconocida'} | {status}")
        for rank, (name, probability) in enumerate(top_three, start=1):
            print(f"  {rank}. {name}: {probability:.2%}")
        results.append({
            "path": path, "expected": expected, "predicted": predicted,
            "top_three": top_three, "correct": correct,
        })

    # Excluye del porcentaje los archivos sin una etiqueta esperada reconocible.
    labeled = sum(result["expected"] is not None for result in results)
    hits = sum(result["correct"] is True for result in results)
    if labeled:
        print(f"\nAciertos: {hits}/{labeled} ({hits / labeled:.2%}) "
              f"sobre imágenes con clase esperada; evaluadas: {len(results)}.", flush=True)
    else:
        print(f"\nEvaluadas: {len(results)} imágenes; ninguna tiene clase esperada reconocible.",
              flush=True)
    _save_prediction_figure(results, images, class_names)


if __name__ == "__main__":
    try:
        run_predictions()
    except (OSError, ValueError) as error:
        raise SystemExit(f"No se pudieron realizar las predicciones: {error}") from error
