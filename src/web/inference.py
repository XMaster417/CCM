"""Carga los artefactos existentes y predice imágenes sin guardarlas en disco."""

from functools import lru_cache
from io import BytesIO
from pathlib import Path
from threading import Lock
import warnings

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from src.scripts.transform_data import IMAGE_SIZE, preprocess_image


ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = ROOT / "public" / "models"
# Los nombres deben coincidir con los archivos exportados de cada entrenamiento.
MODELS = {
    "efficientnet": {
        "label": "EfficientNet-B0", "filename": "efficientnet.keras",
        "classes_path": ROOT / "data" / "augmented" / "sets" / "train" / "metadata.csv",
        "normalize": False,
    },
    "cnn_v1": {
        "label": "CNN v1", "filename": "cnn_v1.keras",
        "classes_path": ROOT / "data" / "sets" / "train" / "metadata.csv",
        "normalize": True,
    },
    "cnn_v2": {
        "label": "CNN v2", "filename": "cnn_v2.keras",
        "classes_path": ROOT / "data" / "augmented" / "sets" / "train" / "metadata.csv",
        "normalize": True,
    },
}
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}


class InvalidImageError(ValueError):
    """El archivo recibido no es una imagen admitida y decodificable."""


class Predictor:
    """Conserva el modelo y las clases en el orden utilizado al entrenar."""

    def __init__(self, model_id):
        self.config = MODELS[model_id]
        model_path = MODELS_DIR / self.config["filename"]
        classes_path = self.config["classes_path"]
        if not model_path.is_file():
            raise FileNotFoundError(
                f"El modelo {self.config['label']} todavía no está disponible. "
                f"Coloca el archivo {self.config['filename']} en public/models/."
            )
        if not classes_path.is_file():
            raise FileNotFoundError(f"No se encontró el CSV de clases: {classes_path}")
        frame = pd.read_csv(classes_path, dtype=str, keep_default_na=False)
        if "class" not in frame or frame.empty or frame["class"].str.strip().eq("").any():
            raise ValueError("El CSV de entrenamiento debe contener clases válidas en class.")
        self.class_names = sorted(frame["class"].unique())
        self.model = tf.keras.models.load_model(model_path, compile=False)
        if self.model.output_shape != (None, len(self.class_names)):
            raise ValueError("Las salidas del modelo no coinciden con las clases de entrenamiento.")
        if self.model.input_shape != (None, IMAGE_SIZE, IMAGE_SIZE, 3):
            raise ValueError(f"El modelo debe aceptar imágenes RGB de {IMAGE_SIZE} × {IMAGE_SIZE}.")
        self._lock = Lock()

    def predict(self, content):
        """Devuelve las tres clases más probables para los bytes de una imagen.

        Args:
            content (bytes): Contenido JPEG, PNG, WebP o BMP recibido por la API.

        Returns:
            list[dict]: Clases y probabilidades originales, ordenadas de mayor a menor.

        Raises:
            InvalidImageError: El contenido no se puede decodificar como imagen admitida.
            RuntimeError: El modelo devuelve probabilidades inválidas.
        """
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with BytesIO(content) as stream:
                    # Verifica el contenido real antes de aplicar el preprocesamiento existente.
                    with Image.open(stream) as source:
                        if source.format not in ALLOWED_FORMATS:
                            raise InvalidImageError("Usa una imagen JPEG, PNG, WebP o BMP.")
                        source.verify()
                    stream.seek(0)
                    with preprocess_image(stream) as image:
                        pixels = np.asarray(image, dtype=np.float32)
        except (OSError, UnidentifiedImageError, ValueError, SyntaxError,
                Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
            raise InvalidImageError(
                "No se pudo leer la imagen. Usa un archivo JPEG, PNG, WebP o BMP válido."
            ) from error

        # Las CNN se entrenan con [0, 1]; EfficientNet incorpora su normalización.
        if self.config["normalize"]:
            pixels /= np.float32(255.0)
        # Serializa la inferencia si llegan solicitudes desde varias pestañas.
        with self._lock:
            scores = np.asarray(self.model(pixels[None, ...], training=False))[0]
        if (scores.shape != (len(self.class_names),) or not np.isfinite(scores).all()
                or np.any(scores < 0) or np.any(scores > 1)):
            raise RuntimeError("El modelo devolvió probabilidades inválidas.")
        indices = np.argsort(-scores, kind="stable")[:3]
        return [
            {"class_name": str(self.class_names[index]), "probability": float(scores[index])}
            for index in indices
        ]


@lru_cache(maxsize=len(MODELS))
def _load_predictor(model_id):
    return Predictor(model_id)


_load_lock = Lock()


def get_predictor(model_id="efficientnet"):
    """Carga una vez cada modelo seleccionado, incluso con solicitudes simultáneas."""
    if model_id not in MODELS:
        raise ValueError("Selecciona un modelo válido.")
    with _load_lock:
        return _load_predictor(model_id)
