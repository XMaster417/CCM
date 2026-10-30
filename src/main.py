"""Ejecuta el flujo completo, entrena EfficientNet e imprime y grafica predicciones."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "test"))
from test_model import run_predictions


def main():
    from models.efficientnet_b0 import train_efficientnet_b0
    from scripts.data_analysis import analyze_dataset
    from scripts.data_augmentation import augment_dataset
    from scripts.load_data import download_data, preview_images
    from scripts.split_data import split_dataset
    from scripts.transform_data import transform_dataset

# ------------------------------------------------------------------------------
# Cargar el dataset
# ------------------------------------------------------------------------------
    try:
        dataset_images = download_data()
        preview_images(dataset_images)
    except Exception as error:
        raise SystemExit(
            f"No se pudo completar el proceso: {error}\n"
            "Revisa la conexión, el espacio disponible y el acceso al dataset en Kaggle. "
            "Si se requiere autenticación, consulta el README."
        ) from error
# ------------------------------------------------------------------------------
# Analisis exploratorio de datos (EDA)
# ------------------------------------------------------------------------------
    analyze_dataset()
# ------------------------------------------------------------------------------
# Transformación y preprocesamiento
# ------------------------------------------------------------------------------
    transform_dataset()
# ------------------------------------------------------------------------------
# División estratificada
# ------------------------------------------------------------------------------
    split_dataset()
# ------------------------------------------------------------------------------
# Augmentación fija en data/augmented
# ------------------------------------------------------------------------------
    augment_dataset()
# ------------------------------------------------------------------------------
# Entrenamiento y evaluación de EfficientNet-B0 sobre data/augmented
# ------------------------------------------------------------------------------
    training = train_efficientnet_b0()
# ------------------------------------------------------------------------------
# Predicciones manuales sobre src/test/images con el modelo recién entrenado
# ------------------------------------------------------------------------------
    run_predictions(training["model"], training["class_names"])


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"No se pudo completar el proceso: {error}") from error
