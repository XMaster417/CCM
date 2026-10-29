"""Descarga, analiza, transforma, divide, aumenta y entrena la CNN de automóviles."""

from models.cnn_v1 import train_model_v1
from models.cnn_v2 import train_model_v2
from scripts.data_augmentation import augment_dataset
from scripts.data_analysis import analyze_dataset
from scripts.load_data import download_data, preview_images
from scripts.split_data import split_dataset
from scripts.transform_data import transform_dataset


def main():
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
# Aumentar solo train y conservar los conjuntos de validación y test
# ------------------------------------------------------------------------------
    augmented = augment_dataset()
# ------------------------------------------------------------------------------
# Entrenamiento y evaluación de la CNN
# ------------------------------------------------------------------------------
    train_model_v2(
        input_dir=augmented["input_dir"],
        sets_dir=augmented["sets_dir"],
    )

if __name__ == "__main__":
    main()
