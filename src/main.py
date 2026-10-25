"""Descarga, analiza y transforma el dataset de imágenes de automóviles."""

from scripts.data_analysis import analyze_dataset
from scripts.load_data import download_data, preview_images
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

if __name__ == "__main__":
    main()
