"""Ejecuta la descarga del dataset y guarda su vista previa."""

from load_data import download_data, preview_images


def main():
    try:
        dataset_images = download_data()
        preview_images(dataset_images)
    except Exception as error:
        raise SystemExit(
            f"No se pudo completar el proceso: {error}\n"
            "Revisa la conexión, el espacio disponible y el acceso al dataset en Kaggle. "
            "Si se requiere autenticación, consulta el README."
        ) from error
    

if __name__ == "__main__":
    main()
