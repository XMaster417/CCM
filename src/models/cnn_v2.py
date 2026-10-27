"""Entrena la CNN con dropout reutilizando el entrenamiento de cnn_v1."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn_v1 import (
    BATCH_SIZE,
    EPOCHS,
    INPUT_DIR,
    ROOT,
    SEED,
    SETS_DIR,
    train_model_v1 as train_base_model,
)

OUTPUT_DIR = ROOT / "public" / "images" / "cnn_v2"


def train_model_v2(input_dir=INPUT_DIR, sets_dir=SETS_DIR, output_dir=OUTPUT_DIR,
                epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED):
    """
    Reutiliza los datos, entrenamiento y reportes con Dropout(0.5).
    
    Args:
        input_dir (str | Path): Carpeta de las imágenes procesadas.
        sets_dir (str | Path): Carpeta con los CSV de train, val y test.
        output_dir (str | Path): Carpeta para los resultados (imagenes).
        epochs (int): Máximo de número de épocas. EarlyStopping puede terminar 
            antes si val_loss no mejora durante PATIENCE épocas.
        batch_size (int): Máximo de imágenes por lote, por defecto 32.
        seed (int): Semilla de inicialización y mezcla, por defecto 42.
        dropout_rate (float): Tasa de dropout después , con 0 no lo agrega.
        model_name (str): Nombre del modelo y prefijo de sus reportes PNG.

    Returns:
        dict: model (modelo con mejores pesos de validación), history
            (dict de Keras con accuracy, loss, val_accuracy, val_loss,
            test_accuracy y test_loss por época), metrics (métricas finales),
            confusion_matrices (matrices 10x10), class_names (orden de clases)
            y output_dir (Path). Solo se exportan los PNG, no el diccionario.
    """
    return train_base_model(
        input_dir=input_dir, 
        sets_dir=sets_dir, 
        output_dir=output_dir,
        epochs=epochs, 
        batch_size=batch_size, 
        seed=seed,
        dropout_rate=0.5, 
        model_name="cnn_v2",
    )


if __name__ == "__main__":
    train_model()
