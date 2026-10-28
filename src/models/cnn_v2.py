"""Busca la tasa de aprendizaje de la CNN con dropout y reporta al ganador."""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn_v1 import (
    BATCH_SIZE,
    EPOCHS,
    INPUT_DIR,
    PATIENCE,
    ROOT,
    SEED,
    SETS_DIR,
    build_model,
    calculate_metrics,
    load_splits,
    make_dataset,
    print_final_metrics,
    save_final_reports,
    save_training_plots,
)

OUTPUT_DIR = ROOT / "public" / "images" / "cnn_v2"
LEARNING_RATES = (0.0001, 0.0003, 0.001, 0.003, 0.01)

def fit_candidate(model, train_dataset, validation_dataset, validation_labels,
                  epochs=EPOCHS):
    """
    Entrena el modelo con una tasa de aprendizaje y guarda los pesos por época.

    Calcula F1 macro sobre toda la validación y conserva los pesos antes de
    que EarlyStopping restaure la mejor época. No evalúa el conjunto de test.

    Args:
        model (tf.keras.Model): Modelo compilado con la tasa que se evaluará.
        train_dataset (tf.data.Dataset): Lotes de entrenamiento mezclados.
        validation_dataset (tf.data.Dataset): Lotes de validación sin mezclar,
            con el mismo orden que validation_labels.
        validation_labels (np.ndarray): Etiquetas reales de validación, de 0 a 9.
        epochs (int): Máximo de épocas. EarlyStopping puede terminar antes si
            val_f1_macro no aumenta durante PATIENCE épocas.

    Returns:
        tuple[dict, list[list[np.ndarray]]]: Historial de Keras con loss,
            accuracy, val_loss, val_accuracy y val_f1_macro, y lista de pesos
            de cada época con pérdidas finitas, en orden de entrenamiento.
    """
    epoch_weights = []

    def evaluate_epoch_metrics(epoch, logs):
        """
        Registra F1 macro de validación y guarda los pesos antes de EarlyStopping.

        Detiene el entrenamiento si la pérdida de train o val no es finita.

        Args:
            epoch (int): Índice de época proporcionado por Keras.
            logs (dict): Métricas de la época que History guardará después.
        """
        if not np.isfinite([logs["loss"], logs["val_loss"]]).all():
            logs["val_f1_macro"] = float("nan")
            model.stop_training = True
            return
        probabilities = model.predict(validation_dataset, verbose=0)
        logs["val_f1_macro"] = calculate_metrics(
            validation_labels, probabilities.argmax(axis=1),
        )["f1"]
        epoch_weights.append(model.get_weights())

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_macro", mode="max", patience=PATIENCE, min_delta=0,
        restore_best_weights=True, verbose=1,
    )
    history = model.fit(
        train_dataset, validation_data=validation_dataset, epochs=epochs,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_epoch_metrics),
            early_stopping,
        ],
        shuffle=False, verbose=1,
    )
    return history.history, epoch_weights


def train_model_v2(input_dir=INPUT_DIR, sets_dir=SETS_DIR, output_dir=OUTPUT_DIR,
                epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, *,
                learning_rates=LEARNING_RATES):
    """
    Compara las tasas de aprendizaje con Dropout(0.5) y selecciona el modelo
    con mayor F1 macro de validación.

    Test se evalúa después de elegir al ganador, usando los pesos que tuvo
    en cada época, sin volver a entrenar ni usar test para seleccionar.
    
    Args:
        input_dir (str | Path): Carpeta de las imágenes procesadas.
        sets_dir (str | Path): Carpeta con los CSV de train, val y test.
        output_dir (str | Path): Carpeta para los resultados (imagenes).
        epochs (int): Máximo de número de épocas. EarlyStopping puede terminar 
            antes si val_f1_macro no aumenta durante PATIENCE épocas.
        batch_size (int): Máximo de imágenes por lote, por defecto 32.
        seed (int): Semilla de inicialización y mezcla, por defecto 42.
        learning_rates (iterable[float]): Tasas positivas y finitas de Adam.
            Se evalúan en orden; si hay empate se conserva la primera.

    Returns:
        dict: model (modelo con mayor F1 macro de validación), history
            (dict de Keras con accuracy, loss, val_accuracy, val_loss,
            val_f1_macro, test_accuracy y test_loss por época), metrics
            (métricas finales),
            confusion_matrices (matrices 10x10), class_names (orden de clases)
            y output_dir (Path), además de best_learning_rate (float) y
            search_results (lista de resultados por tasa). Las épocas se
            numeran desde 1. Solo se exportan los PNG, no el diccionario.
    """
    learning_rates = tuple(learning_rates)
    tf.config.set_visible_devices([], "GPU")
    splits, class_names = load_splits(input_dir, sets_dir)
    search_results = []
    best_result, best_history, best_epoch_weights = None, None, None
    print(
        f"Búsqueda en CPU | tasas: {learning_rates} | épocas máximas: {epochs} | "
        f"batch: {batch_size} | semilla: {seed}",
        flush=True,
    )

    for index, learning_rate in enumerate(learning_rates, start=1):
        # Reiniciar tanto la inicialización como la mezcla para cada candidato.
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        train_dataset = make_dataset(
            splits["train"]["paths"], splits["train"]["labels"], batch_size,
            shuffle=True, seed=seed,
        )
        validation_dataset = make_dataset(
            splits["val"]["paths"], splits["val"]["labels"], batch_size,
        )
        model = build_model(
            dropout_rate=0.5, model_name="cnn_v2", learning_rate=learning_rate,
        )
        print(
            f"\nCandidato {index}/{len(learning_rates)} | learning_rate={learning_rate:g}",
            flush=True,
        )
        history, epoch_weights = fit_candidate(
            model, train_dataset, validation_dataset, splits["val"]["labels"],
            epochs,
        )
        valid = all(
            np.isfinite(history[metric]).all()
            for metric in ("loss", "val_loss", "val_f1_macro")
        )
        result = {
            "learning_rate": learning_rate,
            "best_val_f1_macro": None,
            "best_epoch": None,
            "epochs_run": len(history["loss"]),
            "valid": valid,
        }
        if valid:
            best_index = int(np.argmax(history["val_f1_macro"]))
            result["best_val_f1_macro"] = float(history["val_f1_macro"][best_index])
            result["best_epoch"] = best_index + 1
            if (
                best_result is None
                or result["best_val_f1_macro"] > best_result["best_val_f1_macro"]
            ):
                best_result = result
                best_history, best_epoch_weights = history, epoch_weights
        else:
            print("Candidato descartado: pérdida no finita.", flush=True)
        search_results.append(result)
        # No conservar los modelos ni las copias de pesos de los perdedores.
        del model, history, epoch_weights, train_dataset, validation_dataset

    if best_result is None:
        raise RuntimeError("Ninguna tasa produjo un entrenamiento con valores finitos.")

    print(
        f"\nMejor tasa de aprendizaje entre las evaluadas: "
        f"{best_result['learning_rate']:g} | "
        f"F1 macro de validación: {best_result['best_val_f1_macro']:.4f} | "
        f"mejor época: {best_result['best_epoch']} | "
        f"épocas ejecutadas: {best_result['epochs_run']}",
        flush=True,
    )
    tf.keras.backend.clear_session()
    model = build_model(
        dropout_rate=0.5, model_name="cnn_v2",
        learning_rate=best_result["learning_rate"],
    )
    datasets = {
        name: make_dataset(data["paths"], data["labels"], batch_size)
        for name, data in splits.items()
    }
    best_history["test_loss"], best_history["test_accuracy"] = [], []
    try:
        for weights in best_epoch_weights:
            model.set_weights(weights)
            scores = model.evaluate(datasets["test"], verbose=0, return_dict=True)
            best_history["test_loss"].append(float(scores["loss"]))
            best_history["test_accuracy"].append(float(scores["accuracy"]))
    finally:
        # El reporte final corresponde a la mejor época de validación.
        model.set_weights(best_epoch_weights[best_result["best_epoch"] - 1])
    del best_epoch_weights

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_training_plots(best_history, output_dir, report_prefix="cnn_v2")
    metrics, matrices = save_final_reports(
        model, datasets, splits, class_names, output_dir, report_prefix="cnn_v2",
    )
    print_final_metrics(metrics, output_dir)
    return {
        "model": model, "history": best_history, "metrics": metrics,
        "confusion_matrices": matrices, "class_names": class_names,
        "output_dir": output_dir,
        "best_learning_rate": best_result["learning_rate"],
        "search_results": search_results,
    }


if __name__ == "__main__":
    train_model_v2()
