"""Compara configuraciones por F1 macro de validación y exporta reportes PNG."""

from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from sklearn.model_selection import ParameterGrid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.cnn_v1 import (
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    ROOT,
    SEED,
    build_model,
    calculate_metrics,
    load_splits,
    make_dataset,
    print_final_metrics,
    save_final_reports,
    save_training_plots,
)
from scripts.data_analysis import plt, save_figure

INPUT_DIR = ROOT / "data" / "augmented" / "images"
SETS_DIR = ROOT / "data" / "augmented" / "sets"
OUTPUT_DIR = ROOT / "public" / "images" / "cnn_v2"
LEARNING_RATES = (0.001,)
DROPOUT_RATES = (0.5,)
POOL_SIZES = ((3, 3), )
L2_RATES = (1e-4)

def save_hyperparameter_comparisons(search_results, output_dir):
    """Guarda las cinco mejores combinaciones por valor de cada parámetro variable.

    Cada hiperparámetro variable actualiza su propia imagen. Las comparaciones
    de parámetros fijos se conservan sin cambios. Cada panel muestra un valor,
    ordenando sus candidatos válidos por F1 macro de validación descendente.
    Si hay menos de cinco candidatos válidos, muestra solo los disponibles.
    Las etiquetas incluyen toda la configuración y el número de parámetros.
    """
    comparisons = (
        ("pool_size", "pooling", "ventanas de max pooling"),
        ("dropout_rate", "dropout", "tasas de dropout"),
        ("learning_rate", "learning", "tasas de aprendizaje"),
        ("l2_rate", "l2", "coeficientes de regularización L2"),
    )
    for parameter, filename, title in comparisons:
        values = sorted({result[parameter] for result in search_results})
        if len(values) <= 1:
            continue
        fig, axes = plt.subplots(len(values), 1, figsize=(12, 4 * len(values)))
        fig.suptitle(f"Comparación de {title}: hasta 5 mejores por valor")
        for ax, value in zip(axes, values):
            candidates = sorted(
                (r for r in search_results
                 if r[parameter] == value and r['valid']
                 and r['best_val_f1_macro'] is not None
                 and np.isfinite(r['best_val_f1_macro'])),
                key=lambda r: r['best_val_f1_macro'], reverse=True,
            )[:5]
            positions = np.arange(len(candidates))
            labels = [
                f"{r['pool_size'][0]}×{r['pool_size'][1]}\n"
                f"lr={r['learning_rate']:g}\ndropout={r['dropout_rate']:g}\n"
                f"l2={r['l2_rate']:g}\n"
                f"{r['num_parameters']:,} parámetros"
                for r in candidates
            ]
            scores = [r['best_val_f1_macro'] for r in candidates]
            bars = ax.bar(positions, scores, color="#3679ac")
            ax.bar_label(bars, fmt="%.4f", padding=3)
            value_label = (f"{value[0]}×{value[1]}" if parameter == "pool_size"
                           else f"{value:g}")
            ax.set(ylim=(0, 1.12), ylabel="Mejor F1 macro de validación",
                   xlabel="Configuración", title=f"{parameter} = {value_label}")
            if not candidates:
                ax.text(0.5, 0.5, "Sin candidatos válidos", ha="center",
                        transform=ax.transAxes)
            ax.set_xticks(positions, labels, fontsize=8)
            ax.grid(axis="y", alpha=0.25)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        save_figure(fig, output_dir, f"{filename}_comparison", prefix="cnn_v2")

def fit_candidate(model, train_dataset, validation_dataset, validation_labels,
                  epochs=EPOCHS):
    """
    Entrena un candidato de hiperparámetros y guarda pesos por época.

    Calcula F1 macro sobre toda la validación y conserva los pesos antes de
    que EarlyStopping restaure la mejor época. No evalúa el conjunto de test.

    Args:
        model (tf.keras.Model): Modelo compilado con los hiperparámetros a evaluar.
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
                learning_rates=LEARNING_RATES, dropout_rates=DROPOUT_RATES,
                pool_sizes=POOL_SIZES, l2_rates=L2_RATES):
    """
    Compara todas las combinaciones de hiperparametros mediante
    ParameterGrid y selecciona el modelo con mayor F1 macro de validación.

    Test se evalúa después de elegir al ganador, usando los pesos que tuvo
    en cada época, sin volver a entrenar ni usar test para seleccionar.
    Se usa stride 2 y padding same en todas las ventanas, conservando
    el tamaño de Flatten y el número de parámetros.
    
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
        dropout_rates (iterable[float]): Dropouts a evaluar; por defecto solo 0.5.
        pool_sizes (iterable[tuple[int, int]]): Ventanas, ordenadas de menor
            a mayor antes de evaluarlas.
        l2_rates (iterable[float]): Coeficientes finitos y no negativos de L2

    Returns:
        dict: model (modelo con mayor F1 macro de validación), history
            (dict de Keras con accuracy, loss, val_accuracy, val_loss,
            val_f1_macro, test_accuracy y test_loss por época), metrics
            (métricas finales),
            confusion_matrices (matrices 10x10), class_names (orden de clases)
            y output_dir (Path), además de best_learning_rate (float),
            best_dropout_rate (float), best_pool_size (tuple[int, int]),
            best_l2_rate (float)
            search_results (hiperparámetros, F1, épocas y num_parameters
            por combinación). Las épocas se numeran
            desde 1. Solo se exportan los reportes PNG.
    """
    learning_rates = tuple(learning_rates)
    dropout_rates = tuple(dropout_rates)
    pool_sizes = tuple(sorted(tuple(size) for size in pool_sizes))
    l2_rates = tuple(float(rate) for rate in l2_rates)
    
    candidates = ParameterGrid({
        "learning_rate": learning_rates,
        "dropout_rate": dropout_rates,
        "pool_size": pool_sizes,
        "l2_rate": l2_rates,
    })
    device = "GPU" if tf.config.list_logical_devices("GPU") else "CPU"
    splits, class_names = load_splits(input_dir, sets_dir)
    search_results = []
    best_result, best_history, best_epoch_weights = None, None, None
    print(
        f"Búsqueda en {device} | learning rates: {learning_rates} | "
        f"dropouts: {dropout_rates} | pooling: {pool_sizes} | "
        f"L2: {l2_rates} | "
        f"combinaciones: {len(candidates)} | "
        f"épocas máximas: {epochs} | "
        f"batch: {batch_size} | semilla: {seed}",
        flush=True,
    )

    for index, params in enumerate(candidates, start=1):
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
        model = build_model(model_name="cnn_v2", **params)
        print(
            f"\nCandidato {index}/{len(candidates)} | "
            f"learning_rate={params['learning_rate']:g} | "
            f"dropout_rate={params['dropout_rate']:g} | "
            f"l2_rate={params['l2_rate']:g} | "
            f"pool_size={params['pool_size']} | parámetros={model.count_params():,}",
            flush=True,
        )
        history, epoch_weights = fit_candidate(
            model, train_dataset, validation_dataset, splits["val"]["labels"],
            epochs,
        )
        valid = bool(history["loss"]) and all(
            np.isfinite(history[metric]).all()
            for metric in ("loss", "val_loss", "val_f1_macro")
        )
        result = {
            **params,
            "best_val_f1_macro": None,
            "best_epoch": None,
            "epochs_run": len(history["loss"]),
            "valid": valid,
            "num_parameters": model.count_params(),
        }
        if valid:
            best_index = int(np.argmax(history["val_f1_macro"]))
            result["best_val_f1_macro"] = float(history["val_f1_macro"][best_index])
            result["best_epoch"] = best_index + 1
            print(
                f"Ventana {params['pool_size']} | "
                f"l2_rate={params['l2_rate']:g} | "
                f"mejor F1 macro val={result['best_val_f1_macro']:.4f} | "
                f"mejor época={result['best_epoch']} | "
                f"épocas ejecutadas={result['epochs_run']} | "
                f"parámetros={result['num_parameters']:,}", flush=True,
            )
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
        raise RuntimeError(
            "Ninguna combinación produjo un entrenamiento con valores finitos."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    save_hyperparameter_comparisons(search_results, output_dir)
    print(
        f"\nMejor combinación entre las evaluadas: "
        f"learning_rate={best_result['learning_rate']:g} | "
        f"dropout_rate={best_result['dropout_rate']:g} | "
        f"l2_rate={best_result['l2_rate']:g} | "
        f"pool_size={best_result['pool_size']} | "
        f"F1 macro de validación: {best_result['best_val_f1_macro']:.4f} | "
        f"mejor época: {best_result['best_epoch']} | "
        f"épocas ejecutadas: {best_result['epochs_run']}",
        flush=True,
    )
    tf.keras.backend.clear_session()
    model = build_model(
        dropout_rate=best_result["dropout_rate"], model_name="cnn_v2",
        learning_rate=best_result["learning_rate"],
        pool_size=best_result["pool_size"],
        l2_rate=best_result["l2_rate"],
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
        "best_dropout_rate": best_result["dropout_rate"],
        "best_pool_size": best_result["pool_size"],
        "best_l2_rate": best_result["l2_rate"],
        "search_results": search_results,
    }


if __name__ == "__main__":
    train_model_v2()
