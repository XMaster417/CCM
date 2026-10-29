"""EfficientNet-B0 sobre augmentación fija en disco y fine tuning parcial.

Primero aprende la cabeza GAP -> Dropout -> Dense y opcionalmente ajusta la base.
Las imágenes procesadas se cargan en RGB, a 224x224 y en [0, 255]: EfficientNet
incorpora su propia normalización. La evaluación no aplica augmentación.
"""

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_analysis import save_figure


ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "augmented" / "images"
SETS_DIR = ROOT / "data" / "augmented" / "sets"
OUTPUT_DIR = ROOT / "public" / "images" / "efficientnet_b0"
ARTIFACTS_DIR = ROOT / "public" / "models"
IMAGE_SIZE = 224
EPOCHS = 30
BATCH_SIZE = 32
LEARNING_RATE = 0.001
FINE_TUNE_EPOCHS = 15
FINE_TUNE_LEARNING_RATE = 0.00001
DROPOUT_RATE = 0.5
PATIENCE = 5
SEED = 42
SPLITS = ("train", "val", "test")
COLORS = {"train": "#3679ac", "val": "#d98735", "test": "#47956a"}


def load_splits(input_dir=INPUT_DIR, sets_dir=SETS_DIR):
    """Lee CSV con class/path relativos a input_dir y valida los tres splits.

    Las clases se numeran por orden alfabético de train.

    Args:
        input_dir (str | Path): Carpeta raíz de las imágenes.
        sets_dir (str | Path): Carpeta con train/metadata.csv,
            val/metadata.csv y test/metadata.csv, con columnas class y path.

    Returns:
        tuple[dict, list[str]]: Splits con paths absolutos y labels int32,
            y nombres de clases en el orden de las salidas del modelo.
    """
    input_dir, sets_dir = Path(input_dir).resolve(), Path(sets_dir).resolve()
    frames = {}
    for name in SPLITS:
        metadata_path = sets_dir / name / "metadata.csv"
        frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        if frame.empty or frame[["class", "path"]].apply(
            lambda column: column.str.strip().eq("")
        ).any().any():
            raise ValueError(f"{metadata_path} está vacío o tiene class/path vacíos.")
        frames[name] = frame

    class_names = sorted(frames["train"]["class"].unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    seen_paths, splits = {}, {}
    for name, frame in frames.items():
        paths = []
        for relative in frame["path"]:
            path = (input_dir / relative).resolve()
            if path in seen_paths:
                raise ValueError(
                    f"Imagen repetida en {seen_paths[path]} y {name}: {relative}"
                )
            seen_paths[path] = name
            paths.append(str(path))
        splits[name] = {
            "paths": paths,
            "labels": frame["class"].map(class_to_index).to_numpy(dtype=np.int32),
        }
    return splits, class_names


def make_dataset(paths, labels, batch_size=BATCH_SIZE, shuffle=False, seed=SEED,
                 image_size=IMAGE_SIZE):
    """Carga RGB float32 en [0, 255], sin dividir por 255 ni augmentar.

    Ajusta con padding negro para conservar proporciones; no altera los archivos.
    Las variantes de train ya están guardadas en data/augmented/images.
    Solo train debe mezclarse para mantener alineadas predicciones y etiquetas
    durante la evaluación de los conjuntos.

    Args:
        paths (list[str]): Rutas de las imágenes.
        labels (np.ndarray): Índices de clase alineados con paths.
        batch_size (int): Máximo de imágenes por lote.
        shuffle (bool): Mezclar las parejas imagen-etiqueta en cada época.
        seed (int): Semilla para la mezcla.
        image_size (int): Alto y ancho de salida en píxeles.

    Returns:
        tf.data.Dataset: Lotes (imágenes, etiquetas), con imágenes float32
            de forma (N, image_size, image_size, 3) y etiquetas de forma (N,).
    """
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            len(paths), 
            seed=seed, 
            reshuffle_each_iteration=True
        )

    def load(path, label):
        """Decodifica una imagen y ajusta su tamaño conservando proporciones.

        Args:
            path (tf.Tensor): Ruta escalar de tipo string.
            label (tf.Tensor): Índice de clase de la imagen.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Imagen RGB float32 en [0, 255]
                de forma (image_size, image_size, 3) y su etiqueta original.
        """
        pixels = tf.io.decode_image(
            tf.io.read_file(path), channels=3, expand_animations=False,
        )
        pixels.set_shape((None, None, 3))
        pixels = tf.image.resize_with_pad(
            tf.cast(pixels, tf.float32), image_size, image_size, antialias=True,
        )
        pixels = tf.clip_by_value(pixels, 0.0, 255.0)
        pixels.set_shape((image_size, image_size, 3))
        return pixels, label

    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(1)


def _compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def build_model(num_classes, learning_rate=LEARNING_RATE,
                dropout_rate=DROPOUT_RATE, image_size=IMAGE_SIZE):
    """Compila ImageNet congelada sin transformaciones de augmentación.

    La primera ejecución descarga los pesos oficiales si no están en caché.
    El modelo guardado espera lotes RGB de image_size cuadrados, en [0, 255].

    Args:
        num_classes (int): Número de clases y de probabilidades de salida.
        learning_rate (float): Tasa de aprendizaje de Adam.
        dropout_rate (float): Fracción de unidades descartadas en la cabeza.
        image_size (int): Alto y ancho de entrada en píxeles, al menos 32.

    Returns:
        tf.keras.Model: Modelo compilado con base congelada, clasificador
            entrenable y salida softmax de forma (N, num_classes).
    """
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet",
        input_shape=(image_size, image_size, 3),
    )
    backbone.trainable = False
    inputs = tf.keras.Input(shape=(image_size, image_size, 3), name="rgb_pixels")
    # Modo inferencia también para BatchNormalization y stochastic depth.
    features = backbone(inputs, training=False)
    features = tf.keras.layers.GlobalAveragePooling2D(name="global_pool")(features)
    features = tf.keras.layers.Dropout(dropout_rate, name="head_dropout")(features)
    outputs = tf.keras.layers.Dense(
        num_classes, activation="softmax", name="car_probabilities",
    )(features)
    model = tf.keras.Model(inputs, outputs, name="efficientnet")
    _compile_model(model, learning_rate)
    return model


def _configure_fine_tuning(model, learning_rate):
    """Ajusta juntos los bloques 6 y 7 y top_conv, con BatchNormalization fija."""
    backbone = model.get_layer("efficientnetb0")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = (
            layer.name.startswith(("block6", "block7a_")) or layer.name == "top_conv"
        ) and not isinstance(layer, tf.keras.layers.BatchNormalization)
    _compile_model(model, learning_rate)


def calculate_metrics(expected, predicted, num_classes):
    """Calcula accuracy y precision/recall/F1 macro para todas las clases.

    Args:
        expected (np.ndarray): Índices de las clases reales.
        predicted (np.ndarray): Índices predichos, alineados con expected.
        num_classes (int): Número total de clases, incluidas las no predichas.

    Returns:
        dict[str, float]: accuracy, precision, recall y f1. Las métricas macro
            asignan cero a las divisiones indefinidas.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=np.arange(num_classes), average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def save_reports(model, datasets, splits, class_names, history, output_dir, *,
                 fine_tune_start_epoch=None):
    """Exporta seis PNG y devuelve métricas y matrices con los mejores pesos.

    Las curvas muestran train/val/test por época. Los reportes finales usan
    los pesos seleccionados por F1 macro de validación.

    Args:
        model (tf.keras.Model): Modelo con los mejores pesos de validación.
        datasets (dict[str, tf.data.Dataset]): Train, val y test sin mezclar,
            en el mismo orden que las etiquetas de splits.
        splits (dict): Rutas y etiquetas devueltas por load_splits.
        class_names (list[str]): Nombres en el orden de las salidas del modelo.
        history (dict[str, list[float]]): Historial de entrenamiento con loss,
            accuracy, val_loss, val_accuracy, test_loss y test_accuracy por época.
        output_dir (str | Path): Carpeta de destino de los seis PNG.

    Returns:
        tuple[dict, dict]: Métricas finales y matrices de confusión por conjunto.
            Cada matriz tiene forma (n_clases, n_clases).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(history["loss"]) + 1)
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, prefix in (("train", ""), ("val", "val_"), ("test", "test_")):
            ax.plot(epochs, history[f"{prefix}{metric}"], label=name,
                    color=COLORS[name], marker="o", markersize=3)
        if fine_tune_start_epoch is not None:
            ax.axvline(fine_tune_start_epoch - 0.5, color="gray", linestyle="--",
                       label="Inicio de fine tuning")
        ax.set(xlabel="Época", ylabel=title, title=f"{title} por conjunto")
        ax.set_xticks(epochs[::max(1, len(epochs) // 10)])
        if metric == "accuracy":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, output_dir, metric, prefix="efficientnet_b0")

    metrics, matrices = {}, {}
    labels = np.arange(len(class_names))
    for name in SPLITS:
        probabilities = model.predict(datasets[name], verbose=0)
        predicted = probabilities.argmax(axis=1)
        expected = splits[name]["labels"]
        metrics[name] = calculate_metrics(expected, predicted, len(class_names))
        matrix = confusion_matrix(expected, predicted, labels=labels)
        matrices[name] = matrix
        side = max(8, len(class_names) * 0.9)
        fig, ax = plt.subplots(figsize=(side + 1, side))
        ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
            ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False,
        )
        ax.set(title=f"Matriz de confusión: {name} ({len(expected)} imágenes)",
               xlabel="Clase predicha", ylabel="Clase real")
        save_figure(fig, output_dir, f"confusion_{name}", prefix="efficientnet_b0")

    metric_names = ("accuracy", "precision", "recall", "f1")
    positions, width = np.arange(len(metric_names)), 0.25
    fig, ax = plt.subplots(figsize=(11, 6))
    for index, name in enumerate(SPLITS):
        bars = ax.bar(
            positions + (index - 1) * width,
            [metrics[name][metric] for metric in metric_names], width,
            label=name, color=COLORS[name],
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
    ax.set_xticks(positions, ["Accuracy", "Precision (macro)", "Recall (macro)",
                             "F1-score (macro)"])
    ax.set(ylim=(0, 1.15), ylabel="Valor", title="Métricas finales por conjunto")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "metrics", prefix="efficientnet_b0")
    return metrics, matrices


def train_efficientnet_b0(input_dir=INPUT_DIR, sets_dir=SETS_DIR,
                         output_dir=OUTPUT_DIR, artifacts_dir=ARTIFACTS_DIR,
                         epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, *,
                         learning_rate=LEARNING_RATE, dropout_rate=DROPOUT_RATE,
                         patience=PATIENCE, image_size=IMAGE_SIZE,
                         fine_tune_epochs=FINE_TUNE_EPOCHS,
                         fine_tune_learning_rate=FINE_TUNE_LEARNING_RATE):
    """Entrena la cabeza y hace fine tuning; selecciona por F1 macro val.

    Usa los originales y variantes fijas de data/augmented, sin aplicar nuevas
    transformaciones durante fit. No genera splits ni busca hiperparámetros.
    Guarda PNG en output_dir y efficientnet.keras en artifacts_dir, sin JSON
    ni subcarpeta adicional; otra ejecución sustituye estos resultados.

    Args:
        input_dir (str | Path): Carpeta raíz de las imágenes.
        sets_dir (str | Path): Carpeta con los CSV de train, val y test,
            cuyas rutas de imágenes son relativas a input_dir.
        output_dir (str | Path): Carpeta de destino de los reportes PNG.
        artifacts_dir (str | Path): Carpeta para efficientnet.keras;
            por defecto public/models.
        epochs (int): Máximo de épocas de entrenamiento.
        batch_size (int): Máximo de imágenes por lote.
        seed (int): Semilla para inicialización y mezcla de train.
        learning_rate (float): Tasa de aprendizaje de Adam.
        dropout_rate (float): Fracción de dropout del clasificador.
        patience (int): Épocas sin mejora de F1 macro val antes de detenerse.
        image_size (int): Alto y ancho de entrada en píxeles, al menos 32.
        fine_tune_epochs (int): Máximo de épocas de ajuste parcial; 0 lo omite.
        fine_tune_learning_rate (float): Tasa de aprendizaje del ajuste parcial.

    Returns:
        dict: model con los mejores pesos, history con métricas de train,
            val y test por época, metrics finales,
            confusion_matrices por conjunto, class_names, config, best_epoch
            numerada desde 1, output_dir, artifacts_dir y model_path.
            Las carpetas y rutas devueltas son objetos Path.
    """
    tf.keras.utils.set_random_seed(seed)
    splits, class_names = load_splits(input_dir, sets_dir)
    model = build_model(
        len(class_names), learning_rate, dropout_rate, image_size,
    )
    datasets = {
        name: make_dataset(data["paths"], data["labels"], batch_size,
                           image_size=image_size)
        for name, data in splits.items()
    }
    train_dataset = make_dataset(
        splits["train"]["paths"], splits["train"]["labels"], batch_size,
        shuffle=True, seed=seed, image_size=image_size,
    )
    config = {
        "model": model.name, "weights": "imagenet", "backbone_trainable": False,
        "image_size": image_size, "input_range": [0, 255],
        "resize": "bilinear_with_pad_antialias", "optimizer": "Adam",
        "learning_rate": learning_rate, "dropout_rate": dropout_rate,
        "fine_tune_epochs": fine_tune_epochs,
        "fine_tune_learning_rate": fine_tune_learning_rate,
        "epochs": epochs, "batch_size": batch_size, "seed": seed,
        "augmentation": {
            "mode": "fixed_on_disk", "dynamic": False,
        },
        "patience": patience, "monitor": "val_f1_macro",
        "input_dir": str(Path(input_dir).resolve()),
        "sets_dir": str(Path(sets_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "artifacts_dir": str(Path(artifacts_dir).resolve()),
        "num_parameters": model.count_params(),
        "trainable_parameters": int(sum(np.prod(w.shape) for w in model.trainable_weights)),
        "stages": [],
    }
    device = "GPU" if tf.config.list_logical_devices("GPU") else "CPU"
    print(
        f"EfficientNet-B0 congelada en {device} | clases: {len(class_names)} | "
        f"entrada: {image_size}x{image_size} | épocas: {epochs} | batch: {batch_size} | "
        f"lr: {learning_rate:g} | dropout: {dropout_rate:g} | semilla: {seed}\n"
        f"Parámetros entrenables: {config['trainable_parameters']:,} / "
        f"{config['num_parameters']:,}\nImágenes: "
        + ", ".join(f"{name}={len(data['labels'])}" for name, data in splits.items()),
        flush=True,
    )

    def evaluate_epoch_metrics(epoch, logs):
        """Registra F1 macro val y loss/accuracy de test antes de EarlyStopping.

        Test se evalúa en modo inferencia y no interviene en la selección
        de pesos ni en la detención temprana.

        Args:
            epoch (int): Índice de época, desde cero, recibido de Keras.
            logs (dict): Métricas de la época; se modifica en el lugar para
                registrar val_f1_macro, test_loss y test_accuracy en History.
                EarlyStopping utiliza únicamente val_f1_macro.
        """
        if not np.isfinite([logs["loss"], logs["val_loss"]]).all():
            raise RuntimeError("Entrenamiento detenido: pérdida no finita.")
        probabilities = model.predict(datasets["val"], verbose=0)
        if not np.isfinite(probabilities).all():
            raise RuntimeError("Entrenamiento detenido: predicciones no finitas.")
        logs["val_f1_macro"] = calculate_metrics(
            splits["val"]["labels"], probabilities.argmax(axis=1), len(class_names),
        )["f1"]
        test_scores = model.evaluate(datasets["test"], verbose=0, return_dict=True)
        logs["test_loss"] = float(test_scores["loss"])
        logs["test_accuracy"] = float(test_scores["accuracy"])
        logs["learning_rate"] = float(
            tf.keras.backend.get_value(model.optimizer.learning_rate)
        )
        print(
            f" — val_f1_macro: {logs['val_f1_macro']:.4f} | "
            f"test_loss: {logs['test_loss']:.4f} | "
            f"test_accuracy: {logs['test_accuracy']:.4f}", flush=True,
        )

    def fit_stage(name, stage_epochs, stage_learning_rate, initial_epoch=0):
        stage = {
            "name": name, "learning_rate": stage_learning_rate,
            "epochs": stage_epochs, "start_epoch": initial_epoch + 1,
            "backbone_trainable": model.get_layer("efficientnetb0").trainable,
            "trainable_parameters": int(
                sum(np.prod(w.shape) for w in model.trainable_weights)
            ),
        }
        config["stages"].append(stage)
        print(
            f"\nEtapa: {name} | lr: {stage_learning_rate:g} | "
            f"épocas máximas: {stage_epochs} | "
            f"parámetros entrenables: {stage['trainable_parameters']:,}",
            flush=True,
        )
        stage_history = model.fit(
            train_dataset, validation_data=datasets["val"],
            initial_epoch=initial_epoch, epochs=initial_epoch + stage_epochs,
            callbacks=[
                tf.keras.callbacks.TerminateOnNaN(),
                tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_epoch_metrics),
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_f1_macro", mode="max", patience=patience,
                    min_delta=0, restore_best_weights=True, verbose=1,
                ),
            ],
            shuffle=False, verbose=1,
        ).history
        stage["epochs_trained"] = len(stage_history["loss"])
        return stage_history

    history = fit_stage("head", epochs, learning_rate)
    fine_tune_start_epoch = None
    if fine_tune_epochs:
        head_best_score = max(history["val_f1_macro"])
        head_best_weights = model.get_weights()
        head_epochs = len(history["loss"])
        fine_tune_start_epoch = head_epochs + 1
        _configure_fine_tuning(model, fine_tune_learning_rate)
        fine_tune_history = fit_stage(
            "fine_tuning", fine_tune_epochs, fine_tune_learning_rate,
            initial_epoch=head_epochs,
        )
        if max(fine_tune_history["val_f1_macro"]) <= head_best_score:
            model.set_weights(head_best_weights)
            print("Se conservan los mejores pesos de la primera etapa.", flush=True)
        del head_best_weights
        for name, values in fine_tune_history.items():
            history[name].extend(values)

    config["backbone_trainable"] = model.get_layer("efficientnetb0").trainable
    config["trainable_parameters"] = config["stages"][-1]["trainable_parameters"]
    history = {name: [float(value) for value in values]
               for name, values in history.items()}
    best_epoch = int(np.argmax(history["val_f1_macro"])) + 1
    metrics, matrices = save_reports(
        model, datasets, splits, class_names, history, output_dir,
        fine_tune_start_epoch=fine_tune_start_epoch,
    )
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifacts_dir / "efficientnet.keras"
    model.save(model_path)
    print(f"\nMejor época por F1 macro de validación: {best_epoch}", flush=True)
    for name, scores in metrics.items():
        print(
            f"  {name:5s} | accuracy: {scores['accuracy']:.2%} | "
            f"precision: {scores['precision']:.4f} | recall: {scores['recall']:.4f} | "
            f"F1 macro: {scores['f1']:.4f}", flush=True,
        )
    print(f"Gráficas: {Path(output_dir).resolve()}\nModelo: {model_path.resolve()}",
          flush=True)
    return {
        "model": model, "history": history, "metrics": metrics,
        "confusion_matrices": matrices, "class_names": class_names,
        "config": config, "best_epoch": best_epoch, "output_dir": Path(output_dir),
        "artifacts_dir": artifacts_dir, "model_path": model_path,
    }


if __name__ == "__main__":
    train_efficientnet_b0()
