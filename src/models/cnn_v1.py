"""Entrena una CNN de diez clases y genera únicamente reportes PNG."""

from pathlib import Path
import sys

import matplotlib

# Guardar las gráficas sin abrir ventanas durante el entrenamiento.
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

# Añadir src a las rutas de importación para reutilizar las funciones de scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.data_analysis import save_figure
from scripts.transform_data import load_normalized_image

# Definir dónde leer las imágenes y los conjuntos, y dónde guardar los PNG.
ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed"
SETS_DIR = ROOT / "data" / "sets"
OUTPUT_DIR = ROOT / "public" / "images" / "cnn_v1"

# Configurar la entrada (alto, ancho, canales), las clases y el entrenamiento.
IMAGE_SHAPE = (224, 224, 3)
NUM_CLASSES = 10
EPOCHS = 30
PATIENCE = 5  # Épocas consecutivas sin aumentar val_f1_macro antes de detenerse.
BATCH_SIZE = 32
SEED = 42
SPLITS = ("train", "val", "test")
COLORS = {"train": "#3679ac", "val": "#d98735", "test": "#47956a"}

def load_splits(input_dir=INPUT_DIR, sets_dir=SETS_DIR):
    """
    Lee y valida los metadatos de train, val y test antes del entrenamiento.

    Asigna un número de 0 a 9 a cada clase, usando el mismo orden alfabético en 
    los tres conjuntos, y detecta rutas repetidas.

    Args:
        input_dir (str | Path): Carpeta de imágenes procesadas.
        sets_dir (str | Path): Carpeta con train/metadata.csv, val/metadata.csv 
            y test/metadata.csv.

    Returns:
        tuple[dict, list[str]]: splits y class_names.
    """
    # Resolver las carpetas y leer los tres CSV como texto.
    input_dir, sets_dir = Path(input_dir).resolve(), Path(sets_dir).resolve()
    frames = {}
    for name in SPLITS:
        metadata_path = sets_dir / name / "metadata.csv"
        if not metadata_path.is_file():
            raise FileNotFoundError(
                f"No existe {metadata_path}. Ejecuta primero split_data.py."
            )
        frame = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
        frames[name] = frame

    # Crear una única correspondencia, por ejemplo Audi_R8 -> 0.
    class_names = sorted(frames["train"]["class"].unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    
    # Verificar las mismas clases y revisar cada ruta de imagen.
    seen_paths, splits = {}, {}
    for name, frame in frames.items():
        paths = []
        for relative in frame["path"]:
            path = (input_dir / relative).resolve()
            # Las imágenes deben existir dentro de input_dir y no repetirse.
            if Path(relative).is_absolute() or not path.is_relative_to(input_dir):
                raise ValueError(f"Ruta fuera de la carpeta de imágenes: {relative}")
            if not path.is_file():
                raise FileNotFoundError(f"No existe la imagen: {path}")
            if path in seen_paths:
                raise ValueError(
                    f"Imagen repetida en {seen_paths[path]} y {name}: {relative}"
                )
            seen_paths[path] = name
            paths.append(str(path))
        # Convertir los nombres a etiquetas conservando el orden de las rutas.
        labels = frame["class"].map(class_to_index).to_numpy(dtype=np.int32)
        splits[name] = {"paths": paths, "labels": labels}
    return splits, class_names

def _load_image(path):
    """
    Adapta una ruta de TensorFlow a la función de normalización existente.

    Args:
        path (bytes | np.ndarray): Ruta codificada en UTF-8 recibida desde
            tf.numpy_function.

    Returns:
        np.ndarray: Imagen float32 de forma (224, 224, 3), con píxeles
            normalizados entre 0 y 1.
    """
    # Extraer el valor si TensorFlow lo entrega dentro de un array de NumPy.
    if isinstance(path, np.ndarray):
        path = path.item()
    
    # Decodificar la ruta y reutilizar la carga que divide los píxeles por 255.
    pixels = load_normalized_image(path.decode("utf-8"))

    return pixels

def make_dataset(paths, labels, batch_size=BATCH_SIZE, shuffle=False, seed=SEED):
    """
    Construye el flujo de carga de imágenes y etiquetas mediante tf.data.

    Args:
        paths (list[str]): Rutas absolutas de las imágenes.
        labels (np.ndarray): Etiquetas de clase de 0 a 9.
        batch_size (int): Máximo de imágenes por lote.
        shuffle (bool): Mezclar el orden en cada época.
        seed (int): Semilla para reproducir la mezcla.

    Returns:
        tf.data.Dataset: Lotes (imágenes, etiquetas), con formas 
            (N, 224, 224, 3) y (N,).
    """
    # Emparejar cada ruta con su etiqueta, aún no se leen los píxeles.
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    
    # Mezclar las parejas, conservando la correspondencia imagen-etiqueta.
    if shuffle:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        """
        Carga una imagen durante la iteración del dataset.

        Args:
            path (tf.Tensor): Ruta escalar de tipo string.
            label (tf.Tensor): Etiqueta escalar de tipo int32.

        Returns:
            tuple[tf.Tensor, tf.Tensor]: Imagen float32 (224, 224, 3) y su
                etiqueta original.
        """
        # Ejecutar la función de NumPy/Pillow dentro del flujo de TensorFlow.
        pixels = tf.numpy_function(_load_image, [path], tf.float32)
        pixels.set_shape(IMAGE_SHAPE)
        return pixels, label

    # Cargar en paralelo, formar lotes y preparar un lote por adelantado.
    return dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(
        batch_size
    ).prefetch(1)

def build_model(dropout_rate=0.0, model_name="cnn_v1"):
    """
    Construye y compila la CNN que recibe una imagen y clasifica su automóvil.
    Recorrido por imagen:
        Entrada:     224 x 224 x 3
        Conv + pool:  56 x  56 x 16
        Conv + pool:  14 x  14 x 32
        Conv + pool:   4 x   4 x 64
        Conv + pool:   1 x   1 x 128
        Flatten:     128 valores
        Dense ReLU:  128 valores
        Softmax:      10 probabilidades que suman aproximadamente 1

    Args:
        dropout_rate (float): Tasa de dropout después de Dense ReLU. Si es 0,
            no se agrega la capa y se conserva la arquitectura original.
        model_name (str): Nombre del modelo de Keras.

    Returns:
        tf.keras.Sequential: Modelo compilado con Adam (0.001), pérdida
            SparseCategoricalCrossentropy y métrica de accuracy multiclase.
    """
    # Declarar la forma de cada imagen.
    layers = [tf.keras.layers.Input(shape=IMAGE_SHAPE)]
    
    # Añadir cuatro bloques convolución 3x3 con ReLU y max pooling 4x4.
    for filters in (16, 32, 64, 128):
        layers.extend([
            tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(pool_size=(4, 4), padding="same"),
        ])
    
    # Aplanar, combinar características y producir probabilidades.
    layers.extend([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
    ])
    if dropout_rate:
        layers.append(tf.keras.layers.Dropout(dropout_rate))

    layers.append(tf.keras.layers.Dense(NUM_CLASSES, activation="softmax"))
    
    # Encadenar las capas y definir cómo aprenderá el modelo.
    model = tf.keras.Sequential(layers, name=model_name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    return model

def save_training_plots(history, output_dir, report_prefix="cnn_v1"):
    """
    Genera las curvas una sola vez, después de terminar el entrenamiento.

    Args:
        history (dict): historia con loss, accuracy para val_ y test_. Train 
            es el promedio de los lotes durante la época.
        output_dir (str | Path): Carpeta existente para guardar PNG.
        report_prefix (str): Prefijo de los archivos PNG.
    """
    output_dir = Path(output_dir)
    epochs = range(1, len(history["loss"]) + 1)
    for metric, title in (("accuracy", "Accuracy"), ("loss", "Loss")):
        fig, ax = plt.subplots(figsize=(10, 6))
        for name, prefix in (("train", ""), ("val", "val_"), ("test", "test_")):
            ax.plot(
                epochs, history[f"{prefix}{metric}"], label=name,
                color=COLORS[name], marker="o", markersize=3,
            )
        ax.set(xlabel="Época", ylabel=title, title=f"{title} por conjunto")
        ax.set_xticks(epochs[::max(1, len(epochs) // 10)])
        if metric == "accuracy":
            ax.set_ylim(0, 1.05)
        else:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.25)
        ax.legend()
        save_figure(fig, output_dir, metric, prefix=report_prefix)

def calculate_metrics(expected, predicted):
    """
    Calcula accuracy y métricas macro con el mismo criterio para los reportes
    finales y el seguimiento de validación por época.

    Args:
        expected (np.ndarray): Etiquetas reales, de 0 a 9.
        predicted (np.ndarray): Clases predichas mediante argmax, de 0 a 9.

    Returns:
        dict: accuracy, precision, recall y f1. Las métricas macro incluyen
            las diez clases y asignan 0 cuando hay una división indefinida.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        expected, predicted, labels=np.arange(NUM_CLASSES), average="macro",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
    }


def save_final_reports(model, datasets, splits, class_names, output_dir,
                       report_prefix="cnn_v1"):
    """
    Calcula las métricas finales y genera cuatro reportes PNG.

    Se ejecuta después de fit, cuando EarlyStopping ya restauró los pesos con
    mayor F1 macro de validación. Las curvas conservan el historial original.

    Args:
        model (tf.keras.Model): Modelo entrenado.
        datasets (dict[str, tf.data.Dataset]): Conjuntos con el
            mismo orden que las etiquetas guardadas en splits.
        splits (dict): Datos de load_splits, se utilizan sus labels reales.
        class_names (list[str]): Nombres de las clases en el orden 0 a 9.
        output_dir (str | Path): Carpeta existente para los cuatro PNG.
        report_prefix (str): Prefijo de los archivos PNG.

    Returns:
        tuple[dict, dict]: metrics contiene accuracy, precision, recall y f1
            por conjunto, matrices contiene una matriz de confusión 10x10
            por conjunto.
    """
    # Preparar los contenedores y el orden completo de etiquetas.
    output_dir = Path(output_dir)
    metrics, matrices = {}, {}
    labels = np.arange(NUM_CLASSES)

    for name in SPLITS:
        # Obtener diez probabilidades por imagen y elegir la mayor (argmax).
        probabilities = model.predict(datasets[name], verbose=0)
        predicted = probabilities.argmax(axis=1)
        expected = splits[name]["labels"]

        # Calcula accuracy, precision, recall y f1 macro.
        metrics[name] = calculate_metrics(expected, predicted)
        # Contar aciertos en la diagonal y errores fuera de la diagonal.
        # Los valores son cantidades de imágenes, no porcentajes.
        matrix = confusion_matrix(expected, predicted, labels=labels)
        matrices[name] = matrix
        fig, ax = plt.subplots(figsize=(12, 11))
        ConfusionMatrixDisplay(matrix, display_labels=class_names).plot(
            ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False,
        )
        ax.set(
            title=f"Matriz de confusión: {name} ({len(expected)} imágenes)",
            xlabel="Clase predicha", ylabel="Clase real",
        )
        save_figure(fig, output_dir, f"confusion_{name}", prefix=report_prefix)

    # Comparar los tres conjuntos en una gráfica de barras de cuatro métricas.
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
    ax.set_xticks(
        positions,
        ["Accuracy", "Precision (macro)", "Recall (macro)", "F1-score (macro)"],
    )
    ax.set(ylim=(0, 1.15), ylabel="Valor", title="Métricas finales por conjunto")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    save_figure(fig, output_dir, "metrics", prefix=report_prefix)
    return metrics, matrices

def train_model_v1(input_dir=INPUT_DIR, sets_dir=SETS_DIR, output_dir=OUTPUT_DIR,
                epochs=EPOCHS, batch_size=BATCH_SIZE, seed=SEED, *,
                dropout_rate=0.0, model_name="cnn_v1"):
    """
    Coordina la lectura de datos, el entrenamiento y los reportes de la CNN.

    Args:
        input_dir (str | Path): Carpeta de las imágenes procesadas.
        sets_dir (str | Path): Carpeta con los CSV de train, val y test.
        output_dir (str | Path): Carpeta para los resultados (imagenes).
        epochs (int): Máximo de número de épocas. EarlyStopping puede terminar 
            antes si val_f1_macro no aumenta durante PATIENCE épocas.
        batch_size (int): Máximo de imágenes por lote, por defecto 32.
        seed (int): Semilla de inicialización y mezcla, por defecto 42.
        dropout_rate (float): Tasa de dropout después de Dense ReLU; 0 lo omite.
        model_name (str): Nombre del modelo y prefijo de sus reportes PNG.

    Returns:
        dict: model (modelo con mayor F1 macro de validación), history
            (dict de Keras con accuracy, loss, val_accuracy, val_loss,
            val_f1_macro, test_accuracy y test_loss por época), metrics
            (métricas finales),
            confusion_matrices (matrices 10x10), class_names (orden de clases)
            y output_dir (Path). Solo se exportan los PNG, no el diccionario.
    """
    # Usar CPU, fijar la semilla y validar metadatos, clases y rutas.
    tf.config.set_visible_devices([], "GPU")
    tf.keras.utils.set_random_seed(seed)
    splits, class_names = load_splits(input_dir, sets_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Preparar los tres datasets sin mezclar.
    datasets = {
        name: make_dataset(data["paths"], data["labels"], batch_size)
        for name, data in splits.items()
    }
    # Crear otro flujo de train mezclado, este es el que actualiza los pesos.
    train_dataset = make_dataset(
        splits["train"]["paths"], splits["train"]["labels"], batch_size,
        shuffle=True, seed=seed,
    )
    
    # Construir el modelo y los callbacks que acompañan cada época.
    model = build_model(dropout_rate=dropout_rate, model_name=model_name)

    def evaluate_epoch_metrics(epoch, logs):
        """
        Registra F1 macro de validación antes de EarlyStopping y las métricas
        de test. Solo el F1 de validación determina cuándo detenerse.

        Args:
            epoch (int): Índice de época proporcionado por Keras.
            logs (dict): Métricas de la época que History guardará después.
        """
        # El dataset no se mezcla: argmax y etiquetas mantienen el mismo orden.
        # Calcular F1 sobre toda la validación, no como promedio de los lotes.
        probabilities = model.predict(datasets["val"], verbose=0)
        predicted = probabilities.argmax(axis=1)
        logs["val_f1_macro"] = calculate_metrics(
            splits["val"]["labels"], predicted,
        )["f1"]

        scores = model.evaluate(datasets["test"], verbose=0, return_dict=True)
        logs["test_loss"] = float(scores["loss"])
        logs["test_accuracy"] = float(scores["accuracy"])

    # Una mejora es un aumento estricto del F1 macro de validación. Tras cinco
    # épocas sin mejorar, detenerse y restaurar los pesos con mayor F1 macro.
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_f1_macro", mode="max", patience=PATIENCE, min_delta=0,
        restore_best_weights=True, verbose=1,
    )
    print(
        f"Entrenamiento en CPU | épocas: {epochs} | batch: {batch_size} | "
        f"semilla: {seed}\nImágenes: "
        + ", ".join(f"{name}={len(data['labels'])}" for name, data in splits.items()),
        flush=True,
    )

    # Registrar val_f1_macro antes de EarlyStopping y del History de Keras.
    # También se conservan las métricas de test de cada época.
    history = model.fit(
        train_dataset, validation_data=datasets["val"], epochs=epochs,
        callbacks=[
            tf.keras.callbacks.LambdaCallback(on_epoch_end=evaluate_epoch_metrics),
            early_stopping,
        ],
        shuffle=False, verbose=1,
    )

    # Generar el historial completo y los reportes finales.
    save_training_plots(history.history, output_dir, report_prefix=model_name)
    metrics, matrices = save_final_reports(
        model, datasets, splits, class_names, 
        output_dir, report_prefix=model_name,
    )
    print(
        "\nMétricas finales con los pesos de mayor F1 macro de validación "
        "(precision, recall y F1 con promedio macro):",
        flush=True,
    )
    for name, scores in metrics.items():
        print(
            f"  {name:5s} | accuracy: {scores['accuracy']:.2%} | "
            f"precision: {scores['precision']:.4f} | recall: {scores['recall']:.4f} | "
            f"F1-score: {scores['f1']:.4f}",
            flush=True,
        )
    print(f"Gráficas guardadas en: {output_dir.resolve()}", flush=True)

    return {
        "model": model, "history": history.history, "metrics": metrics,
        "confusion_matrices": matrices, "class_names": class_names,
        "output_dir": output_dir,
    }

if __name__ == "__main__":
    train_model_v1()
