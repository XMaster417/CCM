"""Divide los metadatos por clase y exporta los conteos en CSV y PNG."""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Importar las funciones de exportación desde main.py o desde este script.
if __package__:
    from .data_analysis import save_figure, save_table
else:
    from data_analysis import save_figure, save_table

# Definir la carpeta raíz y las rutas de entrada y salida.
ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "data" / "processed"
SETS_DIR = ROOT / "data" / "sets"
IMAGES_DIR = ROOT / "public" / "images" / "split"
DOCUMENTS_DIR = ROOT / "public" / "documents"

# Configurar los porcentajes enteros de cada conjunto; deben sumar 100.
TRAIN_PERCENT = 70
VAL_PERCENT = 15
TEST_PERCENT = 15
SEED = 42
SPLITS = ("train", "val", "test")


def assign_splits(metadata, seed=SEED):
    """
    Divide los metadatos mediante dos separaciones estratificadas por clase.

    Primero separa train del resto y después divide el resto en val y test.
    Ordena las filas y usa una semilla para repetir la misma asignación.

    Args:
        metadata (pd.DataFrame): Metadatos con las columnas class y path.
        seed (int): Semilla para las dos separaciones de scikit-learn.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Metadatos ordenados con la columna
            split agregada, y tabla de conteos por clase y conjunto.
    """
    percentages = {
        "train": TRAIN_PERCENT, "val": VAL_PERCENT, "test": TEST_PERCENT,
    }

    # Verificar que los porcentajes sean enteros positivos y sumen 100.
    if any(not isinstance(value, int) or value <= 0
            for value in percentages.values()) or sum(percentages.values()) != 100:
        raise ValueError("Los porcentajes deben ser enteros positivos y sumar 100.")

    # Ordenar por clase y ruta, y numerar las filas desde cero.
    ordered = metadata.sort_values(["class", "path"]).reset_index(drop=True)

    # Separar train del resto, conservando la proporción de clases con stratify.
    train, remaining = train_test_split(
        ordered,                                        # conjunto ordenado
        test_size=(VAL_PERCENT + TEST_PERCENT) / 100,   # porcentaje test + val
        random_state=seed,                              
        stratify=ordered["class"],                      # muestreo estratificado
    )

    # Dividir el resto: para 70/15/15, test recibe 15 / (15 + 15) = 50% del resto.
    val, test = train_test_split(
        remaining,                                              # 30% de datos
        test_size=TEST_PERCENT / (VAL_PERCENT + TEST_PERCENT),  # 50% c/u
        random_state=seed,
        stratify=remaining["class"],
    )

    # Etiquetar las filas usando los índices que conservó cada subconjunto.
    ordered["split"] = ""
    for name, subset in zip(SPLITS, (train, val, test)):
        ordered.loc[subset.index, "split"] = name

    # Contar las asignaciones reales y comprobar que cada clase esté representada.
    counts = pd.crosstab(ordered["class"], ordered["split"]).reindex(
        columns=SPLITS, fill_value=0
    )
    counts.columns.name = None
    if (counts <= 0).any().any():
        raise ValueError(
            "No hay suficientes imágenes por clase para representar todas las "
            "clases en los tres conjuntos con los porcentajes configurados."
        )
    return ordered, counts


def save_reports(counts, images_dir=IMAGES_DIR, documents_dir=DOCUMENTS_DIR):
    """
    Exporta los conteos por clase a CSV, tabla PNG y tres gráficas de barras.

    Args:
        counts (pd.DataFrame): Conteos con las clases como índice y las
            columnas train, val y test.
        images_dir (str | Path): Carpeta de destino de las imágenes PNG.
        documents_dir (str | Path): Carpeta de destino del CSV de conteos.

    Returns:
        pd.DataFrame: Tabla exportada con las columnas class, train, val,
            test y total.
    """
    # Convertir las rutas y crear las carpetas de los reportes si no existen.
    images_dir, documents_dir = Path(images_dir), Path(documents_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)

    # Convertir el índice de clases en una columna y sumar cada fila.
    table = counts.rename_axis("class").reset_index()
    table["total"] = table[list(SPLITS)].sum(axis=1)

    # Guardar los mismos conteos como CSV y como una tabla visual PNG.
    save_table(
        table, images_dir, "conteos_clases", "Imágenes por clase y conjunto",
        documents_dir, csv_prefix="split", figure_prefix="split",
    )

    # Definir el color y el título de cada conjunto.
    colors = {"train": "#3679ac", "val": "#d98735", "test": "#47956a"}
    titles = {"train": "Train", "val": "Validation", "test": "Test"}
    for name in SPLITS:
        # Dibujar una barra horizontal por clase y escribir su cantidad al lado.
        fig, ax = plt.subplots(figsize=(11, 6))
        bars = ax.barh(counts.index, counts[name], color=colors[name])
        ax.bar_label(bars, padding=3)
        # Dejar espacio para las etiquetas, configurar los ejes y guardar el PNG.
        ax.set_xlim(0, counts[name].max() * 1.15)
        ax.set_xlabel("Número de imágenes")
        ax.set_title(f"{titles[name]}: {int(counts[name].sum())} imágenes")
        save_figure(fig, images_dir, name, prefix="split")

    return table

def split_dataset(input_dir=INPUT_DIR, output_dir=SETS_DIR, seed=SEED):
    """
    Lee, valida y divide los metadatos, y genera los reportes de la separación.

    Guarda un metadata.csv por conjunto. Las rutas de las imágenes permanecen
    relativas a input_dir y las imágenes no se copian ni se modifican.

    Args:
        input_dir (str | Path): Carpeta con metadata.csv y sus imágenes.
        output_dir (str | Path): Carpeta donde se crean train, val y test.
        seed (int): Semilla para repetir la asignación.

    Returns:
        dict: input_dir (Path), seed (int), counts (pd.DataFrame con los
            conteos exportados) y totals (dict con la cantidad por conjunto).
    """
    # Resolver las rutas completas y validar la semilla.
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()

    # Leer todas las columnas como texto y conservar las celdas vacías como "".
    metadata = pd.read_csv(
        input_dir / "metadata.csv", dtype=str, keep_default_na=False
    )

    # Construir las rutas completas y detectar referencias repetidas.
    paths = [(input_dir / path).resolve() for path in metadata["path"]]
    if len(set(paths)) != len(paths):
        raise ValueError("metadata.csv contiene rutas de imágenes repetidas.")

    for relative, path in zip(metadata["path"], paths):
        # Comprobar que cada ruta esté dentro de la entrada y que el archivo exista.
        if Path(relative).is_absolute() or not path.is_relative_to(input_dir):
            raise ValueError(f"La ruta debe ser relativa a la entrada: {relative}")
        if not path.is_file():
            raise FileNotFoundError(f"No existe la imagen: {path}")

    # Asignar las imágenes y guardar únicamente las columnas originales del CSV.
    assigned, counts = assign_splits(metadata, seed)
    for name in SPLITS:
        destination = output_dir / name
        destination.mkdir(parents=True, exist_ok=True)
        # Filtrar las filas del conjunto actual y exportarlas sin el índice.
        assigned.loc[assigned["split"].eq(name), metadata.columns].to_csv(
            destination / "metadata.csv", index=False, encoding="utf-8-sig"
        )

    # Generar los reportes y sumar las imágenes por conjunto.
    table = save_reports(counts)
    totals = {name: int(counts[name].sum()) for name in SPLITS}

    # Mostrar los resultados y devolver el resumen.
    print(f"\nEntrada: {input_dir}")
    print(table.to_string(index=False))
    for name, count in totals.items():
        print(f"{name}: {count} imágenes ({count / len(metadata):.2%})")

    return {
        "input_dir": input_dir, "seed": seed,
        "counts": table, "totals": totals,
    }

if __name__ == "__main__":
    split_dataset()
