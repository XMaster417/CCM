"""Análisis del dataset; imágenes en public/images/eda y CSV en public/documents.

No descarga, elimina, modifica ni divide las imágenes originales.
"""
import hashlib
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "public" / "images" / "eda"
DOCUMENTS_DIR = ROOT / "public" / "documents"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SEED = 42
DHASH_DISTANCE = 5
METRICS = [
    "width", "height", "megapixels", "aspect_ratio", "size_kib",
    "brightness", "contrast", "sharpness", "dark_fraction", "light_fraction",
]

def save_csv(frame, output_dir, name):
    """
    Exporta una tabla como CSV con el prefijo que identifica al análisis.

    Args:
        frame (pd.DataFrame): Tabla que se guardará, con sus filas y columnas.
        output_dir (str): Carpeta de destino.
        name (str): Nombre base para el archivo csv.
    """
    frame.to_csv(
        output_dir / f"analisis_{name}.csv",    # dirección y nombre del archivo
        index=False,                            # no guarda indice de tabla
        encoding="utf-8-sig"                    # permite leer acentos
    )

def save_figure(fig, output_dir, name):
    """
    Guarda una imagen como png con fondo blanco a 150 dpi.

    Args:
        fig (matplotlib.figure.Figure): Figura que contiene la imagen.
        output_dir (str): Carpeta de destino.
        name (str): Nombre base para el archivo csv.
    """
    if fig._suptitle is not None:   # hay titulo, reserva espacio arriba
        fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=2)
    else:                           # no hay titulo, solo acomoda la imagen
        fig.tight_layout()

    fig.savefig(output_dir / f"eda_{name}.png", dpi=150, facecolor="white")
    plt.close(fig)

def save_table(frame, output_dir, name, title, documents_dir=DOCUMENTS_DIR):
    """
    Exporta una tabla a CSV y crea una versión visual de sus datos en PNG.
    El csv contiene los valores originales, en la imagen solo dos decimales.

    Args:
        frame (pd.DataFrame): Tabla pequeña que se desea exportar y dibujar.
        output_dir (str): Carpeta de destino de la imagen PNG.
        name (str): Nombre base para el archivo csv e imagen.
        title (str): Título que se mostrará encima de la tabla en el PNG.
        documents_dir (str): Carpeta de destino del archivo CSV.
    """
    save_csv(frame, documents_dir, name)
    display = frame.copy()
    
    for column in display.select_dtypes(include="number"):
        display[column] = display[column].map(lambda value: f"{value:,.2f}")

    fig, ax = plt.subplots(figsize=(max(10, len(display.columns) * 1.65),
                                    max(2.5, len(display) * 0.38 + 1.4)))
    ax.axis("off")
    ax.set_title(title, pad=20)
    table = ax.table(cellText=display.values, colLabels=display.columns,
                     cellLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dce9f5")

    save_figure(fig, output_dir, name)

def inspect_image(path, raw_dir):
    """
    Lee una imagen y devuelve su identificación, huellas digitales y métricas.

    Procesa una imagen a la vez. Corrige su orientación y utiliza una versión 
    RGB para comparar píxeles. Calcula brillo, contraste y nitidez sobre una 
    miniatura en gris de hasta 256 píxeles por lado, sin ampliarla.

    Args:
        path (str): Ruta de la imagen que se inspeccionará.
        raw_dir (str): Carpeta raíz del dataset.

    Returns:
        dict: Contiene la información de la imagen.
    """
    relative = path.relative_to(raw_dir)
    row = {
        "path": relative.as_posix(),
        "class": relative.parts[0] if len(relative.parts) > 1 else "(sin carpeta)",
        "filename": path.name,
        "extension": path.suffix.lower(),
        "readable": False,
        "error": "",
        "warning": "",
    }

    # Obtener el tamaño y huella (sirve para duplicados) de la imagen
    row["size_bytes"] = path.stat().st_size
    row["size_kib"] = row["size_bytes"] / 1024
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    row["sha256"] = digest.hexdigest()

    # Obtener la orientación de la imagen
    with Image.open(path) as source:
        row["format"] = source.format
        row["mode"] = source.mode
        row["stored_width"], row["stored_height"] = source.size
        source.load()  # Fuerza la decodificación completa, no solo la cabecera.
        rgb = ImageOps.exif_transpose(source).convert("RGB")

    # Obtener las medidas de la imagen
    row["width"], row["height"] = rgb.size
    row["megapixels"] = rgb.width * rgb.height / 1_000_000
    row["aspect_ratio"] = rgb.width / rgb.height
    row["orientation"] = (
        "horizontal" if rgb.width > rgb.height else
        "vertical" if rgb.width < rgb.height else "cuadrada"
    )

    # Encontrar una huella parecida
    pixels_digest = hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode())
    pixels_digest.update(rgb.tobytes())
    row["pixel_sha256"] = pixels_digest.hexdigest()
    gray = rgb.convert("L")
    hash_pixels = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS))
    bits = hash_pixels[:, 1:] > hash_pixels[:, :-1]
    row["dhash"] = f"{int.from_bytes(np.packbits(bits).tobytes(), 'big'):016x}"
    gray.thumbnail((256, 256), Image.Resampling.LANCZOS)

    # Obtener el brillo, contraste e intensidad de las imagenes
    values = np.asarray(gray, dtype=np.float32)
    row["brightness"] = float(values.mean())
    row["contrast"] = float(values.std())
    row["dark_fraction"] = float((values <= 5).mean())
    row["light_fraction"] = float((values >= 250).mean())

    if min(values.shape) >= 3:
        laplacian = (
            values[:-2, 1:-1] + values[2:, 1:-1]        # Arriba + abajo 
            + values[1:-1, :-2] + values[1:-1, 2:]      # + Izquierda + derecha 
            - 4 * values[1:-1, 1:-1]                    # - 4 (centro)
        )
        row["sharpness"] = float(laplacian.var())
    else:
        row["sharpness"] = float("nan")
    
    row["readable"] = True
    return row

def audit_metadata(images, metadata):
    """Compara el inventario de imágenes con el CSV y reúne sus incidencias.

    Identifica cada imagen mediante la combinación de clase y nombre de
    archivo, porque distintas clases pueden contener nombres iguales.

    Args:
        images (pd.DataFrame): Inventario generado con inspect_image. 
        metadata (pd.DataFrame): Contenido de metadata.csv.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Contiene todas las imágenes y los 
            metadatos coincidentes.
    """
    issues = []
    actual_keys = set(zip(images["class"], images["filename"]))
    declared_keys = set(zip(metadata["class"], metadata["filename"]))

    # Revisa claves que están en el csv pero no en imagenes
    for class_name, filename in sorted(declared_keys - actual_keys):
        issues.append(
            {"type": "metadata_without_image", "class": class_name,
            "filename": filename, "detail": "No coincide clase + archivo"}
        )

    # Revisa claves que están en las imagenes pero no en el csv
    for class_name, filename in sorted(actual_keys - declared_keys):
        issues.append({"type": "image_without_metadata", "class": class_name,
                        "filename": filename, 
                        "detail": "No coincide clase + archivo"}
        )

    # Revisa claves repetidas en el csv
    duplicates = metadata[metadata.duplicated(["class", "filename"], keep=False)]
    for _, row in duplicates.iterrows():
        issues.append({"type": "duplicate_metadata_key", "class": row["class"],
                        "filename": row["filename"], 
                        "detail": "Clave repetida en CSV"}
        )

    # Revisa claves no legibles
    for _, row in images[~images["readable"]].iterrows():
        issues.append({"type": "unreadable_image", "class": row["class"],
                        "filename": row["filename"], "detail": row["error"]}
        )

    # Revisa advertencias en las imagenes
    for _, row in images[images["warning"].ne("")].iterrows():
        issues.append({"type": "image_warning", "class": row["class"],
                        "filename": row["filename"], "detail": row["warning"]}
        )

    # Revisa campos de texto vacios en el csv.
    for column in metadata.columns:
        missing = metadata[column].str.strip().eq("")
        if missing.any():
            issues.append({"type": "empty_metadata_field", 
                            "class": "",
                            "filename": "", 
                            "detail": f"{column}: {int(missing.sum())}"}
            )
    # Conserva las incidencias de claves repetidas sin multiplicar imágenes
    # al juntarlas.
    merged = images.merge(metadata.drop_duplicates(["class", "filename"]),
                            on=["class", "filename"], 
                            how="left", 
                            validate="many_to_one"
    )
    return merged, pd.DataFrame(
                        issues, 
                        columns=["type", "class", "filename", "detail"]
                    )

def duplicate_analysis(images):
    """
    Encuentra archivos idénticos, píxeles idénticos y pares visualmente 
    similares.
    Agrupa por SHA-256 para detectar identidad exacta. Después compara los dHash 
    de todos los pares legibles

    Args:
        images (pd.DataFrame): Inventario con las columnas path, class,
            readable, sha256, pixel_sha256 y dhash.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: tabla de duplicados exactos y 
            candidatos.
    """
    members = []

    # Busca duplicados exactamente iguales
    for kind, column in [("bytes", "sha256"), ("pixels_rgb", "pixel_sha256")]:
        subset = images[images[column].notna()]
        
        for digest, group in subset.groupby(column, sort=True):
            if len(group) < 2:
                continue
            for _, row in group.iterrows():
                members.append({"type": kind, "group": digest, 
                                "path": row["path"], "class": row["class"], 
                                "cross_class": group["class"].nunique() > 1}
                )
    
    exact = pd.DataFrame(members, 
                        columns=["type", "group", 
                                "path", "class", "cross_class"]
                        )
    valid = images[images["readable"]].to_dict("records")
    hashes = [int(row["dhash"], 16) for row in valid]
    pairs = []

    # Buscar parecidos o posibles duplicados
    for index, left in enumerate(valid):
        for other in range(index + 1, len(valid)):
            right = valid[other]
            if left["pixel_sha256"] == right["pixel_sha256"]:
                continue
            distance = (hashes[index] ^ hashes[other]).bit_count()
            if distance <= DHASH_DISTANCE:
                pairs.append({"path_a": left["path"], "path_b": right["path"],
                            "class_a": left["class"], "class_b": right["class"],
                            "distance": distance, 
                            "cross_class": left["class"] != right["class"]}
                )
    
    candidates = pd.DataFrame(pairs, columns=[
        "path_a", "path_b", "class_a", "class_b", "distance", "cross_class",
    ])
    candidates = candidates.sort_values(["distance", "path_a", "path_b"]).reset_index(drop=True)
    
    return exact, candidates


def gallery(paths, labels, raw_dir, output_dir, name, title, columns=4):
    """Construye un mosaico de miniaturas con una etiqueta por imagen.

    Args:
        paths (list[str]): Rutas relativas de las imágenes que aparecerán.
        labels (list[str]): Textos para las miniaturas.
        raw_dir (Path): Carpeta raíz que se combina con cada ruta relativa.
        output_dir (Path): Carpeta donde se guardará el mosaico.
        name (str): Nombre base del PNG.
        title (str): Título del mosaico.
        columns (int): Número positivo de columnas.
    """
    if not paths:
        # Evita mostrar candidatos de una ejecución anterior si ya no existen.
        (output_dir / f"eda_{name}.png").unlink(missing_ok=True)
        return

    # Calcula el numero de filas necesarias.
    rows = math.ceil(len(paths) / columns)
    fig, axes = plt.subplots(rows, columns, 
                             figsize=(columns * 3.6, rows * 2.7), squeeze=False
                )
    for ax in axes.flat:
        ax.axis("off")
    
    for ax, relative, label in zip(axes.flat, paths, labels):
        with Image.open(raw_dir / relative) as source:
            thumb = ImageOps.exif_transpose(source).convert("RGB")
            thumb.thumbnail((480, 320), Image.Resampling.LANCZOS)
        ax.imshow(thumb)
        ax.set_title(label, fontsize=8)
    fig.suptitle(title, fontsize=13)
    save_figure(fig, output_dir, name)

def make_charts(
    valid, classes, numeric, 
    metadata, creators, exact, 
    candidates, raw_dir, output_dir, documents_dir=DOCUMENTS_DIR
):
    """
    Genera las tablas, gráficas y muestras visuales del análisis exploratorio.
    Guarda tablas de clases y estadísticas en CSV y PNG, gráficas de 
    distribución, dimensiones, peso, formatos, calidad y creadores. También 
    genera mosaicos, extremos y duplicados, además de un CSV de extremos.

    Args:
        valid (pd.DataFrame): Imágenes legibles.
        classes (pd.DataFrame): Resumen por clase.
        numeric (pd.DataFrame): Estadísticas descriptivas de las métricas.
        metadata (pd.DataFrame): Metadatos originales.
        creators (pd.DataFrame): Concentración por clase de creadores.
        exact (pd.DataFrame): Grupos de duplicados exactos.
        candidates (pd.DataFrame): Pares similares.
        raw_dir (Path): Carpeta raíz donde están las imágenes originales.
        output_dir (Path): Carpeta existente donde se guardaran las imagenes.
        documents_dir (Path): Carpeta existente donde se guardaran los CSV.
    """
    # Guarda las tablas de clases y estadisticas.
    save_table(classes.rename(columns={
        "class": "Clase", "images": "Imágenes", "readable": "Legibles",
        "percent": "% del total", "median_kib": "Mediana KiB",
        "median_width": "Ancho mediano", "median_height": "Alto mediano",
    }), output_dir, "tabla_clases", "Resumen por clase (dimensiones en píxeles)", documents_dir)
    
    save_table(numeric.round(3), output_dir, 
                "tabla_estadisticas", "Distribuciones de las imágenes legibles", documents_dir
    )
    
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(classes["class"], classes["images"], color="#3679ac")
    ax.bar_label(bars, padding=3)
    ax.set_xlim(0, classes["images"].max() * 1.15)
    ax.set_xlabel("Número de imágenes")
    ax.set_title("Distribución de clases")
    save_figure(fig, output_dir, "clases")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].scatter(valid["width"], valid["height"], alpha=0.25, s=10)
    axes[0, 0].set(xlabel="Ancho (px)", ylabel="Alto (px)", title="Dimensiones después de orientación EXIF")

    # Agrupa resolución, proporción y peso en histogramas que guarda
    for ax, column, title in [
        (axes[0, 1], "megapixels", "Resolución (megapíxeles)"),
        (axes[1, 0], "aspect_ratio", "Relación de aspecto (ancho/alto)"),
        (axes[1, 1], "size_kib", "Peso del archivo (KiB)"),
    ]:
        ax.hist(valid[column], bins=35, color="#3679ac", edgecolor="white")
        ax.set(xlabel=title, ylabel="Imágenes")
    save_figure(fig, output_dir, "dimensiones_peso")

    # Cuenta los formatos, modos de color y orientación
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, column, title in zip(axes, ["format", "mode", "orientation"],
                                  ["Formato real", "Modo de color original", "Orientación"]):
        counts = valid[column].value_counts()
        bars = ax.bar(counts.index, counts.values, color="#3679ac")
        ax.bar_label(bars, padding=3)
        ax.set_title(title)
        ax.set_ylim(0, counts.max() * 1.16)
    save_figure(fig, output_dir, "formatos_color_orientacion")

    # Dibuja brillo, contraste y nitidez.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for column_index, (column, title) in enumerate([
        ("brightness", "Brillo medio (0–255)"), ("contrast", "Contraste (desviación gris)"),
        ("sharpness", "Nitidez aproximada (varianza laplaciana)"),
    ]):
        values = valid[column].dropna()
        axes[0, column_index].hist(values, bins=35, color="#3679ac", edgecolor="white")
        axes[0, column_index].set(xlabel=title, ylabel="Imágenes")
        groups = [valid.loc[valid["class"].eq(name), column].dropna() for name in classes["class"]]
        axes[1, column_index].boxplot(groups)
        axes[1, column_index].set_xticks(range(1, len(classes) + 1), classes["class"], rotation=75)
        axes[1, column_index].set_ylabel(title)
    fig.suptitle("Indicadores exploratorios: escala de grises, lado mayor de hasta 256 px")
    save_figure(fig, output_dir, "calidad_visual")

    # Representa los creadores más frecuentes y proporción
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top = metadata["creator"].replace("", "(sin creador)").value_counts().head(10).sort_values()
    axes[0].barh(top.index, top.values, color="#3679ac")
    axes[0].set(title="Diez creadores más frecuentes (registros CSV)", xlabel="Registros")
    ordered = creators.sort_values("top_creator_percent")
    axes[1].barh(ordered["class"], ordered["top_creator_percent"], color="#d98735")
    axes[1].set(title="Participación del creador principal por clase", xlabel="% de registros", xlim=(0, 100))
    save_figure(fig, output_dir, "procedencia_creadores")

    # Selecciona 4 muestras por cada clase
    selection = []
    for _, group in valid.groupby("class", sort=True):
        selection.append(group.sample(n=min(4, len(group)), random_state=SEED))
    sample = pd.concat(selection, ignore_index=True)
    gallery(sample["path"].tolist(), [f"{r['class']} / {r['filename']}" for _, r in sample.iterrows()],
            raw_dir, output_dir, "muestras_clases", "Cuatro ejemplos por clase · semilla 42")

    # Busca y guarda valores extremos en csv e imagenes
    extremes = []
    for column in ["brightness", "contrast", "sharpness", "aspect_ratio", "megapixels"]:
        ranked = valid[valid[column].notna()].sort_values([column, "path"])
        if ranked.empty:
            continue
        for side, subset in [("mínimo", ranked.head(2)), ("máximo", ranked.tail(2))]:
            for _, row in subset.iterrows():
                extremes.append({"metric": column, "side": side, "value": row[column],
                                 "path": row["path"], "class": row["class"]})
    save_csv(pd.DataFrame(extremes), documents_dir, "extremos")
    gallery([r["path"] for r in extremes],
            [f"{r['metric']} {r['side']}: {r['value']:.2f}\n{r['path']}" for r in extremes],
            raw_dir, output_dir, "extremos", "Extremos medidos · requieren interpretación visual")

    # Prepara las imagenes duplicadas exactamente y los candidatos.
    exact_sample = exact[exact["type"].eq("pixels_rgb")].head(24)
    gallery(exact_sample["path"].tolist(),
            [f"Grupo {r['group'][:8]} · {r['path']}" for _, r in exact_sample.iterrows()],
            raw_dir, output_dir, "duplicados_exactos",
            "Píxeles idénticos · hasta 24 archivos ordenados por grupo", columns=2)

    for cross_class, suffix in [(True, "entre_clases"), (False, "misma_clase")]:
        chosen = candidates[candidates["cross_class"].eq(cross_class)].head(8)
        paths, labels = [], []
        for _, row in chosen.iterrows():
            for side in ["a", "b"]:
                paths.append(row[f"path_{side}"])
                labels.append(f"dHash={row['distance']} · {row[f'path_{side}']}")
        gallery(paths, labels, raw_dir, output_dir, f"similares_{suffix}",
                "Candidatos visuales por pares consecutivos · no son duplicados confirmados")

def analyze_dataset(raw_dir=RAW_DIR, output_dir=OUTPUT_DIR,
                    documents_dir=DOCUMENTS_DIR):
    """
    Lleva a cabo la inspección del dataset local y genera sus resultados.
    Lee los metadatos, inventaría las imágenes, calcula sus características,
    revisa la correspondencia de etiquetas y busca duplicados. Después prepara
    los resúmenes por clase, las estadísticas y las salidas visuales.

    Args:
        raw_dir (str): Carpeta que contiene metadata.csv y las imágenes.
        output_dir (str): Carpeta donde se guardarán las imagenes del análisis.
        documents_dir (str): Carpeta donde se guardarán los CSV del análisis.
    """
    # Convertir str a rutas
    raw_dir, output_dir = Path(raw_dir), Path(output_dir)
    documents_dir = Path(documents_dir)
    metadata_path = raw_dir / "metadata.csv"

    # Conseguir los datos.
    metadata = pd.read_csv(metadata_path, dtype=str, keep_default_na=False)
    paths = sorted(path for path in raw_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    output_dir.mkdir(parents=True, exist_ok=True)
    documents_dir.mkdir(parents=True, exist_ok=True)
    records = []

    # Agrega una ficha por foto
    for index, path in enumerate(paths, start=1):
        records.append(inspect_image(path, raw_dir))
        if index % 250 == 0 or index == len(paths):
            print(f"Análisis: {index}/{len(paths)} imágenes inspeccionadas", flush=True)

    # Revisa las imagenes e identifica problemas
    images, issues = audit_metadata(pd.DataFrame(records), metadata)
    if not issues.empty:
        print(f"Incidencias detectadas: {len(issues)}", flush=True)
    valid = images[images["readable"]].copy()

    # Resumir información de las imagenes
    classes = images.groupby("class", sort=True).agg(
        images=("path", "size"), readable=("readable", "sum"),
    ).reset_index()
    class_metrics = valid.groupby("class").agg(
        median_kib=("size_kib", "median"), median_width=("width", "median"),
        median_height=("height", "median"),
    ).reset_index()
    classes = classes.merge(class_metrics, on="class", how="left")

    # Calcula los porcentajes y estadisticas de las clases 
    classes["percent"] = classes["images"] / len(images) * 100
    classes = classes.sort_values(["images", "class"], ascending=[False, True]).reset_index(drop=True)
    numeric = valid[METRICS].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    numeric = numeric.rename_axis("metric").reset_index()

    # Buscar duplicados y posibles repetidos.
    exact, candidates = duplicate_analysis(images)
    save_csv(exact, documents_dir, "duplicados_exactos")
    save_csv(candidates, documents_dir, "similares_candidatos")
    creators = []
    for class_name, group in metadata.groupby("class", sort=True):
        counts = group["creator"].replace("", "(sin creador)").value_counts()
        creators.append({"class": class_name, "creators": int(counts.size),
                         "top_creator": counts.index[0], "top_creator_images": int(counts.iloc[0]),
                         "top_creator_percent": float(counts.iloc[0] / len(group) * 100)})
    creators = pd.DataFrame(creators)
    save_table(creators, output_dir, "tabla_creadores", "Concentración de creadores por clase (metadata)", documents_dir)
    source_counts = metadata["source"].replace("", "(sin fuente)").value_counts()
    save_table(source_counts.rename_axis("Fuente").reset_index(name="Registros"),
               output_dir, "tabla_fuentes", "Procedencia declarada en metadata.csv", documents_dir)
    make_charts(valid, classes, numeric, metadata, creators, exact, candidates,
                raw_dir, output_dir, documents_dir)

    print(f"Análisis terminado: {len(images)} imágenes, {len(valid)} legibles, {len(classes)} clases.", flush=True)
    print(f"Imagenes: {output_dir / 'eda_*.png'}", flush=True)
    print(f"Documentos: {documents_dir / 'analisis_*.csv'}", flush=True)

if __name__ == "__main__":
    analyze_dataset()
