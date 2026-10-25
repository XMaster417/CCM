"""Limpia y transforma el dataset usando los resultados del análisis exploratorio.

Conserva data/raw. No aplica data augmentation ni divide el dataset.
El resumen se imprime en consola y se devuelve como diccionario, sin reportes
de decisiones en CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps


# Preparar las carpetas del proyecto y el tamaño de las imagenes.
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
ANALYSIS_DIR = ROOT / "public" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
IMAGE_SIZE = 224


def select_images(metadata, exact, candidates):
    """
    Lee los grupos exactos y los pares dHash de los CSV del análisis. Se 
    prioriza la ruta alfabética para elegir una copia. Una imagen parecida solo 
    descarta una imagen si su parecido coincide directamente con una imagen que 
    sí se conserva.

    Args:
        metadata (pd.DataFrame): Metadatos originales de las imagenes.
        exact (pd.DataFrame): Grupos de duplicados exactos del análisis.
        candidates (pd.DataFrame): Pares de imagenes similares del análisis.

    Returns:
        tuple[pd.DataFrame, dict]: Imagenes seleccionadas excluidas.
    """
    # Copiar los metadatos y construir una ruta por cada imagen.
    valid = metadata.copy()
    valid["path"] = valid["class"] + "/" + valid["filename"]

    # Cada imagen comienza apuntando a si misma.
    parents = {path: path for path in valid["path"]}

    def root(path):
        """
        Encuentra el representante del grupo exacto al que pertenece una imagen.

        Args:
            path (str): Ruta relativa de una imagen incluida en parents.

        Returns:
            str: Ruta del representante del grupo exacto.
        """
        # Acortar el camino al representante para las siguientes consultas.
        while parents[path] != path:
            parents[path] = parents[parents[path]]
            path = parents[path]
        return path

    # Unir los miembros de cada grupo de duplicados exactos.
    for _, group in exact.groupby(["type", "group"], sort=True):
        members = [p for p in group["path"] if p in parents]
        for path in members[1:]:
            parents[root(path)] = root(members[0])

    # Buscar grupos donde una misma imagen tiene distintas etiquetas de clase.
    valid["exact_group"] = valid["path"].map(root)
    class_counts = valid.groupby("exact_group")["class"].nunique()
    conflict_groups = set(class_counts[class_counts > 1].index)

    # Relacionar los grupos que contienen imagenes visualmente parecidas.
    neighbors = {group: set() for group in class_counts.index}
    for pair in candidates.itertuples(index=False):
        if pair.path_a not in parents or pair.path_b not in parents:
            continue
        left, right = root(pair.path_a), root(pair.path_b)
        if pair.class_a != pair.class_b:
            # Excluir ambos grupos si las etiquetas no coinciden.
            conflict_groups.update([left, right])
        elif left != right:
            # Registrar el parecido en ambos sentidos.
            neighbors[left].add(right)
            neighbors[right].add(left)

    # Separar los conflictos y elegir una copia exacta por orden de ruta.
    conflicts = valid[valid["exact_group"].isin(conflict_groups)]
    eligible = valid[~valid["exact_group"].isin(conflict_groups)].copy()
    ranked = eligible.sort_values("path")
    representatives = ranked.drop_duplicates("exact_group")

    # Descartar un parecido solo si ya se conservó un grupo vecino.
    retained_groups, retained_paths, similar_paths = set(), [], []
    for row in representatives.itertuples(index=False):
        if neighbors[row.exact_group] & retained_groups:
            similar_paths.append(row.path)
        else:
            retained_groups.add(row.exact_group)
            retained_paths.append(row.path)

    # Preparar la tabla final y contar las exclusiones sin duplicarlas.
    selected = valid[valid["path"].isin(retained_paths)].sort_values("path").copy()
    summary = {
        "input_images": len(metadata),
        "label_conflicts": len(conflicts),
        "exact_duplicates_removed": len(eligible) - len(representatives),
        "similar_duplicates_removed": len(similar_paths),
        "output_images": len(selected),
        "conflict_paths": sorted(conflicts["path"].tolist()),
        "exact_duplicate_paths": sorted(eligible.loc[
            ~eligible["path"].isin(representatives["path"]), "path"].tolist()),
        "similar_duplicate_paths": sorted(similar_paths),
    }
    return selected, summary

def preprocess_image(path, image_size=IMAGE_SIZE):
    """
    Corrige la orientación, convierte a RGB y ajusta el tamaño sin deformar.
    Conserva toda la imagen y completa el espacio restante con relleno negro.

    Args:
        path (str | Path): Ruta de la imagen original.
        image_size (int): Ancho y alto de la imagen de salida, en pixeles.

    Returns:
        Image.Image: Imagen RGB reescalada con relleno negro.
    """
    # Abrir la imagen, aplicar su orientación y unificar los canales de color.
    with Image.open(path) as source:
        rgb = ImageOps.exif_transpose(source).convert("RGB")
        # Ajustar la imagen dentro de un cuadrado, centrada y sin recortar.
        resized = ImageOps.pad(
            rgb, (image_size, image_size), method=Image.Resampling.LANCZOS,
            color=(0, 0, 0), centering=(0.5, 0.5),
        )
    return resized

def load_normalized_image(path):
    """
    Normaliza mediante dividir en /255 los pixeles, acotando los valores entre 0
    y 1.

    Args:
        path (str | Path): Ruta del JPG procesado que se cargará para la CNN.

    Returns:
        np.ndarray: Pixeles RGB de tipo float32 con forma (alto, ancho, 3).
    """
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.float32) / np.float32(255.0)

def transform_dataset(raw_dir=RAW_DIR, output_dir=PROCESSED_DIR,
                      image_size=IMAGE_SIZE, analysis_dir=ANALYSIS_DIR):
    """
    Genera imágenes procesadas y sus metadatos, utilizando metadata.csv y los 
    CSV de duplicados 

    Args:
        raw_dir (str | Path): Carpeta que con datos crudos.
        output_dir (str | Path): Carpeta donde se guardan los JPG y su metadata.
        image_size (int): Ancho y alto de las imagenes procesadas, en pixeles.
        analysis_dir (str | Path): Carpeta que el analisis de duplicados.

    Returns:
        dict: Resumen con los conteos y las rutas de las imagenes excluidas.
    """
    # Convertir las carpetas a rutas completas y proteger los originales.
    raw_dir, output_dir = Path(raw_dir).resolve(), Path(output_dir).resolve()

    # Leer los metadatos y los resultados existentes del análisis.
    analysis_dir = Path(analysis_dir)
    source_metadata = pd.read_csv(
        raw_dir / "metadata.csv", dtype=str, keep_default_na=False
    )
    exact = pd.read_csv(analysis_dir / "analisis_duplicados_exactos.csv", 
                        dtype=str,
                        keep_default_na=False
    )
    candidates = pd.read_csv(analysis_dir / "analisis_similares_candidatos.csv",
                                keep_default_na=False
    )

    # Seleccionar las imagenes que se conservarán.
    selected, summary = select_images(source_metadata, exact, candidates)

    # Conservar las columnas originales y agregar las rutas de las nuevas imagenes.
    metadata = selected[list(source_metadata.columns)].reset_index(drop=True).copy()
    metadata["source_path"] = selected["path"].to_numpy()
    metadata["path"] = [f"{class_name}/{filename}" for class_name, filename in
                        zip(metadata["class"], metadata["filename"])]

    # Procesar y guardar una imagen a la vez dentro de su clase.
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(metadata.itertuples(index=False), start=1):
        image = preprocess_image(raw_dir / row.source_path, image_size)
        target = output_dir / row.path
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="JPEG", quality=95, subsampling=0)
        if index % 250 == 0 or index == len(selected):
            print(f"Transformación: {index}/{len(selected)} imágenes", flush=True)

    # Guardar solo los metadatos de las imagenes seleccionadas.
    metadata.to_csv(output_dir / "metadata.csv", 
                    index=False, encoding="utf-8-sig"
    )

    # Mostrar los conteos y los nombres de las imagenes que se excluyeron.
    print(f"\nOriginales: {summary['input_images']}", flush=True)
    print(f"Excluidas por conflicto de etiquetas: {summary['label_conflicts']}")
    for path in summary["conflict_paths"]:
        print(f"  {path}")

    print(f"Copias exactas retiradas: {summary['exact_duplicates_removed']}")
    for path in summary["exact_duplicate_paths"]:
        print(f"  {path}")

    print(f"Copias similares retiradas: {summary['similar_duplicates_removed']}")
    for path in summary["similar_duplicate_paths"]:
        print(f"  {path}")

    print(f"Resultado: {summary['output_images']} imágenes RGB de \
            {image_size} x {image_size}"
    )
    print(f"Salida: {output_dir}", flush=True)
    return summary

if __name__ == "__main__":
    # Ejecutar la transformación cuando se abre este archivo como script.
    transform_dataset()
