from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src.paths import GRID_OUTPUT, DF_DATA

logger = logging.getLogger(__name__)

_DEFAULT_ID_COL = "customer_code"


def _load_ids(path: Path, id_col: str = _DEFAULT_ID_COL) -> np.ndarray:
    """Liest Kunden-IDs aus der Master-Pickle-Datei."""

    df = pd.read_pickle(path)
    if id_col not in df.columns:
        raise ValueError(
            f"Spalte '{id_col}' nicht in {path.name} gefunden. "
            f"Vorhandene Spalten: {list(df.columns)}"
        )
    return df[id_col].astype(str).to_numpy()


def _iter_square_pickles(grid_root: Path):
    """Liefert alle Square-PKL-Dateien, ueberspringt bereits konvertierte."""

    for path in sorted(grid_root.rglob("*.pkl")):
        if any("_long" in part or part == "long" for part in path.parts):
            continue
        yield path


def _make_output_path(input_path: Path) -> Path:
    """Leitet den Long-Ausgabepfad aus dem Square-Eingabepfad ab."""

    stem = input_path.stem.removesuffix("_square") + "_long"
    return input_path.parent / "long" / f"{stem}.pkl"



def square_to_long(
    matrix: pd.DataFrame | np.ndarray,
    *,
    fallback_ids: np.ndarray | None = None,
    include_diagonal: bool = False,
) -> pd.DataFrame:
    """
    Wandelt eine quadratische Distanzmatrix ins Long-Format.

    Args:
        matrix:           Quadratische Distanzmatrix (DataFrame).
        fallback_ids:     Kunden-IDs, falls matrix ein numpy-Array ist.
        include_diagonal: Wenn True, werden auch i==j-Paare ausgegeben.

    Returns:
        DataFrame mit Spalten Customer_ID_1, Customer_ID_2, dissimilarity.

    Raises:
        ValueError: Wenn die Matrix nicht quadratisch ist oder
                    fallback_ids nicht zur Matrixgroesse passt.
    """

    if isinstance(matrix, pd.DataFrame):
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError(f"DataFrame ist nicht quadratisch: shape={matrix.shape}")
        values = matrix.to_numpy()
        ids = matrix.index.astype(str).to_numpy()
    else:
        values = np.asarray(matrix)
        if values.ndim != 2 or values.shape[0] != values.shape[1]:
            raise ValueError(f"Kein quadratisches 2D-Array: shape={values.shape}")
        n = values.shape[0]
        if fallback_ids is not None:
            if len(fallback_ids) != n:
                raise ValueError(
                    f"fallback_ids hat {len(fallback_ids)} Eintraege, "
                    f"Matrix aber {n}x{n}."
                )
            ids = fallback_ids.astype(str)
        else:
            ids = np.arange(n).astype(str)

    k = 0 if include_diagonal else 1
    i_idx, j_idx = np.triu_indices(len(ids), k=k)

    return pd.DataFrame({
        "Customer_ID_1": pd.array(ids[i_idx], dtype="string"),
        "Customer_ID_2": pd.array(ids[j_idx], dtype="string"),
        "dissimilarity":  values[i_idx, j_idx].astype(np.float64, copy=False),
    })


def convert_all(
    grid_root: Path = GRID_OUTPUT,
    *,
    id_source: Path = DF_DATA,
    id_col: str = _DEFAULT_ID_COL,
    include_diagonal: bool = False,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Konvertiert alle Square-PKLs unter grid_root ins Long-Format.

    Fuer jeden Unterordner (NAICS/, HS/, APP/) wird ein Ordner long/
    angelegt. Eine registry_long.csv wird pro Unterordner gespeichert.

    Args:
        grid_root:        Wurzelverzeichnis der Grid-Outputs.
        id_source:        Master-Pickle mit Kunden-IDs.
        id_col:           Name der ID-Spalte.
        include_diagonal: Wenn True, werden auch i==j-Paare ausgegeben.
        overwrite:        Wenn True, werden bestehende Long-Dateien ueberschrieben.

    Returns:
        Registry-DataFrame mit Spalten source_file, long_file, status, n_rows.
    """
    grid_root = grid_root.resolve()
    if not grid_root.exists():
        raise FileNotFoundError(f"grid_root existiert nicht: {grid_root}")

    fallback_ids = _load_ids(id_source, id_col)
    rows: list[dict] = []

    for src in _iter_square_pickles(grid_root):
        dst = _make_output_path(src)

        if dst.exists() and not overwrite:
            logger.info("Uebersprungen (existiert): %s", src.name)
            rows.append({"source_file": str(src), "long_file": str(dst), "status": "skipped"})
            continue

        try:
            obj = pd.read_pickle(src)
            long_df = square_to_long(
                obj,
                fallback_ids=fallback_ids,
                include_diagonal=include_diagonal,
            )
        except Exception as exc:
            logger.warning("Uebersprungen (Fehler): %s — %s", src.name, exc)
            rows.append({"source_file": str(src), "long_file": None, "status": f"error: {exc}"})
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        long_df.to_pickle(dst, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info("[%s] %s -> %s  (%d Zeilen)",
                     src.parent.name, src.name, dst.name, len(long_df))

        rows.append({
            "source_file": str(src),
            "long_file":   str(dst),
            "status":      "ok",
            "n_rows":      len(long_df),
        })

    registry = pd.DataFrame(rows)

    if not registry.empty:
        for folder, group in registry.groupby(
            registry["long_file"].apply(lambda p: Path(p).parent if p else None)
        ):
            if folder is None:
                continue
            reg_path = Path(folder) / "registry_long.csv"
            group.to_csv(reg_path, index=False)
            logger.info("Registry gespeichert: %s", reg_path)

    logger.info("Fertig: %d konvertiert, %d uebersprungen, %d Fehler",
                 sum(r["status"] == "ok" for r in rows),
                 sum(r["status"] == "skipped" for r in rows),
                 sum(r["status"].startswith("error") for r in rows))

    return registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    registry = convert_all(overwrite=False)
    print("\nRegistry:")
    print(registry)
