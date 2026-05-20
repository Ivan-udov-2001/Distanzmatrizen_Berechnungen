"""
Grid-Search Wrapper fuer die Distanzberechnungen.
Erzeugt systematisch Distanzmatrizen ueber alle Parameter-Kombinationen
und speichert sie als .pkl mit sprechenden Dateinamen und Registry.
"""

from pathlib import Path
import pickle
import hashlib
import json
import pandas as pd
import logging

from sklearn.model_selection import ParameterGrid
from dataclasses import dataclass, field
from typing import Callable
from src.paths import GRID_OUTPUT
from src.naics import pairwise_naics_dist_slim
from src.hs import pairwise_hs_dist_slim
from src.application import pairwise_app_dist_slim

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────────
# Hilfsfunktionen
# ────────────────────────────────────────────────────────────────────────

def _signature(params: dict) -> str:
    """
    Kurze Hash-Signatur fuer Deduplizierung,
    gleiche Parameter ergeben gleichen Hash.
    """
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.md5(s.encode()).hexdigest()[:8]


def _clean_param_name(s: str) -> str:
    """
    Ersetzt Unterstriche durch Bindestriche fuer Dateinamen.
    """
    return str(s).replace("_", "-")


@dataclass(frozen=True)
class NameSchema:
    prefix: str
    keys: tuple[str, ...]
    formatters: dict[str, Callable] = field(default_factory=dict)

_SCHEMAS: dict[str, NameSchema] = {
    "naics": NameSchema(
        prefix="naics",
        keys=("metric", "weight_scheme", "alpha_penalty",
              "use_primary_only", "n_digits", "output"),
        formatters={
            "alpha_penalty":    lambda v: f"a{v:.2f}".replace(".", "p"),
            "use_primary_only": lambda v: "pp-only" if v else None,
            "n_digits":         lambda v: f"d{v}",
        },
    ),

    "hs": NameSchema(
        prefix="hs",
        keys=("metric", "inner_metric", "weight_scheme", "k", "output"),
        formatters={
            "k":             lambda v: f"k{v}",
        },
    ),
    "app": NameSchema(
        prefix="app",
        keys=("metric", "inner_metric", "weight_scheme",
              "k", "product_filter", "output"),
        formatters={
            "k":             lambda v: f"k{v}",
            "product_filter": lambda v: f"pf{_clean_param_name(v)}" if v is not None else None,
        }
    )
}


def _pretty_name(schema: str, idx: int, params: dict, include_hash: bool = False) -> str:
    """
    Erzeugt einen sprechenden Dateinamen fuer Grid-Matrizen.

    Das Schema bestimmt Praefix und Formatierungsregeln (siehe _SCHEMAS).
    Parameter die nicht im Schema definiert sind, werden ignoriert.
    """

    name_schema = _SCHEMAS[schema]
    parts = [f"{name_schema.prefix}_{idx:03d}"]
    for key in name_schema.keys:
        if key not in params:
            continue
        value = params[key]
        if key in name_schema.formatters:
            formatted = name_schema.formatters[key](value)
            if formatted is not None:
                parts.append(formatted)
        else:
            parts.append(_clean_param_name(str(value)))
    if include_hash:
        parts.append(_signature(params))
    return "_".join(parts)


# ────────────────────────────────────────────────────────────────────────
# Generische Grid-Search-Schleife
# ────────────────────────────────────────────────────────────────────────

def _grid_compute_generic(
    df: pd.DataFrame,
    param_grid,
    *,
    distance_fn: Callable,
    fixed_cols: dict,
    schema: str,
    output: Path,
    include_hash_in_name: bool = False,
) -> pd.DataFrame:
    """
    Generische Grid-Search-Schleife fuer alle Distanzmatrizen.
    """
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)

    combos = list(ParameterGrid(param_grid))
    logger.info("Grid hat %d Kombinationen", len(combos))
    logger.info("Zielordner: %s", out.resolve())

    seen_signatures = set()
    registry_rows = []

    for raw_params in combos:
        sig = _signature(raw_params)
        if sig in seen_signatures:
            logger.debug("[skip duplicate] %s", raw_params)
            continue
        seen_signatures.add(sig)

        idx = len(registry_rows)
        name = _pretty_name(schema, idx, raw_params,
                           include_hash=include_hash_in_name)

        logger.info("[%d/%d] [%s] %s", idx + 1, len(combos), name, raw_params)

        D = distance_fn(df, **fixed_cols, **raw_params)

        pkl_path = out / f"{name}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(D, f, protocol=pickle.HIGHEST_PROTOCOL)

        registry_rows.append({
            "name": name,
            "file": str(pkl_path),
            **raw_params,
        })

    registry = pd.DataFrame(registry_rows)
    registry.to_csv(out / "registry.csv", index=False)

    logger.info("Fertig: %d Matrizen gespeichert", len(registry))
    logger.info("Registry: %s", out / "registry.csv")

    return registry


def grid_compute_hs_matrices(
    df: pd.DataFrame,
    param_grid,
    *,
    id_col: str = "customer_code",
    hs_col: str = "hs6_code",
    output: Path = GRID_OUTPUT / "HS",
    include_hash_in_name: bool = False,
) -> pd.DataFrame:
    """
    Berechnet fuer jede Parameter-Kombination eine HS-Distanzmatrix
    und speichert sie als .pkl im Zielordner.

    param_grid muss 'output' als Parameter enthalten ('square' und/oder 'long').

    Rueckgabe: registry DataFrame [name, file, <parameter...>]
    """
    return _grid_compute_generic(
        df, param_grid,
        distance_fn=pairwise_hs_dist_slim,
        fixed_cols={"id_col": id_col, "hs_col": hs_col},
        schema="hs",
        output=output,
        include_hash_in_name=include_hash_in_name,
    )


def grid_compute_naics_matrices(
    df: pd.DataFrame,
    param_grid,
    *,
    primary_col: str,
    secondary_col: str | None = None,
    id_col: str | None = None,
    output: Path = GRID_OUTPUT / "NAICS",
    include_hash_in_name: bool = False,
) -> pd.DataFrame:
    """
    Berechnet fuer jede Parameter-Kombination eine NAICS-Distanzmatrix
    und speichert sie als .pkl im Zielordner.

    param_grid muss 'output' als Parameter enthalten ('square' und/oder 'long').

    Rueckgabe: registry DataFrame [name, file, <parameter...>]
    """
    return _grid_compute_generic(
        df, param_grid,
        distance_fn=pairwise_naics_dist_slim,
        fixed_cols={
            "primary_col": primary_col,
            "secondary_col": secondary_col,
            "id_col": id_col,
        },
        schema="naics",
        output=output,
        include_hash_in_name=include_hash_in_name,
    )


def grid_compute_app_matrices(
    df: pd.DataFrame,
    param_grid,
    *,
    id_col: str = "customer_code",
    am_col: str = "application_material_set_h2",
    output: Path = GRID_OUTPUT / "APP",
    include_hash_in_name: bool = False,
) -> pd.DataFrame:
    """
    Berechnet fuer jede Parameter-Kombination eine APP-Distanzmatrix
    und speichert sie als .pkl im Zielordner.

    param_grid muss 'output' als Parameter enthalten ('square' und/oder 'long').

    Rueckgabe: registry DataFrame [name, file, <parameter...>]
    """
    return _grid_compute_generic(
        df, param_grid,
        distance_fn=pairwise_app_dist_slim,
        fixed_cols={"id_col": id_col, "am_col": am_col},
        schema="app",
        output=output,
        include_hash_in_name=include_hash_in_name,
    )