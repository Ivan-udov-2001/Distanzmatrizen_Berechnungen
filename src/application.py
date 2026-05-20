from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Set, Dict, List
from numba import njit, prange

from src.set_metrics import (
    _parse_cell,
    _pairwise_dist_core,
)
from src.core_metrics import _alloc_pairs


# ------------------------------------------------------------
# Application-Code Distanzfunktionen (Set-Engine und Punkt-Engine)
# ------------------------------------------------------------

# Produkt-zu-Applikation Mapping: definiert relevante AM-Codes pro Produkt.
# Wird von pairwise_app_dist_slim zum Filtern verwendet.

PRODUCT_APP_FILTER: Dict[str, set[str]] = {
    "0601": {"2.1", "2.5", "2.6", "2.7", "3.1", "3.3", "5.6"},
    "0602": {"1.1", "1.2", "1.3", "1.4", "1.7", "2.5", "3.1"},
    "0604": {"2.1", "2.2", "2.3", "2.6", "3.1", "3.2", "3.3", "4.4", "5.6"},
    "0201": {"1.9", "1.10", "1.11", "1.12", "2.1", "2.2", "2.3", "2.4", "3.1", "3.2", "5.6"},
}


def _parse_app_part(code: str):
    """
    Extrahiert den Applikationsteil (A.B) aus einem AM-Code.

    Akzeptiert vollstaendige Codes ('1.5_2.4.5') und partielle Codes ('6.1_' oder '6.1').
    Der Material-Teil nach '_' wird ignoriert.

    Rueckgabe:
        Tupel (A, B) bei gueltigem Code, z.B. ('1', '5') fuer '1.5_2.4.5'.
        None wenn der Code nicht geparst werden kann.
    """
    s = str(code).strip()

    if "_" in s:
        s = s.split("_", 1)[0]

    parts = s.split(".")
    if len(parts) != 2:
        return None

    A = parts[0].strip()
    B = parts[1].strip()

    if not A or not B:
        return None

    return A, B


def _get_or_create_id(m, key):
    """
    Kodiert APP-Code Teile (A, B) als Integer fuer die Numba-Berechnung.
    Neue Eintraege erhalten automatisch die naechste freie ID.
    """
    v = m.get(key)
    if v is None:
        v = len(m) + 1  # start at 1
        m[key] = v
    return v


def _build_app_only_sets(df: pd.DataFrame, id_col: str, am_col: str,
                         allowed_apps: set[str] | None = None) -> Dict[str, Set[str]]:
    """
    Erstellt ein Dictionary von Kunden-ID zu App-only Code-Sets.
    Der Material-Teil der AM-Codes wird ignoriert.
    """

    code_sets: Dict[str, Set[str]] = {}
    for customer_id, am_value in (df[[id_col, am_col]].dropna(subset=[id_col]).itertuples(index=False, name=None)):
        customer_id = str(customer_id)
        parsed_codes = _parse_cell(am_value)
        app_only = set()
        for code in parsed_codes:
            parts = _parse_app_part(code)
            if parts is not None:
                app_code = f"{parts[0]}.{parts[1]}"
                if allowed_apps is None or app_code in allowed_apps:
                    app_only.add(app_code)
        code_sets[customer_id] = app_only
    return code_sets


def _build_app_flat(df: pd.DataFrame, col: str,
                   customer_col: str = "customer_code",
                   allowed_apps: set[str] | None = None):
    """
    Baut die Flat-Darstellung fuer Cartesian, Chamfer und kNN auf.

    Jeder AM-Code wird als (A, B) Integer-Paar gespeichert,
    sodass Numba direkt damit rechnen kann.

    Rueckgabe: customer_ids, flat_codes, offsets, id_mappings
    """

    df = df.copy()
    df["appmat_list"] = df[col].apply(_parse_cell)
    customer_ids = df[customer_col].astype(str).to_numpy()
    N = len(df)

    lists = df["appmat_list"].to_list()

    parsed_lists = []
    for code_list in lists:
        seen = set()
        unique_pairs = []
        for code in code_list:
            parts = _parse_app_part(code)
            if parts is not None:
                A, B = parts
                if allowed_apps is not None and f"{A}.{B}" not in allowed_apps:
                    continue
                key = (A, B)
                if key not in seen:
                    seen.add(key)
                    unique_pairs.append((A, B))
        parsed_lists.append(unique_pairs)

    lengths = np.array([len(x) for x in parsed_lists], dtype=np.int32)
    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    total = int(offsets[-1])
    flat_codes = np.zeros((total, 2), dtype=np.int32)

    id_map_A, id_map_B = {}, {}
    idx = 0
    for pair_list in parsed_lists:
        for A, B in pair_list:
            flat_codes[idx, 0] = _get_or_create_id(id_map_A, A)
            flat_codes[idx, 1] = _get_or_create_id(id_map_B, B)
            idx += 1
    return customer_ids, flat_codes, offsets, {"A": id_map_A, "B": id_map_B}


_SQRT2 = 1.41421356     # math.sqrt(2), wird als poly_concave Gewichtung fuer A verwendet

@njit(inline="always")
def _app_dist_flexible(code1, code2, metric_id, ws_id, gating):
    """
    Punkt-Distanzfunktion fuer zwei AM-Codes als (A, B) Integer-Paare.

    metric_id:
        0 = exact     -> 0.0 nur wenn A und B identisch
        1 = hamming   -> mittlere Abweichung ueber A und B
        2 = w_hamming -> gewichtetes Hamming (A staerker gewichtet)
        3 = block     -> feste Gewichte (3.0 fuer A, 1.0 fuer B)
        4 = lcp       -> laengster gemeinsamer Praefix

    gating: True = kein gemeinsames A -> sofort 1.0
    """
    if code1[0] == 0 or code1[1] == 0:
        return 1.0
    if code2[0] == 0 or code2[1] == 0:
        return 1.0

    m0 = 0 if code1[0] == code2[0] else 1
    m1 = 0 if code1[1] == code2[1] else 1

    if gating and m0 == 1:
        return 1.0

    if metric_id == 0:
        return 0.0 if (m0 == 0 and m1 == 0) else 1.0
    if metric_id == 1:
        return (m0 + m1) / 2.0
    if metric_id == 2:
        if ws_id == 0:
            w0, w1 = 2.0, 1.0
        elif ws_id == 1:
            w0, w1 = 1.0, 0.5
        elif ws_id == 2:
            w0, w1 = 4.0, 1.0
        else:
            w0, w1 = _SQRT2, 1.0
        return (w0 * m0 + w1 * m1) / (w0 + w1)
    if metric_id == 3:
        w0, w1 = 3.0, 1.0
        return (w0 * m0 + w1 * m1) / (w0 + w1)
    if metric_id == 4:
        if m0 == 1:
            return 1.0
        if m1 == 1:
            return 0.5
        return 0.0
    return 0.0


@njit
def _cartesian_mean_app(flat_codes, offsets, i, j,
                       metric_id, ws_id, gating, empty_value=1.0):
    """
    Mittlere Distanz zwischen allen AM-Code-Paaren zweier Kunden.
    """

    a0 = offsets[i]; a1 = offsets[i + 1]
    b0 = offsets[j]; b1 = offsets[j + 1]
    na = a1 - a0
    nb = b1 - b0
    if na == 0 or nb == 0:
        return empty_value
    total = 0.0
    for ii in range(a0, a1):
        c1 = flat_codes[ii]
        for jj in range(b0, b1):
            total += _app_dist_flexible(c1, flat_codes[jj], metric_id, ws_id, gating)
    return total / (na * nb)



@njit(parallel=True)
def _all_pairs_cartesian_app(flat_codes, offsets, metric_id, ws_id, gating):
    """
    Berechnet cartesian_mean_app fuer alle Kunden-Paare parallel.
    """

    N = len(offsets) - 1
    out_i, out_j, out_d, start = _alloc_pairs(N)

    for i in prange(N - 1):
        k = start[i]
        for j in range(i + 1, N):
            out_i[k] = i
            out_j[k] = j
            out_d[k] = _cartesian_mean_app(flat_codes, offsets, i, j,
                                          metric_id, ws_id, gating)
            k += 1
    return out_i, out_j, out_d


@njit
def _chamfer_symmetric_app(flat_codes, offsets, i, j,
                          metric_id, ws_id, gating, empty_value=1.0):
    """
    Symmetrische Chamfer-Distanz fuer AM-Codes.
    """

    a0 = offsets[i]; a1 = offsets[i + 1]
    b0 = offsets[j]; b1 = offsets[j + 1]
    na = a1 - a0
    nb = b1 - b0
    if na == 0 or nb == 0:
        return empty_value

    sum_min_a = 0.0
    for ii in range(a0, a1):
        c1 = flat_codes[ii]
        best = 1.0
        for jj in range(b0, b1):
            d = _app_dist_flexible(c1, flat_codes[jj], metric_id, ws_id, gating)
            if d < best:
                best = d
                if best == 0.0:
                    break
        sum_min_a += best
    mean_a = sum_min_a / na

    sum_min_b = 0.0
    for jj in range(b0, b1):
        c2 = flat_codes[jj]
        best = 1.0
        for ii in range(a0, a1):
            d = _app_dist_flexible(c2, flat_codes[ii], metric_id, ws_id, gating)
            if d < best:
                best = d
                if best == 0.0:
                    break
        sum_min_b += best
    mean_b = sum_min_b / nb

    return 0.5 * (mean_a + mean_b)


@njit(parallel=True)
def _all_pairs_chamfer_app(flat_codes, offsets, metric_id, ws_id, gating):
    """
    Berechnet chamfer_symmetric_app fuer alle Kunden-Paare parallel.
    """

    N = len(offsets) - 1
    out_i, out_j, out_d, start = _alloc_pairs(N)

    for i in prange(N - 1):
        k = start[i]
        for j in range(i + 1, N):
            out_i[k] = i
            out_j[k] = j
            out_d[k] = _chamfer_symmetric_app(flat_codes, offsets, i, j,
                                             metric_id, ws_id, gating)
            k += 1
    return out_i, out_j, out_d


@njit
def _topk_mean_to_other_app(flat_codes, a0, a1, b0, b1, k,
                            metric_id, ws_id, gating, empty_value=1.0):
    """
    Mittlerer Abstand der k naechsten Nachbarn von Menge A zu Menge B.
    """

    na = a1 - a0
    nb = b1 - b0
    if na == 0 or nb == 0:
        return empty_value

    kk = k
    if kk > nb:
        kk = nb

    total = 0.0
    for ii in range(a0, a1):
        c1 = flat_codes[ii]
        best = np.empty(kk, dtype=np.float32)
        for t in range(kk):
            best[t] = 1.0

        for jj in range(b0, b1):
            d = _app_dist_flexible(c1, flat_codes[jj], metric_id, ws_id, gating)
            if d >= best[kk - 1]:
                continue
            pos = kk - 1
            while pos > 0 and d < best[pos - 1]:
                best[pos] = best[pos - 1]
                pos -= 1
            best[pos] = d
            if kk == 1 and best[0] == 0.0:
                break

        s = 0.0
        for t in range(kk):
            s += best[t]
        total += (s / kk)
    return total / na


@njit
def _knn_mean_symmetric_app(flat_codes, offsets, i, j, k,
                           metric_id, ws_id, gating, empty_value=1.0):
    """
    Symmetrischer kNN-Mittelwert: Durchschnitt aus beiden Richtungen A->B und B->A.
    """

    a0 = offsets[i]; a1 = offsets[i + 1]
    b0 = offsets[j]; b1 = offsets[j + 1]
    na = a1 - a0
    nb = b1 - b0
    if na == 0 or nb == 0:
        return empty_value
    mean_a = _topk_mean_to_other_app(flat_codes, a0, a1, b0, b1, k,
                                     metric_id, ws_id, gating, empty_value)
    mean_b = _topk_mean_to_other_app(flat_codes, b0, b1, a0, a1, k,
                                     metric_id, ws_id, gating, empty_value)
    return 0.5 * (mean_a + mean_b)


@njit(parallel=True)
def _all_pairs_knn_mean_app(flat_codes, offsets, k, metric_id, ws_id, gating):
    """
    Berechnet knn_mean_symmetric_app fuer alle Kunden-Paare parallel.
    """

    N = len(offsets) - 1
    out_i, out_j, out_d, start = _alloc_pairs(N)

    for i in prange(N - 1):
        kk = start[i]
        for j in range(i + 1, N):
            out_i[kk] = i
            out_j[kk] = j
            out_d[kk] = _knn_mean_symmetric_app(flat_codes, offsets, i, j, k,
                                               metric_id, ws_id, gating)
            kk += 1
    return out_i, out_j, out_d


def pairwise_app_dist_slim(
    df: pd.DataFrame,
    *,
    id_col: str = "customer_code",
    am_col: str = "application_material_set_h2",
    metric: str = "cartesian_mean",
    output: str = "square",
    weights: dict[str, float] | None = None,
    k: int = 3,
    inner_metric: str = "hamming",
    weight_scheme: str = "linear",
    gating: bool = False,
    product_filter: str | None = None,
) -> pd.DataFrame:
    """
    Berechnet paarweise Distanzen zwischen Kunden basierend auf AM-Code-Sets.
    Der Material-Teil wird ignoriert, nur der Applikationsteil (A.B) wird verwendet.

    metric:         Set-Engine:   jaccard, overlap, dice, + weighted Varianten
                    Punkt-Engine: cartesian_mean, chamfer, knn_mean

    inner_metric:   Punkt-Distanzfunktion (nur fuer Punkt-Engine)
                    'exact' | 'hamming' | 'w_hamming' | 'block_weights' | 'lcp'

    weight_scheme:  Gewichtungsschema fuer w_hamming
                    'linear' | 'exponential' | 'poly_convex' | 'poly_concave'

    product_filter: Filtert AM-Codes auf produktspezifische Applikationen
    """

    SET_METRICS    = {"jaccard", "jaccard_weighted",
                      "overlap", "overlap_weighted",
                      "dice", "dice_weighted"}

    PREFIX_METRICS = {"cartesian_mean", "chamfer", "knn_mean"}

    metric_l = metric.lower()

    # Eingaben validieren
    if output not in {"long", "square"}:
        raise ValueError("output muss 'long' oder 'square' sein")

    if metric_l not in SET_METRICS | PREFIX_METRICS:
        raise ValueError(
            f"Unbekannte Metrik '{metric}'. "
            f"Erwartet: {sorted(SET_METRICS | PREFIX_METRICS)}"
        )

    # Produktfilter aufloesen
    if product_filter is not None:
        product_filter = str(product_filter)
        if product_filter not in PRODUCT_APP_FILTER:
            raise ValueError(
                f"Unbekanntes Produkt '{product_filter}'. "
                f"Verfuegbar: {sorted(PRODUCT_APP_FILTER.keys())}"
            )
        allowed_apps = PRODUCT_APP_FILTER[product_filter]
    else:
        allowed_apps = None

    # Set-Engine
    if metric_l in SET_METRICS:
        sets = _build_app_only_sets(df, id_col=id_col, am_col=am_col, allowed_apps=allowed_apps)
        return _pairwise_dist_core(sets, metric_l, output, weights)

    # Punkt-Engine
    _inner_metric_map = {
        "exact":         0,
        "hamming":       1,
        "w_hamming":     2,
        "block_weights": 3,
        "lcp":           4,
    }
    _ws_map = {
        "linear":         0,
        "exponential":    1,
        "poly_convex":    2,
        "poly_concave":   3,
    }

    inner_metric_l = inner_metric.lower()
    weight_scheme_l = weight_scheme.lower()

    if inner_metric_l not in _inner_metric_map:
        raise ValueError(f"inner_metric muss eines von {sorted(_inner_metric_map)} sein")
    if weight_scheme_l not in _ws_map:
        raise ValueError(f"weight_scheme muss eines von {sorted(_ws_map)} sein")

    metric_id = _inner_metric_map[inner_metric_l]
    ws_id = _ws_map[weight_scheme_l]

    # Daten vorbereiten
    customer_ids, flat_codes, offsets, _ = _build_app_flat(
        df, col=am_col, customer_col=id_col, allowed_apps=allowed_apps
    )

    # Aggregation
    if metric_l == "cartesian_mean":
        i_idx, j_idx, dist = _all_pairs_cartesian_app(
            flat_codes, offsets, metric_id, ws_id, gating)
    elif metric_l == "chamfer":
        i_idx, j_idx, dist = _all_pairs_chamfer_app(
            flat_codes, offsets, metric_id, ws_id, gating)
    else:
        i_idx, j_idx, dist = _all_pairs_knn_mean_app(
            flat_codes, offsets, k, metric_id, ws_id, gating)

    # Ausgabe
    if output == "square":
        N = len(df)
        distance_matrix = np.zeros((N, N), dtype=np.float32)
        distance_matrix[i_idx, j_idx] = dist
        distance_matrix[j_idx, i_idx] = dist
        return pd.DataFrame(distance_matrix, index=customer_ids, columns=customer_ids)

    return pd.DataFrame({
        "Customer_ID_1": customer_ids[i_idx],
        "Customer_ID_2": customer_ids[j_idx],
        "dissimilarity": dist,
    })