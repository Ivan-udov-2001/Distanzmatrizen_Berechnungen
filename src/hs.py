from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Set, Dict, List
from numba import njit, prange

from src.core_metrics import (
    _lcp_dist,
    _hamming_gated,
    _w_hamming_flexible,
    _block_weighted_hamming,
    _alloc_pairs,
)
from src.set_metrics import (
    _parse_cell,
    _idf_weights,
    _pairwise_dist_core,
    _IDF_CLIP_MAX,
)


# -----------------------------------------------------------
# HS-Code Distanzfunktionen (Set-Engine und Punkt-Engine)
# -----------------------------------------------------------

def _normalize_hs_codes(items: List[str], level: int) -> Set[str]:
    """
    Normalisiert HS-Codes auf eine feste Ziffernanzahl.

    Nicht-Ziffern werden entfernt, zu kurze Codes werden
    rechts mit Nullen aufgefuellt und auf ein level Stellen gekuerzt.
    Wird von Set-Engine (Jaccard, Dice, Overlap) verwendet.
    """

    normalized_set: Set[str] = set()
    for item in items:
        digits = "".join(ch for ch in item if ch.isdigit())
        if digits:
            normalized_code = (digits + "0" * level)[:level]
            normalized_set.add(normalized_code)
    return normalized_set


def _normalize_hs6_int(code: str) -> int:
    """
    Normalisiert einen HS-Code auf 6 Stellen als Integer.
    Wird von Cartesian, Chamfer und kNN verwendet.

    - Nicht Ziffern werden entfernt
    - Zu kurze Codes werden rechts mit Nullen aufgefuellt
    - Zu lange Codes werden auf 6 Stellen gekuerzt
    - Rueckgabe: -1 bei ungueltigem oder leerem Code
    """

    s = "".join(ch for ch in str(code) if ch.isdigit())
    if not s:
        return -1

    s = s.ljust(6, "0")[:6]
    return int(s)


def _normalize_hs6_digits(code_int: int, level: int = 6) -> np.ndarray:
    """
    Wandelt einen normalisierten HS-Code in ein uint8-Ziffern-Array um.

    Beispiel: 850420 -> [8, 5, 0, 4, 2, 0]
    Ungueltige Codes (<0) -> [0, 0, 0, 0, 0, 0]
    """

    if code_int < 0:
        return np.zeros(level, dtype=np.uint8)
    s = str(code_int).zfill(6)[:level]
    return np.array([int(c) for c in s], dtype=np.uint8)



def _build_flat_representation(df: pd.DataFrame, level: int = 6) -> tuple:
    """
    Baut die Flat-Darstellung fuer Cartesian, Chamfer und kNN auf.

    Jeder HS-Code wird als uint8-Ziffern-Array der Laenge level gespeichert,
    sodass die Hamming-Funktionen aus core_metrics direkt angewendet werden koennen.

    Rueckgabe:
        flat    -- Array der Form (total, level), dtype uint8
        offsets -- Offset-Array der Laenge N+1 fuer den Zugriff per Kunde
    """

    lists = df["hs6_list"].to_list()
    N = len(lists)

    lengths = np.array([len(x) for x in lists], dtype=np.int32)

    offsets = np.zeros(N + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)

    total = int(offsets[-1])
    flat = np.zeros((total, level), dtype=np.uint8)

    k = 0
    for x in lists:
        for code_int in x:
            flat[k] = _normalize_hs6_digits(code_int, level=level)
            k += 1

    return flat, offsets

def _prepare_hs6_dataframe(df_master: pd.DataFrame,
                          id_col: str = "input_id",
                          hs_col: str = "hs6_code"
) -> pd.DataFrame:
    """
    Bereitet einen DataFrame fuer die HS-Code Distanzberechnung vor.

    - Kunden-ID einheitlich als String speichern
    - HS-Codes werden geparst, normalisiert und dedupliziert
    - Ungueltige Codes werden herausgefiltert
    """

    df = df_master.copy()
    df[id_col] = df[id_col].astype(str)
    df["hs6_list"] = df[hs_col].apply(_parse_cell)

    df["hs6_list"] = df["hs6_list"].apply(
        lambda xs: sorted({
            norm for x in xs
            if (norm := _normalize_hs6_int(x)) >= 0
        })
    )

    return df


def _build_hs_sets(df: pd.DataFrame, level: int = 6,
                  id_col: str = "customer_id",
                  hs_col: str = "hs_code") -> Dict[str, Set[str]]:
    """
    Erstellt ein Dictionary von Kunden-ID zu normalisierten HS-Codes-Sets.
    Wird von Set-Engine (Jaccard, Dice, Overlap) verwendet.
    """
    code_sets: Dict[str, Set[str]] = {}
    for customer_id, hs_value in (df[[id_col, hs_col]].dropna(subset=[id_col]).itertuples(index=False, name=None)):
        customer_id = str(customer_id)
        parsed_codes = _parse_cell(hs_value)
        code_sets[customer_id] = _normalize_hs_codes(parsed_codes, level)
    return code_sets


@njit(inline="always")
def _hs6_dissim_general(a_row: np.ndarray, b_row: np.ndarray,
                       metric_id: int, weight_scheme: int, gating: bool) -> float:
    """
    Berechnet die Distanz zwischen zwei HS-Codes als uint8-Ziffern-Arrays.
    Wird intern von Cartesian, Chamfer und kNN aufgerufen.

    metric_id:
        0 = LCP           -> kein gemeinsames Kapitel (Stellen 0-1) -> 1.0
        1 = Hamming       -> _hamming_gated
        2 = Gew. Hamming  -> _w_hamming_flexible
        3 = Block         -> _block_weighted_hamming

    weight_scheme (nur fuer metric_id=2):
        0=linear,
        1=exponential,
        2=poly_convex,
        3=poly_concave

    gating (fuer metric_id = 1, 2, 3):
        True  = erste 2 Ziffern muessen uebereinstimmen, sonst 1.0
        False = alle Stellen werden verglichen

    Alle Distanzen sind normalisiert auf [0, 1].
    """

    if metric_id == 0:
        if a_row[0] != b_row[0] or a_row[1] != b_row[1]:
            return 1.0
        return _lcp_dist(a_row, b_row, True)

    elif metric_id == 1:
        return _hamming_gated(a_row, b_row, True, gating)

    elif metric_id == 2:
        return _w_hamming_flexible(a_row, b_row, True, gating, weight_scheme)

    else:
        return _block_weighted_hamming(a_row, b_row, True, gating)


@njit
def _cartesian_mean(flat, offsets, i, j, metric_id, weight_scheme, gating):
    """
    Mittlere Distanz zwischen allen Code-Paaren zweier Kunden.
    """

    a0 = offsets[i]; a1 = offsets[i + 1]
    b0 = offsets[j]; b1 = offsets[j + 1]

    na = a1 - a0
    nb = b1 - b0

    if na == 0 or nb == 0:
        return 1.0

    total = 0.0

    for ii in range(a0, a1):
        for jj in range(b0, b1):
            total += _hs6_dissim_general(flat[ii], flat[jj], metric_id, weight_scheme, gating)

    return total / (na * nb)


@njit(parallel=True)
def _all_pairs_cartesian(flat, offsets, metric_id, weight_scheme, gating):
    """
    Berechnet cartesian_mean fuer alle Kunden-Paare parallel.
    """

    N = len(offsets) - 1
    out_i, out_j, out_d, start = _alloc_pairs(N)

    for i in prange(N - 1):
        k = start[i]
        for j in range(i + 1, N):
            out_i[k] = i
            out_j[k] = j
            out_d[k] = _cartesian_mean(flat, offsets, i, j, metric_id, weight_scheme, gating)
            k += 1

    return out_i, out_j, out_d


@njit
def _chamfer_symmetric(flat, offsets, i, j, metric_id, weight_scheme, gating):
    """
    Symmetrische Chamfer-Distanz: Mittel aus minimalem Abstand in beide Richtungen.
    """

    a0 = offsets[i]; a1 = offsets[i + 1]
    b0 = offsets[j]; b1 = offsets[j + 1]
    na = a1 - a0
    nb = b1 - b0

    if na == 0 or nb == 0:
        return 1.0

    # A -> B
    sum_min_a = 0.0
    for ii in range(a0, a1):
        best = 1.0
        for jj in range(b0, b1):
            d = _hs6_dissim_general(flat[ii], flat[jj], metric_id, weight_scheme, gating)
            if d < best:
                best = d
                if best == 0.0:
                    break
        sum_min_a += best
    mean_a = sum_min_a / na

    # B -> A
    sum_min_b = 0.0
    for jj in range(b0, b1):
        best = 1.0
        for ii in range(a0, a1):
            d = _hs6_dissim_general(flat[jj], flat[ii], metric_id, weight_scheme, gating)
            if d < best:
                best = d
                if best == 0.0:
                    break
        sum_min_b += best
    mean_b = sum_min_b / nb

    return 0.5 * (mean_a + mean_b)


@njit(parallel=True)
def _all_pairs_chamfer(flat, offsets, metric_id, weight_scheme, gating):
    """
    Berechnet chamfer_symmetric fuer alle Kunden-Paare parallel.
    """

    N = len(offsets) - 1
    out_i, out_j, out_d, start = _alloc_pairs(N)

    for i in prange(N - 1):
        k = start[i]
        for j in range(i + 1, N):
            out_i[k] = i
            out_j[k] = j
            out_d[k] = _chamfer_symmetric(flat, offsets, i, j, metric_id, weight_scheme, gating)
            k += 1

    return out_i, out_j, out_d


@njit
def _topk_mean_to_other(flat, a0, a1, b0, b1, k, metric_id, weight_scheme, gating):
    """
    Mittlerer Abstand der k naechsten Nachbarn von Menge A zu Menge B.
    """

    na = a1 - a0
    nb = b1 - b0
    if na == 0 or nb == 0:
        return 1.0

    kk = k
    if kk > nb:
        kk = nb

    total = 0.0

    for ii in range(a0, a1):
        best = np.empty(kk, dtype=np.float32)
        for t in range(kk):
            best[t] = 1.0

        for jj in range(b0, b1):
            d = _hs6_dissim_general(flat[ii], flat[jj], metric_id, weight_scheme, gating)

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
def _knn_mean_symmetric(flat, offsets, i, j, k, metric_id, weight_scheme, gating):
    """
    Symmetrischer kNN-Mittelwert: Durchschnitt aus beiden Richtungen A->B und B->A.
    """

    a0 = offsets[i]; a1 = offsets[i + 1]
    b0 = offsets[j]; b1 = offsets[j + 1]
    na = a1 - a0
    nb = b1 - b0

    if na == 0 or nb == 0:
        return 1.0

    mean_a = _topk_mean_to_other(flat, a0, a1, b0, b1, k, metric_id, weight_scheme, gating)
    mean_b = _topk_mean_to_other(flat, b0, b1, a0, a1, k, metric_id, weight_scheme, gating)

    return 0.5 * (mean_a + mean_b)


@njit(parallel=True)
def _all_pairs_knn_mean(flat, offsets, k, metric_id, weight_scheme, gating):
    """
    Berechnet knn_mean_symmetric fuer alle Kunden-Paare parallel.
    """

    N = len(offsets) - 1
    out_i, out_j, out_d, start = _alloc_pairs(N)

    for i in prange(N - 1):
        kk = start[i]
        for j in range(i + 1, N):
            out_i[kk] = i
            out_j[kk] = j
            out_d[kk] = _knn_mean_symmetric(flat, offsets, i, j, k, metric_id, weight_scheme, gating)
            kk += 1

    return out_i, out_j, out_d


def pairwise_hs_dist_slim(
    df: pd.DataFrame,
    *,
    id_col: str = "customer_code",
    hs_col: str = "hs6_code",
    level: int = 6,
    metric: str = "cartesian_mean",
    output: str = "square",
    weights: dict[str, float] | None = None,
    k: int = 3,
    inner_metric: str = "hamming",
    weight_scheme: str = "linear",
    gating: bool = True,
) -> pd.DataFrame:
    """
    Berechnet paarweise Distanzen zwischen Kunden basierend auf HS-Code-Sets.

    metric:        Aggregations-Strategie
                   Set-Engine:   jaccard, overlap, dice, + weighted Varianten
                   Punkt-Engine: cartesian_mean, chamfer, knn_mean

    inner_metric:  Punkt-Distanzfunktion (nur fuer cartesian_mean, chamfer, knn_mean)
                   'lcp'           -> Longest Common Prefix
                   'hamming'       -> Hamming-Distanz
                   'w_hamming'     -> Gewichtetes Hamming
                   'block_weights' -> Block-gewichtetes Hamming

    weight_scheme: Gewichtungsschema (nur fuer inner_metric='w_hamming')
                   'linear' | 'exponential' | 'poly_convex' | 'poly_concave'

    gating:        Erste 2 Ziffern (Kapitel) muessen uebereinstimmen, sonst 1.0
                   Nur relevant fuer Hamming-Varianten. (LCP hat immer Kapitel-Check)
    """

    SET_METRICS         = {"jaccard", "jaccard_weighted",
                           "overlap", "overlap_weighted",
                           "dice", "dice_weighted"}

    AGGREGATION_METRICS = {"cartesian_mean", "chamfer", "knn_mean"}

    metric_l = metric.lower()

    if output not in {"long", "square"}:
        raise ValueError("output muss 'long' oder 'square' sein")

    if metric_l not in SET_METRICS | AGGREGATION_METRICS:
        raise ValueError(
            f"Unbekannte Metrik '{metric}'. "
            f"Erwartet {sorted(SET_METRICS | AGGREGATION_METRICS)}"
        )

    # Set-Engine
    if metric_l in SET_METRICS:
        sets = _build_hs_sets(df, level=level, id_col=id_col, hs_col=hs_col)
        idf = weights if weights is not None else _idf_weights(sets, clip_max=_IDF_CLIP_MAX)
        return _pairwise_dist_core(sets, metric_l, output, idf)

    # Punkt-Engine
    inner_metric_l = inner_metric.lower()
    _inner_metric_map = {
        "lcp":           0,
        "hamming":       1,
        "w_hamming":     2,
        "block_weights": 3,
    }
    if inner_metric_l not in _inner_metric_map:
        raise ValueError(
            f"inner_metric muss eines von {sorted(_inner_metric_map)} sein"
        )
    metric_id = _inner_metric_map[inner_metric_l]

    _ws_map = {
        "linear":       0,
        "exponential":  1,
        "poly_convex":  2,
        "poly_concave": 3,
    }
    if weight_scheme.lower() not in _ws_map:
        raise ValueError(
            f"weight_scheme muss eines von {sorted(_ws_map)} sein"
        )
    ws_id = _ws_map[weight_scheme.lower()]

    # Daten vorbereiten
    df_prep = _prepare_hs6_dataframe(df, id_col=id_col, hs_col=hs_col)
    ids = df_prep[id_col].to_numpy()
    flat, offsets = _build_flat_representation(df_prep, level=level)

    # Aggregation
    if metric_l == "cartesian_mean":
        i_idx, j_idx, dist = _all_pairs_cartesian(flat, offsets, metric_id, ws_id, gating)
    elif metric_l == "chamfer":
        i_idx, j_idx, dist = _all_pairs_chamfer(flat, offsets, metric_id, ws_id, gating)
    elif metric_l == "knn_mean":
        i_idx, j_idx, dist = _all_pairs_knn_mean(flat, offsets, k, metric_id, ws_id, gating)

    # Ausgabe
    if output == "square":
        N = len(df_prep)
        distance_matrix = np.zeros((N, N), dtype=np.float32)
        distance_matrix[i_idx, j_idx] = dist
        distance_matrix[j_idx, i_idx] = dist
        return pd.DataFrame(distance_matrix, index=ids, columns=ids)

    return pd.DataFrame({
        "Customer_ID_1": ids[i_idx],
        "Customer_ID_2": ids[j_idx],
        "dissimilarity": dist,
    })
