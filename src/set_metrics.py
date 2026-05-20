from __future__ import annotations

import math
import numpy as np
import pandas as pd
from collections import Counter
from typing import Set, Dict, List

_IDF_CLIP_MAX = 8.0         # Daempft den Einfluss sehr seltener Codes

# ----------------------------------------------------------------------------------
# Generische Set-Distanzfunktionen fuer HS- und AM-Codes (APP-Codes)
# ----------------------------------------------------------------------------------

def _parse_cell(val) -> List[str]:
    """
    Parst eine Zelle in eine Liste eindeutiger Code-Strings.

    Unterstuetzt Einzelwerte, Listen, Sets, Arrays sowie String-Darstellungen
    davon (z.B. "{'1.1', '2.3'}").
    Duplikate werden entfernt, die Reihenfolge bleibt erhalten.
    """

    if val is None:
        return []

    # Container-Typen direkt iterieren - pd.isna() wuerde bei Arrays ein Array zurueckgeben
    if isinstance(val, (list, tuple, set, np.ndarray)):
        items = []
        for x in val:
            if x is None:
                continue
            sx = str(x).strip().strip("'").strip('"').strip()
            if sx and sx.lower() not in {"nan", "none"}:
                items.append(sx)
        return list(dict.fromkeys(items))

    if pd.isna(val):
        return []

    s = str(val).strip()
    if not s or s.lower() in {"nan", "none"}:
        return []

    # Klammern entfernen falls der Wert als Set- oder Listen-String gespeichert wurde
    if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
        s = s[1:-1].strip()

    # Gaengige Trennzeichen durch Leerzeichen ersetzen
    for sep in [",", ";", "|", "/", "\n", "\t"]:
        s = s.replace(sep, " ")

    tokens = []
    for item in s.split():
        item = item.strip().strip("'").strip('"').strip()
        if item:
            tokens.append(item)
    return list(dict.fromkeys(tokens))


# ----------------------------------------------------------------------------------
# Einzelpaar-Distanzfunktionen (Referenzimplementierung und Utility)
# ----------------------------------------------------------------------------------

def _d_jaccard(A: Set[str], B: Set[str]) -> float:
    """
    Jaccard-Distanz: 1 - |A∩B| / |A∪B|.
    Leere Sets → 1.0.
    """

    if not A and not B:
        return 1.0
    if not A or not B:
        return 1.0
    return 1.0 - (len(A & B) / len(A | B))


def _d_overlap(A: Set[str], B: Set[str]) -> float:
    """
    Overlap-Distanz: 1 - |A∩B| / min(|A|,|B|).
    Leere Sets → 1.0.
    """

    if not A and not B:
        return 1.0
    if not A or not B:
        return 1.0
    return 1.0 - (len(A & B) / min(len(A), len(B)))


def _d_dice(A: Set[str], B: Set[str]) -> float:
    """
    Dice-Distanz: 1 - 2|A∩B| / (|A|+|B|).
    Leere Sets → 1.0.
    """

    if not A and not B:
        return 1.0
    if not A or not B:
        return 1.0
    return 1.0 - (2.0 * len(A & B) / (len(A) + len(B)))


# ----------------------------------------------------------------------------------
# IDF-Gewichtung
# ----------------------------------------------------------------------------------

def _idf_weights(sets: Dict[str, Set[str]], clip_max: float | None = None) -> Dict[str, float]:
    """
    Berechnet Inverse Document Frequency (IDF) Gewichte fuer alle Codes
    ueber alle Kunden-Sets.

    Seltene Codes erhalten ein hoeheres Gewicht als haeufige.
    Mit clip_max kann das maximale Gewicht begrenzt werden,
    um den Einfluss sehr seltener Codes zu daempfen.
    """

    code_frequency = Counter(code for S in sets.values() for code in S)
    N = max(1, len(sets))
    weights: Dict[str, float] = {}
    for code, count in code_frequency.items():
        weight = math.log(N / count)
        if clip_max is not None:
            weight = min(weight, clip_max)
        weights[code] = weight
    return weights


# ----------------------------------------------------------------------------------
# Einzelpaar-Distanzfunktionen (gewichtet)
# ----------------------------------------------------------------------------------

def _d_jaccard_weighted(A: Set[str], B: Set[str], w: Dict[str, float]) -> float:
    """
    Gewichtete Jaccard-Distanz mit IDF-Gewichten.

    Identische Sets geben immer 0.0 zurueck, auch wenn alle
    Gewichte 0 sind. Das verhindert fehlerhafte Distanzen 1.0
    bei Codes die bei allen Kunden vorkommen (IDF = 0).
    """

    if not A or not B:
        return 1.0
    if A == B:
        return 0.0
    inter = sum(w.get(c, 0.0) for c in (A & B))
    union = sum(w.get(c, 0.0) for c in (A | B))
    return 1.0 - (inter / union) if union > 0 else 1.0


def _d_dice_weighted(A: Set[str], B: Set[str], w: Dict[str, float]) -> float:
    """
    Gewichtete Dice-Distanz mit IDF-Gewichten.

    Identische Sets geben immer 0.0 zurueck, auch wenn alle
    Gewichte 0 sind. Das verhindert fehlerhafte Distanzen 1.0
    bei Codes die bei allen Kunden vorkommen (IDF = 0).
    """

    if not A or not B:
        return 1.0
    if A == B:
        return 0.0
    inter = sum(w.get(c, 0.0) for c in (A & B))
    denom = sum(w.get(c, 0.0) for c in A) + sum(w.get(c, 0.0) for c in B)
    return 1.0 - (2.0 * inter / denom) if denom > 0 else 1.0


def _d_overlap_weighted(A: Set[str], B: Set[str], w: Dict[str, float]) -> float:
    """
    Gewichtete Overlap-Distanz mit IDF-Gewichten.

    Identische Sets geben immer 0.0 zurueck, auch wenn alle
    Gewichte 0 sind. Das verhindert fehlerhafte Distanzen 1.0
    bei Codes die bei allen Kunden vorkommen (IDF = 0).
    """

    if not A or not B:
        return 1.0
    if A == B:
        return 0.0
    inter = sum(w.get(c, 0.0) for c in (A & B))
    denom = min(sum(w.get(c, 0.0) for c in A), sum(w.get(c, 0.0) for c in B))
    return 1.0 - (inter / denom) if denom > 0 else 1.0


# ----------------------------------------------------------------------------------
# Vektorisierte paarweise Distanzberechnung (M @ M.T)
# ----------------------------------------------------------------------------------

def _pairwise_dist_core(sets: Dict[str, Set[str]], metric: str,
                        output: str,
                        weights: dict[str, float] | None = None) -> pd.DataFrame:
    """
    Berechnet paarweise Set-Distanzen fuer alle Kunden-Paare.

    Wird intern von HS- und AM-Distanzfunktionen verwendet.
    Die Reihenfolge der IDs entspricht der Eingabereihenfolge.

    Verwendet Matrixmultiplikation (M @ M.T) zur Berechnung aller
    Intersection-Groessen in einem Schritt statt O(n²) Python-Schleifen.

    Parameter:
        sets     -- Dictionary {Kunden-ID: Set von Code-Strings}
        metric   -- 'jaccard', 'dice', 'overlap' oder deren '_weighted' Variante
        output   -- 'square' fuer n×n-Matrix, 'long' fuer Paar-DataFrame
        weights  -- optionale IDF-Gewichte; wenn None werden sie berechnet

    Rueckgabe:
        pd.DataFrame im gewaehlten Ausgabeformat (float32)
    """

    ids = list(sets.keys())
    n = len(ids)

    # Binaere Matrix: Zeile = Kunde, Spalte = eindeutiger Code
    all_codes = sorted({c for s in sets.values() for c in s})
    code_to_idx = {c: i for i, c in enumerate(all_codes)}
    m = len(all_codes)

    # float64 fuer praezise Zwischenberechnung, Ergebnis wird am Ende auf float32 gecastet
    M = np.zeros((n, m), dtype=np.float64)
    for i, cid in enumerate(ids):
        for c in sets[cid]:
            M[i, code_to_idx[c]] = 1.0

    # Set-Laengen und Intersection-Matrix
    sizes = M.sum(axis=1)
    inter = M @ M.T

    empty = sizes == 0
    empty_mask = empty[:, None] | empty[None, :]

    # Gewichtete Metriken vorbereiten
    is_weighted = metric.endswith("_weighted")

    if is_weighted:
        idf = weights if weights is not None else _idf_weights(sets, clip_max=_IDF_CLIP_MAX)
        w_vec = np.array([idf.get(c, 0.0) for c in all_codes])
        M_w = M * w_vec[None, :]
        w_inter = M_w @ M.T
        w_sizes = M_w.sum(axis=1)

        # Identische Sets erkennen (A == B → 0.0, auch wenn alle Gewichte 0)
        identical = (inter == sizes[:, None]) & (inter == sizes[None, :]) & ~empty_mask

    # Metrik-Formeln auf der gesamten Matrix
    with np.errstate(divide="ignore", invalid="ignore"):

        if metric == "jaccard":
            union = sizes[:, None] + sizes[None, :] - inter
            dist = np.where(union > 0, 1.0 - inter / union, 1.0)

        elif metric == "dice":
            denom = sizes[:, None] + sizes[None, :]
            dist = np.where(denom > 0, 1.0 - 2.0 * inter / denom, 1.0)

        elif metric == "overlap":
            min_sizes = np.minimum(sizes[:, None], sizes[None, :])
            dist = np.where(min_sizes > 0, 1.0 - inter / min_sizes, 1.0)

        elif metric == "jaccard_weighted":
            w_union = w_sizes[:, None] + w_sizes[None, :] - w_inter
            dist = np.where(w_union > 0, 1.0 - w_inter / w_union, 1.0)
            dist = np.where(identical, 0.0, dist)

        elif metric == "dice_weighted":
            w_denom = w_sizes[:, None] + w_sizes[None, :]
            dist = np.where(w_denom > 0, 1.0 - 2.0 * w_inter / w_denom, 1.0)
            dist = np.where(identical, 0.0, dist)

        elif metric == "overlap_weighted":
            w_min = np.minimum(w_sizes[:, None], w_sizes[None, :])
            dist = np.where(w_min > 0, 1.0 - w_inter / w_min, 1.0)
            dist = np.where(identical, 0.0, dist)

        else:
            raise ValueError(f"Unbekannte Metrik: {metric}")

    # Leere Sets → 1.0, Diagonale → 0.0
    dist = np.where(empty_mask, 1.0, dist)
    np.fill_diagonal(dist, 0.0)

    # Ausgabe
    if output == "square":
        return pd.DataFrame(dist.astype(np.float32), index=ids, columns=ids)

    if output == "long":
        i_idx, j_idx = np.triu_indices(n, k=1)
        return pd.DataFrame({
            "Customer_ID_1": [ids[i] for i in i_idx],
            "Customer_ID_2": [ids[j] for j in j_idx],
            "dissimilarity": np.round(dist[i_idx, j_idx], 6).astype(np.float32),
        })

    raise ValueError("output muss 'square' oder 'long' sein")