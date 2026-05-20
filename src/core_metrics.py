from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit


# ------------------------------------------------------------
# Kern-Distanzfunktionen fuer NAICS- und HS-Codes
# Alle Funktionen arbeiten auf uint8-Ziffern-Arrays.
# ------------------------------------------------------------

def _clean_code_series(series: pd.Series) -> pd.Series:
    """
    Ersetzt gaengige Null-Darstellungen durch NaN,
    damit leere Eintraege nicht faelschlicherweise als
    Zahlen interpretiert werden.

    """
    return series.replace(
        ["null", "NULL", "Null", "None", "none", "nan", "NaN", "NA", "N/A", "", " ", "  ", "   ", "\t"],
        np.nan,
    )

def _normalize_codes_to_uint8(series: pd.Series, n_digits: int) -> np.ndarray:
    """
    Normalisiert NAICS-artige Codes auf feste Laenge und
    kodiert sie als uint8-Ziffern-Array (Wertebereich 0..9).

    - Nicht-Ziffern werden entfernt
    - Zu kurze Codes werden rechts mit '0' aufgefuellt
    - Rueckgabe: Array der Form (N, n_digits), dtype uint8

    """
    code_array = (
        _clean_code_series(series)
        .fillna("")
        .astype(str)
        .str.replace(r"[^0-9]", "", regex=True)
        .str.ljust(n_digits, "0")
        .str[:n_digits]
    ).to_numpy()

    N = len(code_array)
    encoded_digits = np.empty((N, n_digits), dtype=np.uint8)

    for i in range(N):
        code_str = code_array[i]

        for k in range(n_digits):
            # ord('0') == 48, daher: Zeichen - 48 = Ziffernwert
            c = ord(code_str[k]) if k < len(code_str) else 48
            d = c - 48
            encoded_digits[i, k] = d if 0 <= d <= 9 else 0
    return encoded_digits


@njit(inline="always")
def _lcp_len(a_row: np.ndarray, b_row: np.ndarray) -> int:
    """
    Laenge des laengsten gemeinsamen Praefixes
    zweier Ziffern-Arrays.

    """
    L = a_row.shape[0]
    for k in range(L):
        if a_row[k] != b_row[k]:
            return k
    return L


@njit(inline="always")
def _lcp_dist(a_row: np.ndarray, b_row: np.ndarray, normalize: bool) -> float:
    """
    LCP-basierte Distanz zwischen zwei Ziffern-Arrays.

    Ein gemeinsamer Praefix kuerzer als 2 Stellen bedeutet
    kein gemeinsamer Sektor -> maximale Distanz.

    """
    L = a_row.shape[0]
    lcp = _lcp_len(a_row, b_row)
    if lcp < 2:
        return 1.0 if normalize else float(L)
    if normalize:
        return 1.0 - (lcp / L)
    return float(L - lcp)


@njit(inline="always")
def _hamming_gated(a_row: np.ndarray, b_row: np.ndarray, normalize: bool, gating: bool) -> float:
    """
    Hamming-Distanz zwischen zwei Ziffern-Arrays.

    Wenn gating=True: Stimmen die ersten 2 Stellen (Sektor) nicht
    ueberein, wird sofort die maximale Distanz 1.0 zurueckgegeben.
    Andernfalls wird nur der Tail ab der LCP-Laenge verglichen.
    Die Distanz bezieht sich dadurch nur auf den verbleibenden
    Bereich nach dem Prefix und nicht auf den gesamten Code.

    """

    L = a_row.shape[0]
    lcp = _lcp_len(a_row, b_row)

    if not gating:
        mismatch_count = 0
        for k in range(L):
            mismatch_count += 1 if a_row[k] != b_row[k] else 0
        return mismatch_count / L if normalize else float(mismatch_count)

    # Kein gemeinsamer Sektor -> maximale Distanz
    if lcp < 2:
        return 1.0 if normalize else float(L)

    tail_start = lcp if lcp > 2 else 2
    tail_mismatch_count = 0
    tail_len = L - tail_start
    for k in range(tail_start, L):
        tail_mismatch_count += 1 if a_row[k] != b_row[k] else 0

    if normalize:
        denom = tail_len if tail_len > 0 else 1
        return tail_mismatch_count / denom
    return float(tail_mismatch_count)

@njit(inline="always")
def _position_weight(k: int, L: int, scheme: int) -> float:
    """
    Positionsabhaengiges Gewicht fuer Stelle k eines Codes der Laenge L.

    Fruehere Stellen (kleines k) deuten bei Uebereinstimmung auf eine
    staerkere Aehnlichkeit hin und werden daher hoeher gewichtet.

    scheme:
        0 = linear              w = L - k
        1 = exponential         w = 0.5^k
        2 = poly_convex         w = (L - k)^2
        3 = poly_concave        w = (L - k)^0.5

    """

    if scheme == 0:
        return float(L - k)
    elif scheme == 1:
        return float(0.5 ** k)
    elif scheme == 2:
        return float((L - k) ** 2)
    else:
        return float((L - k) ** 0.5)



@njit(inline="always")
def _w_hamming_flexible(a_row: np.ndarray, b_row: np.ndarray,
                        normalize: bool, gating: bool,
                        weight_scheme: int) -> float:
    """
    Gewichtetes Hamming: frueheren Positionen wird mehr Gewicht gegeben.

    Wenn gating=True: Stimmt der Sektor (erste 2 Stellen) nicht
    ueberein, wird sofort 1.0 zurueckgegeben.
    Das Gewichtungsschema wird ueber weight_scheme gesteuert
    (siehe _position_weight).
    """
    L = a_row.shape[0]
    lcp = _lcp_len(a_row, b_row)

    if not gating:
        w_sum = 0.0
        w_val = 0.0
        for k in range(L):
            w = _position_weight(k, L, weight_scheme)
            w_sum += w
            if a_row[k] != b_row[k]:
                w_val += w
        return (w_val / w_sum) if normalize else w_val

    # Kein gemeinsamer Sektor -> maximale Distanz
    if lcp < 2:
        return 1.0 if normalize else float(L)

    tail_start = lcp if lcp > 2 else 2
    w_sum = 0.0
    w_val = 0.0
    for k in range(tail_start, L):
        w = _position_weight(k, L, weight_scheme)
        w_sum += w
        if a_row[k] != b_row[k]:
            w_val += w

    if normalize:
        denom = w_sum if w_sum > 0 else 1
        return w_val / denom
    return w_val



@njit(inline="always")
def _block_weighted_hamming(a_row: np.ndarray, b_row: np.ndarray, normalize: bool, gating: bool) -> float:
    """
    Block-gewichtetes Hamming nach NAICS-Hierarchieebenen.

    Sektor (Stellen 0-1) wird 10x staerker gewichtet als die
    Branchen-Ebene, Teilsektor und Industriegruppe (2-3)
    liegen dazwischen.
    """
    L = a_row.shape[0]
    lcp = _lcp_len(a_row, b_row)

    # Gewichte nach NAICS-Hierarchie
    w_sektor = 10.0
    w_teilsektor = 3.0
    w_branche = 1.0

    if not gating:
        w_sum = 0.0
        w_val = 0.0
        for k in range(L):
            if k < 2:
                w = w_sektor
            elif k < 4:
                w = w_teilsektor
            else:
                w = w_branche
            w_sum += w
            if a_row[k] != b_row[k]:
                w_val += w
        return (w_val / w_sum) if normalize else w_val

    # Kein gemeinsamer Sektor -> maximale Distanz
    if lcp < 2:
        return 1.0 if normalize else float(L)

    tail_start = lcp if lcp > 2 else 2
    w_sum = 0.0
    w_val = 0.0
    for k in range(tail_start, L):
        if k < 2:
            w = w_sektor
        elif k < 4:
            w = w_teilsektor
        else:
            w = w_branche
        w_sum += w
        if a_row[k] != b_row[k]:
            w_val += w
    if normalize:
        denom = w_sum if w_sum > 0 else 1
        return w_val / denom
    return w_val


@njit
def _alloc_pairs(N: int):
    """
    Alloziert Ausgabe-Arrays und berechnet Start-Offsets
    fuer die parallele Paar-Iteration der oberen Dreiecksmatrix.

    Wird von allen all_pairs_*-Funktionen in hs.py und application.py
    als Setup-Schritt verwendet.

    Rueckgabe:
        out_i  -- Index-Array i (M Elemente)
        out_j  -- Index-Array j (M Elemente)
        out_d  -- Distanz-Array (M Elemente, float32)
        start  -- Offset pro Zeile i fuer die prange-Schleife (N Elemente)
    """
    M = N * (N - 1) // 2

    out_i = np.empty(M, dtype=np.int64)
    out_j = np.empty(M, dtype=np.int64)
    out_d = np.empty(M, dtype=np.float32)

    start = np.empty(N, dtype=np.int64)
    s = 0
    for i in range(N - 1):
        start[i] = s
        s += (N - 1 - i)
    start[N - 1] = s

    return out_i, out_j, out_d, start