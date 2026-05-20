from __future__ import annotations

import numpy as np
import pandas as pd
from numba import njit, prange

from src.core_metrics import (
    _clean_code_series,
    _normalize_codes_to_uint8,
    _hamming_gated,
    _w_hamming_flexible,
    _lcp_dist,
    _block_weighted_hamming,
)

# -----------------------------------------------------------------
# NAICS-Code Distanzfunktionen (paarweise, mit Gating und Quelle)
# -----------------------------------------------------------------

@njit(parallel=True)
def _pairwise_naics_min_dist(
    A_p: np.ndarray,
    A_s: np.ndarray,
    has_s: np.ndarray,
    normalize: bool,
    alpha_penalty: float,
    metric_id: int,
    gating: bool,
    weight_scheme: int = 0
):
    """
    Berechnet paarweise NAICS-Distanzen fuer alle Kunden-Paare parallel.

    Fuer jedes Paar (i, j) werden bis zu vier Kandidaten verglichen:
    pp, ps, sp, ss (Primary/Secondary Kombination).

    has_s stellt sicher dass fehlende Secondary-Codes (als 000000 kodiert)
    nie in die Distanzberechnung eingehen, verhindert faelschlicherweise
    niedrige Distanzen durch kuenstliche Nullen-Codes.

    Rueckgabe:
        out -- Distanzmatrix (N, N), dtype float32
        src -- Quellmatrix (N, N), dtype uint8 (0=pp, 1=ps, 2=sp, 3=ss)
    """

    N, L = A_p.shape
    out = np.empty((N, N), dtype=np.float32)
    src = np.empty((N, N), dtype=np.uint8)

    for i in prange(N):
        out[i, i] = 0.0
        src[i, i] = 0
        for j in range(i + 1, N):
            if metric_id == 0:
                d_pp = _hamming_gated(A_p[i], A_p[j], normalize, gating)
                d_ps = _hamming_gated(A_p[i], A_s[j], normalize, gating)
                d_sp = _hamming_gated(A_s[i], A_p[j], normalize, gating)
                d_ss = _hamming_gated(A_s[i], A_s[j], normalize, gating)

            elif metric_id == 1:
                d_pp = _w_hamming_flexible(A_p[i], A_p[j], normalize, gating, weight_scheme)
                d_ps = _w_hamming_flexible(A_p[i], A_s[j], normalize, gating, weight_scheme)
                d_sp = _w_hamming_flexible(A_s[i], A_p[j], normalize, gating, weight_scheme)
                d_ss = _w_hamming_flexible(A_s[i], A_s[j], normalize, gating, weight_scheme)

            elif metric_id == 2:
                d_pp = _lcp_dist(A_p[i], A_p[j], normalize)
                d_ps = _lcp_dist(A_p[i], A_s[j], normalize)
                d_sp = _lcp_dist(A_s[i], A_p[j], normalize)
                d_ss = _lcp_dist(A_s[i], A_s[j], normalize)

            else:
                d_pp = _block_weighted_hamming(A_p[i], A_p[j], normalize, gating)
                d_ps = _block_weighted_hamming(A_p[i], A_s[j], normalize, gating)
                d_sp = _block_weighted_hamming(A_s[i], A_p[j], normalize, gating)
                d_ss = _block_weighted_hamming(A_s[i], A_s[j], normalize, gating)

            # Besten Kandidaten unter Beruecksichtigung der Verfuegbarkeit waehlen
            best = d_pp
            best_src = 0

            if has_s[j]:
                cand = d_ps + alpha_penalty
                if cand < best:
                    best = cand
                    best_src = 1

            if has_s[i]:
                cand = d_sp + alpha_penalty
                if cand < best:
                    best = cand
                    best_src = 2

            if has_s[i] and has_s[j]:
                cand = d_ss + 2.0 * alpha_penalty
                if cand < best:
                    best = cand
                    best_src = 3

            out[i, j] = np.float32(best)
            out[j, i] = np.float32(best)
            src[i, j] = best_src
            src[j, i] = best_src

    return out, src


def pairwise_naics_dist_slim(
    df: pd.DataFrame,
    *,
    primary_col: str = "primary_naics_code",
    secondary_col: str = "secondary_naics_code",
    n_digits: int = 6,
    id_col: str = "customer_code",
    output: str = "square",
    normalize: bool = True,
    alpha_penalty: float = 0.20,
    use_primary_only: bool = False,
    metric: str = "hamming",
    weight_scheme: str = "linear",
    gating: bool = True,
    return_source: bool = True,
) -> pd.DataFrame:
    """
    Berechnet paarweise NAICS-Distanzen zwischen allen Kunden.

    primary_col:   Spaltenname des primaeren NAICS-Codes
    secondary_col: Spaltenname des sekundaeren NAICS-Codes (optional)
    n_digits:      Anzahl der Stellen (Standard: 6)
    id_col:        Spaltenname der Kunden-ID (Standard: Index)
    output:        'long' oder 'square'
    normalize:     Distanz auf [0, 1] normalisieren
    alpha_penalty: Strafterm fuer Secondary-Code Verwendung
    metric:        'hamming' | 'w_hamming' | 'lcp' | 'block_weights'
    weight_scheme: Gewichtungsschema fuer w_hamming
    gating:        Erste 2 Stellen muessen uebereinstimmen, sonst 1.0
    return_source: Quellspalte (pp/ps/sp/ss) im Long-Format ergaenzen
    """

    metric_l = metric.lower()

    _metric_map = {
        "hamming":          0,
        "w_hamming":        1,
        "lcp":              2,
        "block_weights":    3,
    }

    # Eingaben validieren
    if output not in {"long", "square"}:
        raise ValueError("output muss 'long' oder 'square' sein")

    if metric_l not in _metric_map:
        raise ValueError(
            f"Unbekannte Metrik '{metric}'. "
            f"Erwartet: {sorted(_metric_map)}"
        )

    metric_id = _metric_map[metric_l]

    # weight_scheme -> ws_id (nur relevant fuer w_hamming)
    weight_scheme_l = weight_scheme.lower()

    _ws_map = {
        "linear":       0,
        "exponential":  1,
        "poly_convex":  2,
        "poly_concave": 3,
    }
    if metric_l == "w_hamming":
        if weight_scheme_l not in _ws_map:
            raise ValueError(
                f"weight_scheme muss eines von {sorted(_ws_map)} sein"
            )
        ws_id = _ws_map[weight_scheme_l]
    else:
        ws_id = 0

    # Kunden-IDs bestimmen
    ids = (
        df[id_col].astype(str).to_numpy()
        if id_col and id_col in df.columns
        else df.index.astype(str).to_numpy()
    )

    # Codes normalisieren
    A_p = _normalize_codes_to_uint8(df[primary_col], n_digits)

    if secondary_col and secondary_col in df.columns:
        secondary_clean = _clean_code_series(df[secondary_col])

        if secondary_clean.notna().any():
            has_s = secondary_clean.notna().to_numpy()
            A_s = _normalize_codes_to_uint8(secondary_clean.fillna(""), n_digits)

        else:
            has_s = np.zeros(len(df), dtype=np.bool_)
            A_s = np.zeros_like(A_p)
    else:
        has_s = np.zeros(len(df), dtype=np.bool_)
        A_s = np.zeros_like(A_p)

    if use_primary_only:
        # Secondary-Code wird ignoriert, nur Primary-Code wird verwendet
        has_s = np.zeros(len(df), dtype=np.bool_)

    # Distanzmatrix berechnen
    val, src_idx = _pairwise_naics_min_dist(
        A_p, A_s, has_s, normalize, float(alpha_penalty), metric_id, gating, ws_id
    )

    # Ausgabe
    if output == "square":
        return pd.DataFrame(val, index=ids, columns=ids)

    # output == "long" (bereits durch Validierung sichergestellt)
    N = len(df)
    iu = np.triu_indices(N, k=1)
    result = pd.DataFrame(
        {
            "Customer_ID_1": ids[iu[0]],
            "Customer_ID_2": ids[iu[1]],
            "dissimilarity": val[iu].astype(np.float32),
        }
    )
    if return_source:
        src_map = np.array(["pp", "ps", "sp", "ss"])
        result["source"] = src_map[src_idx[iu]]
    return result
