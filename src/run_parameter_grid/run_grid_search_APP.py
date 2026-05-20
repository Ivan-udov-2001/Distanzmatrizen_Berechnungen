from src.parameter_grid import grid_compute_app_matrices
from src.paths import DF_DATA
from src.paths import GRID_OUTPUT
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)


# App-only Parameter-Grid
# Material wird ignoriert. gating=True fix, k ∈ [2..10] bei knn_mean

APP_PARAM_GRID = [

    # Set-Engine
    {
        "metric": ["jaccard", "overlap", "dice",
                   "jaccard_weighted", "overlap_weighted", "dice_weighted"],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output": ["square"],
    },

    # Prefix-Engine mit exact
    {
        "metric":       ["cartesian_mean", "chamfer"],
        "inner_metric": ["exact"],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },
    {
        "metric":       ["knn_mean"],
        "inner_metric": ["exact"],
        "k":            [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },

    # Engine mit hamming
    {
        "metric":       ["cartesian_mean", "chamfer"],
        "inner_metric": ["hamming"],
        "gating":       [True],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },
    {
        "metric":       ["knn_mean"],
        "inner_metric": ["hamming"],
        "gating":       [True],
        "k":            [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },

    # Engine mit w_hamming
    {
        "metric":        ["cartesian_mean", "chamfer"],
        "inner_metric":  ["w_hamming"],
        "weight_scheme": ["linear", "exponential", "poly_convex", "poly_concave"],
        "gating":        [True],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":        ["square"],
    },
    {
        "metric":        ["knn_mean"],
        "inner_metric":  ["w_hamming"],
        "weight_scheme": ["linear", "exponential", "poly_convex", "poly_concave"],
        "gating":        [True],
        "k":             [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":        ["square"],
    },

    # Engine mit block_weights
    {
        "metric":       ["cartesian_mean", "chamfer"],
        "inner_metric": ["block_weights"],
        "gating":       [True],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },
    {
        "metric":       ["knn_mean"],
        "inner_metric": ["block_weights"],
        "gating":       [True],
        "k":            [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },

    # ─── Prefix-Engine mit lcp (kein gating, kein weight_scheme) ───
    # Hinweis: bei nur 2 Positionen (A, B) ist LCP mathematisch identisch
    # zu Hamming. Trotzdem für Vollständigkeit/Vergleich aufnehmen.
    {
        "metric":       ["cartesian_mean", "chamfer"],
        "inner_metric": ["lcp"],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },
    {
        "metric":       ["knn_mean"],
        "inner_metric": ["lcp"],
        "k":            [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "product_filter": [None, "0601", "0602", "0604", "0201"],
        "output":       ["square"],
    },
]


if __name__ == "__main__":
    df = pd.read_pickle(DF_DATA)


    registry_test = grid_compute_app_matrices(
        df,
        APP_PARAM_GRID,
        id_col="customer_code",
        am_col="application_material_set_h2",
        output= GRID_OUTPUT / "APP",
        )
    print("\nRegistry:")
    print(registry_test)

