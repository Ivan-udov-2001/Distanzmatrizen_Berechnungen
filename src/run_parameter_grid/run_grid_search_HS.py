from src.parameter_grid import grid_compute_hs_matrices
from src.paths import DF_DATA
from src.paths import GRID_OUTPUT
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)


# HS-Code Parameter-Grid
# level=6 fix, gating=True fix, k ∈ [2..10] bei knn_mean

HS_PARAM_GRID = [

    # Set-Engine
    {
        "metric": ["jaccard", "overlap", "dice",
                   "jaccard_weighted", "overlap_weighted", "dice_weighted"],
        "level": [6],
        "output": ["square"],
    },

    # Prefix-Engine mit lcp
    {
        "metric": ["cartesian_mean", "chamfer"],
        "inner_metric": ["lcp"],
        "level": [6],
        "output": ["square"],
    },
    {
        "metric": ["knn_mean"],
        "inner_metric": ["lcp"],
        "level": [6],
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "output": ["square"],
    },

    # Engine mit hamming
    {
        "metric": ["cartesian_mean", "chamfer"],
        "inner_metric": ["hamming"],
        "level": [6],
        "gating": [True],
        "output": ["square"],
    },
    {
        "metric": ["knn_mean"],
        "inner_metric": ["hamming"],
        "level": [6],
        "gating": [True],
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "output": ["square"],
    },

    # Engine mit w_hamming
    {
        "metric": ["cartesian_mean", "chamfer"],
        "inner_metric": ["w_hamming"],
        "weight_scheme": ["linear", "exponential", "poly_convex", "poly_concave"],
        "level": [6],
        "gating": [True],
        "output": ["square"],
    },
    {
        "metric": ["knn_mean"],
        "inner_metric": ["w_hamming"],
        "weight_scheme": ["linear", "exponential", "poly_convex", "poly_concave"],
        "level": [6],
        "gating": [True],
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "output": ["square"],
    },

    # Engine mit block_weights
    {
        "metric": ["cartesian_mean", "chamfer"],
        "inner_metric": ["block_weights"],
        "level": [6],
        "gating": [True],
        "output": ["square"],
    },
    {
        "metric": ["knn_mean"],
        "inner_metric": ["block_weights"],
        "level": [6],
        "gating": [True],
        "k": [2, 3, 4, 5, 6, 7, 8, 9, 10],
        "output": ["square"],
    },
]


if __name__ == "__main__":
    df = pd.read_pickle(DF_DATA)

    registry = grid_compute_hs_matrices(
        df,
        HS_PARAM_GRID,
        id_col="customer_code",
        hs_col="hs6_code",
        output= GRID_OUTPUT / "HS",
        )
    print("\nRegistry:")
    print(registry)


