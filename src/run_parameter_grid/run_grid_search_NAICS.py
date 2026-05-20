from src.parameter_grid import grid_compute_naics_matrices
from src.paths import DF_DATA
from src.paths import GRID_OUTPUT
import pandas as pd
import logging
logging.basicConfig(level=logging.INFO)

# NAICS Parameter-Grid
# n_digits=6, normalize=True, gating=True, alpha_penalty=0.15 — alle fix


NAICS_PARAM_GRID = [

    # hamming
    {
        "metric":        ["hamming"],
        "n_digits":      [6],
        "gating":        [True],
        "alpha_penalty": [0.15],
        "normalize":     [True],
        "use_primary_only": [True, False],
        "output":        ["square"],
    },

    # w_hamming
    {
        "metric":        ["w_hamming"],
        "weight_scheme": ["linear", "exponential", "poly_convex", "poly_concave"],
        "n_digits":      [6],
        "gating":        [True],
        "alpha_penalty": [0.15],
        "normalize":     [True],
        "use_primary_only": [True, False],
        "output":        ["square"],
    },

    # lcp
    {
        "metric":        ["lcp"],
        "n_digits":      [6],
        "alpha_penalty": [0.15],
        "normalize":     [True],
        "use_primary_only": [True, False],
        "output":        ["square"],
    },

    # block_weights
    {
        "metric":        ["block_weights"],
        "n_digits":      [6],
        "gating":        [True],
        "alpha_penalty": [0.15],
        "normalize":     [True],
        "use_primary_only": [True, False],
        "output":        ["square"],
    },
]


if __name__ == "__main__":
    df = pd.read_pickle(DF_DATA)

    registry_test = grid_compute_naics_matrices(
        df,
        NAICS_PARAM_GRID,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output= GRID_OUTPUT / "NAICS",
        )
    print("\nRegistry:")
    print(registry_test)


