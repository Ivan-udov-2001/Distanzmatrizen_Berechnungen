import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.naics import pairwise_naics_dist_slim
from src.hs import pairwise_hs_dist_slim
from src.application import pairwise_app_dist_slim

_DATA_ = Path(__file__).resolve().parent.parent.parent / "data"


@pytest.fixture(scope="module")
def df_real():
    path = _DATA_ / "df_final_master_pseudo.pkl"
    if not path.exists():
        pytest.skip("Testdaten nicht vorhanden")
    cols = ["customer_code", "primary_naics_code", "secondary_naics_code", "hs6_code", "application_material_set_h2"]
    return pd.read_pickle(path)[cols].head(20).copy()

@pytest.fixture(scope="module")
def df_single(df_real):
    return df_real.head(1).copy()


@pytest.fixture(scope="module")
def df_nan_secondary(df_real):
    df = df_real.copy()
    df.loc[df.index[:5], "secondary_naics_code"] = np.nan
    return df


@pytest.fixture(scope="module")
def df_all_nan_secondary(df_real):
    df = df_real.copy()
    df["secondary_naics_code"] = np.nan
    return df


@pytest.mark.parametrize("metric", ["hamming", "w_hamming", "lcp", "block_weights"])
def test_naics_all_metrics_long(df_real, metric):
    result = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric=metric,
        output="long",
    )
    n = len(df_real)
    assert len(result) == n * (n - 1) // 2
    assert result["dissimilarity"].between(0.0, 1.0).all()


@pytest.mark.parametrize("metric", ["hamming", "w_hamming", "lcp", "block_weights"])
def test_naics_all_metrics_square(df_real, metric):
    result = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric=metric,
        output="square",
    )
    assert result.shape == (len(df_real), len(df_real))
    assert np.allclose(result.values, result.values.T, atol=1e-5)
    assert np.allclose(np.diag(result.values), 0.0, atol=1e-5)


@pytest.mark.parametrize("weight_scheme", ["linear", "exponential", "poly_convex", "poly_concave"])
def test_naics_w_hamming_all_schemes(df_real, weight_scheme):
    result = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="w_hamming",
        weight_scheme=weight_scheme,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_naics_long_square_consistent(df_real):
    long = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
        return_source=False,
    )
    square = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="square",
    )
    for _, row in long.iterrows():
        assert abs(row["dissimilarity"] - square.loc[row["Customer_ID_1"], row["Customer_ID_2"]]) < 1e-4


def test_naics_use_primary_only_pp(df_real):
    result = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
        use_primary_only=True,
    )
    assert set(result["source"].unique()) == {"pp"}


def test_naics_return_source_columns(df_real):
    result_with = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
        return_source=True,
    )
    result_without = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
        return_source=False,
    )
    assert "source" in result_with.columns
    assert set(result_with["source"].unique()).issubset({"pp", "ps", "sp", "ss"})
    assert "source" not in result_without.columns


def test_naics_nan_secondary_no_error(df_nan_secondary):
    result = pairwise_naics_dist_slim(
        df_nan_secondary,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_naics_all_nan_secondary_only_pp(df_all_nan_secondary):
    result = pairwise_naics_dist_slim(
        df_all_nan_secondary,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
    )
    assert set(result["source"].unique()) == {"pp"}


def test_naics_alpha_penalty_influence(df_real):
    r_low = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
        alpha_penalty=0.0,
    )
    r_high = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
        alpha_penalty=1.0,
    )
    assert (r_high["source"] != "pp").sum() <= (r_low["source"] != "pp").sum()


@pytest.mark.parametrize("n_digits", [2, 4, 6])
def test_naics_n_digits(df_real, n_digits):
    result = pairwise_naics_dist_slim(
        df_real,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        n_digits=n_digits,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_naics_individual_customer_zero_couples(df_single):
    result = pairwise_naics_dist_slim(
        df_single,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        metric="hamming",
        output="long",
    )
    assert len(result) == 0


@pytest.mark.parametrize("metric", [
    "jaccard", "overlap", "dice",
    "jaccard_weighted", "overlap_weighted", "dice_weighted"
])
def test_hs_set_metrics_long(df_real, metric):
    result = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric=metric,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


@pytest.mark.parametrize("metric,inner_metric", [
    ("cartesian_mean", "lcp"),
    ("cartesian_mean", "hamming"),
    ("cartesian_mean", "w_hamming"),
    ("cartesian_mean", "block_weights"),
    ("chamfer", "lcp"),
    ("chamfer", "hamming"),
    ("knn_mean", "lcp"),
    ("knn_mean", "hamming"),
])
def test_hs_point_engine_combinations(df_real, metric, inner_metric):
    result = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric=metric,
        inner_metric=inner_metric,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


@pytest.mark.parametrize("weight_scheme", ["linear", "exponential", "poly_convex", "poly_concave"])
def test_hs_w_hamming_all_schemes(df_real, weight_scheme):
    result = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme=weight_scheme,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_hs_long_square_consistent(df_real):
    long = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="long",
    )
    square = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="square",
    )
    for _, row in long.iterrows():
        assert abs(row["dissimilarity"] - square.loc[row["Customer_ID_1"], row["Customer_ID_2"]]) < 1e-4


@pytest.mark.parametrize("level", [2, 4, 6])
def test_hs_level_parameter(df_real, level):
    result = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        level=level,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


@pytest.mark.parametrize("k", [1, 3, 5])
def test_hs_knn_k_parameter(df_real, k):
    result = pairwise_hs_dist_slim(
        df_real,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="knn_mean",
        inner_metric="hamming",
        k=k,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_hs_individual_customer_zero_couples(df_single):
    result = pairwise_hs_dist_slim(
        df_single,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="long",
    )
    assert len(result) == 0


@pytest.mark.parametrize("metric", [
    "jaccard", "overlap", "dice",
    "jaccard_weighted", "overlap_weighted", "dice_weighted"
])
def test_app_set_metrics_long(df_real, metric):
    result = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric=metric,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


@pytest.mark.parametrize("metric,inner_metric", [
    ("cartesian_mean", "exact"),
    ("cartesian_mean", "hamming"),
    ("cartesian_mean", "w_hamming"),
    ("cartesian_mean", "block_weights"),
    ("cartesian_mean", "lcp"),
    ("chamfer", "exact"),
    ("chamfer", "hamming"),
    ("knn_mean", "hamming"),
    ("knn_mean", "exact"),
])
def test_app_point_engine_combinations(df_real, metric, inner_metric):
    result = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric=metric,
        inner_metric=inner_metric,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


@pytest.mark.parametrize("weight_scheme", ["linear", "exponential", "poly_convex", "poly_concave"])
def test_app_w_hamming_all_schemes(df_real, weight_scheme):
    result = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme=weight_scheme,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_app_gating_increases_distance(df_real):
    r_without = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        gating=False,
        output="long",
    )
    r_with = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        gating=True,
        output="long",
    )
    assert (r_with["dissimilarity"].values >= r_without["dissimilarity"].values - 1e-5).all()


def test_app_long_square_consistent(df_real):
    long = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="long",
    )
    square = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="square",
    )
    for _, row in long.iterrows():
        assert abs(row["dissimilarity"] - square.loc[row["Customer_ID_1"], row["Customer_ID_2"]]) < 1e-4


@pytest.mark.parametrize("product_filter", ["0601", "0602", "0604", "0201"])
def test_app_all_productfilter(df_real, product_filter):
    result = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        product_filter=product_filter,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_app_productfilter_change_distances(df_real):
    Without = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        output="long",
    )
    With = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        product_filter="0601",
        output="long",
    )
    assert not np.allclose(Without["dissimilarity"].values, With["dissimilarity"].values, atol=1e-4)


@pytest.mark.parametrize("k", [1, 3, 5])
def test_app_knn_k_parameter(df_real, k):
    result = pairwise_app_dist_slim(
        df_real,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="knn_mean",
        inner_metric="hamming",
        k=k,
        output="long",
    )
    assert result["dissimilarity"].between(0.0, 1.0).all()


def test_app_same_customer_distance_zero(df_real):
    df_dup = pd.concat([df_real.head(1), df_real.head(1)], ignore_index=True)
    df_dup["customer_code"] = ["A", "B"]
    result = pairwise_app_dist_slim(
        df_dup,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="long",
    )
    assert result["dissimilarity"].iloc[0] == pytest.approx(0.0, abs=1e-5)


def test_app_individual_customer_zero_couples(df_single):
    result = pairwise_app_dist_slim(
        df_single,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="long",
    )
    assert len(result) == 0