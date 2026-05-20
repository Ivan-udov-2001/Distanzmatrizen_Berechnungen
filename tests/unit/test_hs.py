import pandas as pd
import numpy as np
import pytest

from src.hs import _normalize_hs6_int, pairwise_hs_dist_slim


def _make_df():
    return pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "hs6_code": [
            {"850420", "850431"},
            {"850420"},
            {"870899"},
        ],
    })


def _make_df_knn():
    return pd.DataFrame({
        "customer_code": ["A", "B"],
        "hs6_code": [
            {"850420", "850431", "850440", "850490"},
            {"850420", "850431", "870899"},
        ],
    })

def _get_pair(result_long, id1, id2):
    mask = (
        ((result_long["Customer_ID_1"] == id1) & (result_long["Customer_ID_2"] == id2)) |
        ((result_long["Customer_ID_1"] == id2) & (result_long["Customer_ID_2"] == id1))
    )
    return result_long.loc[mask, "dissimilarity"].values[0]


def test_normalize_hs6_int_six_digits_unchanged():
    assert _normalize_hs6_int("850420") == 850420


def test_normalize_hs6_int_right_padded():
    assert _normalize_hs6_int("8504") == 850400


def test_normalize_hs6_int_invalid_returns_minus_one():
    assert _normalize_hs6_int("") == -1
    assert _normalize_hs6_int("abc") == -1


def test_hs_square_output_shape_and_diagonal():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="square",
    )
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result.values), 0.0)


def test_hs_long_output_number_of_pairs():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="long",
    )
    assert len(result) == 3
    assert set(result.columns) >= {"Customer_ID_1", "Customer_ID_2", "dissimilarity"}

def test_hs_invalid_output_raises():
    with pytest.raises(ValueError):
        pairwise_hs_dist_slim(
            _make_df(),
            id_col="customer_code",
            hs_col="hs6_code",
            output="wrong"
        )

def test_hs_invalid_metric_raises():
    with pytest.raises(ValueError):
        pairwise_hs_dist_slim(
            _make_df(),
            id_col="customer_code",
            hs_col="hs6_code",
            metric="invalid"
        )


def test_hs_jaccard_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)

def test_hs_overlap_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="overlap",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.0, abs=1e-7)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)

def test_hs_dice_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="dice",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/3, abs=1e-7)


def test_hs_jaccard_weighted_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard_weighted",
        output="square"
    )
    assert result.loc["A", "B"] < 1.0
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)


def test_hs_dice_weighted_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="dice_weighted",
        output="square"
    )
    assert result.loc["A", "B"] < 1.0
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)


def test_hs_overlap_weighted_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="overlap_weighted",
        output="square"
    )
    assert result.loc["A", "B"] < 1.0
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)

def test_hs_cartesian_lcp_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="lcp",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/6, abs=1e-7)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)

def test_hs_cartesian_hamming_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="hamming",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)

def test_hs_cartesian_w_hamming_linear_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="linear",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)

def test_hs_cartesian_w_hamming_exponential_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="exponential",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)

def test_hs_cartesian_block_weights_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="block_weights",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)


def test_hs_cartesian_w_hamming_poly_convex_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="poly_convex",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)


def test_hs_cartesian_w_hamming_poly_concave_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="poly_concave",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)

def test_hs_chamfer_lcp_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="chamfer",
        inner_metric="lcp",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/12, abs=1e-7)

def test_hs_chamfer_hamming_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="chamfer",
        inner_metric="hamming",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)


def test_hs_chamfer_w_hamming_linear_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="chamfer",
        inner_metric="w_hamming",
        weight_scheme="linear",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)


def test_hs_chamfer_block_weights_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="chamfer",
        inner_metric="block_weights",
        gating=True,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)

def test_hs_knn_lcp_k2_reference():
    result = pairwise_hs_dist_slim(
        _make_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="knn_mean",
        inner_metric="lcp",
        k=2,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/6, abs=1e-7)


def test_hs_knn_lcp_k2():
    result = pairwise_hs_dist_slim(
        _make_df_knn(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="knn_mean",
        inner_metric="lcp",
        k=2,
        output="square"
    )
    assert result.loc["A", "B"] >= 0.0
    assert result.loc["A", "B"] <= 1.0


def test_hs_knn_lcp_k3():
    result = pairwise_hs_dist_slim(
        _make_df_knn(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="knn_mean",
        inner_metric="lcp",
        k=3,
        output="square"
    )
    assert result.loc["A", "B"] >= 0.0
    assert result.loc["A", "B"] <= 1.0


def test_hs_knn_lcp_k4():
    result = pairwise_hs_dist_slim(
        _make_df_knn(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric="knn_mean",
        inner_metric="lcp",
        k=4,
        output="square"
    )
    assert result.loc["A", "B"] >= 0.0
    assert result.loc["A", "B"] <= 1.0


def test_hs_empty_code_set():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "hs6_code": [{"850420"},
                     set()],
    })
    result = pairwise_hs_dist_slim(
        df,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="jaccard",
        output="square")
    assert result.loc["A", "B"] == pytest.approx(1.0, abs=1e-7)


def test_hs_invalid_inner_metric_raises():
    with pytest.raises(ValueError):
        pairwise_hs_dist_slim(
            _make_df(),
            id_col="customer_code",
            hs_col="hs6_code",
            metric="cartesian_mean",
            inner_metric="xxx"
        )