import pandas as pd
import numpy as np
import pytest

from src.application import _parse_app_part, pairwise_app_dist_slim


def _make_df():
    return pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "application_material_set_h2": [
            {"2.1_1.1.1", "3.1_1.1.1"},
            {"2.1_9.9.9"},
            {"5.6_1.1.1"},
        ],
    })


def _make_df_knn():
    return pd.DataFrame({
        "customer_code": ["A", "B"],
        "application_material_set_h2": [
            {"2.1_1.1.1", "3.1_1.1.1", "5.6_1.1.1", "2.5_1.1.1"},
            {"2.1_9.9.9", "3.1_9.9.9", "4.3_2.2.2"},
        ],
    })


def _get_pair(result_long, id1, id2):
    mask = (
        ((result_long["Customer_ID_1"] == id1) & (result_long["Customer_ID_2"] == id2)) |
        ((result_long["Customer_ID_1"] == id2) & (result_long["Customer_ID_2"] == id1))
    )
    return result_long.loc[mask, "dissimilarity"].values[0]


def test_parse_app_part_full_code():
    assert _parse_app_part("1.5_2.4.5") == ("1", "5")


def test_parse_app_part_partial_with_underscore():
    assert _parse_app_part("6.1_") == ("6", "1")


def test_parse_app_part_partial_without_material():
    assert _parse_app_part("6.1") == ("6", "1")


def test_parse_app_part_invalid_returns_none():
    assert _parse_app_part("invalid") is None
    assert _parse_app_part("") is None
    assert _parse_app_part("1.2.3") is None


def test_app_square_output_shape_and_diagonal():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="square"
    )
    assert result.shape == (3, 3)
    assert np.allclose(np.diag(result.values), 0.0)

def test_app_unknown_product_filter_raises():
    with pytest.raises(ValueError):
        pairwise_app_dist_slim(
            _make_df(),
            id_col="customer_code",
            am_col="application_material_set_h2",
            metric="jaccard",
            output="square",
            product_filter="9999"
        )

def test_app_invalid_metric_raises():
    with pytest.raises(ValueError):
        pairwise_app_dist_slim(
            _make_df(),
            id_col="customer_code",
            am_col="application_material_set_h2",
            metric="invalid",
            output="square"
        )


def test_app_jaccard_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)

def test_app_overlap_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="overlap",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.0, abs=1e-7)


def test_app_dice_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="dice",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/3, abs=1e-7)


def test_app_jaccard_weighted_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard_weighted",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.7304, abs=1e-3)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-4)


def test_app_dice_weighted_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="dice_weighted",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5753, abs=1e-3)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-4)



def test_app_overlap_weighted_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="overlap_weighted",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-4)


def test_app_cartesian_exact_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="exact",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)


def test_app_cartesian_hamming_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)
    assert result.loc["A", "C"] == pytest.approx(1.0, abs=1e-7)


def test_app_cartesian_lcp_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="lcp",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)


def test_app_cartesian_w_hamming_linear_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="linear",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/3, abs=1e-7)


def test_app_cartesian_block_weights_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="block_weights",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.375, abs=1e-7)


def test_app_chamfer_exact_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="chamfer",
        inner_metric="exact",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)

def test_app_chamfer_hamming_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="chamfer",
        inner_metric="hamming",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.125, abs=1e-7)


def test_app_chamfer_lcp_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="chamfer",
        inner_metric="lcp",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)


def test_app_chamfer_w_hamming_linear_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="chamfer",
        inner_metric="w_hamming",
        weight_scheme="linear",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/6, abs=1e-7)


def test_app_chamfer_block_weights_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="chamfer",
        inner_metric="block_weights",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.1875, abs=1e-7)


def test_app_knn_hamming_k2_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="knn_mean",
        inner_metric="hamming",
        k=2,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.25, abs=1e-7)


def test_app_knn_exact_k2_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="knn_mean",
        inner_metric="exact",
        k=2,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.5, abs=1e-7)


def test_app_cartesian_w_hamming_exponential_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="exponential",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1/3, abs=1e-7)


def test_app_cartesian_w_hamming_poly_convex_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="poly_convex",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.4, abs=1e-7)


def test_app_cartesian_w_hamming_poly_concave_reference():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="w_hamming",
        weight_scheme="poly_concave",
        output="square"
    )
    expected = (np.sqrt(2) / (np.sqrt(2) + 1)) / 2
    assert result.loc["A", "B"] == pytest.approx(expected, abs=1e-7)


def test_app_empty_code_set():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "application_material_set_h2": [{"2.1_1.1.1"}, set()],
    })
    result = pairwise_app_dist_slim(
        df,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(1.0, abs=1e-4)


def test_app_knn_hamming_k2():
    result = pairwise_app_dist_slim(
        _make_df_knn(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="knn_mean",
        inner_metric="hamming",
        k=2,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(0.53125, abs=1e-4)


def test_app_knn_hamming_k3():
    result = pairwise_app_dist_slim(
        _make_df_knn(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="knn_mean",
        inner_metric="hamming",
        k=3,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(95/144, abs=1e-7)


def test_app_knn_hamming_k4():
    result = pairwise_app_dist_slim(
        _make_df_knn(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="knn_mean",
        inner_metric="hamming",
        k=4,
        output="square"
    )
    assert result.loc["A", "B"] == pytest.approx(17/24, abs=1e-4)


def test_app_product_filter_keeps_relevant_application():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "application_material_set_h2": [
            {"2.1_1.1.1"},
            {"2.1_9.9.9"},
        ],
    })
    result = pairwise_app_dist_slim(
        df,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="long",
        product_filter="0601"
    )
    assert result.loc[0, "dissimilarity"] == 0.0

def test_app_product_filter_removes_irrelevant_codes():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "application_material_set_h2": [
            {"1.5_1.1.1", "2.1_1.1.1"},
            {"2.1_9.9.9"},
        ],
    })
    result = pairwise_app_dist_slim(
        df,
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="long",
        product_filter="0601"
    )
    assert result.loc[0, "dissimilarity"] == 0.0


def test_app_long_output_number_of_pairs():
    result = pairwise_app_dist_slim(
        _make_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="jaccard",
        output="long",
    )
    assert len(result) == 3
    assert set(result.columns) >= {"Customer_ID_1", "Customer_ID_2", "dissimilarity"}


def test_app_invalid_output_raises():
    with pytest.raises(ValueError):
        pairwise_app_dist_slim(
            _make_df(),
            id_col="customer_code",
            am_col="application_material_set_h2",
            metric="jaccard",
            output="wrong"
        )


def test_app_invalid_inner_metric_raises():
    with pytest.raises(ValueError):
        pairwise_app_dist_slim(
            _make_df(),
            id_col="customer_code",
            am_col="application_material_set_h2",
            metric="cartesian_mean",
            inner_metric="xxx"
        )


