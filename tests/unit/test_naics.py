import pandas as pd
import numpy as np
import pytest

from src.naics import pairwise_naics_dist_slim


def _make_df():
    return pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "primary_naics_code": ["332710", "332720", "541330"],
        "secondary_naics_code": [None, None, None],
    })


def _get_pair(result_long, id1, id2):
    mask = (
        ((result_long["Customer_ID_1"] == id1) & (result_long["Customer_ID_2"] == id2)) |
        ((result_long["Customer_ID_1"] == id2) & (result_long["Customer_ID_2"] == id1))
    )
    return result_long.loc[mask, "dissimilarity"].values[0]


def test_naics_output_square_shape_and_diagonal():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="square",
    )

    assert result.shape == (3, 3)
    assert list(result.index) == ["A", "B", "C"]
    assert list(result.columns) == ["A", "B", "C"]
    assert np.allclose(np.diag(result.values), 0.0)


def test_naics_output_long_number_of_pairs():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
    )

    assert len(result) == 3  # 3 choose 2
    assert set(result.columns) >= {"Customer_ID_1", "Customer_ID_2", "dissimilarity"}


def test_naics_invalid_output_raises():
    with pytest.raises(ValueError):
        pairwise_naics_dist_slim(
            _make_df(),
            primary_col="primary_naics_code",
            id_col="customer_code",
            output="sqaure",
        )


def test_naics_invalid_metric_raises():
    with pytest.raises(ValueError):
        pairwise_naics_dist_slim(
            _make_df(),
            primary_col="primary_naics_code",
            id_col="customer_code",
            metric="invalid",
        )


def test_naics_lcp_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="lcp",
    )

    assert abs(_get_pair(result, "A", "B") - 1/3) < 1e-7
    assert _get_pair(result, "A", "C") == 1.0
    assert _get_pair(result, "B", "C") == 1.0


def test_naics_hamming_gated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        gating=True,
    )

    assert _get_pair(result, "A", "B") == 0.5
    assert _get_pair(result, "A", "C") == 1.0
    assert _get_pair(result, "B", "C") == 1.0


def test_naics_hamming_ungated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        gating=False,
    )

    assert abs(_get_pair(result, "A", "B") - 1/6) < 1e-10
    assert abs(_get_pair(result, "A", "C") - 5/6) < 1e-10


def test_naics_w_hamming_linear_gated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="w_hamming",
        weight_scheme="linear",
        gating=True,
    )

    assert abs(_get_pair(result, "A", "B") - 2/3) < 1e-10
    assert _get_pair(result, "A", "C") == 1.0


def test_naics_w_hamming_exponential_gated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="w_hamming",
        weight_scheme="exponential",
        gating=True,
    )

    assert abs(_get_pair(result, "A", "B") - 2/3) < 1e-10


def test_naics_w_hamming_poly_convex_gated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="w_hamming",
        weight_scheme="poly_convex",
        gating=True,
    )

    assert abs(_get_pair(result, "A", "B") - 0.8) < 1e-10


def test_naics_w_hamming_poly_concave_gated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="w_hamming",
        weight_scheme="poly_concave",
        gating=True,
    )

    expected = np.sqrt(2) / (np.sqrt(2) + 1)
    assert abs(_get_pair(result, "A", "B") - expected) < 1e-7


def test_naics_block_weights_gated_reference():
    result = pairwise_naics_dist_slim(
        _make_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="block_weights",
        gating=True,
    )

    assert _get_pair(result, "A", "B") == 0.5
    assert _get_pair(result, "A", "C") == 1.0


def test_naics_use_primary_only_ignores_secondary():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["336111", "541330"],
        "secondary_naics_code": ["541330", "336111"],  # swapped secondaries
    })

    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        gating=True,
        use_primary_only=True,
        return_source=True,
    )

    assert result.loc[0, "source"] == "pp"
    assert result.loc[0, "dissimilarity"] == 1.0


def test_naics_null_secondary_is_not_used_as_real_secondary():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["336111", "541330"],
        "secondary_naics_code": ["null", "null"],
    })

    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        alpha_penalty=0.15,
        return_source=True,
    )

    assert result.loc[0, "source"] == "pp"


def test_naics_identical_primary_with_null_secondary():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["336111", "336111"],
        "secondary_naics_code": ["null", "null"],
    })

    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        alpha_penalty=0.15,
        return_source=True,
    )

    assert result.loc[0, "dissimilarity"] == 0.0
    assert result.loc[0, "source"] == "pp"


def test_naics_one_missing_primary_max_distance():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["null", "332710"],
        "secondary_naics_code": [None, None],
    })
    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        gating=True,
    )
    assert _get_pair(result, "A", "B") == 1.0


def test_naics_secondary_wins():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["332710", "541330"],
        "secondary_naics_code": ["541330", None],
    })
    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        gating=True,
        alpha_penalty=0.01,
        return_source=True,
    )

    assert result.loc[0, "source"] == "sp"
    assert result.loc[0, "dissimilarity"] == pytest.approx(0.01, abs=1e-7)