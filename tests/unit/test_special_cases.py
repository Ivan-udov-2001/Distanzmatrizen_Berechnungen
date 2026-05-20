import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.naics import pairwise_naics_dist_slim
from src.hs import pairwise_hs_dist_slim
from src.application import pairwise_app_dist_slim


_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _naics_df():
    return pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "primary_naics_code": ["332710", "332720", "541330"],
        "secondary_naics_code": [None, None, None],
    })


def _hs_df():
    return pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "hs6_code": [
            {"850420", "850431"},
            {"850420"},
            {"870899"},
        ],
    })


def _app_df():
    return pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "application_material_set_h2": [
            {"2.1_1.1.1", "3.1_1.1.1"},
            {"2.1_9.9.9"},
            {"5.6_1.1.1"},
        ],
    })


def _get_pair(df_long, id1, id2):
    mask = (
            ((df_long["Customer_ID_1"] == id1) & (df_long["Customer_ID_2"] == id2)) |
            ((df_long["Customer_ID_1"] == id2) & (df_long["Customer_ID_2"] == id1))
    )
    return df_long.loc[mask, "dissimilarity"].values[0]


@pytest.mark.parametrize("metric", ["hamming", "w_hamming", "lcp", "block_weights"])
def test_naics_symmetry(metric):
    M = pairwise_naics_dist_slim(
        _naics_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="square",
        metric=metric,
    )
    assert np.allclose(M.values, M.values.T, atol=1e-6)


@pytest.mark.parametrize("metric,kw", [
    ("jaccard", {}),
    ("dice", {}),
    ("overlap", {}),
    ("cartesian_mean", {"inner_metric": "lcp"}),
    ("chamfer", {"inner_metric": "lcp"}),
    ("knn_mean", {"inner_metric": "lcp", "k": 2}),
])
def test_hs_symmetry(metric, kw):
    M = pairwise_hs_dist_slim(
        _hs_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric=metric,
        output="square",
        **kw,
    )
    assert np.allclose(M.values, M.values.T, atol=1e-6)


@pytest.mark.parametrize("metric,kw", [
    ("jaccard", {}),
    ("dice", {}),
    ("overlap", {}),
    ("cartesian_mean", {"inner_metric": "hamming"}),
    ("chamfer", {"inner_metric": "hamming"}),
    ("knn_mean", {"inner_metric": "hamming", "k": 2}),
])
def test_app_symmetry(metric, kw):
    M = pairwise_app_dist_slim(
        _app_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric=metric,
        output="square",
        **kw,
    )
    assert np.allclose(M.values, M.values.T, atol=1e-6)


@pytest.mark.parametrize("metric", ["hamming", "w_hamming", "lcp", "block_weights"])
def test_naics_range_of_values(metric):
    M = pairwise_naics_dist_slim(
        _naics_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="square",
        metric=metric,
    )
    assert M.values.min() >= -1e-6
    assert M.values.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("metric,kw", [
    ("jaccard", {}),
    ("chamfer", {"inner_metric": "hamming", "gating": True}),
    ("knn_mean", {"inner_metric": "lcp", "k": 2}),
])
def test_hs_range_of_values(metric, kw):
    M = pairwise_hs_dist_slim(
        _hs_df(),
        id_col="customer_code",
        hs_col="hs6_code",
        metric=metric,
        output="square",
        **kw,
    )
    assert M.values.min() >= -1e-6
    assert M.values.max() <= 1.0 + 1e-6


@pytest.mark.parametrize("metric,kw", [
    ("jaccard", {}),
    ("knn_mean", {"inner_metric": "hamming", "k": 2}),
])
def test_app_range_of_values(metric, kw):
    M = pairwise_app_dist_slim(
        _app_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric=metric,
        output="square",
        **kw,
    )
    assert M.values.min() >= -1e-6
    assert M.values.max() <= 1.0 + 1e-6




def test_naics_ss_wins():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["541330", "611310"],
        "secondary_naics_code": ["332710", "332710"],
    })
    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="hamming",
        gating=True,
        alpha_penalty=0.15,
        return_source=True,
    )
    assert result.loc[0, "source"] == "ss"
    assert result.loc[0, "dissimilarity"] == pytest.approx(0.30, abs=1e-6)


def test_naics_lcp_2_hamming():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["330000", "339999"],
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
    assert result.loc[0, "dissimilarity"] == pytest.approx(1.0, abs=1e-6)


def test_naics_lcp_1_lcp_metrik():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "primary_naics_code": ["336350", "326350"],
        "secondary_naics_code": [None, None],
    })
    result = pairwise_naics_dist_slim(
        df,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="lcp",
        gating=True,
    )
    assert result.loc[0, "dissimilarity"] == pytest.approx(1.0, abs=1e-6)


def test_naics_w_hamming_linear_ungated():
    result = pairwise_naics_dist_slim(
        _naics_df(),
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output="long",
        metric="w_hamming",
        weight_scheme="linear",
        gating=False,
    )
    assert abs(_get_pair(result, "A", "C") - 20 / 21) < 1e-6


def test_hs_leading_zeros():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "hs6_code": [{"010121"}, {"010129"}],
    })
    result = pairwise_hs_dist_slim(
        df,
        id_col="customer_code",
        hs_col="hs6_code",
        metric="cartesian_mean",
        inner_metric="lcp",
        output="square",
    )
    assert result.loc["A", "B"] == pytest.approx(1 / 6, abs=1e-6)


def test_hs_level_4():
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "hs6_code": [{"850420"}, {"850431"}],
    })
    result = pairwise_hs_dist_slim(
        df,
        id_col="customer_code",
        hs_col="hs6_code",
        level=4,
        metric="cartesian_mean",
        inner_metric="lcp",
        output="square",
    )
    assert result.loc["A", "B"] == pytest.approx(0.0, abs=1e-6)


def test_app_gating_increases_distance():
    r_without_gating = pairwise_app_dist_slim(
        _app_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        gating=False,
        output="square",
    )
    r_with_gating = pairwise_app_dist_slim(
        _app_df(),
        id_col="customer_code",
        am_col="application_material_set_h2",
        metric="cartesian_mean",
        inner_metric="hamming",
        gating=True,
        output="square",
    )
    assert (r_with_gating.values >= r_without_gating.values - 1e-6).all()