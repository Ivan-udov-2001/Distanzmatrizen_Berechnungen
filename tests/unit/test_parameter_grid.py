import pytest
import pandas as pd
from pathlib import Path
from src.parameter_grid import (
    _signature, _clean_param_name, _pretty_name,
    grid_compute_naics_matrices,
    grid_compute_hs_matrices,
    grid_compute_app_matrices,
)

def test_signature_deterministic():
    params = {"metric": "hamming", "gating": True}
    assert _signature(params) == _signature(params)


def test_signature_order_doesnt_matter():
    assert _signature({"a": 1, "b": 2}) == _signature({"b": 2, "a": 1})


def test_signature_different_parameter_different_hash():
    assert _signature({"metric": "hamming"}) != _signature({"metric": "lcp"})


def test_clean_param_name_underline_to_hyphen():
    assert _clean_param_name("w_hamming") == "w-hamming"


def test_clean_param_name_without_underline_unchanged():
    assert _clean_param_name("hamming") == "hamming"


def test_pretty_name_naics_format():
    name = _pretty_name("naics", 0, {"metric": "hamming", "output": "square"})
    assert name.startswith("naics_000")


def test_pretty_name_index_correct():
    name_0 = _pretty_name("naics", 0, {"metric": "hamming", "output": "square"})
    name_5 = _pretty_name("naics", 5, {"metric": "hamming", "output": "square"})
    assert "000" in name_0
    assert "005" in name_5


def test_pretty_name_hs_prefix():
    name = _pretty_name("hs", 0, {"metric": "jaccard", "output": "square"})
    assert name.startswith("hs_000")


def test_pretty_name_app_prefix():
    name = _pretty_name("app", 0, {"metric": "jaccard", "output": "square"})
    assert name.startswith("app_000")


def test_grid_compute_naics_create_files(tmp_path):
    df = pd.DataFrame({
        "customer_code": ["A", "B", "C"],
        "primary_naics_code": ["332710", "332720", "541330"],
        "secondary_naics_code": [None, None, None],
    })
    param_grid = [{"metric": ["hamming"], "output": ["square"]}]

    registry = grid_compute_naics_matrices(
        df,
        param_grid,
        primary_col="primary_naics_code",
        secondary_col="secondary_naics_code",
        id_col="customer_code",
        output=tmp_path / "NAICS",
    )

    assert len(registry) == 1
    assert (tmp_path / "NAICS" / "registry.csv").exists()
    assert Path(registry.iloc[0]["file"]).exists()


def test_grid_compute_hs_erstellt_dateien(tmp_path):
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "hs6_code": [{"850420"}, {"870899"}],
    })
    param_grid = [{"metric": ["jaccard"], "output": ["square"]}]

    registry = grid_compute_hs_matrices(
        df,
        param_grid,
        id_col="customer_code",
        hs_col="hs6_code",
        output=tmp_path / "HS",
    )

    assert len(registry) == 1
    assert (tmp_path / "HS" / "registry.csv").exists()
    assert Path(registry.iloc[0]["file"]).exists()


def test_grid_compute_app_erstellt_dateien(tmp_path):
    df = pd.DataFrame({
        "customer_code": ["A", "B"],
        "application_material_set_h2": [{"2.1_1.1.1"}, {"5.6_1.1.1"}],
    })
    param_grid = [{"metric": ["jaccard"], "output": ["square"]}]

    registry = grid_compute_app_matrices(
        df,
        param_grid,
        id_col="customer_code",
        am_col="application_material_set_h2",
        output=tmp_path / "APP",
    )

    assert len(registry) == 1
    assert (tmp_path / "APP" / "registry.csv").exists()
    assert Path(registry.iloc[0]["file"]).exists()