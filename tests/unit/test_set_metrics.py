import math
import pytest


from src.set_metrics import (
    _parse_cell,
    _idf_weights,
    _pairwise_dist_core,
    _d_jaccard,
    _d_dice,
    _d_overlap,
    _d_jaccard_weighted,
    _d_dice_weighted,
    _d_overlap_weighted,
)


def test_parse_cell_different_formats_same_output():
    expectation = {"850420", "720839"}

    # Python-Set
    assert set(_parse_cell({"850420", "720839"})) == expectation

    # Python-Liste
    assert set(_parse_cell(["850420", "720839"])) == expectation

    # Comma-String
    assert set(_parse_cell("850420, 720839")) == expectation

    # Set-String
    assert set(_parse_cell("{'850420', '720839'}")) == expectation


def test_parse_cell_app_codes():
    expectation = {"1.3", "1.10"}

    assert set(_parse_cell({"1.3", "1.10"})) == expectation
    assert set(_parse_cell(["1.3", "1.10"])) == expectation
    assert set(_parse_cell("{'1.3', '1.10'}")) == expectation
    assert set(_parse_cell({"1.3", "1.10", "", " "})) == expectation


def test_parse_cell_null_variants():
    assert  _parse_cell(None) == []
    assert _parse_cell("") == []
    assert _parse_cell(" ") == []
    assert _parse_cell(set()) == []
    assert _parse_cell("None") == []
    assert _parse_cell("NaN") == []
    assert _parse_cell("nan") == []
    assert _parse_cell([]) == []


def test_parse_cell_remove_duplicates():
    result = _parse_cell(["850420", "850420", "720839"])
    assert len(result) == 2
    assert set(result) == {"850420", "720839"}


def test_parse_cell_null_values_in_list():
    result = _parse_cell(["850420", None, "nan", "720839", "None", "", " ", "850420"])
    assert set(result) == {"850420", "720839"}


def test_idf_weights():
    sets = {
        "1": {"850420", "720839"},
        "2": {"850420"},
        "3": {"850420"},
    }
    weights = _idf_weights(sets)

    assert weights["850420"] == pytest.approx(math.log(3 / 3), abs=1e-6)

    assert weights["720839"] == pytest.approx(math.log(3 / 1), abs=1e-6)


def test_idf_weights_clip_max():
    sets = {
        "1": {"850420", "720839"},
        "2": {"850420"},
        "3": {"850420"},
    }
    weights = _idf_weights(sets, clip_max=0.5)

    assert weights["850420"] == pytest.approx(0.0, abs=1e-7)

    assert weights["720839"] == pytest.approx(0.5, abs=1e-7)


def _make_hs_sets():
    return {
        "1": {"850420", "850431", "720839"},
        "2": {"850420", "850431"},
        "3": {"870899"},
    }


def test_pairwise_jaccard_reference():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "jaccard", "square")

    assert result.loc["1", "2"] == pytest.approx(1 / 3, abs=1e-7)

    assert result.loc["1", "3"] == pytest.approx(1.0, abs=1e-7)

    assert result.loc["2", "3"] == pytest.approx(1.0, abs=1e-7)


def test_pairwise_dice_reference():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "dice", "square")

    assert result.loc["1", "2"] == pytest.approx(0.2, abs=1e-7)

    assert result.loc["1", "3"] == pytest.approx(1.0, abs=1e-7)

    assert result.loc["2", "3"] == pytest.approx(1.0, abs=1e-7)


def test_pairwise_overlap_reference():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "overlap", "square")

    assert result.loc["1", "2"] == pytest.approx(0.0, abs=1e-7)

    assert result.loc["1", "3"] == pytest.approx(1.0, abs=1e-7)

    assert result.loc["2", "3"] == pytest.approx(1.0, abs=1e-7)


def test_pairwise_jaccard_weighted_reference():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "jaccard_weighted", "square")

    assert result.loc["1", "2"] == pytest.approx(0.5753, abs=1e-4)
    assert result.loc["1", "3"] == pytest.approx(1.0, abs=1e-4)
    assert result.loc["2", "3"] == pytest.approx(1.0, abs=1e-4)


def test_pairwise_dice_weighted_reference():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "dice_weighted", "square")

    assert result.loc["1", "2"] == pytest.approx(0.4038, abs=1e-4)
    assert result.loc["1", "3"] == pytest.approx(1.0, abs=1e-4)
    assert result.loc["2", "3"] == pytest.approx(1.0, abs=1e-4)

def test_pairwise_overlap_weighted_reference():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "overlap_weighted", "square")

    assert result.loc["1", "2"] == pytest.approx(0.0, abs=1e-4)
    assert result.loc["1", "3"] == pytest.approx(1.0, abs=1e-4)
    assert result.loc["2", "3"] == pytest.approx(1.0, abs=1e-4)

def test_pairwise_square_format():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "jaccard", "square")

    assert result.shape == (3, 3)
    assert list(result.index) == ["1", "2", "3"]
    assert list(result.columns) == ["1", "2", "3"]


def test_pairwise_long_format():
    sets = _make_hs_sets()
    result = _pairwise_dist_core(sets, "jaccard", "long")

    assert len(result) == 3
    assert list(result.columns) == ["Customer_ID_1", "Customer_ID_2", "dissimilarity"]

def test_pairwise_invalid_metric():
    sets = _make_hs_sets()
    with pytest.raises(ValueError):
        _pairwise_dist_core(sets, "xxx", "square")


def test_pairwise_invalid_output():
    sets = _make_hs_sets()
    with pytest.raises(ValueError):
        _pairwise_dist_core(sets, "jaccard", "xxx")

def test_d_jaccard():
    assert _d_jaccard({"850420", "850431", "720839"}, {"850420", "850431"}) == pytest.approx(1/3, abs=1e-7)
    assert _d_jaccard(set(), set()) == 1.0
    assert _d_jaccard({"850420"}, {"850420"}) == 0.0

def test_d_dice():
    assert _d_dice({"850420", "850431", "720839"}, {"850420", "850431"}) == pytest.approx(0.2, abs=1e-7)
    assert _d_dice(set(), set()) == 1.0
    assert _d_dice({"850420"}, {"850420"}) == 0.0

def test_d_overlap():
    assert _d_overlap({"850420", "850431", "720839"}, {"850420", "850431"}) == pytest.approx(0.0, abs=1e-7)
    assert _d_overlap(set(), set()) == 1.0
    assert _d_overlap({"850420"}, {"850420"}) == 0.0

def test_d_jaccard_weighted():
    weights = {"850420": 1.0, "850431": 1.0, "720839": 2.0}
    assert _d_jaccard_weighted({"850420", "850431", "720839"}, {"850420", "850431"}, weights) == pytest.approx(0.5, abs=1e-7)
    assert _d_jaccard_weighted(set(), set(), weights) == 1.0
    assert _d_jaccard_weighted({"850420"}, {"850420"}, weights) == 0.0


def test_d_dice_weighted():
    weights = {"850420": 1.0, "850431": 1.0, "720839": 2.0}
    assert _d_dice_weighted({"850420", "850431", "720839"}, {"850420", "850431"}, weights) == pytest.approx(1/3, abs=1e-7)
    assert _d_dice_weighted(set(), set(), weights) == 1.0
    assert _d_dice_weighted({"850420"}, {"850420"}, weights) == 0.0


def test_d_overlap_weighted():
    weights = {"850420": 1.0, "850431": 1.0, "720839": 2.0}
    assert _d_overlap_weighted({"850420", "850431", "720839"}, {"850420", "850431"}, weights) == pytest.approx(0.0, abs=1e-7)
    assert _d_overlap_weighted(set(), set(), weights) == 1.0
    assert _d_overlap_weighted({"850420"}, {"850420"}, weights) == 0.0