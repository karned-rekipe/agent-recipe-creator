from adapters.output.recipe_agent._fuzzy import _make_fuzzy_matcher


def test_empty_candidates_returns_none():
    matcher = _make_fuzzy_matcher(80)
    assert matcher("farine", []) is None


def test_exact_match():
    matcher = _make_fuzzy_matcher(80)
    candidates = [{"uuid": "u-1", "name": "farine de blé"}]
    result = matcher("farine de blé", candidates)
    assert result is not None
    assert result["uuid"] == "u-1"


def test_fuzzy_match_above_threshold():
    matcher = _make_fuzzy_matcher(80)
    candidates = [{"uuid": "u-1", "name": "farine de blé T55"}]
    result = matcher("farine de blé", candidates)
    assert result is not None
    assert result["uuid"] == "u-1"


def test_no_match_below_threshold():
    matcher = _make_fuzzy_matcher(99)
    candidates = [{"uuid": "u-1", "name": "farine de blé"}]
    result = matcher("curcuma en poudre", candidates)
    assert result is None


def test_returns_best_matching_candidate():
    matcher = _make_fuzzy_matcher(80)
    candidates = [
        {"uuid": "u-1", "name": "sucre blanc"},
        {"uuid": "u-2", "name": "sucre roux"},
        {"uuid": "u-3", "name": "sel fin"},
    ]
    result = matcher("sucre roux", candidates)
    assert result is not None
    assert result["uuid"] == "u-2"


def test_single_candidate_below_threshold():
    matcher = _make_fuzzy_matcher(80)
    candidates = [{"uuid": "u-1", "name": "poivre noir moulu"}]
    result = matcher("beurre demi-sel", candidates)
    assert result is None


def test_threshold_boundary_exact_at_100():
    matcher = _make_fuzzy_matcher(100)
    candidates = [{"uuid": "u-1", "name": "sel"}]
    result = matcher("sel", candidates)
    assert result is not None
    assert result["uuid"] == "u-1"


def test_multiple_candidates_picks_closest():
    matcher = _make_fuzzy_matcher(70)
    candidates = [
        {"uuid": "u-1", "name": "tomate cerise"},
        {"uuid": "u-2", "name": "tomate"},
        {"uuid": "u-3", "name": "pomme"},
    ]
    result = matcher("tomate", candidates)
    assert result is not None
    assert result["name"] in ("tomate", "tomate cerise")
