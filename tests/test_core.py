"""The model object, formulas and the shared helpers."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo import _utils as U
from pricingtools_rmo._core import PricingError, PricingWarning
from pricingtools_rmo._formula import ParsedFormula, expr_vars, r_cut
from pricingtools_rmo.model import fit_glm


def test_formula_parts_and_offsets():
    pf = ParsedFormula("AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + "
                       "offset(log(Exposure)) + offset(log(BM))")
    assert pf.response == "AantalClaims"
    assert pf.offsets == ["log(Exposure)", "log(BM)"]
    assert "offset" not in pf.rhs
    assert expr_vars("log(`AUTO GEWICHT` / 12)") == ["AUTO GEWICHT"]
    assert expr_vars("I(KM^2)") == ["KM"]


def test_cut_labels_as_in_r():
    c = r_cut([18, 24, 30, 70], [17, 24, 40, 65, 99])
    assert list(c.categories) == ["(17,24]", "(24,40]", "(40,65]", "(65,99]"]
    assert list(c.astype(str)) == ["(17,24]", "(17,24]", "(24,40]", "(65,99]"]


def test_prediction_reevaluates_offsets(dat, m_freq):
    nd = dat.iloc[:5].copy()
    p1 = m_freq.predict(nd)
    p2 = m_freq.predict(nd.assign(Exposure=1.0))
    np.testing.assert_allclose(p1 / p2, nd.Exposure.to_numpy(), rtol=1e-12)
    np.testing.assert_allclose(m_freq.predict(nd, drop_offsets=[0]), p2,
                               rtol=1e-12)


def test_training_predictions_match_fitted(dat, m_freq):
    np.testing.assert_allclose(m_freq.predict(m_freq.data), m_freq.fitted,
                               rtol=1e-10)


def test_rhs_vars_include_offset_vars(m_freq):
    assert set(m_freq.rhs_vars()) == {"LEEFTIJD", "REGIO", "BOEKJAAR", "Exposure"}
    assert m_freq.offset_vars() == ["Exposure"]


def test_unseen_levels_named(dat, m_freq):
    nd = dat.iloc[:10].copy()
    nd["BOEKJAAR"] = 2025
    un = U.unseen_levels(m_freq, nd)
    assert list(un) == ["BOEKJAAR"]
    assert un["BOEKJAAR"].values == ["2025"] and un["BOEKJAAR"].n == 10
    msg = U.unseen_message(un, "new frequency", "premium_impact")
    assert "'BOEKJAAR' = 2025 (10 rows)" in msg and "offset" not in msg


def test_model_rate_keeps_bm_and_drops_exposure():
    rng = np.random.default_rng(3)
    n = 4000
    d = pd.DataFrame({"R": rng.choice(["a", "b"], n), "Exposure": rng.uniform(.2, 1, n),
                      "BM": rng.choice([1, .7], n)})
    d["y"] = rng.poisson(d.Exposure * d.BM * np.exp(-1 + .3 * (d.R == "b")))
    m = fit_glm("y ~ R + offset(log(Exposure)) + offset(log(BM))", d)
    r = U.model_rate(m, d, "Exposure", "t")
    base = m.predict(d.assign(Exposure=1.0, BM=1.0))
    np.testing.assert_allclose(r, base * d.BM, rtol=1e-10)
    r0 = U.model_rate(m, d, "Exposure", "t", keep_offsets=False)
    np.testing.assert_allclose(r0, base, rtol=1e-10)


def test_apply_at_validates(dat):
    out = U.apply_at(dat.iloc[:3], {"BOEKJAAR": 2020}, "f")
    assert (out.BOEKJAAR == 2020).all()
    with pytest.raises(PricingError, match="NOPE"):
        U.apply_at(dat, {"NOPE": 1}, "f")
    with pytest.raises(PricingError, match="not a valid value"):
        U.apply_at(dat.iloc[:3], {"REGIO": "Oost"}, "f")


def test_grid_helpers():
    assert U.nice_step((18, 80), 50, True) == 2      # as in R
    assert U.list_values(np.arange(18, 81), 50, True)  # ages listed per value
    assert U.nice_step((800, 2400), 50, False) == 50
    np.testing.assert_array_equal(U.step_grid(800, 2400, 50, 1325)[:3], [800, 850, 900])
    assert 1325 in U.step_grid(800, 2400, 50, 1325)
    assert U.level_chr([1e5, 2.5, 832.65]) == ["100000", "2.5", "832.65"]


def test_exposure_bins_and_gini():
    rng = np.random.default_rng(1)
    s = rng.uniform(size=1000)
    w = np.ones(1000)
    b = U.exposure_bins(s, w, 10)
    assert set(b) == set(range(1, 11))
    assert np.isnan(U.gini_exposure(s, np.zeros(1000), w))
    g = U.gini_exposure(s, s, w)
    assert 0 < g < 1


def test_grouping_column_derived_and_checked():
    rng = np.random.default_rng(5)
    n = 3000
    d = pd.DataFrame({"S": rng.integers(1, 6, n), "Exposure": rng.uniform(.3, 1, n)})
    d["y"] = rng.poisson(d.Exposure * .3)
    d["G"] = np.where(d.S >= 4, "4-5", d.S.astype(str))
    m = fit_glm("y ~ G + offset(log(Exposure))", d)
    new = pd.DataFrame({"S": [1, 4, 5]})
    g = U.grouping_column(new, "S", "G", m, None, "t")
    assert list(g) == ["1", "4-5", "4-5"]
    with pytest.raises(PricingError, match="do not occur"):
        U.grouping_column(pd.DataFrame({"S": [6]}), "S", "G", m, None, "t")
