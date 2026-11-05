"""make_pdp: the port of tests/testthat/test-pdp.R, the PDP tests of
test-offset-link.R and test-data-table-input.R, and the matching tests of
test-review-findings.R."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pricingtools_rmo import _utils as U
from pricingtools_rmo._core import PricingError, PricingMessage, PricingWarning
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.pdp import make_pdp

from conftest import trace, trace_y


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fun(*args, **kw)


def test_make_pdp_computes_the_exposure_weighted_mean_at_exposure_1(dat, m_freq):
    p_pdp = make_pdp(m_freq, dat, "REGIO", metric="Frequency")
    assert isinstance(p_pdp, go.Figure)

    # manual reference: exposure-weighted mean response prediction at Exposure = 1
    nd = dat.copy()
    nd["Exposure"] = 1.0
    nd["REGIO"] = pd.Categorical(["Zuid"] * len(nd), categories=dat.REGIO.cat.categories)
    ref_zuid = np.average(m_freq.predict(nd), weights=np.exp(m_freq.offset))
    pdp_trace = trace(p_pdp, "PDP (model)")
    pdp_zuid = np.asarray(pdp_trace.y)[list(pdp_trace.x).index("Zuid")]
    assert abs(pdp_zuid - ref_zuid) < 1e-8

    # PDP level must be of the same order as the portfolio frequency
    obs_tot = dat.AantalClaims.sum() / dat.Exposure.sum()
    assert np.all(np.abs(np.asarray(pdp_trace.y) / obs_tot - 1) < 1)


def test_make_pdp_works_for_severity_and_deprecates_transform(dat, m_freq, m_sev):
    assert isinstance(make_pdp(m_sev, dat, "LEEFTIJD", metric="Severity"), go.Figure)
    with pytest.warns(PricingWarning, match="deprecated"):
        make_pdp(m_freq, dat, "REGIO", transform=np.exp)


# ---- test-offset-link.R ----------------------------------------------------------


def _dol(seed=51, n=5000):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], n)),
                      "Exposure": np.round(rng.uniform(.3, 1, n), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(-2 + .4 * (d.R == "Z")))
    d["Maanden"] = d.Exposure * 12
    d["SCHADELAST"] = d.AantalClaims * rng.gamma(3, 700, n)
    return d


def test_make_pdp_neutralises_a_non_log_offset_and_keeps_the_collapse():
    dol = _dol()
    m = fit_glm("AantalClaims ~ R + offset(Exposure)", dol, family="poisson")
    pd_ = trace(_quiet(make_pdp, m, dol, "R", metric="Frequency"), "PDP (model)")
    b = m.coef
    manual = {lv: np.exp(b.iloc[0] + b.iloc[1] * (lv == "Z")) for lv in ["N", "Z"]}
    np.testing.assert_allclose(np.asarray(pd_.y),
                               [manual[x] for x in pd_.x], rtol=1e-8)


def _bm_book(seed=71, nb=15000):
    rng = np.random.default_rng(seed)
    db = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "O", "Z"], nb)),
                       "L": np.round(rng.uniform(18, 80, nb)),
                       "Exposure": np.round(rng.uniform(.25, 1, nb), 3)})
    db["BM"] = np.where(db.L < 30, 1.3, np.where(db.L > 60, .6, .9))
    db["AantalClaims"] = rng.poisson(db.Exposure * db.BM * np.exp(
        -2.2 + .012 * db.L + .3 * (db.R == "Z")))
    db["SCHADELAST"] = 0.0
    return db


def test_a_second_offset_does_not_end_up_on_the_exposure_axis_pdp():
    # the PDP average follows the same column as its own observed line, and
    # keeps each policy's own BM, as that line does (PDP-05): only the
    # exposure offset comes out
    db = _bm_book()
    mb = fit_glm("AantalClaims ~ R + L + offset(log(Exposure)) + offset(log(BM))",
                 db, family="poisson")
    pd_ = trace(make_pdp(mb, db, "R", metric="Frequency"), "PDP (model)")

    def hand(wt, bm=True):
        out = {}
        for lv in db.R.cat.categories:
            z = db.copy()
            z["R"] = pd.Categorical([lv] * len(z), categories=db.R.cat.categories)
            r = (U.predict_no_offset(mb, z, [0]) if bm
                 else U.predict_no_offset(mb, z))
            out[lv] = np.average(r, weights=wt)
        return out

    assert mb.offset_exprs[0] == "log(Exposure)"
    h = hand(db.Exposure)
    np.testing.assert_allclose(np.asarray(pd_.y), [h[x] for x in pd_.x], rtol=1e-8)
    # the two weightings really do differ here, so that was a live choice
    h2 = hand(db.Exposure * db.BM)
    assert not np.allclose(list(h.values()), list(h2.values()), rtol=1e-8)
    # keep_offsets = FALSE puts every policy at BM = 1, as before
    pd0 = trace(make_pdp(mb, db, "R", metric="Frequency", keep_offsets=False),
                "PDP (model)")
    h0 = hand(db.Exposure, bm=False)
    np.testing.assert_allclose(np.asarray(pd0.y), [h0[x] for x in pd0.x], rtol=1e-8)
    # the axis says what the rate is on
    p = make_pdp(mb, db, "R")
    assert p.layout.yaxis.title.text == "Frequency (claims / exposure), incl. BM"
    with pytest.raises(PricingError, match="TRUE or FALSE"):
        make_pdp(mb, db, "R", keep_offsets=None)


# ---- test-review-findings.R ------------------------------------------------------


def rv_book(n=20000, seed=1):
    """The review's book (rv_book() of test-review-findings.R)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 85, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    eta = (np.log(0.08) + 0.6 * np.exp(-(d.LEEFTIJD - 18) / 8)
           + d.REGIO.map({"Noord": 0, "Zuid": .2, "West": .35}).astype(float)
           + 0.3 * np.log(d.GEWICHT / 1300))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta))
    d["SCHADELAST"] = np.where(d.AantalClaims > 0,
                               d.AantalClaims * rng.gamma(1.5, 2500 / 1.5, n), 0.0)
    return d


def test_the_pdp_sits_on_the_observed_line_with_a_bonus_malus_offset():
    d1 = rv_book()
    rng = np.random.default_rng(2)
    d1["BM"] = rng.choice([1, .85, .7, .55], len(d1))
    eta = np.log(0.08) + d1.REGIO.map({"Noord": 0, "Zuid": .2, "West": .35}).astype(float)
    d1["AantalClaims"] = rng.poisson(d1.Exposure * d1.BM * np.exp(eta))
    m = fit_glm("AantalClaims ~ REGIO + offset(log(Exposure)) + offset(log(BM))",
                d1, family="poisson")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingMessage)
        p = make_pdp(m, d1, "REGIO")
    np.testing.assert_allclose(trace_y(p, "PDP (model)"), trace_y(p, "Observed"),
                               rtol=0.05)


def test_a_glm_nb_model_works_in_make_pdp():
    rv = rv_book()
    m = _quiet(fit_glm, "AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
               rv, family="negbin")
    p = _quiet(make_pdp, m, rv, "REGIO")
    assert isinstance(p, go.Figure)
    assert np.all(np.isfinite(trace_y(p, "PDP (model)")))


# ---- test-data-table-input.R -----------------------------------------------------


def _ddt(seed=23, n=20000):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp"], n)),
        "KM": np.round(np.maximum(rng.lognormal(np.log(12000), .4, n), 1000), -2),
        "Exposure": np.round(rng.uniform(.3, 1, n), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.6 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD) + 0.3 * (d.REGIO == "Stad")
        + 0.4 * (np.log(d.KM) - np.log(12000))))
    d["SCHADELAST"] = d.AantalClaims * 1500.0
    return d


def test_make_pdp_accepts_other_table_input():
    # R tests a data.table and a tibble; in Python anything with to_pandas()
    # or a dict of columns is read as a DataFrame at the entry point
    ddt = _ddt()
    m_dt = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   ddt, family="poisson")

    class Table:                       # a stand-in for a polars/arrow table
        def __init__(self, df):
            self.df = df

        def to_pandas(self):
            return self.df

    p1 = make_pdp(m_dt, Table(ddt), "REGIO", metric="Frequency")
    p2 = make_pdp(m_dt, {c: ddt[c] for c in ddt.columns}, "REGIO")
    assert isinstance(p1, go.Figure)
    np.testing.assert_allclose(trace_y(p1, "Observed"), trace_y(p2, "Observed"))
    np.testing.assert_allclose(trace_y(p1, "PDP (model)"), trace_y(p2, "PDP (model)"))


# ---- details of the port --------------------------------------------------------


def test_make_pdp_grid_collapse_and_layout(dat, m_freq):
    # a continuous predictor gets a grid of grid_res points and a line
    p = make_pdp(m_freq, dat, "LEEFTIJD", grid_res=50)
    pdp = trace(p, "PDP (model)")
    assert len(pdp.x) == 50 if dat.LEEFTIJD.nunique() > 50 else len(pdp.x) == dat.LEEFTIJD.nunique()
    assert pdp.mode == "lines+markers"
    # the collapse to profiles changes nothing: the same mean over every row
    nd = dat.copy()
    nd["Exposure"] = 1.0
    nd["LEEFTIJD"] = float(pdp.x[3])
    np.testing.assert_allclose(pdp.y[3], np.average(m_freq.predict(nd),
                                                    weights=dat.Exposure),
                               rtol=1e-10)
    assert p.layout.title.text == "Partial Dependence – LEEFTIJD – Frequency"
    assert p.layout.yaxis2.title.text == "Exposure"
    assert p.pricing_config["toImageButtonOptions"]["filename"] == "pdp"
    # a categorical predictor: markers only, in level order
    pr = make_pdp(m_freq, dat, "REGIO")
    assert trace(pr, "Observed").mode == "markers"
    assert pr.layout.xaxis.categoryarray == ("Noord", "West", "Zuid")
    with pytest.raises(PricingError, match="not found"):
        make_pdp(m_freq, dat.assign(X=1), "X")
