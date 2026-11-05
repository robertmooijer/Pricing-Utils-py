"""premium_impact() and the rates behind it: the port of test-impact.R, the
premium_impact tests of test-collinearity-spotlight.R, test-offset-link.R,
test-y-range.R and test-review-findings.R."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pricingtools_rmo import _utils as U
from pricingtools_rmo._core import PricingError, PricingWarning
from pricingtools_rmo.impact import premium_impact
from pricingtools_rmo.lift import double_lift
from pricingtools_rmo.model import fit_glm


def warn_msgs(f, *a, **k):
    """Every warning a call raises (R's rv_warns), and its result."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = f(*a, **k)
    return out, [str(x.message) for x in w]


def no_warnings(f, *a, **k):
    out, msgs = warn_msgs(f, *a, **k)
    assert not msgs, msgs
    return out


def quiet(f, *a, **k):
    """suppressWarnings()."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


# ---- test-impact.R -------------------------------------------------------------


def test_identical_models_produce_zero_dislocation(dat, m_freq, m_sev):
    imp0 = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                          model_freq_old=m_freq, model_sev_old=m_sev,
                          by="REGIO")
    assert np.max(np.abs(imp0.policy.ChangePct)) < 1e-8
    assert len(imp0.by_level) == 3
    assert isinstance(imp0.plot, go.Figure)


def test_an_offset_on_another_column_is_neutralised_and_said_out_loud(dat):
    # setting exposure_col to 1 leaves offset(log(Ex)) untouched, so the
    # "rate" was still a per-record prediction - silently, because an
    # Exposure column did exist and was duly set to 1
    d3 = dat.copy()
    d3["Ex"] = d3.Exposure
    m3 = fit_glm("AantalClaims ~ REGIO + offset(log(Ex))", d3,
                 family="poisson")
    with pytest.warns(PricingWarning, match="offsets on 'Ex'"):
        r = U.model_rate(m3, d3, "Exposure", "t")
    # one rate per region, not one per record
    assert len(np.unique(np.round(r, 10))) == len(d3.REGIO.cat.categories)
    # and it equals the model's own rate at Ex = 1
    d1 = d3.assign(Ex=1.0)
    np.testing.assert_allclose(r, m3.predict(d1), rtol=1e-10)


def test_rebase_separates_rate_level_change_from_dislocation(dat, m_freq,
                                                             m_sev):
    # old premium column = 2x the new rate -> rate-level change -50%,
    # rebased dislocation 0
    d2 = dat.copy()
    new_rate = m_freq.predict(d2.assign(Exposure=1.0)) * m_sev.predict(d2)
    d2["OldPrem"] = 2 * new_rate * d2.Exposure
    # m_freq has factor(BOEKJAAR), and a premium column has no year (RATE-10)
    with pytest.warns(PricingWarning,
                      match="priced at the level of its own year"):
        imp2 = premium_impact(d2, model_freq_new=m_freq, model_sev_new=m_sev,
                              old_premium_col="OldPrem")
    assert abs(imp2.stats.rate_change + 0.5) < 1e-10
    assert np.max(np.abs(imp2.policy.ChangePct)) < 1e-8
    assert len(imp2.largest_increases) == 10
    assert len(imp2.largest_decreases) == 10


# ---- test-collinearity-spotlight.R: contribution and spotlight -------------------


@pytest.fixture(scope="module")
def dc():
    rng = np.random.default_rng(91)
    nc = 20000
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, nc)),
        "GEWICHT": np.round(np.clip(rng.normal(1350, 250, nc), 800, 2400), -1),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp", "Rand"], nc),
                                categories=["Dorp", "Rand", "Stad"]),
        "Exposure": np.round(rng.uniform(.3, 1, nc), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.7 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD)
        + 0.0003 * (d.GEWICHT - 1350)))
    d["HUIDIGE_PREMIE"] = d.Exposure * 250
    return d


@pytest.fixture(scope="module")
def m_new(dc):
    return fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   dc, family="poisson")


def test_contributions_sum_to_the_portfolio_change(dc, m_new):
    imp = premium_impact(dc, model_freq_new=m_new,
                         old_premium_col="HUIDIGE_PREMIE", by="REGIO")
    assert {"ExposureShare", "Contribution"} <= set(imp.by_level.columns)
    assert imp.by_level.ExposureShare.sum() == pytest.approx(1, abs=1e-8)

    # levels of one variable decompose the change in the total premium
    # (RATE-04): exactly 0 after a rebase, the rate-level change without one
    assert imp.by_level.Contribution.sum() == pytest.approx(0, abs=1e-8)
    imp_nr = premium_impact(dc, model_freq_new=m_new,
                            old_premium_col="HUIDIGE_PREMIE", by="REGIO",
                            rebase=False)
    assert imp_nr.by_level.Contribution.sum() == pytest.approx(
        100 * imp_nr.stats.rate_change, rel=1e-8)

    # also when the old premium is not flat, which is where weighting the
    # mean change by exposure did not add up
    dc3 = dc.copy()
    dc3["HUIDIGE_PREMIE"] = dc3.Exposure * 250 * np.where(
        dc3.REGIO == "Stad", 1.3, 1)
    imp3 = premium_impact(dc3, model_freq_new=m_new,
                          old_premium_col="HUIDIGE_PREMIE", by="REGIO")
    assert imp3.by_level.Contribution.sum() == pytest.approx(0, abs=1e-8)
    bl = imp3.by_level
    np.testing.assert_allclose(bl.Contribution,
                               bl.PremiumShare * bl.TotalChangePct,
                               rtol=1e-8)
    assert bl.PremiumShare.sum() == pytest.approx(1, abs=1e-8)
    # largest contribution first within a variable
    assert np.all(np.diff(np.abs(bl.Contribution)) <= 0)


def test_the_spotlight_reports_a_subset_without_filtering_the_book(dc, m_new):
    imp = premium_impact(dc, model_freq_new=m_new,
                         old_premium_col="HUIDIGE_PREMIE",
                         spotlight="REGIO == 'Stad' & LEEFTIJD < 25")
    sel = ((dc.REGIO == "Stad") & (dc.LEEFTIJD < 25)).to_numpy()

    assert imp.spotlight is not None
    assert imp.spotlight.n_rows == sel.sum()
    # the book itself is untouched: every row is still in policy
    assert len(imp.policy) == len(dc)
    assert imp.spotlight.exposure_share == pytest.approx(
        dc.Exposure[sel].sum() / dc.Exposure.sum(), rel=1e-8)
    assert isinstance(imp.spotlight.summary, pd.DataFrame)
    # the histogram carries the subset as a second series
    assert [t.name for t in imp.plot.data] == ["Whole portfolio", "Spotlight"]
    assert np.sum(imp.plot.data[1].y) == pytest.approx(
        dc.Exposure[sel].sum(), rel=1e-10)

    # a boolean vector must give the same answer as the expression
    imp2 = premium_impact(dc, model_freq_new=m_new,
                          old_premium_col="HUIDIGE_PREMIE", spotlight=sel)
    assert imp.spotlight.mean_change == imp2.spotlight.mean_change
    # and so must a Series, R's && and the expression in other words
    imp3 = premium_impact(dc, model_freq_new=m_new,
                          old_premium_col="HUIDIGE_PREMIE",
                          spotlight=pd.Series(sel))
    imp4 = premium_impact(dc, model_freq_new=m_new,
                          old_premium_col="HUIDIGE_PREMIE",
                          spotlight='!(REGIO %in% c("Dorp", "Rand")) '
                                    '&& !(LEEFTIJD >= 25)')
    assert imp3.spotlight.mean_change == imp.spotlight.mean_change
    assert imp4.spotlight.n_rows == imp.spotlight.n_rows
    assert imp4.spotlight.mean_change == imp.spotlight.mean_change


def test_spotlight_input_is_validated(dc, m_new):
    with pytest.raises(PricingError, match="one TRUE/FALSE per row"):
        premium_impact(dc, model_freq_new=m_new,
                       old_premium_col="HUIDIGE_PREMIE",
                       spotlight=[True, False])
    with pytest.warns(PricingWarning, match="no usable rows"):
        premium_impact(dc, model_freq_new=m_new,
                       old_premium_col="HUIDIGE_PREMIE",
                       spotlight="LEEFTIJD > 999")
    # an expression that cannot be evaluated says why
    with pytest.raises(PricingError, match="could not be evaluated"):
        premium_impact(dc, model_freq_new=m_new,
                       old_premium_col="HUIDIGE_PREMIE",
                       spotlight="NOPE == 1")
    # a number per row is not a TRUE/FALSE per row
    with pytest.raises(PricingError, match="one TRUE/FALSE per row"):
        premium_impact(dc, model_freq_new=m_new,
                       old_premium_col="HUIDIGE_PREMIE",
                       spotlight="LEEFTIJD + 1")


# ---- test-offset-link.R: rates, offsets and like for like ------------------------


@pytest.fixture(scope="module")
def dol():
    rng = np.random.default_rng(51)
    nol = 5000
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], nol),
                                          categories=["N", "Z"]),
                      "Exposure": np.round(rng.uniform(.3, 1, nol), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure
                                    * np.exp(-2 + .4 * (d.R == "Z")))
    return d


def test_offsets_from_the_formula_add_up(dol):
    # R's test also passes an offset as glm()'s argument; fit_glm takes
    # offsets from the formula only, any number of them, and predict() sums
    # them all
    dd = dol.copy()
    dd["const"] = 0.1
    m = fit_glm("AantalClaims ~ R + offset(log(Exposure)) + offset(const)",
                dd, family="poisson")
    eta_bare = m.model_matrix(dd)[:, m.keep] @ m.beta
    eta_pred = m.predict(dd, type="link")
    # what predict adds is exactly what we take out
    np.testing.assert_allclose(U.offset_value(m, dd), eta_pred - eta_bare,
                               rtol=1e-10)
    # taking everything out removes both
    np.testing.assert_allclose(
        U.model_rate(m, dd, "Exposure", "t", keep_offsets=False),
        np.exp(eta_bare), rtol=1e-10)
    # by default only the exposure goes, and the second offset stays in
    np.testing.assert_allclose(U.model_rate(m, dd, "Exposure", "t"),
                               np.exp(eta_bare + dd.const), rtol=1e-10)
    assert set(U.offset_vars(m)) == {"Exposure", "const"}


def _bm_book(seed, nk, with_l=False):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], nk),
                                          categories=["N", "Z"]),
                      "BM_KLASSE": rng.integers(1, 6, nk),
                      "Exposure": np.round(rng.uniform(.1, 1, nk), 3)})
    d["BM"] = np.array([1, .9, .8, .7, .6])[d.BM_KLASSE - 1]
    d["AantalClaims"] = rng.poisson(d.Exposure * d.BM
                                    * np.exp(-2 + .3 * (d.R == "Z")))
    return d, rng


def test_a_bonus_malus_offset_stays_in_the_rate_the_exposure_goes():
    dk, _ = _bm_book(81, 20000)
    mk = fit_glm("AantalClaims ~ R + offset(log(Exposure)) + offset(log(BM))",
                 dk, family="poisson")
    base_rate = np.exp(mk.coef.iloc[0] + mk.coef.iloc[1] * (dk.R == "Z"))
    # default: a year at the policy's own BM
    np.testing.assert_allclose(U.model_rate(mk, dk, "Exposure", "t"),
                               base_rate * dk.BM, rtol=1e-10)
    d1 = dk.assign(Exposure=1.0)
    np.testing.assert_allclose(U.model_rate(mk, dk, "Exposure", "t"),
                               mk.predict(d1), rtol=1e-10)
    # everything out: the class without discount
    np.testing.assert_allclose(
        U.model_rate(mk, dk, "Exposure", "t", keep_offsets=False),
        base_rate, rtol=1e-10)
    assert U.kept_offsets([mk], "Exposure") == ["BM"]
    assert U.kept_offsets([mk], "Exposure", False) == []

    # one expression holding both cannot be split, and says so
    mm = fit_glm("AantalClaims ~ R + offset(log(Exposure * BM))", dk,
                 family="poisson")
    with pytest.warns(PricingWarning, match="cannot be separated"):
        r = U.model_rate(mm, dk, "Exposure", "t")
    np.testing.assert_allclose(
        r, np.exp(mm.coef.iloc[0] + mm.coef.iloc[1] * (dk.R == "Z")),
        rtol=1e-10)


def test_premium_impact_compares_a_premium_incl_discount_like_for_like():
    dk, rng = _bm_book(83, 20000)
    dk["SCHADELAST"] = np.where(dk.AantalClaims > 0,
                                dk.AantalClaims * rng.gamma(3, 700, len(dk)),
                                0.0)
    mf = fit_glm("AantalClaims ~ R + offset(log(Exposure)) + offset(log(BM))",
                 dk, family="poisson")
    ds = dk[dk.AantalClaims > 0].copy()
    ds["AvgLoss"] = ds.SCHADELAST / ds.AantalClaims
    ms = fit_glm("AvgLoss ~ R", ds, family="gamma", weights="AantalClaims")

    # an "old" annual premium identical to the new tariff, discount included:
    # the true change is zero for every policy
    d1 = dk.assign(Exposure=1.0)
    dk["OudePremie"] = mf.predict(d1) * ms.predict(d1)

    pi = premium_impact(dk, model_freq_new=mf, model_sev_new=ms,
                        old_premium_col="OudePremie",
                        old_premium_basis="rate")
    assert np.max(np.abs(pi.policy.ChangePct)) < 1e-8
    assert pi.stats.offsets_kept == ["BM"]
    s = pi.summary
    assert s.Value[s.Metric == "Offsets kept in the rate"].item() == "BM"

    # the old behaviour, taking the discount out of the new side only, made
    # an unchanged tariff look like a shift onto the discounted classes
    pi0 = premium_impact(dk, model_freq_new=mf, model_sev_new=ms,
                         old_premium_col="OudePremie",
                         old_premium_basis="rate", keep_offsets=False)
    m0 = pi0.policy.ChangePct.groupby(dk.BM_KLASSE.to_numpy()).mean()
    assert m0[5] - m0[1] > 50
    # and with keep_offsets=False a premium before discount is the match
    dk["OudePremie_exBM"] = dk.OudePremie / dk.BM
    pi1 = premium_impact(dk, model_freq_new=mf, model_sev_new=ms,
                         old_premium_col="OudePremie_exBM",
                         old_premium_basis="rate", keep_offsets=False)
    assert np.max(np.abs(pi1.policy.ChangePct)) < 1e-8
    assert pi1.stats.offsets_kept == []

    # old models against new models: the changes do not depend on the choice
    pa = premium_impact(dk, model_freq_new=mf, model_sev_new=ms,
                        model_freq_old=mf, model_sev_old=ms)
    pb = premium_impact(dk, model_freq_new=mf, model_sev_new=ms,
                        model_freq_old=mf, model_sev_old=ms,
                        keep_offsets=False)
    np.testing.assert_allclose(pa.policy.ChangePct, pb.policy.ChangePct,
                               atol=1e-10)

    # double_lift follows the same rule
    # every policy has the same new/old ratio here, which double_lift rightly
    # warns about; the point is the level and what is reported
    dl = quiet(double_lift, dk, model_freq_new=mf, model_sev_new=ms,
               old_premium_col="OudePremie", old_premium_basis="rate",
               n_bins=2)
    assert dl.stats.offsets_kept == ["BM"]
    assert dl.stats.rate_level_change == pytest.approx(0, abs=1e-10)

    with pytest.raises(PricingError, match="TRUE or FALSE"):
        premium_impact(dk, model_freq_new=mf, old_premium_col="OudePremie",
                       keep_offsets=None)


# ---- test-y-range.R --------------------------------------------------------------


def test_premium_impact_exposes_x_range_and_y_range_on_its_histogram(
        dat, m_freq, m_sev):
    imp = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                         model_freq_old=m_freq, model_sev_old=m_sev)
    assert imp.plot.layout.xaxis.autorange is True

    imp2 = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                          model_freq_old=m_freq, model_sev_old=m_sev,
                          x_range=(-25, 25), y_range=(0, 5000))
    assert list(imp2.plot.layout.xaxis.range) == [-25, 25]
    assert list(imp2.plot.layout.yaxis.range) == [0, 5000]
    assert imp2.plot.layout.xaxis.autorange is False
    # fixing the view must not change the statistics
    assert imp.stats.rate_change == imp2.stats.rate_change


def test_an_invalid_range_is_rejected(dat, m_freq):
    with pytest.raises(PricingError, match="x_range"):
        premium_impact(dat, model_freq_new=m_freq, model_freq_old=m_freq,
                       x_range=(5, 5))
    with pytest.raises(PricingError, match="y_range"):
        premium_impact(dat, model_freq_new=m_freq, model_freq_old=m_freq,
                       y_range=(np.nan, 1))


# ---- the histogram and the tables ----------------------------------------------


def test_histogram_is_pre_binned_into_60_bars(dc, m_new):
    imp = premium_impact(dc, model_freq_new=m_new,
                         old_premium_col="HUIDIGE_PREMIE")
    bar = imp.plot.data[0]
    assert bar.name == "Exposure" and len(bar.x) == 60
    # every unit of exposure lands in exactly one bar
    assert np.sum(bar.y) == pytest.approx(dc.Exposure.sum(), rel=1e-10)
    lo, hi = imp.policy.ChangePct.min(), imp.policy.ChangePct.max()
    np.testing.assert_allclose(np.sum(bar.width), hi - lo, rtol=1e-10)
    assert imp.plot.layout.showlegend is False
    assert imp.plot.layout.title.text.startswith(
        "Premium impact (rebased to old total) – rate-level change")
    assert imp.plot.pricing_config["toImageButtonOptions"]["filename"] == \
        "impact"
    # the median line sits at the reported median
    assert imp.plot.layout.shapes[1].x0 == imp.stats.quantiles["p50"]
    # winners and losers are the extremes of the change
    assert imp.largest_increases.ChangePct.iloc[0] == hi
    assert imp.largest_decreases.ChangePct.iloc[0] == lo
    assert np.all(np.diff(imp.largest_increases.ChangePct) <= 0)
    # Row points back into the data
    r = imp.largest_increases.Row.iloc[0]
    assert imp.policy.Exposure.iloc[r] == dc.Exposure.iloc[r]


# ---- test-review-findings.R ---------------------------------------------------------


def rv_book(n=20000, seed=1, theta=np.inf):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 85, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "BRANDSTOF": pd.Categorical(
            rng.choice(["Benzine", "Diesel", "Waterstof"], n,
                       p=[.7, .295, .005]),
            categories=["Benzine", "Diesel", "Waterstof"]),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "BOEKJAAR": rng.integers(2021, 2025, n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    eta = (np.log(0.08) + 0.6 * np.exp(-(d.LEEFTIJD - 18) / 8)
           + d.REGIO.astype(str).map({"Noord": 0, "Zuid": .2,
                                      "West": .35}).to_numpy()
           + 0.3 * np.log(d.GEWICHT / 1300))
    fr = rng.gamma(theta, 1 / theta, n) if np.isfinite(theta) else 1
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta) * fr)
    d["SCHADELAST"] = np.where(d.AantalClaims > 0,
                               d.AantalClaims * rng.gamma(1.5, 2500 / 1.5, n),
                               0.0)
    d["PREMIE_OUD"] = d.Exposure * 300 * np.exp(0.2 * (d.REGIO == "West"))
    return d


def rv_sev(d):
    s = d[d.AantalClaims > 0].copy()
    s["AvgLoss"] = s.SCHADELAST / s.AantalClaims
    return s


@pytest.fixture(scope="module")
def rv():
    return rv_book()


@pytest.fixture(scope="module")
def rvs(rv):
    return rv_sev(rv)


def test_premium_impact_contributions_add_up_to_the_change_in_total_premium(
        rv, rvs):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                 rv, family="poisson")
    ms = fit_glm("AvgLoss ~ REGIO", rvs, family="gamma",
                 weights="AantalClaims")
    im = premium_impact(rv, mf, ms, old_premium_col="PREMIE_OUD",
                        by="REGIO", rebase=True)
    assert im.by_level.Contribution.sum() == pytest.approx(0, abs=1e-6)


def test_an_unknown_level_at_scoring_is_not_reported_as_an_offset_problem(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                 rv, family="poisson")
    n1 = rv.iloc[:20].copy()
    n1["REGIO"] = n1.REGIO.astype(str).astype(object)
    n1.loc[n1.index[0], "REGIO"] = "Oost"
    with pytest.raises(PricingError) as e:
        premium_impact(n1, mf, None, old_premium_col="PREMIE_OUD")
    msg = str(e.value)
    assert "offset" not in msg
    assert "'REGIO' = Oost (1 row)" in msg
    assert "new frequency model" in msg


def test_at_prices_every_policy_at_one_accounting_year(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + factor(BOEKJAAR)"
                 " + offset(log(Exposure))", rv, family="poisson")
    # against a premium column the year is pointed out ...
    with pytest.warns(PricingWarning, match="own year"):
        premium_impact(rv, mf, old_premium_col="PREMIE_OUD")
    # ... and at={"BOEKJAAR": 2024} prices every row at that year
    im = premium_impact(rv, mf, old_premium_col="PREMIE_OUD",
                        at={"BOEKJAAR": 2024})
    nd = rv.assign(BOEKJAAR=2024, Exposure=1.0)
    np.testing.assert_allclose(im.policy.NewRate, mf.predict(nd), rtol=1e-10)
    # both sides with the year is like for like, so no warning
    no_warnings(premium_impact, rv, mf, model_freq_old=mf)
    # a year the model never saw is named, with the advice to use 'at'
    n2 = rv.iloc[:10].copy()
    n2["BOEKJAAR"] = 2025
    with pytest.raises(PricingError, match="'BOEKJAAR' = 2025.*'at'"):
        premium_impact(n2, mf, old_premium_col="PREMIE_OUD", year_col=None)
    # R's "must be named" for list(2024): here 'at' must be a dict
    with pytest.raises(PricingError, match="'at' must be a dict"):
        premium_impact(rv, mf, old_premium_col="PREMIE_OUD", at=[2024],
                       year_col=None)
    with pytest.raises(PricingError, match="NOPE"):
        premium_impact(rv, mf, old_premium_col="PREMIE_OUD",
                       at={"NOPE": 1}, year_col=None)
    with pytest.raises(PricingError, match="not a valid value for 'REGIO'"):
        premium_impact(rv, mf, old_premium_col="PREMIE_OUD",
                       at={"REGIO": "Oost"}, year_col=None)


def test_scoring_outside_the_fitted_range_is_pointed_out(rv):
    mf = fit_glm("AantalClaims ~ ns(LEEFTIJD, 3) + REGIO"
                 " + offset(log(Exposure))", rv, family="poisson")
    n1 = rv.iloc[:50].copy()
    n1.loc[n1.index[:2], "LEEFTIJD"] = 99
    with pytest.warns(PricingWarning,
                      match=r"'LEEFTIJD' \(2 rows outside 18 to 85\)"):
        premium_impact(n1, mf, old_premium_col="PREMIE_OUD")


def test_dropped_rows_are_counted_per_reason(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                 rv, family="poisson")
    d1 = rv.copy()
    d1.loc[d1.index[:3], "PREMIE_OUD"] = np.nan
    d1.loc[d1.index[5], "Exposure"] = 0
    with pytest.warns(PricingWarning,
                      match=r"4 row\(s\) dropped \(no usable old premium: 4, "
                            r"no positive exposure: 1\)"):
        im = premium_impact(d1, mf, old_premium_col="PREMIE_OUD")
    assert im.stats.n_dropped == 4 and im.stats.n_rows == len(d1) - 4
    assert 5 not in set(im.policy.Row)
    s = im.summary
    assert s.Value[s.Metric == "Rows used / dropped"].item() == \
        "%d / 4" % (len(d1) - 4)
    with pytest.raises(PricingError, match="provide at least one NEW model"):
        premium_impact(rv, old_premium_col="PREMIE_OUD")
    with pytest.raises(PricingError, match="not both"):
        premium_impact(rv, mf, model_freq_old=mf,
                       old_premium_col="PREMIE_OUD")
