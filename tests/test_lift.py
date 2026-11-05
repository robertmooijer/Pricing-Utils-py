"""model_lift() and double_lift(): the port of test-lift.R and the lift tests
of test-offset-link.R and test-review-findings.R."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pricingtools_rmo import _utils as U
from pricingtools_rmo._core import PricingError, PricingWarning
from pricingtools_rmo.lift import double_lift, model_lift
from pricingtools_rmo.model import fit_glm


def warn_msgs(f, *a, **k):
    """Every warning a call raises (R's rv_warns), and its result."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = f(*a, **k)
    return out, [str(x.message) for x in w]


def no_warnings(f, *a, **k):
    """expect_no_warning() / expect_silent()."""
    out, msgs = warn_msgs(f, *a, **k)
    assert not msgs, msgs
    return out


def quiet(f, *a, **k):
    """suppressWarnings()."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


# A model that knows the truth must separate risk; a model that knows
# nothing must not. Those two ends bracket every lift statistic.

NL = 40000


def lift_book(nl, seed=77):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, nl)),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp"], nl),
                                categories=["Dorp", "Stad"]),
        "Exposure": np.round(rng.uniform(.3, 1, nl), 3)})
    lin = (-1.6 + 0.035 * np.maximum(0, 30 - d.LEEFTIJD)
           + 0.45 * (d.REGIO == "Stad"))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(lin))
    d["SCHADELAST"] = d.AantalClaims * 1500.0
    return d


@pytest.fixture(scope="module")
def dl():
    return lift_book(NL)


@pytest.fixture(scope="module")
def m_good(dl):
    return fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   dl, family="poisson")


@pytest.fixture(scope="module")
def m_null(dl):
    return fit_glm("AantalClaims ~ 1 + offset(log(Exposure))", dl,
                   family="poisson")


# ---- test-lift.R ---------------------------------------------------------------


def test_lift_separates_a_real_model_from_an_empty_one(dl, m_good, m_null):
    g = model_lift(m_good, actual_col="AantalClaims", data=dl)
    # the intercept-only model predicts one rate for everyone, so the ten
    # bins are ten arbitrary slices of an identical prediction; that is
    # exactly what the tie warning is for
    with pytest.warns(PricingWarning,
                      match="only 1 distinct predicted value"):
        z = model_lift(m_null, actual_col="AantalClaims", data=dl)

    assert len(g.table) == 10
    assert g.gini > 0.08            # the real model orders risk
    assert abs(z.gini) < 0.03       # the intercept-only model cannot
    assert g.gini > z.gini

    # the actual rate should rise across the bins of a working model
    assert g.table.ActualRate.iloc[9] > g.table.ActualRate.iloc[0]
    assert isinstance(g.plot, go.Figure)
    assert isinstance(g.plot_lorenz, go.Figure)
    # the plots as in R: two series, the Gini and its interval in the titles
    assert [t.name for t in g.plot.data] == ["Predicted", "Actual"]
    assert [t.name for t in g.plot_lorenz.data] == ["No discrimination",
                                                    "Lorenz"]
    assert g.plot.layout.title.text.startswith(
        "Lift – 10 bins of equal exposure (out-of-sample)"
        "  ·  Gini %.3f (90%%: " % g.gini)
    assert g.plot.pricing_config["toImageButtonOptions"]["filename"] == "lift"
    assert g.plot_lorenz.pricing_config["toImageButtonOptions"]["filename"] \
        == "lorenz"
    assert len(g.plot_lorenz.data[1].x) == 400
    assert g.plot_lorenz.data[1].x[0] == 0
    assert g.plot_lorenz.data[1].x[-1] == pytest.approx(1, abs=1e-12)


def test_bins_hold_roughly_equal_exposure(dl, m_good):
    g = model_lift(m_good, actual_col="AantalClaims", data=dl, n_bins=5)
    assert len(g.table) == 5
    assert np.max(np.abs(g.table.ExposureShare - 0.2)) < 0.01
    assert g.table.ExposureShare.sum() == pytest.approx(1, abs=1e-8)


def test_model_lift_falls_back_to_the_models_own_rows(m_good):
    g = model_lift(m_good, actual_col="AantalClaims")
    assert g.stats.in_sample is True
    assert g.stats.n_rows == NL
    assert "(in-sample)" in g.plot.layout.title.text


def test_double_lift_prefers_the_model_that_is_right_where_they_differ(
        dl, m_good, m_null):
    dbl = double_lift(dl, model_freq_new=m_good, model_freq_old=m_null,
                      actual_col="AantalClaims")
    assert len(dbl.table) == 10
    assert dbl.stats.winner == "new"
    assert dbl.stats.mad_new < dbl.stats.mad_old
    # the ratio must be increasing across the bins by construction
    assert np.all(np.diff(dbl.table.RateRatio) > 0)
    assert isinstance(dbl.plot, go.Figure)
    assert [t.name for t in dbl.plot.data] == ["A/E old", "A/E new"]
    assert dbl.plot.pricing_config["toImageButtonOptions"]["filename"] == \
        "double-lift"


def test_double_lift_against_itself_shows_no_winner_and_a_flat_ae(dl, m_good):
    # comparing a model with itself gives every policy a ratio of exactly 1
    with pytest.warns(PricingWarning,
                      match="only 1 distinct predicted value"):
        dbl = double_lift(dl, model_freq_new=m_good, model_freq_old=m_good,
                          actual_col="AantalClaims")
    np.testing.assert_allclose(dbl.table.AE_New, dbl.table.AE_Old,
                               rtol=1e-10)
    assert dbl.stats.mad_new == pytest.approx(dbl.stats.mad_old, rel=1e-10)
    assert abs(dbl.stats.rate_level_change) < 1e-10
    assert dbl.stats.winner == "tie"


def test_the_rate_level_change_is_reported_whether_or_not_we_rebase(
        dl, m_good):
    # the level difference belongs to the two tariffs, not to the decision to
    # rebase: deriving it from the scaling factor reported 0 for rebase =
    # FALSE, which is the one case where the difference is still on screen
    d2 = dl.copy()
    d2["OldPrem"] = U.model_rate(m_good, dl, "Exposure", "t") * 0.8 * \
        dl.Exposure

    # every policy has the same new/old ratio here, so the tie warning
    # fires (in R as well, where the test lets it through)
    def f(rb):
        return quiet(double_lift, d2, model_freq_new=m_good,
                     old_premium_col="OldPrem",
                     actual_col="AantalClaims",
                     rebase=rb).stats.rate_level_change
    assert f(True) == pytest.approx(0.25, rel=1e-8)
    assert f(False) == pytest.approx(0.25, rel=1e-8)


def test_a_severity_only_lift_weights_by_claims_not_exposure():
    # a severity model predicts a mean per claim, so exposure is the wrong
    # denominator: predicted x exposure does not reconcile with total loss
    rng = np.random.default_rng(41)
    nn = 5000
    dd = pd.DataFrame({
        "REGIO": pd.Categorical(rng.choice(["N", "O", "Z"], nn,
                                           p=[.5, .3, .2]),
                                categories=["N", "O", "Z"]),
        "LEEFTIJD": np.round(rng.uniform(18, 75, nn)),
        "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    # claims per policy vary strongly by region, so the two weights differ
    lam = dd.REGIO.astype(str).map({"N": .4, "O": 1.2, "Z": 3.0}).to_numpy()
    dd["AantalClaims"] = rng.poisson(lam)
    mu = np.exp(7 + .012 * dd.LEEFTIJD + .35 * (dd.REGIO == "Z")).to_numpy()
    dd["SCHADELAST"] = [0.0 if k == 0 else
                        float(np.sum(rng.gamma(3, m / 3, k)))
                        for k, m in zip(dd.AantalClaims, mu)]
    dsv = dd[dd.AantalClaims > 0].copy()
    dsv["AvgLoss"] = dsv.SCHADELAST / dsv.AantalClaims
    ms = fit_glm("AvgLoss ~ REGIO + LEEFTIJD", dsv, family="gamma",
                 weights="AantalClaims")

    lf_ = model_lift(None, ms, dsv, actual_col="SCHADELAST")
    # predicted total now reconciles with the realised total
    assert lf_.stats.total_predicted == pytest.approx(
        lf_.stats.total_actual, rel=0.02)
    # the bins hold equal claim counts, not equal exposure
    assert lf_.table.Exposure.sum() == pytest.approx(
        dsv.AantalClaims.sum(), rel=1e-8)
    assert "equal AantalClaims" in lf_.plot.layout.title.text
    # and the unit-mismatch warning is gone: weighting by exposure used to
    # put the predicted total a factor of three below the realised one
    no_warnings(model_lift, None, ms, dsv, actual_col="SCHADELAST")
    with pytest.warns(PricingWarning, match="not the same quantity"):
        model_lift(None, ms, dsv, actual_col="SCHADELAST",
                   weight_col="Exposure")

    # a frequency lift is untouched: still weighted by exposure
    mf = fit_glm("AantalClaims ~ REGIO + LEEFTIJD + offset(log(Exposure))",
                 dsv, family="poisson")
    lf2 = model_lift(mf, None, dsv, actual_col="AantalClaims")
    assert lf2.table.Exposure.sum() == pytest.approx(dsv.Exposure.sum(),
                                                     rel=1e-8)

    # and an explicit weight_col overrides the automatic choice
    lw = quiet(model_lift, None, ms, dsv, actual_col="SCHADELAST",
               weight_col="Exposure")
    assert lw.table.Exposure.sum() == pytest.approx(dsv.Exposure.sum(),
                                                    rel=1e-8)


def test_a_portfolio_without_losses_gives_na_rather_than_nan_for_the_gini(
        dl, m_good):
    d0 = dl.copy()
    d0["AantalClaims"] = 0
    g = model_lift(m_good, actual_col="AantalClaims", data=d0)
    # Python has one missing number, NaN; what R's NA protects is the title,
    # which says NA rather than NaN
    assert np.isnan(g.gini)
    assert "Gini NA" in g.plot.layout.title.text
    assert "nan" not in g.plot.layout.title.text.lower()
    assert isinstance(g.plot, go.Figure)


def test_mismatched_units_are_caught(dl, m_good, m_null):
    # a frequency model compared against loss amounts: the overall A/E is
    # then off by orders of magnitude and the chart would be meaningless
    with pytest.warns(PricingWarning, match="not the same quantity"):
        model_lift(m_good, data=dl, actual_col="SCHADELAST")
    with pytest.warns(PricingWarning, match="not the same quantity"):
        double_lift(dl, model_freq_new=m_good, model_freq_old=m_null,
                    actual_col="SCHADELAST", n_boot=0)
    # matching units stay silent
    no_warnings(model_lift, m_good, data=dl, actual_col="AantalClaims")


def test_lift_functions_validate_their_input(dl, m_good, m_null):
    with pytest.raises(PricingError, match="at least one model"):
        model_lift()
    with pytest.raises(PricingError, match="n_bins"):
        model_lift(m_good, data=dl, n_bins=1)
    with pytest.raises(PricingError, match="NOPE"):
        model_lift(m_good, data=dl, actual_col="NOPE")
    with pytest.raises(PricingError, match="old"):
        double_lift(dl, model_freq_new=m_good)
    # an existing column, so the mutual-exclusion check is what fires
    with pytest.raises(PricingError, match="not both"):
        double_lift(dl, model_freq_new=m_good, model_freq_old=m_null,
                    old_premium_col="Exposure")
    with pytest.raises(PricingError, match="n_bins"):
        double_lift(dl, model_freq_new=m_good, model_freq_old=m_null,
                    n_bins=1)
    with pytest.raises(PricingError, match="y_range"):
        model_lift(m_good, data=dl, y_range=(2, 1))


# ---- test-offset-link.R ------------------------------------------------------------


def test_model_lift_reads_a_bm_model_against_its_own_claims():
    rng = np.random.default_rng(82)
    nk = 30000
    dk = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z", "O"], nk),
                                           categories=["N", "O", "Z"]),
                       "L": np.round(rng.uniform(18, 80, nk)),
                       "BM_KLASSE": rng.integers(1, 6, nk),
                       "Exposure": np.round(rng.uniform(.1, 1, nk), 3)})
    dk["BM"] = np.array([1, .9, .8, .7, .6])[dk.BM_KLASSE - 1]
    dk["AantalClaims"] = rng.poisson(
        dk.Exposure * dk.BM * np.exp(-2 + .3 * (dk.R == "Z")
                                     + .01 * (dk.L - 40)))
    mk = fit_glm("AantalClaims ~ R + L + offset(log(Exposure))"
                 " + offset(log(BM))", dk, family="poisson")

    # in-sample a Poisson GLM reproduces the claim total exactly, so the lift
    # has to as well; with the BM taken out it read the average discount
    lf = model_lift(mk, None, data=dk, actual_col="AantalClaims")
    assert lf.stats.total_actual / lf.stats.total_predicted == \
        pytest.approx(1, abs=1e-8)
    assert lf.stats.offsets_kept == ["BM"]
    lf0 = quiet(model_lift, mk, None, data=dk, actual_col="AantalClaims",
                keep_offsets=False)
    assert lf0.stats.total_actual / lf0.stats.total_predicted == \
        pytest.approx(dk.AantalClaims.sum() / np.sum(mk.fitted / dk.BM),
                      rel=1e-8)
    assert lf0.stats.total_actual / lf0.stats.total_predicted < 0.9
    # the axis says what the rate is on
    assert "incl. BM" in lf.plot.layout.yaxis.title.text
    assert "incl." not in lf0.plot.layout.yaxis.title.text

    with pytest.raises(PricingError, match="TRUE or FALSE"):
        model_lift(mk, None, data=dk, actual_col="AantalClaims",
                   keep_offsets=None)


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


@pytest.fixture(scope="module")
def rv_mf(rv):
    return fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   rv, family="poisson")


@pytest.fixture(scope="module")
def rv_ms(rvs):
    return fit_glm("AvgLoss ~ REGIO", rvs, family="gamma",
                   weights="AantalClaims")


def test_a_double_lift_between_two_severity_models_is_weighted_by_claims(rvs):
    m1 = fit_glm("AvgLoss ~ REGIO + log(GEWICHT)", rvs, family="gamma",
                 weights="AantalClaims")
    m0 = fit_glm("AvgLoss ~ REGIO", rvs, family="gamma",
                 weights="AantalClaims")
    dl_ = quiet(double_lift, rvs, model_sev_new=m1, model_sev_old=m0,
                rebase=False)
    tb = dl_.table
    tot_ae = (np.sum(tb.ActualRate * tb.Exposure)
              / np.sum(tb.Exposure * tb.ActualRate / tb.AE_Old))
    assert tot_ae == pytest.approx(1, rel=0.02)
    assert dl_.stats.weight_col == "AantalClaims"
    assert tb.Exposure.sum() == pytest.approx(rvs.AantalClaims.sum(),
                                              rel=1e-12)


def test_model_lift_reports_a_bootstrap_interval_around_the_gini(rv, rv_mf):
    lf = model_lift(rv_mf, data=rv, actual_col="AantalClaims", seed=4)
    assert len(lf.gini_ci) == 2
    assert lf.gini_ci[0] < lf.gini < lf.gini_ci[1]
    assert lf.stats.gini_ci is lf.gini_ci
    assert "(90%: " in lf.plot.layout.title.text
    # the same seed gives the same interval; n_boot = 0 switches it off
    np.testing.assert_array_equal(
        model_lift(rv_mf, data=rv, actual_col="AantalClaims",
                   seed=4).gini_ci, lf.gini_ci)
    l0 = model_lift(rv_mf, data=rv, actual_col="AantalClaims", n_boot=0)
    assert np.all(np.isnan(l0.gini_ci))
    assert "90%" not in l0.plot.layout.title.text
    with pytest.raises(PricingError, match="n_boot"):
        model_lift(rv_mf, data=rv, n_boot=1.5)
    with pytest.raises(PricingError, match="conf_level"):
        model_lift(rv_mf, data=rv, conf_level=0)


def test_large_losses_that_decide_the_gini_or_the_verdict_are_named(
        rv, rv_mf, rv_ms):
    mf0 = fit_glm("AantalClaims ~ REGIO + offset(log(Exposure))", rv,
                  family="poisson")
    # without large losses a modest interval is shown, not warned about
    no_warnings(model_lift, rv_mf, rv_ms, data=rv, seed=1)
    # ten large losses carrying a third of the total
    d1 = rv.copy()
    rng = np.random.default_rng(5)
    big = rng.choice(np.flatnonzero(d1.AantalClaims > 0), 10, replace=False)
    d1.loc[d1.index[big], "SCHADELAST"] = 150000.0
    assert U.top_share(d1.SCHADELAST) > 0.10
    with pytest.warns(PricingWarning, match="ten largest records carry"):
        model_lift(rv_mf, rv_ms, data=d1, seed=1)
    dl_, w = warn_msgs(double_lift, d1, rv_mf, rv_ms, model_freq_old=mf0,
                       model_sev_old=rv_ms, seed=1)
    assert dl_.stats.winner == "inconclusive"
    assert any("ten largest records carry" in m for m in w)


def test_the_double_lift_names_a_winner_only_when_the_interval_excludes_0():
    nl = 30000
    d = lift_book(nl)
    m_good = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                     d, family="poisson")
    m_null = fit_glm("AantalClaims ~ 1 + offset(log(Exposure))", d,
                     family="poisson")
    m_reg = fit_glm("AantalClaims ~ REGIO + LEEFTIJD + offset(log(Exposure))",
                    d.iloc[::2], family="poisson")
    clear = double_lift(d, model_freq_new=m_good, model_freq_old=m_null,
                        actual_col="AantalClaims", seed=1)
    assert clear.stats.winner == "new"
    assert clear.stats.mad_diff_ci[0] > 0
    assert clear.stats.mad_diff == clear.stats.mad_old - clear.stats.mad_new
    assert "): new</sup>" in clear.plot.layout.title.text
    # two models fitted on the same signal cannot be told apart; without
    # large losses behind it that is the verdict, not a warning
    close = no_warnings(double_lift, d, model_freq_new=m_good,
                        model_freq_old=m_reg, actual_col="AantalClaims",
                        seed=1)
    assert close.stats.winner == "inconclusive"
    assert close.stats.mad_diff_ci[0] <= 0
    assert close.stats.mad_diff_ci[1] >= 0
    # without the bootstrap the point values decide, as before
    pt = double_lift(d, model_freq_new=m_good, model_freq_old=m_reg,
                     actual_col="AantalClaims", n_boot=0)
    assert pt.stats.winner in ("new", "old")
    assert np.all(np.isnan(pt.stats.mad_diff_ci))
    assert "<sup>" not in pt.plot.layout.title.text


def test_a_severity_model_against_a_frequency_model_is_refused(
        rvs, rv_mf, rv_ms):
    with pytest.raises(PricingError, match="one side is a severity model"):
        double_lift(rvs, model_sev_new=rv_ms, model_freq_old=rv_mf,
                    model_sev_old=rv_ms)


def test_dropped_rows_are_reported_per_reason(rv, rv_mf, rv_ms):
    d1 = rv.copy()
    d1.loc[d1.index[:3], "SCHADELAST"] = np.nan
    with pytest.warns(PricingWarning, match="missing 'SCHADELAST': 3"):
        lf = model_lift(rv_mf, rv_ms, data=d1, n_boot=0)
    assert lf.stats.n_dropped == 3 and lf.stats.n_rows == len(d1) - 3
    # the first warning, as R's tryCatch(warning = conditionMessage) reads it
    _, w = warn_msgs(double_lift, d1, rv_mf, rv_ms,
                     old_premium_col="PREMIE_OUD", n_boot=0)
    assert "missing 'SCHADELAST': 3" in w[0]
    assert "exposure" not in w[0]


def test_claims_without_payment_are_pointed_out(rv):
    # the make_rating_table() half of this R test is in the rating-table
    # tests
    d1 = rv.copy()
    rng = np.random.default_rng(12)
    nil = (d1.AantalClaims > 0) & (rng.uniform(size=len(d1)) < 0.15)
    d1.loc[nil, "SCHADELAST"] = 0.0
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                 d1, family="poisson")
    s1 = rv_sev(d1)
    ms = fit_glm("AvgLoss ~ REGIO", s1[s1.SCHADELAST > 0], family="gamma",
                 weights="AantalClaims")
    with pytest.warns(PricingWarning, match="without payment"):
        model_lift(mf, ms, data=d1, n_boot=0)


def test_at_prices_every_policy_for_the_lift_too(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + factor(BOEKJAAR)"
                 " + offset(log(Exposure))", rv, family="poisson")
    lf = model_lift(mf, data=rv, actual_col="AantalClaims", n_boot=0,
                    at={"BOEKJAAR": 2024})
    rate = mf.predict(rv.assign(BOEKJAAR=2024, Exposure=1.0))
    assert lf.stats.total_predicted == pytest.approx(
        np.sum(rate * rv.Exposure), rel=1e-10)
    with pytest.raises(PricingError, match="not a valid value"):
        double_lift(rv, mf, old_premium_col="PREMIE_OUD",
                    actual_col="AantalClaims", at={"REGIO": "Oost"})
