"""agg_all, make_plot, plot_glm_predictor and make_rating_plot: the port of
tests/testthat/test-agg-plots.R, the A/E tests of test-offset-link.R, the
agg/make_plot tests of test-data-table-input.R and the matching tests of
test-review-findings.R."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy import stats

from pricingtools_rmo._core import PricingError, PricingWarning
from pricingtools_rmo.ae import plot_glm_predictor
from pricingtools_rmo.agg import agg_all
from pricingtools_rmo.colors import ta_blue, ta_gold
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.plot_oneway import make_plot
from pricingtools_rmo.rating_plot import make_rating_plot

from conftest import trace, trace_y


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fun(*args, **kw)


def _tr(fig, name):
    """The named trace, or None (R's helper returns NULL)."""
    for t in fig.data:
        if t.name == name:
            return t
    return None


def _per_level(values, groups):
    """A per-level total as a dict label -> value, R's tapply(..., sum)."""
    s = pd.Series(np.asarray(values, dtype=float)).groupby(
        pd.Series(np.asarray(groups, dtype=object)).astype(str)).sum()
    return s.to_dict()


def _at(d, xs):
    return np.array([d[str(x)] for x in xs], dtype=float)


def test_agg_all_works_with_custom_column_names_and_validates_input(dat):
    dat2 = dat.rename(columns={"Exposure": "EXPO"})
    a = agg_all(dat2, "REGIO", by_year=True, exposure_col="EXPO")
    assert {"Exposure", "ClaimCount", "Loss", "Frequency", "Severity",
            "Year"} <= set(a.columns)
    with pytest.raises(PricingError, match="DOES_NOT_EXIST"):
        agg_all(dat, "REGIO", False, exposure_col="DOES_NOT_EXIST")


def test_the_volume_bars_follow_the_metrics_own_denominator(dat):
    # a severity is a mean per claim, so exposure bars would say how much
    # business is in a level, not how many claims the mean rests on - and
    # those routinely run in opposite directions
    pf = make_plot(dat, "REGIO", "Frequency", ta_blue, "F",
                   display="color", by_year=False)
    ps = make_plot(dat, "REGIO", "Severity", ta_blue, "S",
                   display="color", by_year=False)
    assert pf.data[0].name == "Exposure"
    assert ps.data[0].name == "Number of claims"
    np.testing.assert_allclose(
        np.sort(np.asarray(pf.data[0].y, dtype=float)),
        np.sort(dat.groupby("REGIO", observed=True).Exposure.sum().to_numpy()),
        rtol=1e-8)
    np.testing.assert_allclose(
        np.sort(np.asarray(ps.data[0].y, dtype=float)),
        np.sort(dat.groupby("REGIO", observed=True).AantalClaims.sum()
                .to_numpy(dtype=float)), rtol=1e-8)
    # the secondary axis is labelled accordingly
    assert ps.layout.yaxis2.title.text == "Number of claims"

    # facet mode uses the same column
    pff = make_plot(dat, "REGIO", "Severity", ta_blue, "S",
                    display="facet", by_year=True)
    assert isinstance(pff, go.Figure)
    assert pff.data[0].name == "Number of claims"


def test_grouping_on_a_column_agg_all_creates_is_refused(dat):
    # grouping on "Exposure" produced a data.frame with two columns of that
    # name rather than an error, and every later lookup then picked one of
    # them at random
    d = pd.DataFrame({"Exposure": [1, 1, 2], "AantalClaims": [0, 1, 2],
                      "SCHADELAST": [0, 500, 900]})
    with pytest.raises(PricingError, match="cannot group on"):
        agg_all(d, "Exposure", False)
    with pytest.raises(PricingError, match="by_year = FALSE"):
        agg_all(dat, "BOEKJAAR", by_year=True)


def test_a_multi_column_model_term_is_refused_with_a_usable_message(dat):
    # model.frame() holds ns(LEEFTIJD, 4) as a matrix; without a check this
    # failed several frames down with "replacement has 0 rows"
    m = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + offset(log(Exposure))",
                dat, family="poisson")
    with pytest.raises(PricingError, match="multi-column model term"):
        plot_glm_predictor(m, "ns(LEEFTIJD, 4)")
    with pytest.raises(PricingError, match="LEEFTIJD"):
        plot_glm_predictor(m, "ns(LEEFTIJD, 4)")
    # the underlying column still works
    assert isinstance(plot_glm_predictor(m, "LEEFTIJD"), go.Figure)


def test_plot_glm_predictor_returns_plotly_objects(m_freq, m_sev):
    assert isinstance(plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=30), go.Figure)
    assert isinstance(plot_glm_predictor(m_sev, "LEEFTIJD", n_bins=20), go.Figure)


def test_fewer_distinct_values_than_n_bins_means_no_binning_at_all(dat, m_freq):
    def obs_x(p):
        return np.sort(np.asarray(trace(p, "Observed").x, dtype=float))
    uniq = np.sort(dat.LEEFTIJD.unique())          # 63 integer values, 18..80

    # n_bins well above the number of distinct values: exact positions,
    # including the minimum (the case that used to drift off its own value)
    p_exact = plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=150)
    np.testing.assert_array_equal(obs_x(p_exact), uniq)
    assert obs_x(p_exact).min() == dat.LEEFTIJD.min()

    # exactly at the boundary: still unbinned
    np.testing.assert_array_equal(
        obs_x(plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=len(uniq))), uniq)

    # An extreme outlier must not shift the other points; with equal-width
    # bins it would drag every boundary along
    d_out = dat.copy()
    d_out.loc[d_out.index[0], "LEEFTIJD"] = 999
    m_out = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                    d_out, family="poisson")
    x_out = obs_x(plot_glm_predictor(m_out, "LEEFTIJD", n_bins=150))
    assert np.all(np.isin(uniq, x_out))
    assert 999 in x_out

    # More distinct values than n_bins: binning kicks in, at most n_bins points
    p_binned = plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=10)
    assert len(obs_x(p_binned)) <= 10


def test_plot_glm_predictor_y_range_is_unused_by_default_and_fixable(m_freq):
    # Default: the axis auto-scales
    p_auto = plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=30)
    assert p_auto.layout.yaxis.autorange is True

    # Fixed: the requested range is honoured and auto-scaling is off
    p_fixed = plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=30, y_range=(0, 0.5))
    assert list(p_fixed.layout.yaxis.range) == [0, 0.5]
    assert p_fixed.layout.yaxis.autorange is not True

    # Two predictors with an identical y_range share the same axis
    p_a = plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=30, y_range=(0, 0.4))
    p_b = plot_glm_predictor(m_freq, "REGIO", y_range=(0, 0.4))
    assert list(p_a.layout.yaxis.range) == list(p_b.layout.yaxis.range)

    with pytest.raises(PricingError, match="y_range"):
        plot_glm_predictor(m_freq, "LEEFTIJD", y_range=(1, 1))


def test_make_rating_plot_renders_main_effects_and_interactions(tbl, tbl2):
    assert isinstance(make_rating_plot(tbl, "REGIO"), go.Figure)
    assert isinstance(make_rating_plot(tbl2, "LEEFTIJD:REGIO"), go.Figure)


def test_make_plot_aggregates_raw_data_internally(dat):
    p_raw = make_plot(dat, "REGIO", "Frequency", ta_blue, "Frequency",
                      "color", True)
    assert isinstance(p_raw, go.Figure)
    assert isinstance(make_plot(dat, "REGIO", "Frequency", ta_blue, "Frequency",
                                "facet", True), go.Figure)
    assert isinstance(make_plot(dat, "REGIO", "Severity", ta_gold, "Severity",
                                "color", False), go.Figure)

    # plotted values must equal the agg_all aggregation
    ovz = agg_all(dat, "REGIO", by_year=True)
    s = ovz[ovz.Year == "2020"].sort_values("REGIO")
    np.testing.assert_allclose(trace_y(p_raw, "2020"), s.Frequency.to_numpy())

    # invalid metric gives a clear match.arg error
    with pytest.raises(PricingError, match="Frequency"):
        make_plot(dat, "REGIO", "Frequentie", ta_blue, "y", "color", False)


def test_observed_points_carry_error_bars_that_match_the_theory():
    rng = np.random.default_rng(61)
    nn = 30000
    dc = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "O", "Z"], nn)),
                       "LFT": np.round(rng.uniform(18, 80, nn)),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    dc["AantalClaims"] = rng.poisson(dc.Exposure * np.exp(-2.2 + .3 * (dc.R == "Z")))
    mc = fit_glm("AantalClaims ~ R + offset(log(Exposure))", dc, family="poisson")

    o = trace(plot_glm_predictor(mc, "R"), "Observed")
    # counts: se = sqrt(phi * sum(V(mu))) / sum(exposure), phi fixed at 1
    hand = {lv: np.sqrt(mc.fitted[(dc.R == lv).to_numpy()].sum())
            / dc.Exposure[dc.R == lv].sum() for lv in ["N", "O", "Z"]}
    np.testing.assert_allclose(np.asarray(o.error_y.array) / stats.norm.ppf(0.975),
                               _at(hand, o.x), rtol=1e-10)

    # a weighted Gamma uses the weighted-mean form and an estimated phi
    dsv = dc[dc.AantalClaims > 0].copy()
    dsv["Avg"] = rng.gamma(3, 700, len(dsv))
    msv = fit_glm("Avg ~ R", dsv, family="gamma", weights="AantalClaims")
    os_ = trace(_quiet(plot_glm_predictor, msv, "R"), "Observed")
    phi = np.sum(msv.residuals("pearson") ** 2) / msv.df_residual
    h2 = {}
    for lv in ["N", "O", "Z"]:
        i = (dsv.R == lv).to_numpy()
        w = dsv.AantalClaims.to_numpy()[i]
        h2[lv] = np.sqrt(phi * np.sum(w * msv.fitted[i] ** 2)) / w.sum()
    np.testing.assert_allclose(np.asarray(os_.error_y.array) / stats.norm.ppf(0.975),
                               _at(h2, os_.x), rtol=1e-8)


def test_the_ci_switches_behave():
    rng = np.random.default_rng(62)
    nn = 8000
    dc = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], nn)),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    dc["AantalClaims"] = rng.poisson(dc.Exposure * .12)
    mc = fit_glm("AantalClaims ~ R + offset(log(Exposure))", dc, family="poisson")
    assert trace(plot_glm_predictor(mc, "R", ci=False), "Observed").error_y.array is None
    assert trace(plot_glm_predictor(mc, "R"), "Observed").error_y.array is not None
    assert np.all(np.asarray(trace(plot_glm_predictor(mc, "R", ci_level=0.99),
                                   "Observed").error_y.array)
                  > np.asarray(trace(plot_glm_predictor(mc, "R"),
                                     "Observed").error_y.array))
    with pytest.raises(PricingError, match="between 0 and 1"):
        plot_glm_predictor(mc, "R", ci_level=1.5)
    with pytest.raises(PricingError, match="between 0 and 1"):
        plot_glm_predictor(mc, "R", ci_level=0)


def test_binned_residuals_are_drawn_as_points_not_a_line():
    diagnostics = pytest.importorskip("pricingtools_rmo.diagnostics")
    rng = np.random.default_rng(63)
    nn = 6000
    dc = pd.DataFrame({"x": rng.uniform(size=nn),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    dc["AantalClaims"] = rng.poisson(dc.Exposure * np.exp(-2 + dc.x))
    mc = fit_glm("AantalClaims ~ x + offset(log(Exposure))", dc, family="poisson")
    mr = _tr(diagnostics.plot_glm_residuals(mc), "Mean residual")
    # the mode is the switch that decides whether a line is actually drawn
    assert mr.mode == "markers"
    assert "lines" not in mr.mode


def _period(seed, bump, nn=20000):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "O", "Z"], nn)),
                      "LFT": np.round(rng.uniform(18, 80, nn)),
                      "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -2.2 + .3 * (d.R == "Z") + bump * (d.R == "O")))
    return d


def test_plot_glm_predictor_evaluates_another_period_when_data_is_given():
    nn = 20000
    d19 = _period(63, 0)
    d20 = _period(64, 0.5)          # "O" deteriorates in the new year
    mc = fit_glm("AantalClaims ~ R + offset(log(Exposure))", d19, family="poisson")

    p_new = plot_glm_predictor(mc, "R", data=d20)
    assert isinstance(p_new, go.Figure)
    o = trace(p_new, "Observed")
    e = trace(p_new, "Predicted")
    # observed = that year's own rate; predicted = the tariff applied to it
    expo = _per_level(d20.Exposure, d20.R)
    hand_o = {k: v / expo[k] for k, v in _per_level(d20.AantalClaims, d20.R).items()}
    hand_e = {k: v / expo[k] for k, v in _per_level(mc.predict(d20), d20.R).items()}
    np.testing.assert_allclose(np.asarray(o.y), _at(hand_o, o.x), rtol=1e-10)
    np.testing.assert_allclose(np.asarray(e.y), _at(hand_e, e.x), rtol=1e-10)

    # in-sample the two series coincide on a level in the model; out of
    # sample they do not, which is the whole point of monitoring
    o0 = trace(plot_glm_predictor(mc, "R"), "Observed")
    e0 = trace(plot_glm_predictor(mc, "R"), "Predicted")
    np.testing.assert_allclose(np.asarray(o0.y), np.asarray(e0.y), rtol=1e-8)
    i_o = list(o.x).index("O")
    assert o.y[i_o] / e.y[i_o] > 1.2

    # the bars show the new period's exposure, not the old one's
    w = trace(p_new, "Exposure")
    np.testing.assert_allclose(np.asarray(w.y), _at(expo, w.x), rtol=1e-8)

    # a predictor outside the model is read from 'data' as well
    assert isinstance(plot_glm_predictor(mc, "LFT", data=d20, n_bins=20), go.Figure)
    assert nn == len(d20)


def test_plot_glm_predictor_on_new_data_reevaluates_weights_and_offsets():
    rng = np.random.default_rng(65)
    nn = 6000
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], nn)),
                      "Maanden": rng.integers(1, 13, nn)})
    d["AantalClaims"] = rng.poisson(d.Maanden / 12 * .3) + 1
    d["Avg"] = rng.gamma(3, 700, nn) * np.where(d.R == "Z", 1.2, 1)
    m_sev = fit_glm("Avg ~ R", d, family="gamma", weights="AantalClaims")
    d2 = d.iloc[rng.choice(nn, 3000, replace=False)].copy()
    d2["Avg"] = d2.Avg * 1.15                     # a severity trend of 15%

    o = trace(plot_glm_predictor(m_sev, "R", data=d2), "Observed")
    hand = {lv: np.average(d2.Avg[d2.R == lv], weights=d2.AantalClaims[d2.R == lv])
            for lv in ["N", "Z"]}
    np.testing.assert_allclose(np.asarray(o.y), _at(hand, o.x), rtol=1e-10)
    # the weight bars are the new period's claim counts
    w = trace(plot_glm_predictor(m_sev, "R", data=d2), "Weight")
    np.testing.assert_allclose(np.asarray(w.y),
                               _at(_per_level(d2.AantalClaims, d2.R), w.x),
                               rtol=1e-8)

    # an offset on an expression also follows (R passes it as glm(offset =);
    # fit_glm takes every offset in the formula)
    m_f = fit_glm("AantalClaims ~ R + offset(log(Maanden / 12))", d,
                  family="poisson")
    o2 = trace(plot_glm_predictor(m_f, "R", data=d2, exposure_col="Maanden"),
               "Observed")
    mo = _per_level(d2.Maanden, d2.R)
    hand2 = {k: v / mo[k] for k, v in _per_level(d2.AantalClaims, d2.R).items()}
    np.testing.assert_allclose(np.asarray(o2.y), _at(hand2, o2.x), rtol=1e-10)


def test_unusable_monitoring_data_is_refused_rather_than_guessed_at():
    rng = np.random.default_rng(66)
    nn = 4000
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], nn)),
                      "Exposure": rng.uniform(.3, 1, nn)})
    d["AantalClaims"] = rng.poisson(d.Exposure * .2)
    mc = fit_glm("AantalClaims ~ R + offset(log(Exposure))", d, family="poisson")

    with pytest.raises(PricingError, match="AantalClaims"):
        plot_glm_predictor(mc, "R", data=d[["R", "Exposure"]])
    with pytest.raises(PricingError, match="at least one row"):
        plot_glm_predictor(mc, "R", data=d.iloc[0:0])
    d_new = d.copy()
    d_new["R"] = d_new.R.astype(str)
    d_new.loc[0, "R"] = "O"                        # a level the model never saw
    # named with its column, its value and its row count
    with pytest.raises(PricingError, match=r"not fitted on: 'R' = O \(1 row\)"):
        plot_glm_predictor(mc, "R", data=d_new)

    # rows with a missing value are dropped, exactly as the fit dropped them
    d_na = d.copy()
    d_na.loc[0:49, "R"] = np.nan
    w = trace(plot_glm_predictor(mc, "R", data=d_na), "Exposure")
    np.testing.assert_allclose(np.sum(w.y), d.Exposure.iloc[50:].sum(), rtol=1e-8)


def test_ratio_true_plots_ae_against_a_reference_line_at_1():
    nn = 20000
    d19 = _period(67, 0)
    d20 = _period(68, 0.4)
    mc = fit_glm("AantalClaims ~ R + offset(log(Exposure))", d19, family="poisson")

    p = plot_glm_predictor(mc, "R", data=d20, ratio=True)
    ae = _tr(p, "A/E")
    assert ae is not None
    assert _tr(p, "Observed") is None              # one line, not two
    assert _tr(p, "Predicted") is None

    # the plotted ratio is sum(actual) / sum(expected) per level
    pr = _per_level(mc.predict(d20), d20.R)
    hand = {k: v / pr[k] for k, v in _per_level(d20.AantalClaims, d20.R).items()}
    np.testing.assert_allclose(np.asarray(ae.y), _at(hand, ae.x), rtol=1e-10)

    # it is exactly the ratio of the two lines of the default plot
    p2 = plot_glm_predictor(mc, "R", data=d20)
    np.testing.assert_allclose(np.asarray(ae.y),
                               np.asarray(trace(p2, "Observed").y)
                               / np.asarray(trace(p2, "Predicted").y), rtol=1e-10)

    # the error bars are the observed bars, carried through the division
    np.testing.assert_allclose(np.asarray(ae.error_y.array),
                               np.asarray(trace(p2, "Observed").error_y.array)
                               / np.asarray(trace(p2, "Predicted").y), rtol=1e-10)
    assert trace(plot_glm_predictor(mc, "R", data=d20, ratio=True, ci=False),
                 "A/E").error_y.array is None

    # the reference line sits at 1 and spans the plotting area
    sh = p.layout.shapes
    assert len(sh) == 1
    assert (sh[0].y0, sh[0].y1) == (1, 1)
    assert sh[0].xref == "paper"
    assert len(p2.layout.shapes) == 0

    # in-sample a level in the model is pinned to 1 by the score equations
    ae0 = trace(plot_glm_predictor(mc, "R", ratio=True), "A/E")
    np.testing.assert_allclose(np.asarray(ae0.y), np.ones(3), rtol=1e-8)

    # labels and axis control follow the metric
    assert "A/E" in p.layout.title.text
    assert "A/E" in p.layout.yaxis.title.text
    pr2 = plot_glm_predictor(mc, "R", data=d20, ratio=True, y_range=(0.8, 1.3))
    assert list(pr2.layout.yaxis.range) == [0.8, 1.3]

    # continuous predictors and weighted models work the same way
    assert isinstance(plot_glm_predictor(mc, "LFT", data=d20, n_bins=20,
                                         ratio=True), go.Figure)
    rng = np.random.default_rng(69)
    dsv = d19[d19.AantalClaims > 0].copy()
    dsv["Avg"] = rng.gamma(3, 700, len(dsv))
    msv = fit_glm("Avg ~ R", dsv, family="gamma", weights="AantalClaims")
    dsv2 = dsv.copy()
    dsv2["Avg"] = dsv2.Avg * 1.2
    a2 = trace(_quiet(plot_glm_predictor, msv, "R", data=dsv2, ratio=True), "A/E")
    num = _per_level(dsv2.AantalClaims * dsv2.Avg, dsv2.R)
    den = _per_level(dsv2.AantalClaims * msv.predict(dsv2), dsv2.R)
    np.testing.assert_allclose(np.asarray(a2.y),
                               _at({k: num[k] / den[k] for k in num}, a2.x),
                               rtol=1e-8)

    with pytest.raises(PricingError, match="TRUE or FALSE"):
        plot_glm_predictor(mc, "R", ratio=None)
    assert nn == len(d20)


# ---- test-offset-link.R: the A/E plot ------------------------------------------


def _dol(seed=51, n=5000):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], n)),
                      "Exposure": np.round(rng.uniform(.3, 1, n), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(-2 + .4 * (d.R == "Z")))
    d["Maanden"] = d.Exposure * 12
    d["SCHADELAST"] = d.AantalClaims * rng.gamma(3, 700, n)
    return d


def test_exp_offset_is_not_read_as_an_exposure_when_it_is_not_one():
    dol = _dol()
    m = fit_glm("AantalClaims ~ R + offset(Exposure)", dol, family="poisson")
    with pytest.warns(PricingWarning, match="not a logarithm"):
        p = plot_glm_predictor(m, "R")
    # falls back to the prior weights rather than exp(Exposure)
    assert p.layout.yaxis2.title.text == "Weight"


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


def test_a_second_offset_does_not_end_up_on_the_exposure_axis_ae():
    # A known relativity carried as an offset - a bonus-malus scale - makes
    # exp(offset) equal to Exposure * BM. Both series divide by the exposure
    # itself, which leaves the A/E untouched and the axis readable.
    db = _bm_book()
    mb = fit_glm("AantalClaims ~ R + L + offset(log(Exposure)) + offset(log(BM))",
                 db, family="poisson")
    assert set(mb.offset_vars()) == {"Exposure", "BM"}
    p = plot_glm_predictor(mb, "R")
    bar, obs, pre = trace(p, "Exposure"), trace(p, "Observed"), trace(p, "Predicted")
    expo = _per_level(db.Exposure, db.R)
    np.testing.assert_allclose(np.asarray(bar.y), _at(expo, bar.x), rtol=1e-8)
    cl = _per_level(db.AantalClaims, db.R)
    np.testing.assert_allclose(np.asarray(obs.y),
                               _at({k: cl[k] / expo[k] for k in cl}, obs.x),
                               rtol=1e-8)
    # and the comparison the plot exists for is unaffected
    np.testing.assert_allclose(np.asarray(obs.y) / np.asarray(pre.y),
                               np.ones(len(obs.y)), rtol=1e-8)


def test_the_exposure_column_falls_back_and_can_be_named():
    rng = np.random.default_rng(72)
    nb = 4000
    db = pd.DataFrame({"R": pd.Categorical(rng.choice(["N", "Z"], nb)),
                       "Ex": np.round(rng.uniform(.3, 1, nb), 3)})
    db["AantalClaims"] = rng.poisson(db.Ex * .15)
    db["SCHADELAST"] = 0.0
    mb = fit_glm("AantalClaims ~ R + offset(log(Ex))", db, family="poisson")
    hand = db.groupby("R", observed=True).Ex.sum().to_numpy()
    # no column called "Exposure": exp(offset) still recovers it
    np.testing.assert_allclose(
        np.asarray(trace(_quiet(plot_glm_predictor, mb, "R"), "Exposure").y),
        hand, rtol=1e-8)
    # naming it is exact
    np.testing.assert_allclose(
        np.asarray(trace(_quiet(plot_glm_predictor, mb, "R", exposure_col="Ex"),
                         "Exposure").y), hand, rtol=1e-12)


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


def test_a_glm_nb_model_works_in_the_ae_plot():
    rv = rv_book()
    m = _quiet(fit_glm, "AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
               rv, family="negbin")
    p = plot_glm_predictor(m, "REGIO")
    assert isinstance(p, go.Figure)
    # the dispersion of a negative binomial is estimated, as R's family name
    # for glm.nb is not "poisson"
    o = trace(p, "Observed")
    assert np.all(np.isfinite(np.asarray(o.error_y.array)))


def test_missing_values_in_the_data_do_not_break_the_ae_plot():
    # the na.exclude test of test-review-findings.R: a Python fit is always
    # na.omit, so the point that remains is that NA rows do not misalign
    rv = rv_book()
    rv.loc[rv.index[::50], "GEWICHT"] = np.nan
    m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + log(GEWICHT) + "
                "offset(log(Exposure))", rv, family="poisson")
    p = plot_glm_predictor(m, "LEEFTIJD")
    np.testing.assert_allclose(np.sum(trace(p, "Exposure").y),
                               rv.Exposure[rv.GEWICHT.notna()].sum(), rtol=1e-10)


# ---- test-data-table-input.R: other table types ------------------------------------


def test_agg_all_and_make_plot_accept_other_table_input(dat):
    # R tests data.table and tibble input; in Python anything with
    # to_pandas() or a dict of columns is read as a DataFrame
    cols = {c: dat[c].to_numpy() for c in
            ["REGIO", "BOEKJAAR", "Exposure", "AantalClaims", "SCHADELAST"]}
    a1 = agg_all(cols, "REGIO", by_year=False)
    a2 = agg_all(dat, "REGIO", by_year=False)
    np.testing.assert_allclose(a1.sort_values("REGIO").Frequency.to_numpy(),
                               a2.sort_values("REGIO").Frequency.to_numpy())
    assert isinstance(make_plot(cols, "REGIO", "Frequency", ta_blue, "F",
                                "color", True), go.Figure)


# ---- details of the port --------------------------------------------------------


def test_agg_all_keeps_first_appearance_order_and_flags_bad_rows():
    d = pd.DataFrame({"G": ["b", "a", "b", None],
                      "Exposure": [1.0, 0.0, np.nan, 1.0],
                      "AantalClaims": [1, 0, 1, 0],
                      "SCHADELAST": [100.0, 0.0, 300.0, 0.0]})
    with pytest.warns(PricingWarning) as rec:
        a = agg_all(d, "G", by_year=False)
    msgs = " ".join(str(w.message) for w in rec)
    assert "1 NA value(s) in column 'Exposure'" in msgs
    assert "1 group(s) with Exposure <= 0" in msgs
    # data.table's by = keeps the order of first appearance and NA as a group
    assert list(a.G.astype(object).fillna("NA")) == ["b", "a", "NA"]
    np.testing.assert_allclose(a.Exposure, [1, 0, 1])
    assert np.isnan(a.Frequency[1])
    np.testing.assert_allclose(a.Severity[0], 200)
    assert list(a.columns) == ["G", "Exposure", "ClaimCount", "Loss",
                               "Frequency", "Severity"]


def test_make_plot_axes_and_traces_follow_r(dat):
    p = make_plot(dat, "REGIO", "Frequency", ta_blue, "F", "color", True)
    assert p.layout.xaxis.categoryarray == ("Noord", "West", "Zuid")
    assert p.data[0].name == "Exposure (all years)"
    assert [t.name for t in p.data[1:]] == ["2019", "2020", "2021", "2022", "2023"]
    assert p.data[1].mode == "markers"
    assert "<b>Year:</b> 2019" in p.data[1].hovertemplate
    assert p.layout.title.text == "Frequency – REGIO"
    assert p.pricing_config["toImageButtonOptions"]["filename"] == "plot"
    # a continuous axis gets a line, a character one is sorted as in R
    pc = make_plot(dat, "LEEFTIJD", "Frequency", ta_blue, "F", "color", False)
    assert pc.data[1].mode == "lines+markers"
    d = pd.DataFrame({"CH": ["b", "a", "C"] * 10, "Exposure": 1.0,
                      "AantalClaims": 1, "SCHADELAST": 1.0})
    ps = make_plot(d, "CH", "Severity", ta_blue, "S", "color", False)
    assert list(ps.data[1].x) == ["C", "a", "b"]      # arrange(): C locale
    assert ps.layout.xaxis.categoryarray == ("a", "b", "C")   # plotly: locale
    with pytest.raises(PricingError, match="by_year"):
        make_plot(dat, "REGIO", "Frequency", ta_blue, "F", "color")


def test_make_plot_facet_mode_panels(dat):
    p = make_plot(dat, "LEEFTIJD", "Frequency", ta_blue, "F", "facet", True)
    years = ["2019", "2020", "2021", "2022", "2023"]
    bars = [t for t in p.data if t.type == "bar"]
    assert len(bars) == len(years)
    assert [t.showlegend for t in bars] == [True] + [False] * 4
    lines = [t for t in p.data if t.type == "scatter" and t.mode == "lines"]
    assert len(lines) == len(years)            # 63 ages > discrete_cutoff
    # the points of each panel are that year's frequencies, the bars are
    # rescaled to the panel's maximum
    a = agg_all(dat, "LEEFTIJD", by_year=True)
    s = a[a.Year == "2021"].sort_values("LEEFTIJD")
    pts = [t for t in p.data if t.type == "scatter" and t.mode == "markers"]
    np.testing.assert_allclose(np.asarray(pts[2].y, dtype=float), s.Frequency)
    np.testing.assert_allclose(np.nanmax(np.asarray(bars[2].y, dtype=float)),
                               s.Frequency.max())
    assert "<b>2021</b>" in [a.text for a in p.layout.annotations]


def _hand_table():
    """A small table with the structure of make_rating_table()."""
    rows = [
        # Variable, Type, Level, LevelNum, Group, Exposure, F, S, XVar, GroupVar, IsThin
        ("LEEFTIJD", "continuous", "40", 40.0, None, 10.0, 1.0, 1.0, "LEEFTIJD", None, False),
        ("LEEFTIJD", "continuous", "20", 20.0, None, 5.0, 1.2, 0.9, "LEEFTIJD", None, False),
        ("REGIO", "categorical", "Noord", np.nan, None, 8.0, 1.0, 1.0, "REGIO", None, False),
        ("REGIO", "categorical", "West", np.nan, None, 3.0, 1.1, 1.05, "REGIO", None, True),
        ("BOEKJAAR:REGIO", "categorical", "2021", 2021.0, "Noord", 4.0, 1.0, np.nan, "BOEKJAAR", "REGIO", False),
        ("BOEKJAAR:REGIO", "categorical", "2022", 2022.0, "Noord", 4.0, 1.1, np.nan, "BOEKJAAR", "REGIO", False),
        ("BOEKJAAR:REGIO", "categorical", "2021", 2021.0, "West", 2.0, 1.0, np.nan, "BOEKJAAR", "REGIO", False),
        ("BOEKJAAR:REGIO", "categorical", "2022", 2022.0, "West", 1.0, 0.9, np.nan, "BOEKJAAR", "REGIO", True),
    ]
    t = pd.DataFrame(rows, columns=["Variable", "Type", "Level", "LevelNum", "Group",
                                    "Exposure", "Factor_Frequency", "Factor_Severity",
                                    "XVar", "GroupVar", "IsThin"])
    t["Factor_Premium"] = t.Factor_Frequency * t.Factor_Severity
    return t


def test_make_rating_plot_traces_follow_the_table():
    t = _hand_table()
    p = make_rating_plot(t, "REGIO")
    assert [x.name for x in p.data] == ["Exposure", "Factor Frequency",
                                        "Factor Severity", "Factor Premium"]
    f = trace(p, "Factor Frequency")
    assert f.mode == "markers" and list(f.x) == ["Noord", "West"]
    # a thin level is dimmed and says so on hover
    assert list(f.marker.opacity) == [1.0, 0.4]
    assert "low claim volume" in f.text[1] and f.text[0] == ""
    assert p.layout.shapes[0].y0 == 1 and p.layout.title.text == "Rating factors – REGIO"
    assert p.pricing_config["toImageButtonOptions"]["filename"] == "rating"
    pc = make_rating_plot(t, "LEEFTIJD")
    assert trace(pc, "Factor Premium").mode == "lines+markers"
    assert trace(pc, "Factor Premium").line.dash == "dash"
    np.testing.assert_allclose(trace_y(pc, "Factor Premium"), [1.0, 1.08])

    pi = make_rating_plot(t, "BOEKJAAR:REGIO")
    assert [x.name for x in pi.data] == ["Exposure", "REGIO = Noord", "REGIO = West"]
    # categorical x categorical: bars summed over the groups
    np.testing.assert_allclose(trace_y(pi, "Exposure"), [6.0, 5.0])
    assert pi.layout.title.text == ("Interaction – BOEKJAAR × REGIO"
                                    "  (Factor Frequency)")
    assert pi.layout.legend.title.text == "REGIO"
    assert pi.pricing_config["toImageButtonOptions"]["filename"] == "interaction"
    assert list(trace(pi, "REGIO = West").marker.opacity) == [1.0, 0.4]
    with pytest.raises(PricingError, match="should be one of"):
        make_rating_plot(t, "BOEKJAAR:REGIO", metric="Claims")
    with pytest.raises(PricingError, match="not found"):
        make_rating_plot(t, "NOPE")


def test_make_rating_plot_on_a_continuous_interaction(tbl2):
    p = make_rating_plot(tbl2, "LEEFTIJD:REGIO")
    names = [t.name for t in p.data]
    assert names == ["Exposure", "REGIO = Noord", "REGIO = West", "REGIO = Zuid"]
    # the table carries exposure per grid cell, so the bars are summed over
    # the groups at each numeric level, on a numeric axis
    rows = tbl2[tbl2.Variable == "LEEFTIJD:REGIO"]
    hand = rows.groupby(rows.LevelNum).Exposure.sum()
    bar = trace(p, "Exposure")
    np.testing.assert_allclose(np.asarray(bar.x, dtype=float), hand.index.to_numpy())
    np.testing.assert_allclose(np.asarray(bar.y, dtype=float), hand.to_numpy())
    assert p.layout.yaxis2.title.text == "Exposure"
    for t in p.data[1:]:
        x = np.asarray(t.x, dtype=float)
        assert np.all(np.diff(x) > 0) and t.mode == "lines+markers"
