"""glm_diagnostics, glm_collinearity and plot_glm_residuals: the port of
tests/testthat/test-diagnostics.R, the glm_collinearity tests of
test-collinearity-spotlight.R and the matching tests of
test-review-findings.R."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pricingtools_rmo._core import PricingError, PricingMessage, PricingWarning
from pricingtools_rmo.diagnostics import (glm_collinearity, glm_diagnostics,
                                          plot_glm_residuals)
from pricingtools_rmo.model import fit_glm

from conftest import trace, trace_y


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fun(*args, **kw)


def rv_book(n=20000, seed=1, theta=np.inf):
    """The review's book (rv_book() of test-review-findings.R)."""
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 85, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "BRANDSTOF": pd.Categorical(rng.choice(["Benzine", "Diesel", "Waterstof"],
                                               n, p=[.7, .295, .005]),
                                    categories=["Benzine", "Diesel", "Waterstof"]),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "BOEKJAAR": rng.integers(2021, 2025, n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    eta = (np.log(0.08) + 0.6 * np.exp(-(d.LEEFTIJD - 18) / 8)
           + d.REGIO.map({"Noord": 0, "Zuid": .2, "West": .35}).astype(float)
           + 0.3 * np.log(d.GEWICHT / 1300))
    fr = rng.gamma(theta, 1 / theta, n) if np.isfinite(theta) else 1
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta) * fr)
    d["SCHADELAST"] = np.where(d.AantalClaims > 0,
                               d.AantalClaims * rng.gamma(1.5, 2500 / 1.5, n), 0.0)
    return d


# ---- test-diagnostics.R --------------------------------------------------------------


def test_glm_diagnostics_summarises_both_models(m_freq, m_sev):
    diag = glm_diagnostics(m_freq, m_sev)
    assert len(diag) == 2
    assert {"Dispersion", "AIC", "DevianceExplained"} <= set(diag.columns)
    # well-specified Poisson: dispersion near 1
    assert abs(diag.Dispersion[diag.Model == "frequency"].iloc[0] - 1) < 0.1
    assert list(diag.columns) == ["Model", "Family", "Link", "N", "Deviance",
                                  "DFResidual", "AIC", "Dispersion",
                                  "DevianceExplained"]
    with pytest.raises(PricingError, match="at least one model"):
        glm_diagnostics()
    with pytest.raises(PricingError, match="must be a glm object"):
        glm_diagnostics("nope")


def test_overdispersion_raises_a_warning(dat):
    # negative binomial data fitted as Poisson: dispersion ~ 1 + mu/size >> 1.2
    rng = np.random.default_rng(7)
    d_od = dat.copy()
    lin = (-2.3 + 0.015 * (d_od.LEEFTIJD - 40)
           + np.where(d_od.REGIO == "Zuid", .25,
                      np.where(d_od.REGIO == "West", .4, 0)))
    mu = d_od.Exposure * np.exp(lin)
    size = 0.05
    d_od["AC_od"] = rng.negative_binomial(size, size / (size + mu))
    m_od = fit_glm("AC_od ~ REGIO + offset(log(Exposure))", d_od,
                   family="poisson")
    with pytest.warns(PricingWarning, match="overdispersion"):
        glm_diagnostics(m_od)


def test_binned_residual_plots_render_and_stay_within_the_band(m_freq):
    p_res = plot_glm_residuals(m_freq)
    assert isinstance(p_res, go.Figure)
    assert isinstance(plot_glm_residuals(m_freq, "LEEFTIJD"), go.Figure)
    assert isinstance(plot_glm_residuals(m_freq, "REGIO"), go.Figure)

    # for a correct model most binned means must lie inside the band
    mr = trace_y(p_res, "Mean residual")
    bd = trace_y(p_res, "±2·SE")
    frac_in = np.mean(np.abs(mr) <= bd)
    assert frac_in > 0.7

    assert p_res.pricing_config["toImageButtonOptions"]["filename"] == "residuals"
    with pytest.raises(PricingError, match="not found"):
        plot_glm_residuals(m_freq, "GEEN_KOLOM")
    with pytest.raises(PricingError, match="at least 2"):
        plot_glm_residuals(m_freq, n_bins=1)
    with pytest.raises(PricingError, match="lo < hi"):
        plot_glm_residuals(m_freq, y_range=(1, 0))


# ---- test-collinearity-spotlight.R: glm_collinearity() ------------------------------


@pytest.fixture(scope="module")
def dc():
    rng = np.random.default_rng(91)
    nc = 20000
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, nc)),
        "GEWICHT": np.round(np.minimum(np.maximum(rng.normal(1350, 250, nc), 800),
                                       2400), -1),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp", "Rand"], nc),
                                categories=["Dorp", "Rand", "Stad"]),
        "Exposure": np.round(rng.uniform(.3, 1, nc), 3)})
    # a near-duplicate of GEWICHT: the pair must light up, the rest must not
    d["GEWICHT_PROXY"] = d.GEWICHT + np.round(rng.normal(0, 25, nc))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.7 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD)
        + 0.0003 * (d.GEWICHT - 1350)))
    return d


def test_collinear_terms_are_flagged_and_independent_ones_are_not(dc):
    m_col = fit_glm("AantalClaims ~ LEEFTIJD + GEWICHT + GEWICHT_PROXY + REGIO"
                    " + offset(log(Exposure))", dc, family="poisson")
    v = _quiet(glm_collinearity, m_col)

    assert {"Term", "DF", "GVIF", "GVIF_scaled", "Flag"} <= set(v.columns)
    assert set(v.Term) == {"LEEFTIJD", "GEWICHT", "GEWICHT_PROXY", "REGIO"}
    flagged = set(v.Term[v.Flag.fillna(False).astype(bool)])
    assert flagged <= {"GEWICHT", "GEWICHT_PROXY"}
    assert {"GEWICHT", "GEWICHT_PROXY"} <= flagged
    assert not v.Flag[v.Term == "REGIO"].iloc[0]
    assert np.all(np.diff(v.GVIF_scaled) <= 0)       # sorted descending
    with pytest.warns(PricingWarning, match="scaled GVIF"):
        glm_collinearity(m_col)


def test_a_clean_model_flags_nothing(dc):
    m_ok = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   dc, family="poisson")
    v = glm_collinearity(m_ok)
    assert not v.Flag.fillna(False).any()
    assert np.all(v.GVIF_scaled < 2)


def test_multi_column_terms_use_the_generalised_form(dc):
    m_ns = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + REGIO + offset(log(Exposure))",
                   dc, family="poisson")
    v = glm_collinearity(m_ns)
    assert v.DF[v.Term == "ns(LEEFTIJD, 4)"].iloc[0] == 4
    assert v.DF[v.Term == "REGIO"].iloc[0] == 2
    # scaled version is the df-th root, so it must be far below the raw GVIF
    assert (v.GVIF_scaled[v.Term == "ns(LEEFTIJD, 4)"].iloc[0]
            < v.GVIF[v.Term == "ns(LEEFTIJD, 4)"].iloc[0])


def test_splines_and_rank_deficiency_do_not_empty_the_table(dc):
    # Many correlated spline columns: det(R) underflows to zero long before
    # the model is singular, so the GVIF must be computed on log-determinants
    m_sp = fit_glm("AantalClaims ~ ns(LEEFTIJD, 5) + ns(GEWICHT, 3) + REGIO"
                   " + offset(log(Exposure))", dc, family="poisson")
    v_sp = glm_collinearity(m_sp)
    assert len(v_sp) == 3
    assert np.all(np.isfinite(v_sp.GVIF_scaled))
    assert not v_sp.Flag.fillna(False).any()

    # A rank-deficient fit leaves an aliased coefficient; it is dropped with
    # a warning and the remaining terms still get a number
    m_rd = fit_glm("AantalClaims ~ ns(LEEFTIJD, 5) + REGIO + LEEFTIJD:REGIO"
                   " + offset(log(Exposure))", dc, family="poisson")
    if m_rd.coef.isna().sum() == 0:
        pytest.skip("fit was not rank deficient here")
    v_rd = _quiet(glm_collinearity, m_rd)
    assert len(v_rd) > 1
    assert np.all(np.isfinite(v_rd.GVIF_scaled))
    # the interaction and REGIO, which it contains, are reported, not flagged
    with pytest.warns(PricingWarning) as rec:
        glm_collinearity(m_rd)
    msgs = [str(w.message) for w in rec]
    assert any("rank" in s for s in msgs)
    assert any("belong to an interaction" in s for s in msgs)
    assert v_rd.Flag[v_rd.InInteraction].isna().all()
    # the variables in the order they first appear in the formula, as R
    # labels the term ("REGIO:LEEFTIJD")
    assert set(v_rd.Term[v_rd.InInteraction]) == {"REGIO", "REGIO:LEEFTIJD"}


def test_a_gvif_that_depends_on_the_coding_is_not_flagged(dc):
    # Centring the numeric side of an interaction leaves the fit identical,
    # and moves the GVIF of the factor's main effect, whose coefficients are
    # then measured at another value of GEWICHT (DIAG-07)
    m_a = fit_glm("AantalClaims ~ GEWICHT * REGIO + LEEFTIJD"
                  " + offset(log(Exposure))", dc, family="poisson")
    dc2 = dc.copy()
    dc2["GEWICHT"] = dc2.GEWICHT - 1350
    m_b = m_a.refit(dc2)
    np.testing.assert_allclose(m_a.fitted, m_b.fitted, rtol=1e-8)
    g_a = _quiet(glm_collinearity, m_a)
    g_b = _quiet(glm_collinearity, m_b)
    ga = g_a.GVIF_scaled[g_a.Term == "REGIO"].iloc[0]
    gb = g_b.GVIF_scaled[g_b.Term == "REGIO"].iloc[0]
    assert ga / gb > 2
    # so neither version flags the terms of the interaction, and a term
    # outside it still gets its flag
    int_terms = {"GEWICHT", "REGIO", "GEWICHT:REGIO"}
    assert set(g_a.Term[g_a.InInteraction]) == int_terms
    assert g_a.Flag[g_a.Term.isin(int_terms)].isna().all()
    assert g_b.Flag[g_b.Term.isin(int_terms)].isna().all()
    fl = g_a.Flag[g_a.Term == "LEEFTIJD"].iloc[0]
    assert not pd.isna(fl) and not fl
    with pytest.warns(PricingWarning, match="belong to an interaction"):
        glm_collinearity(m_a)


def test_a_model_with_fewer_than_two_terms_returns_nothing(dc):
    m1 = fit_glm("AantalClaims ~ LEEFTIJD + offset(log(Exposure))", dc,
                 family="poisson")
    with pytest.warns(PricingMessage, match="fewer than two"):
        v = glm_collinearity(m1)
    assert len(v) == 0
    assert list(v.columns) == ["Term", "DF", "GVIF", "GVIF_scaled",
                               "InInteraction", "Flag"]
    with pytest.raises(PricingError, match="glm object"):
        glm_collinearity("nope")


# ---- test-review-findings.R -----------------------------------------------------------


def test_na_rows_give_the_same_result_as_complete_rows():
    # R: "na.exclude gives the same result as na.omit". A Python model is
    # always fitted as with na.omit, so the port checks that a model on data
    # with missing values matches one on the complete rows, in every
    # diagnostic that reads its residuals.
    from pricingtools_rmo.interactions import plot_residual_heatmap
    from pricingtools_rmo.qq_influence import plot_glm_qq
    rv = rv_book()
    d1 = rv.copy()
    d1.loc[d1.index[::50], "GEWICHT"] = np.nan
    f = ("AantalClaims ~ LEEFTIJD + REGIO + log(GEWICHT)"
         " + offset(log(Exposure))")
    m_x = fit_glm(f, d1, family="poisson")
    m_o = fit_glm(f, d1.dropna(subset=["GEWICHT"]), family="poisson")
    assert glm_diagnostics(m_x).Dispersion.iloc[0] == pytest.approx(
        glm_diagnostics(m_o).Dispersion.iloc[0], rel=1e-10)
    assert isinstance(plot_glm_residuals(m_x, "LEEFTIJD"), go.Figure)
    h_o = plot_residual_heatmap(m_o, "LEEFTIJD", "REGIO").cells
    h_x = plot_residual_heatmap(m_x, "LEEFTIJD", "REGIO").cells
    h_o = h_o.sort_values(["gx", "gy"]).reset_index(drop=True)
    h_x = h_x.sort_values(["gx", "gy"]).reset_index(drop=True)
    np.testing.assert_allclose(h_x.Expected, h_o.Expected, rtol=1e-8)
    np.testing.assert_allclose(plot_glm_qq(m_x, seed=3).residuals,
                               plot_glm_qq(m_o, seed=3).residuals, rtol=1e-8)


def test_binned_deviance_residuals_of_a_correct_poisson_model_stay_in_the_band():
    rv = rv_book()
    m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + log(GEWICHT)"
                " + offset(log(Exposure))", rv, family="poisson")
    p = plot_glm_residuals(m, residual_type="deviance")
    y = trace_y(p, "Mean residual")
    band = np.asarray(p.data[0].y, dtype=float)
    assert np.mean(np.abs(y) > band) < 0.15
    # the band follows the bin instead of being one number
    assert len(np.unique(np.round(band, 6))) > 1
    assert p.layout.yaxis.title.text == "Mean deviance residual, centred"


def test_binned_residuals_are_drawn_as_points_not_a_line(m_freq):
    # DIAG-03 (test-agg-plots.R pins it in R)
    p = plot_glm_residuals(m_freq)
    assert trace(p, "Mean residual").mode == "markers"
    p = plot_glm_residuals(m_freq, "REGIO")
    t = trace(p, "Mean residual")
    assert t.mode == "markers" and t.error_y.type == "data"
    assert list(p.layout.xaxis.categoryarray) == ["Noord", "West", "Zuid"]
