"""plot_residual_heatmap and detect_interactions: the port of
tests/testthat/test-interactions.R, the interaction tests of
test-awkward-names.R and those of test-review-findings.R.

A GLM that is missing a known interaction must be caught by the cell-level
tools, precisely because the one-way checks cannot see it."""
import json
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pricingtools_rmo._core import PricingError, PricingMessage, PricingWarning
from pricingtools_rmo.interactions import (detect_interactions,
                                           plot_residual_heatmap)
from pricingtools_rmo.model import fit_glm


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fun(*args, **kw)


def _regio(rng, n, levels=("Stad", "Rand", "Dorp")):
    return pd.Categorical(rng.choice(list(levels), n),
                          categories=sorted(levels))


def make_missed_interaction(n=60000, seed=11):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": _regio(rng, n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    jong = np.maximum(0, 30 - d.LEEFTIJD)
    reg = d.REGIO.map({"Stad": .25, "Rand": .10, "Dorp": 0}).astype(float)
    lin = (-1.9 + 0.025 * jong + reg
           + 0.060 * jong * (d.REGIO == "Stad"))      # the interaction
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(lin))
    d["SCHADELAST"] = d.AantalClaims * 1500.0
    return d


@pytest.fixture(scope="module")
def dsim():
    return make_missed_interaction()


@pytest.fixture(scope="module")
def m_gap(dsim):
    return fit_glm("AantalClaims ~ ns(LEEFTIJD, 5) + REGIO + offset(log(Exposure))",
                   dsim, family="poisson")


def test_one_way_ae_is_blind_to_the_missing_interaction(dsim, m_gap):
    # Score equations of a canonical link: fitted totals equal observed
    # totals for every level of a categorical variable in the model
    mu = m_gap.predict(dsim)
    g = pd.DataFrame({"a": dsim.AantalClaims, "e": mu, "r": dsim.REGIO})
    s = g.groupby("r", observed=True)[["a", "e"]].sum()
    assert np.all(np.abs(s.a / s.e - 1) < 1e-8)


def test_detect_interactions_ranks_the_true_pair_first(m_gap):
    r = detect_interactions(m_gap, n_bins=8, n_sim=100, seed=1)
    assert isinstance(r, pd.DataFrame)
    assert {"VarX", "VarY", "Z", "P", "Method", "MaxAE",
            "MaxAE_ExposureShare"} <= set(r.columns)
    assert list(r.columns) == ["VarX", "VarY", "Cells", "Claims", "Deviance",
                               "DF", "Z", "P", "Method", "Dispersion", "MaxAE",
                               "MaxAE_Claims", "MaxAE_ExposureShare"]
    assert r.Method.iloc[0] == "simulation"
    assert sorted([r.VarX.iloc[0], r.VarY.iloc[0]]) == ["LEEFTIJD", "REGIO"]
    assert r.Z.iloc[0] > 5
    assert r.P.iloc[0] < 0.05
    # the worst cell is materially, not just significantly, off
    assert abs(r.MaxAE.iloc[0] - 1) > 0.15


def test_no_interaction_present_means_no_signal(dsim):
    rng = np.random.default_rng(3)
    d = dsim.copy()
    # same marginals, but the claims now follow a purely additive truth
    jong = np.maximum(0, 30 - d.LEEFTIJD)
    lin = (-1.9 + 0.025 * jong
           + d.REGIO.map({"Stad": .25, "Rand": .10, "Dorp": 0}).astype(float))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(lin))
    m_ok = fit_glm("AantalClaims ~ ns(LEEFTIJD, 5) + REGIO + offset(log(Exposure))",
                   d, family="poisson")
    r = detect_interactions(m_ok, n_bins=8, n_sim=100, seed=1)
    assert r.Z.iloc[0] < 4          # nothing stands out
    assert r.P.iloc[0] > 0.01


def test_plot_residual_heatmap_returns_a_plot_plus_its_cell_table(m_gap):
    p = plot_residual_heatmap(m_gap, "LEEFTIJD", "REGIO", n_bins=8)
    assert isinstance(p, go.Figure)
    assert p.pricing_config["toImageButtonOptions"]["filename"] == "heatmap"

    cells = p.cells
    assert isinstance(cells, pd.DataFrame)
    assert {"AE", "Claims", "Exposure", "IsThin"} <= set(cells.columns)

    # the young/Stad corner must be the underpriced one
    young = cells.gx.cat.categories[0]
    sel = cells.gx.astype(str) == young
    ae_young = dict(zip(cells.gy[sel].astype(str), cells.AE[sel]))
    assert ae_young["Stad"] > 1.10
    assert ae_young["Dorp"] < 1.00


def test_thin_cells_are_not_coloured_and_z_range_is_validated(dsim, m_gap):
    d = dsim.copy()
    d["RAAR"] = pd.Categorical(np.where(np.arange(len(d)) < 20, "Zeldzaam",
                                        "Normaal"))
    m = fit_glm("AantalClaims ~ REGIO + RAAR + offset(log(Exposure))", d,
                family="poisson")
    p = plot_residual_heatmap(m, "REGIO", "RAAR", min_claims=30)
    cells = p.cells
    assert cells.IsThin.any()

    # a thin cell keeps its A/E in the table but is blank in the plotted z
    z = np.array(p.data[0].z, dtype=float)
    assert np.isnan(z).any()
    assert np.isfinite(cells.AE[cells.IsThin]).all()
    # and is marked with a cross instead
    assert p.data[1].marker.symbol == "x"

    with pytest.raises(PricingError, match="z_range"):
        plot_residual_heatmap(m_gap, "LEEFTIJD", "REGIO", z_range=(2, 1))


def _max_ae(y, mu, w, g):
    df = pd.DataFrame({"a": w * y, "e": w * mu, "g": g})
    s = df.groupby("g", observed=True)[["a", "e"]].sum()
    return float(np.max(np.abs(s.a / s.e - 1)))


def _max_mean_ratio(y, mu, w, g):
    df = pd.DataFrame({"r": w * y / mu, "w": w, "g": g})
    s = df.groupby("g", observed=True)[["r", "w"]].sum()
    return float(np.max(np.abs(s.r / s.w - 1)))


def _link_book(seed=31, nn=40000, with_r=True):
    rng = np.random.default_rng(seed)
    dd = pd.DataFrame({"L": np.round(rng.uniform(18, 80, nn)),
                       "R": pd.Categorical(rng.choice(["N", "O", "Z"], nn),
                                           categories=["N", "O", "Z"]),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    eta = -2.3 + .015 * dd.L + (.3 * (dd.R == "Z") if with_r else 0)
    dd["AantalClaims"] = rng.poisson(dd.Exposure * np.exp(eta))
    return dd, rng


def test_each_link_pins_a_different_quantity_to_exactly_1():
    # The documented blind spot rests on the score equations weighting each
    # observation by (dmu/deta)/V(mu). That weight is 1 for a canonical link,
    # leaving sum(x*(y - mu)) = 0, so the A/E is exact. Gamma with a log link
    # is NOT canonical and pins the weighted mean of the RATIO instead. These
    # numbers are what the documentation claims, so they are pinned here.
    dd, rng = _link_book()
    nn = len(dd)
    dd["SCHADELAST"] = dd.AantalClaims * rng.gamma(3, 700, nn)
    dsv = dd[dd.AantalClaims > 0].copy()
    dsv["Avg"] = dsv.SCHADELAST / dsv.AantalClaims

    # Poisson log: canonical, so the TOTALS match exactly
    mp = fit_glm("AantalClaims ~ L + R + offset(log(Exposure))", dd,
                 family="poisson")
    w1 = np.ones(nn)
    assert _max_ae(dd.AantalClaims, mp.fitted, w1, dd.R) < 1e-10
    assert _max_mean_ratio(dd.AantalClaims, mp.fitted, w1, dd.R) > 1e-4

    # Gamma log: not canonical, so the mean RATIO matches instead
    mg = fit_glm("Avg ~ L + R", dsv, family="gamma", link="log",
                 weights="AantalClaims")
    wg = dsv.AantalClaims.to_numpy(dtype=float)
    assert _max_mean_ratio(dsv.Avg, mg.fitted, wg, dsv.R) < 1e-6
    assert _max_ae(dsv.Avg, mg.fitted, wg, dsv.R) > 1e-6

    # Gamma inverse: canonical again, so the totals match and the ratio does not
    # (statsmodels warns that these links leave the family's domain; R does
    # not, so the warning is silenced)
    mi = _quiet(fit_glm, "Avg ~ L + R", dsv, family="gamma", link="inverse",
                weights="AantalClaims")
    assert _max_ae(dsv.Avg, mi.fitted, wg, dsv.R) < 1e-8
    assert _max_mean_ratio(dsv.Avg, mi.fitted, wg, dsv.R) > 1e-6

    # A saturated model is blind whatever the link, which is the second route
    # (R passes start values for the identity link; statsmodels starts from
    # the mean and needs none)
    ms = _quiet(fit_glm, "AantalClaims ~ R", dd, family="poisson",
                link="identity")
    assert _max_ae(dd.AantalClaims, ms.fitted, w1, dd.R) < 1e-10


def test_the_severity_residue_is_far_too_small_to_price_on():
    # The claim is that a severity model is blind in practice, not by
    # identity: a real omitted interaction has to move the one-way A/E by
    # something you could see, and it does not.
    dd, rng = _link_book(with_r=False)
    dsv = dd[dd.AantalClaims > 0].copy()
    dsv["Avg"] = rng.gamma(3, 700, len(dsv))
    wg = dsv.AantalClaims.to_numpy(dtype=float)

    m0 = fit_glm("Avg ~ L + R", dsv, family="gamma", link="log",
                 weights="AantalClaims")
    base = _max_ae(dsv.Avg, m0.fitted, wg, dsv.R)

    # now plant a 35% interaction the model is not given
    dsv["Avg2"] = dsv.Avg * np.where((dsv.R == "Z") & (dsv.L > 50), 1.35, 1)
    m1 = fit_glm("Avg2 ~ L + R", dsv, family="gamma", link="log",
                 weights="AantalClaims")
    planted = _max_ae(dsv.Avg2, m1.fitted, wg, dsv.R)

    assert base < 0.001         # numerical floor, ~0.01%
    assert planted < 0.01       # a 35% effect still moves it under 1%
    assert planted > base       # it is visible in principle

    # and the cell-level test finds what the margins hide
    r = _quiet(detect_interactions, m1, vars=["L", "R"], n_bins=4, n_sim=99)
    assert np.nanmax(np.abs(r.MaxAE - 1)) > 10 * planted


# ---- test-awkward-names.R ---------------------------------------------------------------


@pytest.fixture(scope="module")
def daw():
    rng = np.random.default_rng(41)
    naw = 15000
    d = pd.DataFrame({
        "REGIO": _regio(rng, naw, ("Stad", "Dorp")),
        "LEEFTIJD": np.round(rng.uniform(18, 80, naw)),
        "POLISJAREN": np.round(rng.uniform(.3, 1, naw), 3)})
    d["AUTO GEWICHT"] = np.round(rng.uniform(800, 2000, naw))
    d["Exposure"] = d.POLISJAREN
    d["AantalClaims"] = rng.poisson(d.POLISJAREN * np.exp(
        -1.7 + 0.3 * (d.REGIO == "Stad") + 0.0002 * (d["AUTO GEWICHT"] - 1400)))
    d["SCHADELAST"] = d.AantalClaims * 1500.0
    return d


def test_a_column_name_with_a_space_survives_the_interaction_scan(daw):
    # the detect_interactions part of "a column name with a space survives
    # the whole rating table"
    m = fit_glm("AantalClaims ~ REGIO + `AUTO GEWICHT` + offset(log(Exposure))",
                daw, family="poisson")
    r = detect_interactions(m, n_bins=5, n_sim=30, seed=1)
    assert {"REGIO", "AUTO GEWICHT"} <= set(r.VarX) | set(r.VarY)
    p = plot_residual_heatmap(m, "AUTO GEWICHT", "REGIO", n_bins=5)
    assert "AUTO GEWICHT" in p.layout.title.text


def test_a_poisson_model_without_an_offset_still_gets_the_simulated_null(daw):
    m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO", daw, family="poisson")
    r = detect_interactions(m, n_bins=5, n_sim=30, seed=1)
    assert set(r.Method) == {"simulation"}

    # a Gamma severity model genuinely cannot use it, and says why
    ds = daw[daw.AantalClaims > 0].copy()
    ds["Gem"] = ds.SCHADELAST / ds.AantalClaims
    # R's data give a constant severity (1500), which R's glm() fits with a
    # deviance of 0; statsmodels' IRLS refuses that perfect fit ("first
    # guess on the deviance function returned a nan"), so the severity gets
    # some spread here. The property under test, the chisq route for a
    # response that is not counts, is unchanged.
    ds["Gem"] = ds.Gem * np.random.default_rng(2).gamma(50, 1 / 50, len(ds))
    m_g = fit_glm("Gem ~ REGIO + LEEFTIJD", ds, family="gamma", link="log",
                  weights="AantalClaims")
    with pytest.warns(PricingMessage, match="not Poisson counts"):
        r2 = detect_interactions(m_g, n_bins=5, n_sim=30)
    assert set(r2.Method) == {"chisq"}


# ---- test-review-findings.R ---------------------------------------------------------------


def test_the_interaction_scan_is_not_fooled_by_overdispersion():
    rng = np.random.default_rng(9)
    n = 6000
    d1 = pd.DataFrame({
        "A": pd.Categorical(rng.integers(1, 6, n)),
        "B": pd.Categorical(rng.integers(1, 7, n)),
        "C": pd.Categorical(rng.integers(1, 5, n)),
        "Exposure": rng.uniform(5, 30, n)})
    mu = d1.Exposure * 0.1 * np.exp(0.1 * d1.A.astype(int))
    d1["AantalClaims"] = rng.poisson(mu * rng.gamma(3, 1 / 3, n))  # dispersion ~1.9
    m = fit_glm("AantalClaims ~ A + B + C + offset(log(Exposure))", d1,
                family="quasipoisson")
    di = _quiet(detect_interactions, m, n_sim=100, seed=1)
    assert np.all(di.P > 0.01)
    # the dispersion of a cell total, not the row-level Pearson one
    phi_row = np.sum(m.residuals("pearson") ** 2) / m.df_residual
    assert di.Dispersion.iloc[0] > phi_row
    assert di.Dispersion.iloc[0] == pytest.approx(
        np.sum((d1.AantalClaims - m.fitted) ** 2) / np.sum(m.fitted)
        * n / m.df_residual, rel=1e-10)
    with pytest.warns(PricingMessage, match="overdispersed"):
        detect_interactions(m, n_sim=10, seed=1)


def test_a_heatmap_with_every_cell_thin_still_builds():
    from test_diagnostics import rv_book
    d1 = rv_book(n=400, seed=5)
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", d1,
                 family="poisson")
    with pytest.warns(PricingWarning, match="none is coloured"):
        p = plot_residual_heatmap(mf, "LEEFTIJD", "REGIO")
    # the figure serialises (plotly_build in R) and holds only the crosses
    json.loads(p.to_json())
    assert [t.type for t in p.data] == ["scatter"]
    assert p.cells.IsThin.all()


def test_detect_interactions_input_is_validated(m_gap):
    with pytest.raises(PricingError, match="glm object"):
        detect_interactions("nope")
    with pytest.raises(PricingError, match="at least two variables"):
        detect_interactions(m_gap, vars=["REGIO"])
    with pytest.raises(PricingError, match="not found in the model data"):
        plot_residual_heatmap(m_gap, "GEEN_KOLOM", "REGIO")
    # top_n keeps the strongest pairs, and the offset is never a candidate
    r = detect_interactions(m_gap, vars=["LEEFTIJD", "REGIO", "Exposure"],
                            n_bins=5, n_sim=20, seed=2, top_n=1)
    assert len(r) == 1
    # the same seed gives the same reference distribution
    r1 = detect_interactions(m_gap, n_bins=5, n_sim=20, seed=4)
    r2 = detect_interactions(m_gap, n_bins=5, n_sim=20, seed=4)
    pd.testing.assert_frame_equal(r1, r2)
