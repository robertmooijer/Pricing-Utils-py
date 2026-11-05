"""plot_glm_qq, plot_glm_influence and the quantile residuals: the port of
tests/testthat/test-qq-influence.R and the glm.nb Q-Q test of
test-review-findings.R."""
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest
from scipy import stats

from pricingtools_rmo._core import PricingError, PricingMessage, PricingWarning
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.qq_influence import (_quantile_residuals, plot_glm_influence,
                                           plot_glm_qq)

NQ = 6000


@pytest.fixture(scope="module")
def dq():
    rng = np.random.default_rng(17)
    d = pd.DataFrame({"x": rng.uniform(size=NQ),
                      "Exposure": np.round(rng.uniform(.3, 1, NQ), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(-2.0 + 1.1 * d.x))
    return d


@pytest.fixture(scope="module")
def mq_pois(dq):
    return fit_glm("AantalClaims ~ x + offset(log(Exposure))", dq,
                   family="poisson")


@pytest.fixture(scope="module")
def dqs():
    rng = np.random.default_rng(18)
    d = pd.DataFrame({"x": rng.uniform(size=4000),
                      "w": 1 + rng.poisson(1.5, 4000)})
    mu = np.exp(7 + .5 * d.x)
    d["y"] = [np.mean(rng.gamma(2.5, m / 2.5, w)) for m, w in zip(mu, d.w)]
    return d


@pytest.fixture(scope="module")
def mq_gam(dqs):
    return fit_glm("y ~ x", dqs, family="gamma", link="log", weights="w")


def _ks_p(r):
    return stats.kstest(r, "norm").pvalue


def test_quantile_residuals_are_standard_normal_on_a_correct_model(mq_pois, mq_gam):
    # the whole point: for counts, deviance residuals are not normal even
    # when the model is right, so the classic Q-Q condemns a correct fit
    r = _quantile_residuals(mq_pois, "t", np.random.default_rng(2))
    assert abs(np.mean(r)) < 0.05
    assert abs(np.std(r, ddof=1) - 1) < 0.05
    assert _ks_p(r) > 0.01
    assert abs(np.mean(np.abs(r) > 1.96) - 0.05) < 0.015

    # deviance residuals on the same data do not manage that
    dv = mq_pois.residuals("deviance")
    dv = (dv - dv.mean()) / dv.std(ddof=1)
    assert _ks_p(dv) < 1e-6

    # Gamma too, and there the response is continuous so no randomisation
    rg = _quantile_residuals(mq_gam, "t")
    assert abs(np.std(rg, ddof=1) - 1) < 0.05
    assert _ks_p(rg) > 0.01
    np.testing.assert_array_equal(_quantile_residuals(mq_gam, "t"), rg)


def test_quantile_residuals_are_the_cdf_transform(mq_pois, mq_gam):
    # R checks against statmod::qresid(); here the defining property is
    # checked directly (and tests/parity compares with the R package).
    # Continuous response: exactly qnorm(F(y)) with the GLM's own Gamma,
    # shape w / phi and mean mu.
    m = mq_gam
    phi = np.sum(m.residuals("pearson") ** 2) / m.df_residual
    w = m.prior_weights
    ref = stats.norm.ppf(stats.gamma.cdf(m.y, a=w / phi,
                                         scale=phi * m.fitted / w))
    np.testing.assert_allclose(_quantile_residuals(m, "t"), ref, rtol=1e-12)

    # Discrete response: u = pnorm(r) lies inside the jump of the fitted
    # CDF at y, and the same seed gives the same draw
    r1 = _quantile_residuals(mq_pois, "t", np.random.default_rng(11))
    r2 = _quantile_residuals(mq_pois, "t", np.random.default_rng(11))
    np.testing.assert_array_equal(r1, r2)
    u = stats.norm.cdf(r1)
    a = stats.poisson.cdf(mq_pois.y - 1, mq_pois.fitted)
    b = stats.poisson.cdf(mq_pois.y, mq_pois.fitted)
    assert np.all(u >= a - 1e-12) and np.all(u <= b + 1e-12)
    # and it is uniform within the jump, not pinned to an end of it
    t = (u - a) / (b - a)
    assert stats.kstest(t, "uniform").pvalue > 0.01


def test_the_qq_plot_flags_a_distribution_that_is_genuinely_wrong(dq, mq_pois):
    # a Poisson fitted to overdispersed counts must show fatter tails
    rng = np.random.default_rng(4)
    d = dq.copy()
    mu = d.Exposure * np.exp(-1.2 + 1.1 * d.x)
    d["Over"] = rng.negative_binomial(.4, .4 / (.4 + mu))
    m_over = fit_glm("Over ~ x + offset(log(Exposure))", d, family="poisson")
    r_ok = _quantile_residuals(mq_pois, "t", rng)
    r_bad = _quantile_residuals(m_over, "t", rng)
    assert np.std(r_bad, ddof=1) > np.std(r_ok, ddof=1)
    assert np.mean(np.abs(r_bad) > 1.96) > np.mean(np.abs(r_ok) > 1.96)


def test_plot_glm_qq_returns_a_plotly_object_and_its_residuals(mq_pois, mq_gam):
    p = plot_glm_qq(mq_pois, seed=1)
    assert isinstance(p, go.Figure)
    assert p.type == "quantile"
    assert len(p.residuals) == NQ
    assert isinstance(plot_glm_qq(mq_gam), go.Figure)
    assert p.pricing_config["toImageButtonOptions"]["filename"] == "qq"
    assert [t.name for t in p.data] == ["95% pointwise", "95% pointwise",
                                        "Perfect fit", "quantile residuals"]

    # seed makes the randomisation reproducible
    np.testing.assert_array_equal(plot_glm_qq(mq_pois, seed=7).residuals,
                                  plot_glm_qq(mq_pois, seed=7).residuals)

    # thinning draws fewer points but leaves the residuals untouched
    b = plot_glm_qq(mq_pois, n_max=200, band=False, seed=1)
    pts = b.data[-1]
    assert len(pts.y) <= 210
    # both extremes are always kept
    r = np.sort(b.residuals)
    assert pts.y[0] == r[0] and pts.y[-1] == r[-1]
    assert len(plot_glm_qq(mq_pois, n_max=200, seed=1).residuals) == NQ

    with pytest.raises(PricingError, match="at least 10"):
        plot_glm_qq(mq_pois, n_max=3)
    with pytest.raises(PricingError, match="glm object"):
        plot_glm_qq("nope")
    with pytest.raises(PricingError, match="lo < hi"):
        plot_glm_qq(mq_pois, y_range=(3, 1))


def test_a_family_without_a_closed_form_cdf_falls_back_and_says_so(dqs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m_ig = fit_glm("y ~ x", dqs, family="inverse.gaussian", link="log")
    with pytest.warns(PricingWarning, match="deviance residuals are shown"):
        p = plot_glm_qq(m_ig)
    assert p.type == "deviance"
    assert isinstance(p, go.Figure)


def test_a_quasipoisson_is_drawn_but_the_caveat_is_raised(dq):
    m_qp = fit_glm("AantalClaims ~ x + offset(log(Exposure))", dq,
                   family="quasipoisson")
    with pytest.warns(PricingWarning, match="no likelihood"):
        plot_glm_qq(m_qp, seed=1)


def test_plot_glm_influence_ranks_rows_and_returns_them(mq_pois):
    p = plot_glm_influence(mq_pois, n_label=5)
    assert isinstance(p, go.Figure)
    inf = p.influential
    assert isinstance(inf, pd.DataFrame)
    assert len(inf) == 5
    assert {"Row", "CooksD", "Leverage", "StdResid", "Actual",
            "Fitted"} <= set(inf.columns)
    # sorted, strongest first, and they really are the strongest
    assert np.all(np.diff(inf.CooksD) <= 0)
    np.testing.assert_allclose(
        inf.CooksD, np.sort(mq_pois.cooks_distance())[::-1][:5], rtol=1e-10)
    # the model's own variables are carried along for context
    assert "x" in inf.columns
    assert p.pricing_config["toImageButtonOptions"]["filename"] == "influence"
    assert [t.name for t in p.data] == ["Rows", "Top 5"]


def test_an_extreme_predictor_value_tops_the_ranking_and_masks_itself(dq):
    rng = np.random.default_rng(5)
    d2 = dq.copy()
    d2["g"] = np.round(rng.normal(1350, 200, NQ), -1)
    d2.loc[0, "g"] = 20000                  # a keying error in a predictor
    d2.loc[0, "AantalClaims"] = 3
    m2 = fit_glm("AantalClaims ~ x + g + offset(log(Exposure))", d2,
                 family="poisson")
    inf = plot_glm_influence(m2, n_label=3).influential

    assert inf.Row.iloc[0] == m2.data.index[0]
    assert inf.CooksD.iloc[0] > inf.CooksD.iloc[1]
    assert inf.Leverage.iloc[0] > 0.5

    # and it hides from a residual check, because at that leverage the fit
    # bends to meet it - which is why the plot needs both axes
    rp = np.abs(m2.residuals("pearson"))
    assert stats.rankdata(-rp)[0] > 100


def test_a_low_exposure_keying_error_is_invisible_to_cooks_distance(dq):
    # The IRLS weight of a Poisson row is its fitted mean, so a sliver of
    # exposure carries no leverage whatever its response. This is a
    # characterisation test: it pins down a real blind spot rather than a
    # behaviour worth relying on.
    d2 = dq.copy()
    d2.loc[0, "Exposure"] = 0.001
    d2.loc[0, "AantalClaims"] = 2
    m2 = fit_glm("AantalClaims ~ x + offset(log(Exposure))", d2,
                 family="poisson")

    rp = np.abs(m2.residuals("pearson"))
    assert stats.rankdata(-rp)[0] == 1          # worst-fitted row there is
    assert m2.hatvalues()[0] < 1e-4             # yet no leverage at all

    # so a residual-based check is the complement that catches it
    r = _quantile_residuals(m2, "t", np.random.default_rng(1))
    assert abs(r[0]) > 3


def test_plot_glm_influence_caps_the_row_count_with_a_message(mq_pois):
    with pytest.warns(PricingMessage, match="max_rows") as rec:
        p = plot_glm_influence(mq_pois, max_rows=1000)
    assert "a sample of 1,000 of 6,000 rows" in str(rec[0].message)
    assert isinstance(p, go.Figure)
    assert "(1,000 rows)" in p.layout.title.text
    with pytest.raises(PricingError, match="at least 1"):
        plot_glm_influence(mq_pois, n_label=0)
    with pytest.raises(PricingError, match="glm object"):
        plot_glm_influence("nope")


def test_a_glm_nb_model_gets_quantile_residuals_in_the_qq_plot():
    # test-review-findings.R: "a glm.nb model works in make_pdp and the A/E
    # plot"; the Q-Q part of it
    from test_diagnostics import rv_book
    rv = rv_book()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                    rv, family="negbin")
    q = plot_glm_qq(m, seed=1)
    assert q.type == "quantile"
    # and on a correctly specified model they are standard normal
    assert _ks_p(q.residuals) > 0.001
