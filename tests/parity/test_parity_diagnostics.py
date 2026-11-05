"""Parity of diagnostics.py, qq_influence.py and interactions.py with the R
package pricingtoolsRmO.

The data are built here, the same formula is fitted in both languages, and
the R function runs on its own glm. R fits with glm.control(epsilon =
1e-15): with its default of 1e-8 (and even 1e-12) the Gamma log-link fit,
whose IRLS converges only linearly, stopped with fitted values 3e-8 away
from the optimum, which the comparison would then measure. Everything
deterministic is compared at rtol 1e-6. The one looser tolerance is the
centred deviance residual of a Gamma model: R integrates its mean with
integrate()'s default rel.tol of 1.2e-4 (an error of up to ~1e-7 here),
scipy's quad to 1.5e-8. Simulated quantities (the Z and P of the
simulation null, randomised quantile residuals) differ by design and are
left out."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo.diagnostics import (glm_collinearity, glm_diagnostics,
                                          plot_glm_residuals)
from pricingtools_rmo.interactions import (detect_interactions,
                                           plot_residual_heatmap)
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.qq_influence import (_quantile_residuals, plot_glm_influence,
                                           plot_glm_qq)

from ._r import as_float, requires_r, run_r

pytestmark = [pytest.mark.parity, requires_r]

RTOL = 1e-6

F_FREQ = ("AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR)"
          " + offset(log(Exposure))")
F_INT = ("AantalClaims ~ ns(LEEFTIJD, 4) + GEWICHT * REGIO + factor(BOEKJAAR)"
         " + offset(log(Exposure))")
F_COL = ("AantalClaims ~ LEEFTIJD + GEWICHT + GEWICHT_PROXY + REGIO"
         " + offset(log(Exposure))")
F_SEV = "AvgLoss ~ LEEFTIJD + REGIO"

R_FIT = """
ctl <- glm.control(epsilon = 1e-15, maxit = 100)
mf <- glm(%s, family = poisson(), data = d, control = ctl)
ms <- glm(%s, family = Gamma(link = "log"), data = s, weights = AantalClaims,
          control = ctl)
""" % (F_FREQ, F_SEV)


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fun(*args, **kw)


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(2026)
    n = 8000
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Rand", "Dorp"], n,
                                           p=[.4, .35, .25]),
                                categories=["Dorp", "Rand", "Stad"]),
        "GEWICHT": np.clip(np.round(rng.normal(1350, 250, n)), 800, 2400),
        "BOEKJAAR": rng.integers(2020, 2024, n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    d["GEWICHT_PROXY"] = d.GEWICHT + np.round(rng.normal(0, 25, n))
    jong = np.maximum(0, 30 - d.LEEFTIJD)
    eta = (-1.8 + 0.01 * (d.LEEFTIJD - 40) + 0.0003 * (d.GEWICHT - 1350)
           + d.REGIO.map({"Stad": .25, "Rand": .1, "Dorp": 0}).astype(float)
           + 0.05 * jong * (d.REGIO == "Stad"))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta))
    s = d[d.AantalClaims > 0].copy()
    s["AvgLoss"] = np.round(rng.gamma(2, 750 * np.exp(0.01 * (s.LEEFTIJD - 40))), 2)
    return d.reset_index(drop=True), s.reset_index(drop=True)


@pytest.fixture(scope="module")
def models(frames):
    d, s = frames
    mf = fit_glm(F_FREQ, d, family="poisson")
    ms = fit_glm(F_SEV, s, family="gamma", link="log", weights="AantalClaims")
    return mf, ms


def _close(a, b, rtol=RTOL, atol=0.0):
    np.testing.assert_allclose(np.asarray(a, dtype=float), as_float(b),
                               rtol=rtol, atol=atol)


def _key(term):
    """A term label without the order of its variables, so patsy's
    'GEWICHT:REGIO' meets R's whichever way round it writes it."""
    return frozenset(term.split(":"))


def test_glm_diagnostics_table(frames, models):
    d, s = frames
    mf, ms = models
    py = glm_diagnostics(mf, ms)
    r = run_r(R_FIT + "out$t <- glm_diagnostics(mf, ms)", {"d": d, "s": s})["t"]
    assert list(py.Model) == r["Model"]
    assert r["Family"] == list(py.Family)    # R's names: "Gamma"
    assert list(py.Link) == r["Link"]
    assert list(py.N) == r["N"]
    assert list(py.DFResidual) == r["DFResidual"]
    for col in ("Deviance", "AIC", "Dispersion", "DevianceExplained"):
        _close(py[col], r[col])


def test_glm_collinearity_table(frames):
    d, _ = frames
    code = """
ctl <- glm.control(epsilon = 1e-15, maxit = 100)
m1 <- glm(%s, family = poisson(), data = d, control = ctl)
m2 <- glm(%s, family = poisson(), data = d, control = ctl)
out$a <- suppressWarnings(glm_collinearity(m1))
out$b <- suppressWarnings(glm_collinearity(m2))
""" % (F_INT, F_COL)
    r = run_r(code, {"d": d})
    for key, f in (("a", F_INT), ("b", F_COL)):
        py = _quiet(glm_collinearity, fit_glm(f, d, family="poisson"))
        rt = pd.DataFrame(r[key])
        assert len(py) == len(rt)
        rt.index = [_key(t) for t in rt.Term]
        rt = rt.loc[[_key(t) for t in py.Term]]
        assert list(py.DF) == list(rt.DF)
        _close(py.GVIF, rt.GVIF.tolist())
        _close(py.GVIF_scaled, rt.GVIF_scaled.tolist())
        assert list(py.InInteraction) == list(rt.InInteraction)
        assert [None if pd.isna(v) else bool(v) for v in py.Flag] == \
            list(rt.Flag)
    # the case with an interaction carries NA flags, the other a TRUE one
    assert any(v is None for v in r["a"]["Flag"])
    assert any(v is True for v in r["b"]["Flag"])


R_RES = """
tr <- function(p) {
  b <- plotly::plotly_build(p)
  dd <- b$x$data
  k <- length(dd)
  list(n = k, x = unlist(dd[[k]]$x), y = unlist(dd[[k]]$y),
       band = if (k == 3) unlist(dd[[1]]$y) else unlist(dd[[k]]$error_y$array),
       ytitle = unlist(b$x$layout$yaxis$title)[1])
}
out$fp <- tr(plot_glm_residuals(mf))
out$fd <- tr(plot_glm_residuals(mf, residual_type = "deviance"))
out$lp <- tr(plot_glm_residuals(mf, "LEEFTIJD", n_bins = 20))
out$ld <- tr(plot_glm_residuals(mf, "LEEFTIJD", n_bins = 20,
                                residual_type = "deviance"))
out$rd <- tr(plot_glm_residuals(mf, "REGIO", residual_type = "deviance"))
out$sd <- tr(plot_glm_residuals(ms, residual_type = "deviance"))
"""


def test_plot_glm_residuals_bins_and_bands(frames, models):
    d, s = frames
    mf, ms = models
    r = run_r(R_FIT + R_RES, {"d": d, "s": s})
    cases = {"fp": (mf, None, 50, "pearson"), "fd": (mf, None, 50, "deviance"),
             "lp": (mf, "LEEFTIJD", 20, "pearson"),
             "ld": (mf, "LEEFTIJD", 20, "deviance"),
             "rd": (mf, "REGIO", 50, "deviance"),
             "sd": (ms, None, 50, "deviance")}
    for key, (m, pred, nb, rt) in cases.items():
        p = plot_glm_residuals(m, pred, n_bins=nb, residual_type=rt)
        rr = r[key]
        assert len(p.data) == rr["n"], key
        last = p.data[-1]
        if pred == "REGIO":
            assert list(last.x) == rr["x"]
            band = last.error_y.array
        else:
            _close(last.x, rr["x"])
            band = p.data[0].y
        # Gamma deviance moments: limited by R's integrate() tolerance
        _close(last.y, rr["y"], atol=2e-7 if key == "sd" else 1e-10)
        _close(band, rr["band"])
        assert p.layout.yaxis.title.text == rr["ytitle"], key


def test_residual_heatmap_cells(frames, models):
    d, s = frames
    mf, _ = models
    code = R_FIT + """
p <- plot_residual_heatmap(mf, "LEEFTIJD", "REGIO", n_bins = 8)
out$cells <- attr(p, "cells")
out$text <- plotly::plotly_build(p)$x$data[[1]]$text
p2 <- plot_residual_heatmap(mf, "GEWICHT", "BOEKJAAR", n_bins = 6,
                            min_claims = 40)
out$cells2 <- attr(p2, "cells")
"""
    r = run_r(code, {"d": d, "s": s})
    for key, args in (("cells", ("LEEFTIJD", "REGIO", 8, 30)),
                      ("cells2", ("GEWICHT", "BOEKJAAR", 6, 40))):
        p = plot_residual_heatmap(mf, args[0], args[1], n_bins=args[2],
                                  min_claims=args[3])
        py, rc = p.cells, r[key]
        assert list(py.gx.astype(str)) == rc["gx"]
        assert list(py.gy.astype(str)) == rc["gy"]
        for col in ("Actual", "Expected", "Exposure", "Claims", "AE"):
            _close(py[col], rc[col])
        assert list(py.IsThin) == rc["IsThin"]
        if key == "cells":
            # the tooltips, R's number formatting included
            assert [list(row) for row in p.data[0].text] == r["text"]


def test_detect_interactions_deterministic_columns(frames, models):
    d, s = frames
    mf, ms = models
    vars_ = ["LEEFTIJD", "REGIO", "GEWICHT", "BOEKJAAR"]
    code = R_FIT + """
v <- c("LEEFTIJD", "REGIO", "GEWICHT", "BOEKJAAR")
out$f <- detect_interactions(mf, vars = v, n_bins = 6, n_sim = 20, seed = 1)
out$f0 <- detect_interactions(mf, n_bins = 6, n_sim = 5, seed = 1)
out$s <- suppressMessages(detect_interactions(ms, vars = v, n_bins = 4))
"""
    r = run_r(code, {"d": d, "s": s})
    py_f = detect_interactions(mf, vars=vars_, n_bins=6, n_sim=20, seed=1)
    py_f0 = detect_interactions(mf, n_bins=6, n_sim=5, seed=1)
    py_s = _quiet(detect_interactions, ms, vars=vars_, n_bins=4)
    for py, key, chisq in ((py_f, "f", False), (py_f0, "f0", False),
                           (py_s, "s", True)):
        rt = pd.DataFrame(r[key])
        # With vars given the pairs are oriented alike. With the default
        # vars the model's own variables come in patsy's term order, which
        # puts a categorical term before a numeric one (REGIO before
        # LEEFTIJD) where R keeps the formula's order; a pair can then be
        # (REGIO, LEEFTIJD) here and (LEEFTIJD, REGIO) in R, so pairs are
        # matched without their order.
        rt.index = [frozenset(p) for p in zip(rt.VarX, rt.VarY)]
        pk = [frozenset(p) for p in zip(py.VarX, py.VarY)]
        assert set(rt.index) == set(pk), key
        if key != "f0":
            assert set(zip(rt.VarX, rt.VarY)) == set(zip(py.VarX, py.VarY))
        rt = rt.loc[pk]
        assert list(py.Cells) == list(rt.Cells)
        assert list(py.DF) == list(rt.DF)
        assert list(py.Method) == list(rt.Method)
        cols = ["Claims", "Deviance", "Dispersion", "MaxAE", "MaxAE_Claims",
                "MaxAE_ExposureShare"]
        if chisq:                       # deterministic there
            cols += ["Z", "P"]
        for col in cols:
            _close(py[col], rt[col].tolist())


def test_plot_glm_influence_top_rows(frames, models):
    d, s = frames
    mf, _ = models
    code = R_FIT + """
inf <- attr(plot_glm_influence(mf, n_label = 10, max_rows = NULL),
            "influential")
out$inf <- inf[, c("Row", "CooksD", "Leverage", "StdResid", "Actual",
                   "Fitted", "LEEFTIJD", "REGIO", "BOEKJAAR")]
"""
    r = run_r(code, {"d": d, "s": s})["inf"]
    py = plot_glm_influence(mf, n_label=10, max_rows=None).influential
    # R's Row is the row name, 1-based; here it is the index label
    assert [int(x) - 1 for x in r["Row"]] == list(py.Row)
    for col in ("CooksD", "Leverage", "StdResid", "Actual", "Fitted",
                "LEEFTIJD", "BOEKJAAR"):
        _close(py[col], r[col])
    assert list(py.REGIO.astype(str)) == r["REGIO"]


def test_quantile_residuals_and_qq_plot_of_a_gamma_model(frames, models):
    d, s = frames
    _, ms = models
    code = R_FIT + """
out$qr <- pricingtoolsRmO:::.quantile_residuals(ms, "t")
b <- plotly::plotly_build(plot_glm_qq(ms, n_max = 200))$x$data
out$names <- vapply(b, function(t) t$name, "")
out$xy <- lapply(b, function(t) list(x = unlist(t$x), y = unlist(t$y)))
"""
    r = run_r(code, {"d": d, "s": s})
    _close(_quantile_residuals(ms, "t"), r["qr"], atol=1e-9)
    p = plot_glm_qq(ms, n_max=200)
    assert [t.name for t in p.data] == r["names"]
    for t, rt in zip(p.data, r["xy"]):
        _close(t.x, rt["x"], atol=1e-9)
        _close(t.y, rt["y"], atol=1e-9)
