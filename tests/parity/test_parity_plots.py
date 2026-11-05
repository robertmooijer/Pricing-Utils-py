"""Parity of agg_all(), plot_glm_predictor() and make_pdp() with the R
package: the aggregate table, and the values behind the plotted traces
(read in R from plotly::plotly_build(p)$x$data)."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo.ae import plot_glm_predictor
from pricingtools_rmo.agg import agg_all
from pricingtools_rmo.colors import ta_blue
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.pdp import make_pdp
from pricingtools_rmo.plot_oneway import make_plot

from ._r import as_float, requires_r, run_r

RTOL = 1e-6

R_HELPERS = """
suppressPackageStartupMessages(library(splines))
# fit to convergence, so the comparison is of the plots and not of
# where glm() stops iterating (its default epsilon is 1e-8)
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
traces <- function(p) {
  b <- plotly::plotly_build(p)$x$data
  lapply(b, function(t) list(name = t$name,
                             x = if (is.factor(t$x)) as.character(t$x) else t$x,
                             y = t$y, e = t$error_y$array))
}
"""


def _book(n=8000, seed=11, bump=0.0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "BOEKJAAR": rng.integers(2020, 2024, n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3),
        "BM": rng.choice([1, .85, .7, .55], n)})
    lin = (-2.1 + 0.015 * (d.LEEFTIJD - 40)
           + np.where(d.REGIO == "Zuid", .25 + bump, np.where(d.REGIO == "West", .4, 0)))
    d["AantalClaims"] = rng.poisson(d.Exposure * d.BM * np.exp(lin))
    sev = rng.gamma(2, (1500 * np.exp(0.01 * (d.LEEFTIJD - 40))) / 2)
    d["SCHADELAST"] = np.where(d.AantalClaims > 0, sev * d.AantalClaims, 0.0)
    return d


def _sev(d):
    s = d[d.AantalClaims > 0].copy()
    s["AvgLoss"] = s.SCHADELAST / s.AantalClaims
    return s


def _cmp_traces(py_fig, r_traces, label):
    """Every trace of the Python figure against R's, by name: x exact (or
    to rtol for bin positions), y and error bars to RTOL."""
    py = {t.name: t for t in py_fig.data}
    rn = [t["name"] for t in r_traces]
    assert list(py) == rn, label
    worst = 0.0
    for rt in r_traces:
        t = py[rt["name"]]
        rx = rt["x"] if isinstance(rt["x"], list) else [rt["x"]]
        if rx and isinstance(rx[0], str):
            assert list(t.x) == rx, (label, rt["name"])
        else:
            np.testing.assert_allclose(np.asarray(t.x, dtype=float),
                                       as_float(rx), rtol=RTOL,
                                       err_msg=label + " x " + rt["name"])
        for py_v, r_v, what in ((t.y, rt["y"], "y"),
                                (None if t.error_y is None else t.error_y.array,
                                 rt.get("e"), "e")):
            if r_v is None:
                assert py_v is None, (label, rt["name"], what)
                continue
            a, b = np.asarray(py_v, dtype=float), as_float(r_v)
            np.testing.assert_allclose(a, b, rtol=RTOL, equal_nan=True,
                                       err_msg=label + " " + what + " " + rt["name"])
            ok = ~np.isnan(b) & (b != 0)
            if ok.any():
                worst = max(worst, float(np.max(np.abs(a[ok] / b[ok] - 1))))
    return worst


@pytest.mark.parity
@requires_r
def test_agg_all_parity():
    d = _book()
    d.loc[d.index[:7], "SCHADELAST"] = np.nan
    out = run_r(R_HELPERS + """
a1 <- suppressWarnings(agg_all(d, "REGIO", by_year = FALSE))
a2 <- suppressWarnings(agg_all(d, "LEEFTIJD", by_year = TRUE))
out$a1 <- a1; out$a2 <- a2
out$a2_year <- as.character(a2$Year); out$a2_levels <- levels(a2$Year)
out$a1_regio <- as.character(a1$REGIO)
p <- suppressWarnings(make_plot(d, "REGIO", "Frequency", ta_blue, "F",
                                display = "color", by_year = TRUE))
out$plot <- traces(p)
""", {"d": d})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a1 = agg_all(d, "REGIO", by_year=False)
        a2 = agg_all(d, "LEEFTIJD", by_year=True)
        p = make_plot(d, "REGIO", "Frequency", ta_blue, "F", display="color",
                      by_year=True)
    # the groups in the same (first appearance) order, the same labels
    assert list(a1.REGIO.astype(str)) == out["a1_regio"]
    np.testing.assert_array_equal(a2.LEEFTIJD.to_numpy(), as_float(out["a2"]["LEEFTIJD"]))
    assert list(a2.Year.astype(str)) == out["a2_year"]
    assert list(a2.Year.cat.categories) == out["a2_levels"]
    worst = 0.0
    for a, r in ((a1, out["a1"]), (a2, out["a2"])):
        for c in ["Exposure", "ClaimCount", "Loss", "Frequency", "Severity"]:
            x, y = a[c].to_numpy(dtype=float), as_float(r[c])
            np.testing.assert_allclose(x, y, rtol=RTOL, equal_nan=True, err_msg=c)
            ok = ~np.isnan(y) & (y != 0)
            worst = max(worst, float(np.max(np.abs(x[ok] / y[ok] - 1))))
    worst = max(worst, _cmp_traces(p, out["plot"], "make_plot"))
    print("agg_all / make_plot max rel diff: %.3g" % worst)


@pytest.mark.parity
@requires_r
def test_plot_glm_predictor_parity():
    d = _book()
    d2 = _book(seed=12, bump=0.3)
    s, s2 = _sev(d), _sev(d2)
    f_freq = ("AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR) + "
              "offset(log(Exposure))")
    m = fit_glm(f_freq, d, family="poisson")
    ms = fit_glm("AvgLoss ~ LEEFTIJD + REGIO", s, family="gamma",
                 weights="AantalClaims")
    calls = {
        "regio": lambda: plot_glm_predictor(m, "REGIO"),
        "age": lambda: plot_glm_predictor(m, "LEEFTIJD"),
        "age_q": lambda: plot_glm_predictor(m, "LEEFTIJD", n_bins=10),
        "age_w": lambda: plot_glm_predictor(m, "LEEFTIJD", n_bins=8,
                                            bin_type="width"),
        "year": lambda: plot_glm_predictor(m, "factor(BOEKJAAR)"),
        "sev": lambda: plot_glm_predictor(ms, "REGIO"),
        "sev_age": lambda: plot_glm_predictor(ms, "LEEFTIJD", n_bins=12),
        "new_regio": lambda: plot_glm_predictor(m, "REGIO", data=d2),
        "new_age": lambda: plot_glm_predictor(m, "LEEFTIJD", data=d2, n_bins=10),
        "new_ratio": lambda: plot_glm_predictor(m, "REGIO", data=d2, ratio=True),
        "new_sev": lambda: plot_glm_predictor(ms, "REGIO", data=s2, ratio=True),
    }
    out = run_r(R_HELPERS + """
m <- glm(AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR) +
           offset(log(Exposure)), family = poisson(), data = d,
         control = ctl)
ms <- glm(AvgLoss ~ LEEFTIJD + REGIO, family = Gamma("log"), data = s,
          weights = AantalClaims, control = ctl)
out$regio <- traces(plot_glm_predictor(m, "REGIO"))
out$age <- traces(plot_glm_predictor(m, "LEEFTIJD"))
out$age_q <- traces(plot_glm_predictor(m, "LEEFTIJD", n_bins = 10))
out$age_w <- traces(plot_glm_predictor(m, "LEEFTIJD", n_bins = 8,
                                       bin_type = "width"))
out$year <- traces(plot_glm_predictor(m, "factor(BOEKJAAR)"))
out$sev <- traces(plot_glm_predictor(ms, "REGIO"))
out$sev_age <- traces(plot_glm_predictor(ms, "LEEFTIJD", n_bins = 12))
out$new_regio <- traces(plot_glm_predictor(m, "REGIO", data = d2))
out$new_age <- traces(plot_glm_predictor(m, "LEEFTIJD", data = d2, n_bins = 10))
out$new_ratio <- traces(plot_glm_predictor(m, "REGIO", data = d2, ratio = TRUE))
out$new_sev <- traces(plot_glm_predictor(ms, "REGIO", data = s2, ratio = TRUE))
""", {"d": d, "d2": d2, "s": s, "s2": s2})
    worst = 0.0
    for k, fun in calls.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = fun()
        worst = max(worst, _cmp_traces(fig, out[k], k))
    print("plot_glm_predictor max rel diff: %.3g" % worst)


@pytest.mark.parity
@requires_r
def test_make_pdp_parity():
    d = _book()
    s = _sev(d)
    m = fit_glm("AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR) + "
                "offset(log(Exposure))", d, family="poisson")
    mb = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure)) + "
                 "offset(log(BM))", d, family="poisson")
    ms = fit_glm("AvgLoss ~ LEEFTIJD + REGIO", s, family="gamma",
                 weights="AantalClaims")
    calls = {
        "regio": lambda: make_pdp(m, d, "REGIO"),
        "age": lambda: make_pdp(m, d, "LEEFTIJD", grid_res=20),
        "year": lambda: make_pdp(m, d, "BOEKJAAR"),
        "bm_regio": lambda: make_pdp(mb, d, "REGIO"),
        "bm_age": lambda: make_pdp(mb, d, "LEEFTIJD", grid_res=100),
        "bm_regio0": lambda: make_pdp(mb, d, "REGIO", keep_offsets=False),
        "bm_age0": lambda: make_pdp(mb, d, "LEEFTIJD", keep_offsets=False),
        "sev": lambda: make_pdp(ms, d, "LEEFTIJD", metric="Severity"),
    }
    out = run_r(R_HELPERS + """
m <- glm(AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR) +
           offset(log(Exposure)), family = poisson(), data = d,
         control = ctl)
mb <- glm(AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure)) +
            offset(log(BM)), family = poisson(), data = d,
          control = ctl)
ms <- glm(AvgLoss ~ LEEFTIJD + REGIO, family = Gamma("log"), data = s,
          weights = AantalClaims, control = ctl)
out$regio <- traces(make_pdp(m, d, "REGIO"))
out$age <- traces(make_pdp(m, d, "LEEFTIJD", grid_res = 20))
out$year <- traces(make_pdp(m, d, "BOEKJAAR"))
out$bm_regio <- traces(make_pdp(mb, d, "REGIO"))
out$bm_age <- traces(make_pdp(mb, d, "LEEFTIJD", grid_res = 100))
out$bm_regio0 <- traces(make_pdp(mb, d, "REGIO", keep_offsets = FALSE))
out$bm_age0 <- traces(make_pdp(mb, d, "LEEFTIJD", keep_offsets = FALSE))
out$sev <- traces(make_pdp(ms, d, "LEEFTIJD", metric = "Severity"))
out$ylab <- plotly::plotly_build(make_pdp(mb, d, "REGIO"))$x$layout$yaxis$title
""", {"d": d, "s": s})
    worst = 0.0
    for k, fun in calls.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig = fun()
        worst = max(worst, _cmp_traces(fig, out[k], k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert make_pdp(mb, d, "REGIO").layout.yaxis.title.text == out["ylab"]
    print("make_pdp max rel diff: %.3g" % worst)


@pytest.mark.parity
@requires_r
def test_make_rating_plot_parity():
    # the same table in both, so this compares the plot and not the table
    from pricingtools_rmo.rating_plot import make_rating_plot
    from pricingtools_rmo.rating_table import make_rating_table
    d = _book()
    m = fit_glm("AantalClaims ~ LEEFTIJD * REGIO + factor(BOEKJAAR) + "
                "offset(log(Exposure))", d, family="poisson")
    s = _sev(d)
    ms = fit_glm("AvgLoss ~ LEEFTIJD + REGIO", s, family="gamma",
                 weights="AantalClaims")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1 = make_rating_table(m, ms, data=d, grid_res=10)
    t1 = t1.copy()
    t1["IsThin"] = t1["IsThin"].astype(bool)
    t1.loc[t1.index[1], "IsThin"] = True
    out = run_r(R_HELPERS + """
t1$Level <- as.character(t1$Level)
for (v in c("Group", "GroupVar")) {
  t1[[v]] <- as.character(t1[[v]]); t1[[v]][t1[[v]] == ""] <- NA }
for (v in c("REGIO", "LEEFTIJD", "BOEKJAAR", "LEEFTIJD:REGIO"))
  out[[v]] <- traces(make_rating_plot(t1, v))
""", {"t1": t1})
    worst = 0.0
    for v in ["REGIO", "LEEFTIJD", "BOEKJAAR", "LEEFTIJD:REGIO"]:
        worst = max(worst, _cmp_traces(make_rating_plot(t1, v), out[v], v))
    print("make_rating_plot max rel diff: %.3g" % worst)
