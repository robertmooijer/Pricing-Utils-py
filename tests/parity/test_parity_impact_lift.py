"""Parity of premium_impact(), model_lift() and double_lift() with the R
package. Everything compared is deterministic: the bootstraps are switched
off (n_boot = 0), since the random streams differ."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo.impact import premium_impact
from pricingtools_rmo.lift import double_lift, model_lift
from pricingtools_rmo.model import fit_glm

from ._r import as_float, requires_r, run_r

RTOL = 1e-6

F_NEW = ("AantalClaims ~ LEEFTIJD + REGIO + factor(BOEKJAAR)"
         " + offset(log(Exposure)) + offset(log(BM))")
F_OLD = "AantalClaims ~ REGIO + offset(log(Exposure)) + offset(log(BM))"
F_SEV = "AvgLoss ~ REGIO + LEEFTIJD"


def _data(n=6000, seed=23):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        # continuous ages: with whole years many policies share a
        # prediction, and the order of exact ties is set by last-bit noise
        # of the matrix product, which differs between R and numpy; the bins
        # of the lifts would then differ for no reason in the functions
        "LEEFTIJD": np.round(rng.uniform(18, 80, n), 6),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp", "Rand"], n,
                                           p=[.3, .5, .2]),
                                categories=["Dorp", "Rand", "Stad"]),
        "BOEKJAAR": rng.integers(2021, 2025, n),
        "BM": rng.choice([1, .85, .7, .55], n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    eta = (-2 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD)
           + 0.3 * (d.REGIO == "Stad") + 0.05 * (d.BOEKJAAR - 2021))
    d["AantalClaims"] = rng.poisson(d.Exposure * d.BM * np.exp(eta))
    d["SCHADELAST"] = np.where(
        d.AantalClaims > 0,
        d.AantalClaims * rng.gamma(2, 1200 * np.exp(0.005 * (d.LEEFTIJD - 40))
                                   / 2), 0.0)
    d["PREMIE_OUD"] = d.Exposure * d.BM * 180 * np.exp(
        0.15 * (d.REGIO == "Stad") + 0.1 * (d.LEEFTIJD < 25))
    # rows to drop: no old premium, and a missing loss on a row without
    # claims (so the severity fit does not see it)
    d.loc[d.index[[3, 50, 700]], "PREMIE_OUD"] = np.nan
    d.loc[d.index[np.flatnonzero(d.AantalClaims == 0)[:2]], "SCHADELAST"] = \
        np.nan
    return d


# R's glm() stops at a relative deviance change of 1e-8, which leaves the
# Gamma coefficients about 1e-5 from the optimum; fit_glm iterates to 1e-10.
# A tighter epsilon puts both on the same optimum, so what is compared is
# the functions, not the two optimisers.
R_CODE = """
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
mf <- glm(AantalClaims ~ LEEFTIJD + REGIO + factor(BOEKJAAR) +
            offset(log(Exposure)) + offset(log(BM)), poisson(), d,
          control = ctl)
mf_old <- glm(AantalClaims ~ REGIO + offset(log(Exposure)) + offset(log(BM)),
              poisson(), d, control = ctl)
ds <- d[d$AantalClaims > 0, ]
ds$AvgLoss <- ds$SCHADELAST / ds$AantalClaims
ms <- glm(AvgLoss ~ REGIO + LEEFTIJD, Gamma("log"), ds, weights = AantalClaims,
          control = ctl)

imp <- suppressWarnings(premium_impact(d, mf, ms,
  old_premium_col = "PREMIE_OUD", by = c("REGIO", "BOEKJAAR"),
  at = list(BOEKJAAR = 2024), spotlight = REGIO == "Stad" & LEEFTIJD < 30))
b <- plotly::plotly_build(imp$plot)$x
out$imp <- list(policy = imp$policy, by_level = imp$by_level,
                stats = imp$stats, summary = imp$summary,
                spot = imp$spotlight[setdiff(names(imp$spotlight), "summary")],
                spot_summary = imp$spotlight$summary,
                inc = imp$largest_increases$Row,
                dec = imp$largest_decreases$Row,
                hist_x = b$data[[1]]$x, hist_y = b$data[[1]]$y,
                hist_w = b$data[[1]]$width, spot_y = b$data[[2]]$y,
                title = b$layout$title$text)

imp2 <- suppressWarnings(premium_impact(d, mf, ms, mf_old, ms,
  rebase = FALSE, by = "REGIO", keep_offsets = FALSE))
out$imp2 <- list(policy = imp2$policy, by_level = imp2$by_level,
                 stats = imp2$stats, summary = imp2$summary)

lf <- suppressWarnings(model_lift(mf, ms, data = d, n_boot = 0))
bl <- plotly::plotly_build(lf$plot_lorenz)$x
out$lf <- list(table = lf$table, gini = lf$gini, stats = lf$stats,
               title = plotly::plotly_build(lf$plot)$x$layout$title$text,
               lor_x = bl$data[[2]]$x, lor_y = bl$data[[2]]$y)
lf2 <- suppressWarnings(model_lift(mf, n_boot = 0, n_bins = 7,
                                   actual_col = "AantalClaims"))
out$lf2 <- list(table = lf2$table, gini = lf2$gini, stats = lf2$stats)

dbl <- suppressWarnings(double_lift(d, mf, ms, model_freq_old = mf_old,
                                    model_sev_old = ms, n_boot = 0))
out$dbl <- list(table = dbl$table, stats = dbl$stats,
                title = plotly::plotly_build(dbl$plot)$x$layout$title$text)
dbl2 <- suppressWarnings(double_lift(d, mf, ms, old_premium_col = "PREMIE_OUD",
                                     rebase = FALSE, n_boot = 0, n_bins = 8))
out$dbl2 <- list(table = dbl2$table, stats = dbl2$stats)
"""


@pytest.fixture(scope="module")
def both():
    d = _data()
    mf = fit_glm(F_NEW, d, family="poisson")
    mf_old = fit_glm(F_OLD, d, family="poisson")
    ds = d[d.AantalClaims > 0].copy()
    ds["AvgLoss"] = ds.SCHADELAST / ds.AantalClaims
    ms = fit_glm(F_SEV, ds, family="gamma", weights="AantalClaims")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        py = dict(
            imp=premium_impact(d, mf, ms, old_premium_col="PREMIE_OUD",
                               by=["REGIO", "BOEKJAAR"],
                               at={"BOEKJAAR": 2024},
                               spotlight="REGIO == 'Stad' & LEEFTIJD < 30"),
            imp2=premium_impact(d, mf, ms, mf_old, ms, rebase=False,
                                by="REGIO", keep_offsets=False),
            lf=model_lift(mf, ms, data=d, n_boot=0),
            lf2=model_lift(mf, n_boot=0, n_bins=7,
                           actual_col="AantalClaims"),
            dbl=double_lift(d, mf, ms, model_freq_old=mf_old,
                            model_sev_old=ms, n_boot=0),
            dbl2=double_lift(d, mf, ms, old_premium_col="PREMIE_OUD",
                             rebase=False, n_boot=0, n_bins=8))
    return py, run_r(R_CODE, {"d": d})


# Percentage changes and contributions sit around 0, where a relative
# tolerance means nothing: they are compared to 1e-5 percentage points
PCT = {"ChangePct", "MeanChangePct", "TotalChangePct", "Contribution",
       "rate_change", "rate_level_change", "mad_diff", "mean_change",
       "median_change", "p5_change", "p95_change", "contribution"}


def _close(a, b, rtol=RTOL, atol=1e-12):
    np.testing.assert_allclose(as_float(a), as_float(b), rtol=rtol,
                               atol=atol)


def _frame(py, r, num, txt=()):
    assert list(py.columns) == list(r), (list(py.columns), list(r))
    for c in num:
        _close(py[c], r[c], atol=1e-5 if c in PCT else 1e-12)
    for c in txt:
        assert [None if v is None else str(v) for v in py[c]] == \
            [None if v is None else str(v) for v in r[c]], c


def _vec(x):
    return [] if x is None else list(np.atleast_1d(x))


@pytest.mark.parity
@requires_r
def test_premium_impact_parity(both):
    py, r = both
    p, q = py["imp"], r["imp"]
    _frame(p.policy, q["policy"],
           ["Exposure", "OldRate", "NewRate", "NewRateRebased", "ChangePct"],
           ["REGIO", "BOEKJAAR"])
    assert list(p.policy.Row + 1) == q["policy"]["Row"]
    _frame(p.by_level, q["by_level"],
           ["Exposure", "MeanChangePct", "OldPremium", "NewPremium",
            "ExposureShare", "PremiumShare", "TotalChangePct",
            "Contribution"], ["Variable", "Level"])
    st = q["stats"]
    for k in ("old_total", "new_total", "rate_change", "share_gt_10",
              "share_gt_25"):
        _close(p.stats[k], st[k])
    _close(list(p.stats.quantiles.values()), st["quantiles"], atol=1e-5)
    assert p.stats.n_rows == st["n_rows"] and p.stats.n_dropped == 3
    assert p.stats.n_dropped == st["n_dropped"]
    assert p.stats.rebase is st["rebase"]
    assert p.stats.offsets_kept == _vec(st["offsets_kept"]) == ["BM"]
    assert list(p.summary.Metric) == q["summary"]["Metric"]
    assert list(p.summary.Value) == q["summary"]["Value"]
    for k, v in q["spot"].items():
        _close(p.spotlight[k], v, atol=1e-5 if k in PCT else 1e-12)
    assert list(p.spotlight.summary.Value) == q["spot_summary"]["Value"]
    assert list(p.largest_increases.Row + 1) == q["inc"]
    assert list(p.largest_decreases.Row + 1) == q["dec"]
    # the pre-binned histogram, bar for bar
    _close(p.plot.data[0].x, q["hist_x"])
    _close(p.plot.data[0].y, q["hist_y"])
    _close(p.plot.data[0].width, q["hist_w"])
    _close(p.plot.data[1].y, q["spot_y"])
    assert p.plot.layout.title.text == q["title"]

    p, q = py["imp2"], r["imp2"]
    _frame(p.policy, q["policy"],
           ["Exposure", "OldRate", "NewRate", "NewRateRebased", "ChangePct"],
           ["REGIO"])
    _frame(p.by_level, q["by_level"],
           ["Exposure", "MeanChangePct", "OldPremium", "NewPremium",
            "ExposureShare", "PremiumShare", "TotalChangePct",
            "Contribution"], ["Variable", "Level"])
    _close(p.stats.rate_change, q["stats"]["rate_change"])
    assert p.stats.offsets_kept == _vec(q["stats"]["offsets_kept"]) == []
    assert list(p.summary.Value) == q["summary"]["Value"]


@pytest.mark.parity
@requires_r
def test_model_lift_parity(both):
    py, r = both
    for key in ("lf", "lf2"):
        p, q = py[key], r[key]
        _frame(p.table, q["table"], ["Exposure", "ExposureShare",
                                     "ActualRate", "PredictedRate", "AE"])
        assert list(p.table.Bin) == q["table"]["Bin"]
        _close(p.gini, q["gini"])
        st = q["stats"]
        for k in ("total_actual", "total_predicted"):
            _close(p.stats[k], st[k])
        for k in ("n_rows", "n_dropped", "in_sample", "n_bins"):
            assert p.stats[k] == st[k], k
        assert np.all(np.isnan(p.gini_ci)) and st["gini_ci"] == [None, None]
        assert p.stats.offsets_kept == _vec(st["offsets_kept"])
    assert py["lf"].stats.n_dropped == 2
    assert py["lf"].plot.layout.title.text == r["lf"]["title"]
    _close(py["lf"].plot_lorenz.data[1].x, r["lf"]["lor_x"])
    _close(py["lf"].plot_lorenz.data[1].y, r["lf"]["lor_y"])


@pytest.mark.parity
@requires_r
def test_double_lift_parity(both):
    py, r = both
    for key in ("dbl", "dbl2"):
        p, q = py[key], r[key]
        _frame(p.table, q["table"], ["Exposure", "RateRatio", "ActualRate",
                                     "AE_New", "AE_Old"])
        assert list(p.table.Bin) == q["table"]["Bin"]
        st = q["stats"]
        for k in ("mad_new", "mad_old", "mad_diff", "rate_level_change"):
            _close(p.stats[k], st[k])
        for k in ("winner", "n_rows", "n_dropped", "weight_col", "rebase",
                  "n_bins"):
            assert p.stats[k] == st[k], k
        assert st["mad_diff_ci"] == [None, None]
        assert p.stats.offsets_kept == _vec(st["offsets_kept"])
    assert py["dbl"].plot.layout.title.text == r["dbl"]["title"]
