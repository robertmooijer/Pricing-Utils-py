"""Parity of fit_glm() with R's glm() and MASS::glm.nb().

[PY-08] The rest of the package is only as close to R as the fit underneath
it, so the fit is compared on its own: the same data and formulas, fitted
in both languages. R fits with glm.control(epsilon = 1e-14), so the
comparison measures the fit and not R's convergence tolerance.

Five models, one R run:

* Poisson with cut(), ns(), factor(), an interaction and two offsets;
  predictions on new data that include an age outside the cut() breaks
  (NA, as in R) and outside the ns() boundary knots (linear extrapolation).
* quasi-Poisson on the same formula: the dispersion and its standard errors.
* Gamma with a log link and prior weights (a severity model).
* Negative binomial against MASS::glm.nb: theta and the fit.
* Gaussian with an identity link and an offset.
* Inverse Gaussian with a log link and prior weights.

Coefficient names differ (patsy against R), so the standard errors are
compared sorted; everything else is compared value by value.

Measured: fitted values and predictions agree to about 2e-14, deviance and
AIC to 1e-15, standard errors to 8e-12 and theta to 5e-14, except the
Gamma and inverse Gaussian severity fits, whose fitted values agree to
about 2e-8 while their deviance agrees to 1e-15: the optimum is flat, and
the two IRLS loops stop on marginally different points of it. The
tolerances below sit just above that."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo.model import fit_glm

from ._r import as_float, requires_r, run_r

pytestmark = [pytest.mark.parity, requires_r]

RTOL = 1e-10
RTOL_FIT = {"gamma": 1e-7, "ig": 1e-7}      # see the module docstring

F_FREQ = ("AantalClaims ~ cut(LEEFTIJD, c(17, 30, 50, 85)) + ns(GEWICHT, 3)"
          " + factor(BOEKJAAR) + REGIO + REGIO:LEEFTIJD"
          " + offset(log(Exposure)) + offset(log(BM))")
F_SEV = "AvgLoss ~ log(GEWICHT) + REGIO + factor(BOEKJAAR)"
F_NB = "AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))"
F_GAUSS = "y ~ LEEFTIJD + REGIO + offset(Exposure)"
F_IG = "AvgLoss ~ LEEFTIJD + REGIO"

R_CODE = """
library(splines)
ctl <- glm.control(epsilon = 1e-14, maxit = 100)
fit_out <- function(m, nd) list(
  fitted = unname(fitted(m)), dev = deviance(m), null = m$null.deviance,
  aic = m$aic, dfr = m$df.residual, rank = m$rank,
  disp = summary(m)$dispersion,
  se = sort(unname(sqrt(diag(vcov(m))))),
  pred = unname(predict(m, nd, type = "response")))
out$pois <- fit_out(glm(%s, family = poisson(), data = d, control = ctl), nd)
out$quasi <- fit_out(glm(%s, family = quasipoisson(), data = d, control = ctl), nd)
out$gamma <- fit_out(glm(%s, family = Gamma(link = "log"), data = s,
                         weights = AantalClaims, control = ctl), nd)
mnb <- MASS::glm.nb(%s, data = d, control = ctl)
out$nb <- c(fit_out(mnb, nd), theta = mnb$theta)
out$gauss <- fit_out(glm(%s, family = gaussian(), data = d, control = ctl), nd)
out$ig <- fit_out(glm(%s, family = inverse.gaussian(link = "log"), data = s,
                      weights = AantalClaims, control = ctl), nd)
""" % (F_FREQ, F_FREQ, F_SEV, F_NB, F_GAUSS, F_IG)


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(2031)
    n = 10000
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "BOEKJAAR": rng.integers(2021, 2025, n),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "BM": rng.choice([.6, .8, 1.0, 1.2], n),
        "Exposure": np.round(rng.uniform(.1, 1, n), 3)})
    eta = (np.log(.08) + .5 * np.exp(-(d.LEEFTIJD - 18) / 10)
           + .0003 * (d.GEWICHT - 1500)
           + d.REGIO.map({"Noord": 0, "Zuid": .2, "West": .35}).astype(float))
    mu = d.Exposure * d.BM * np.exp(eta)
    # overdispersed counts, so quasi-Poisson and the negative binomial have
    # something to estimate
    d["AantalClaims"] = rng.poisson(mu * rng.gamma(2.0, 0.5, n))
    d["SCHADELAST"] = np.where(
        d.AantalClaims > 0,
        d.AantalClaims * rng.gamma(2.0, 1200 * (d.GEWICHT / 1500) ** .5),
        0.0)
    d["y"] = (5 + .02 * d.LEEFTIJD + (d.REGIO == "Zuid").to_numpy(dtype=float)
              + d.Exposure + rng.normal(0, .5, n))
    s = d[d.AantalClaims > 0].copy()
    s["AvgLoss"] = s.SCHADELAST / s.AantalClaims
    nd = d.iloc[:400].copy()
    nd.loc[nd.index[:5], "LEEFTIJD"] = 90.0      # outside the cut() breaks
    nd.loc[nd.index[5:10], "GEWICHT"] = 2600.0   # outside the ns() knots
    return {"d": d, "s": s, "nd": nd}


@pytest.fixture(scope="module")
def r_out(frames):
    return run_r(R_CODE, frames)


@pytest.fixture(scope="module")
def py_fits(frames):
    d, s = frames["d"], frames["s"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "pois": fit_glm(F_FREQ, d, family="poisson", tol=1e-14),
            "quasi": fit_glm(F_FREQ, d, family="quasipoisson", tol=1e-14),
            "gamma": fit_glm(F_SEV, s, family="gamma", weights="AantalClaims",
                             tol=1e-14),
            "nb": fit_glm(F_NB, d, family="negbin", tol=1e-14),
            "gauss": fit_glm(F_GAUSS, d, family="gaussian", tol=1e-14),
            "ig": fit_glm(F_IG, s, family="inverse.gaussian",
                          weights="AantalClaims", tol=1e-14)}


def _close(a, b, rtol=RTOL):
    np.testing.assert_allclose(np.asarray(a, dtype=float), as_float(b),
                               rtol=rtol, equal_nan=True)


MODELS = ["pois", "quasi", "gamma", "nb", "gauss", "ig"]


@pytest.mark.parametrize("which", MODELS)
def test_fit_glm_matches_r_on_the_fit(which, r_out, py_fits):
    r, m = r_out[which], py_fits[which]
    assert m.rank == r["rank"]
    assert m.df_residual == r["dfr"]
    _close(m.fitted, r["fitted"], RTOL_FIT.get(which, RTOL))
    _close(m.deviance, r["dev"])
    _close(m.null_deviance, r["null"])
    _close(m.dispersion_summary, r["disp"], RTOL_FIT.get(which, RTOL))
    _close(np.sort(np.sqrt(np.diag(m.vcov.to_numpy()))), r["se"],
           RTOL_FIT.get(which, 1e-9))
    if r["aic"] is None:           # quasi-Poisson has no likelihood
        assert np.isnan(m.aic)
    else:
        _close(m.aic, r["aic"])


@pytest.mark.parametrize("which", MODELS)
def test_fit_glm_predicts_new_data_as_r_does(which, r_out, py_fits, frames):
    # NA where cut() has no band, linear extrapolation past the ns() knots
    pred = py_fits[which].predict(frames["nd"])
    _close(pred, r_out[which]["pred"], RTOL_FIT.get(which, RTOL))
    if which in ("pois", "quasi"):
        assert np.isnan(pred[:5]).all()
        assert np.isfinite(pred[5:]).all()


def test_the_negative_binomial_theta_matches_glm_nb(r_out, py_fits):
    assert py_fits["nb"].theta == pytest.approx(r_out["nb"]["theta"], rel=1e-10)
