"""Parity of screen_features() with the R package.

Boosting itself cannot match (other random streams, another xgboost
version), so this compares what is deterministic - the candidates and why
some are skipped, InModel, the near-duplicate table, the split sizes, the
messages - and, qualitatively, the ranking and the verdict."""
import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from pricingtools_rmo._core import PricingMessage, PricingWarning
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.screen import screen_features

from ._r import requires_r, run_r


def _data(n=10000, seed=19):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp"], n),
                                categories=["Dorp", "Stad"]),
        "KM": np.round(np.maximum(rng.lognormal(np.log(12000), .5, n), 1000),
                       -2),
        "RUIS": np.round(rng.normal(size=n), 2),
        "CHR": rng.choice(["p", "q", "r"], n).astype(object),
        "CONST_N": 1.0,
        "Exposure": np.round(rng.uniform(0.3, 1, n), 3)})
    d["KM_PROXY"] = d.KM + np.round(rng.normal(0, 200, n))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.6 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD)
        + 0.35 * (d.REGIO == "Stad")
        + 0.6 * (np.log(d.KM) - np.log(12000))))
    d["SCHADELAST"] = d.AantalClaims * 1500.0
    return d


@pytest.mark.parity
@requires_r
def test_screen_features_parity():
    d = _data()
    m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", d,
                family="poisson")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        r = screen_features(m, seed=1, nrounds=150, nthread=2,
                            early_stopping_rounds=15, n_shap=0)
    py_msgs = [str(w.message) for w in rec if w.category is PricingMessage]
    py_warns = [str(w.message) for w in rec if w.category is PricingWarning]

    out = run_r("""
m <- glm(AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure)),
         family = poisson(), data = d)
msgs <- character(0); warns <- character(0)
r <- withCallingHandlers(
  screen_features(m, seed = 1, nrounds = 150, nthread = 2,
                  early_stopping_rounds = 15, n_shap = 0),
  message = function(c) {
    msgs <<- c(msgs, sub("\\n$", "", conditionMessage(c)))
    invokeRestart("muffleMessage") },
  warning = function(c) {
    warns <<- c(warns, conditionMessage(c))
    invokeRestart("muffleWarning") })
out$feature <- r$features$Feature
out$inmodel <- r$features$InModel
out$cor <- r$correlated
out$stats <- r$stats[c("objective", "n_train", "n_valid", "n_test",
                       "n_rows_used")]
out$stage <- r$summary$Stage
out$verdict <- r$verdict
out$msgs <- msgs
out$warns <- warns
""", {"d": d})

    # the same candidates, the same InModel flags
    f = r.features
    assert sorted(f.Feature) == sorted(out["feature"])
    r_in = dict(zip(out["feature"], out["inmodel"]))
    assert all(r_in[x] == bool(y) for x, y in zip(f.Feature, f.InModel))

    # the near-duplicate table is computed on the full data: exact
    rc = pd.DataFrame(out["cor"])
    assert list(r.correlated.VarX) == list(rc.VarX)
    assert list(r.correlated.VarY) == list(rc.VarY)
    np.testing.assert_allclose(r.correlated.Correlation, rc.Correlation,
                               atol=1e-4)

    # split sizes, objective, stage labels
    for k in ("objective", "n_train", "n_valid", "n_test", "n_rows_used"):
        assert r.stats[k] == out["stats"][k], k
    assert list(r.summary.Stage) == list(out["stage"])

    # the messages and warnings, verdict aside
    assert [s for s in py_msgs if "missing main" not in s
            and "material" not in s and "adequate" not in s] == \
        [s for s in out["msgs"] if "missing main" not in s
         and "material" not in s and "adequate" not in s]
    assert sorted(py_warns) == sorted(out["warns"])

    # qualitatively: the mileage signal leads and the verdict agrees
    assert f.Feature[0] in ("KM", "KM_PROXY")
    assert out["feature"][0] in ("KM", "KM_PROXY")
    assert r.verdict.split(" (")[0] == out["verdict"].split(" (")[0]
