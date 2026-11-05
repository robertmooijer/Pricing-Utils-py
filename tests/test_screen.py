"""screen_features(): the port of tests/testthat/test-screen.R, the
screen_features test of test-data-table-input.R and the SCR-08 test of
test-review-findings.R.

Boosting results differ from R's (another random stream, another xgboost
build), so every test keeps the property the R test states, not its
numbers."""
import inspect
import math
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

xgb = pytest.importorskip("xgboost")

from pricingtools_rmo._core import PricingError, PricingMessage, PricingWarning
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.screen import (_screen_levels, _screen_matrix,
                                     screen_features)


def _quiet(fun, *args, **kw):
    """suppressWarnings(suppressMessages(...))"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingWarning)
        warnings.simplefilter("ignore", PricingMessage)
        return fun(*args, **kw)


# A baseline that deliberately omits one genuinely predictive feature,
# alongside a pure-noise feature and a near-duplicate of the useful one.
def _make_dscr(nn=40000, seed=19):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, nn)),
        "REGIO": pd.Categorical(rng.choice(["Stad", "Dorp"], nn),
                                categories=["Dorp", "Stad"]),
        "KM": np.round(np.maximum(rng.lognormal(np.log(12000), .5, nn), 1000),
                       -2),
        "RUIS": np.round(rng.normal(size=nn), 2),
        "Exposure": np.round(rng.uniform(0.3, 1, nn), 3)})
    d["KM_PROXY"] = d.KM + np.round(rng.normal(0, 200, nn))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.6 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD)
        + 0.35 * (d.REGIO == "Stad")
        + 0.45 * (np.log(d.KM) - np.log(12000))))      # KM matters, RUIS does not
    return d


@pytest.fixture(scope="module")
def dscr():
    return _make_dscr()


@pytest.fixture(scope="module")
def m_scr(dscr):
    return fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   dscr, family="poisson")


@pytest.fixture(scope="module")
def res(m_scr):
    return _quiet(screen_features, m_scr, seed=4, nrounds=300, nthread=2,
                  early_stopping_rounds=20, n_shap=0)


def _pd(f, name):
    return float(f.loc[f.Feature == name, "PermDeviance"].iloc[0])


def test_screen_features_returns_the_documented_structure(res):
    assert all(k in res for k in ("summary", "verdict", "features",
                                  "gain_note", "correlated", "stats", "plot"))
    assert len(res.summary) == 3
    assert isinstance(res.plot, go.Figure)
    assert res.stats.objective == "count:poisson"
    # no scorable model is handed back
    assert not any(isinstance(v, xgb.Booster) for v in res.values())
    assert list(res.features.columns) == ["Feature", "InModel",
                                          "PermDeviance", "Gain"]
    assert list(res.summary.columns) == ["Stage", "Deviance", "Change"]


def test_a_genuinely_predictive_feature_outranks_pure_noise(res):
    f = res.features
    # KM and KM_PROXY are near-duplicates, so they split the signal between
    # them and which of the two comes first is arbitrary. What must hold is
    # that the mileage signal tops the list and that noise does not.
    assert f.Feature[0] in ("KM", "KM_PROXY")
    assert _pd(f, "KM") > 0
    assert _pd(f, "KM_PROXY") > 0
    assert _pd(f, "RUIS") < _pd(f, "KM")
    # noise carries no usable signal, whatever Gain suggests
    assert _pd(f, "RUIS") < 1


def test_features_already_in_the_baseline_show_no_incremental_value(res):
    f = res.features
    inm = f[f.Feature.isin(["LEEFTIJD", "REGIO"])]
    assert inm.InModel.all()
    assert inm.PermDeviance.abs().max() < _pd(f, "KM")


def test_the_verdict_points_at_the_missing_main_effect(res):
    assert "main effects" in res.verdict
    assert res.stats.pct_depth1 < 0


def test_near_duplicate_candidates_are_reported(res):
    assert isinstance(res.correlated, pd.DataFrame)
    c = res.correlated
    assert ((c.VarX == "KM") & (c.VarY == "KM_PROXY")).any()


def test_awkward_columns_are_dropped_instead_of_crashing(dscr):
    # Every one of these used to trigger "contrasts can be applied only to
    # factors with 2 or more levels" once features = None picked them up
    rng = np.random.default_rng(2)
    nn = len(dscr)
    d2 = dscr.copy()
    d2["CONST_F"] = pd.Categorical(["N"] * nn)                   # constant factor
    d2["CONST_N"] = 1.0                                           # constant numeric
    d2["CHR"] = rng.choice(["p", "q", "r"], nn).astype(object)    # character column
    d2["WIDE"] = pd.Categorical(rng.integers(1, 201, nn))         # 200 levels
    d2["DATUM"] = (pd.Timestamp("2024-01-01")
                   + pd.to_timedelta(rng.integers(1, 366, nn), unit="D"))
    m2 = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                 d2, family="poisson")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingMessage)
        with pytest.warns(PricingWarning, match="constant") as rec:
            r2 = screen_features(m2, seed=2, nrounds=100, nthread=2,
                                 early_stopping_rounds=15, n_shap=0)
    msgs = [str(w.message) for w in rec]
    assert any("skipped (too many levels): WIDE" in s for s in msgs)
    assert any("skipped (unsupported type): DATUM" in s for s in msgs)
    assert isinstance(r2.features, pd.DataFrame)
    # dropped
    assert not set(r2.features.Feature) & {"CONST_F", "CONST_N", "WIDE",
                                           "DATUM"}
    # a character column is usable and keeps a stable width across splits
    assert "CHR" in set(r2.features.Feature)
    assert "KM" in set(r2.features.Feature)


def test_a_single_level_factor_no_longer_breaks_the_design_matrix(dscr):
    levs = _screen_levels(dscr, "REGIO")
    one = pd.DataFrame({"REGIO": pd.Categorical(
        ["Stad"] * 5, categories=dscr.REGIO.cat.categories)})
    m = _screen_matrix(one, "REGIO", levs)
    assert m.x.shape[1] == len(dscr.REGIO.cat.categories)  # width follows the level map
    assert m.x.sum() == 5                                    # one indicator per row
    assert m.names == ["REGIODorp", "REGIOStad"]


def test_an_uncorrelatable_pair_does_not_corrupt_the_near_duplicate_table():
    # R's which(arr.ind) skips NA correlations while the matching logical
    # mask returns one NA per skipped pair, so the values shifted onto the
    # wrong pairs and phantom rows appeared with VarX/VarY recycled
    rng = np.random.default_rng(31)
    n = 1500
    dc = pd.DataFrame({"Exposure": rng.uniform(.5, 1, n)})
    dc["A"] = rng.normal(size=n)
    dc["B"] = dc.A * 3 + rng.normal(0, .01, n)            # genuine near-duplicate
    first = np.arange(1, n + 1) <= n / 2
    dc["P"] = np.where(first, rng.normal(size=n), np.nan)  # no overlap with Q
    dc["Q"] = np.where(~first, rng.normal(size=n), np.nan)
    dc["AantalClaims"] = rng.poisson(dc.Exposure * .12)
    mc = fit_glm("AantalClaims ~ 1 + offset(log(Exposure))", dc,
                 family="poisson")

    r = _quiet(screen_features, mc, features=["A", "B", "P", "Q"], n_shap=0,
               nrounds=20, nthread=2, seed=1)
    assert len(r.correlated) == 1
    assert not r.correlated.Correlation.isna().any()
    assert {r.correlated.VarX[0], r.correlated.VarY[0]} == {"A", "B"}
    assert abs(r.correlated.Correlation[0]) > 0.99

    # and the pair that could not be compared is reported rather than hidden
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingMessage)
        with pytest.warns(PricingWarning, match="could not be correlated"):
            screen_features(mc, features=["A", "B", "P", "Q"], n_shap=0,
                            nrounds=20, nthread=2, seed=1)


def test_unsupported_families_and_bad_splits_are_refused(dscr, m_scr):
    m_id = fit_glm("AantalClaims ~ LEEFTIJD", dscr, family="gaussian")
    with pytest.raises(PricingError, match="supported"):
        screen_features(m_id)
    with pytest.raises(PricingError, match="split"):
        screen_features(m_scr, split=(0.5, 0.2, 0.2))
    # a Poisson model without an offset is not a count model here either
    m_no = fit_glm("AantalClaims ~ LEEFTIJD", dscr, family="poisson")
    with pytest.raises(PricingError, match="poisson/log"):
        screen_features(m_no)


def test_max_rows_defaults_to_every_row_and_accepts_inf_or_none():
    # the default used to cap at 1e6 and required None to lift it, which is
    # a convention you had to know about
    assert inspect.signature(screen_features).parameters["max_rows"].default \
        == math.inf

    rng = np.random.default_rng(81)
    nn = 3000
    dm = pd.DataFrame({"x": rng.uniform(size=nn),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    dm["z"] = rng.uniform(size=nn)
    dm["AantalClaims"] = rng.poisson(dm.Exposure * np.exp(-2 + dm.x))
    mm = fit_glm("AantalClaims ~ 1 + offset(log(Exposure))", dm,
                 family="poisson")

    def run(**kw):
        return _quiet(screen_features, mm, features=["x", "z"], n_shap=0,
                      nrounds=20, nthread=2, seed=1, **kw)

    assert run().stats.n_rows_used == nn                    # default: all of them
    assert run(max_rows=None).stats.n_rows_used == nn       # still works
    assert run(max_rows=math.inf).stats.n_rows_used == nn
    assert run(max_rows=1000).stats.n_rows_used == 1000

    # the sample is announced
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingWarning)
        with pytest.warns(PricingMessage, match="sampling 1,000 of 3,000 rows"):
            screen_features(mm, features=["x", "z"], n_shap=0, nrounds=20,
                            nthread=2, seed=1, max_rows=1000)

    # and a nonsensical value is refused rather than silently ignored
    with pytest.raises(PricingError, match="at least 1"):
        run(max_rows=0)
    with pytest.raises(PricingError, match="single number"):
        run(max_rows=[10, 20])
    with pytest.raises(PricingError, match="single number"):
        run(max_rows="veel")


def test_a_single_candidate_does_not_lose_the_run_to_xgb_importance():
    # xgboost 1.7 (R) cannot read the dump of a booster built on one column,
    # and xgb.importance is called after every expensive stage is done. Gain
    # is the column this function says not to rank on, so it degrades to NA.
    rng = np.random.default_rng(82)
    nn = 2500
    d1 = pd.DataFrame({"x": rng.uniform(size=nn),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    d1["AantalClaims"] = rng.poisson(d1.Exposure * np.exp(-2 + d1.x))
    m1 = fit_glm("AantalClaims ~ 1 + offset(log(Exposure))", d1,
                 family="poisson")

    r = _quiet(screen_features, m1, features="x", n_shap=0, nrounds=20,
               nthread=2, seed=1)
    assert len(r.features) == 1
    assert pd.api.types.is_float_dtype(r.features.PermDeviance)
    assert not np.isnan(r.features.PermDeviance[0])    # the useful column survives
    assert isinstance(r.plot, go.Figure)


def test_tree_method_is_hist_by_default_and_passed_through():
    # hist is the algorithm xgboost itself made default in 2.0, and it is
    # much faster than the old default
    assert inspect.signature(screen_features).parameters["tree_method"] \
        .default == "hist"

    rng = np.random.default_rng(83)
    nn = 4000
    dt = pd.DataFrame({"x": rng.uniform(size=nn), "z": rng.uniform(size=nn),
                       "Exposure": np.round(rng.uniform(.3, 1, nn), 3)})
    dt["AantalClaims"] = rng.poisson(dt.Exposure * np.exp(-2 + 1.5 * dt.x))
    mt = fit_glm("AantalClaims ~ 1 + offset(log(Exposure))", dt,
                 family="poisson")

    def run(**kw):
        return _quiet(screen_features, mt, features=["x", "z"], n_shap=0,
                      nrounds=60, nthread=2, seed=5, **kw)

    r_hist = run()
    r_exact = run(tree_method="exact")
    # the real signal wins under either method
    assert r_hist.features.Feature[0] == "x"
    assert r_exact.features.Feature[0] == "x"
    assert r_hist.features.PermDeviance[0] > 0

    with pytest.raises(xgb.core.XGBoostError):
        run(tree_method="onzin")


# ---- test-data-table-input.R ----------------------------------------------


def test_model_data_from_non_pandas_input_is_handled():
    # R: a model fitted on a data.table, whose `d[, cols, drop = FALSE]` is
    # an expression rather than a column selection. Here: a model fitted on
    # a plain dict of columns, which fit_glm turns into a DataFrame once.
    rng = np.random.default_rng(23)
    ndt = 20000
    ddt = {"LEEFTIJD": np.round(rng.uniform(18, 80, ndt)),
           "REGIO": rng.choice(["Stad", "Dorp"], ndt),
           "KM": np.round(np.maximum(rng.lognormal(np.log(12000), .4, ndt),
                                     1000), -2),
           "RUIS": np.round(rng.normal(size=ndt), 2),
           "Exposure": np.round(rng.uniform(.3, 1, ndt), 3)}
    ddt["AantalClaims"] = rng.poisson(ddt["Exposure"] * np.exp(
        -1.6 + 0.02 * np.maximum(0, 30 - ddt["LEEFTIJD"])
        + 0.3 * (ddt["REGIO"] == "Stad")
        + 0.4 * (np.log(ddt["KM"]) - np.log(12000))))
    ddt["SCHADELAST"] = ddt["AantalClaims"] * 1500.0
    m_dt = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                   ddt, family="poisson")
    assert isinstance(m_dt.data, pd.DataFrame)

    # the exact call that failed in R: the correlation check indexes columns
    r = _quiet(screen_features, m_dt, seed=5, nrounds=100, nthread=2,
               early_stopping_rounds=15, n_shap=0)
    assert isinstance(r.features, pd.DataFrame)
    assert "KM" in set(r.features.Feature)


# ---- test-review-findings.R -------------------------------------------------


def _rv_book(n=20000, seed=1):
    """The review's book (rv_book() in test-review-findings.R)."""
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
    reg = d.REGIO.astype(str).map({"Noord": 0, "Zuid": .2, "West": .35})
    eta = (np.log(0.08) + 0.6 * np.exp(-(d.LEEFTIJD - 18) / 8)
           + reg.to_numpy(dtype=float) + 0.3 * np.log(d.GEWICHT / 1300))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta))
    d["SCHADELAST"] = np.where(d.AantalClaims > 0,
                               d.AantalClaims * rng.gamma(1.5, 2500 / 1.5, n),
                               0.0)
    d["PREMIE_OUD"] = d.Exposure * 300 * np.exp(0.2 * (d.REGIO == "West"))
    return d


def test_screen_features_does_not_take_the_loss_amount_as_a_candidate():
    rv = _rv_book()
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))",
                 rv, family="poisson")
    sc = _quiet(screen_features, mf, seed=1, nrounds=100, n_shap=0, nthread=1)
    assert "SCHADELAST" not in set(sc.features.Feature)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingWarning)
        with pytest.warns(PricingMessage,
                          match="left out of the candidates: SCHADELAST"):
            screen_features(mf, seed=1, nrounds=20, n_shap=0, nthread=1)
    # asked for by name it stays, with a warning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingMessage)
        with pytest.warns(PricingWarning, match="outcome or volume column"):
            r = screen_features(mf, features=["GEWICHT", "SCHADELAST"], seed=1,
                                nrounds=20, n_shap=0, nthread=1)
    assert "SCHADELAST" in set(r.features.Feature)


# ---- beyond the R suite: the paths the R tests leave unexercised -----------


def test_a_gamma_severity_model_screens_with_prior_weights(m_sev):
    # reg:gamma with the claim counts as prior weights; the claim count (the
    # weight), the loss amount and the exposure are outcome columns
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PricingWarning)
        with pytest.warns(PricingMessage, match="left out of the candidates: "
                                                "AantalClaims, SCHADELAST, "
                                                "Exposure"):
            r = screen_features(m_sev, seed=3, nrounds=100, nthread=2,
                                early_stopping_rounds=15, n_shap=0)
    assert r.stats.objective == "reg:gamma"
    assert set(r.features.Feature) == {"LEEFTIJD", "REGIO", "BOEKJAAR"}
    assert r.features.InModel.tolist() == [f in ("LEEFTIJD", "REGIO")
                                           for f in r.features.Feature]
    assert np.isfinite(r.stats.deviance_baseline)


def test_the_shap_interaction_ranking_finds_a_built_in_pair():
    rng = np.random.default_rng(7)
    n = 20000
    d = pd.DataFrame({"A": rng.uniform(size=n), "B": rng.uniform(size=n),
                      "C": rng.uniform(size=n),
                      "Exposure": rng.uniform(.5, 1, n)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.5 + 1.2 * (d.A > .5) * (d.B > .5)))
    m = fit_glm("AantalClaims ~ A + B + offset(log(Exposure))", d,
                family="poisson")
    r = _quiet(screen_features, m, seed=1, nrounds=300, nthread=2,
               early_stopping_rounds=20, n_shap=1000)
    it = r.interactions
    assert isinstance(it, pd.DataFrame) and len(it) == 3
    assert (it.Strength.diff().dropna() <= 0).all()     # sorted, strongest first
    assert {it.VarX[0], it.VarY[0]} == {"A", "B"}
    assert r.stats.best_iteration["depth2"] >= 1


def test_the_plot_follows_the_r_layout(res):
    fig = res.plot
    bar = fig.data[0]
    assert bar.orientation == "h"
    fp = res.features.sort_values("PermDeviance", kind="stable")
    assert list(bar.y) == list(fp.Feature)
    assert fig.layout.title.text == "Incremental value over the baseline"
    assert fig.layout.xaxis.title.text == "Deviance increase when shuffled"
    assert fig.pricing_config["toImageButtonOptions"]["filename"] == "screen"
    inm = [t for t in bar.text if "already in the model" in t]
    assert len(inm) == 2
