"""Parity of make_rating_table() with the R package pricingtoolsRmO.

The data are built here, the same formulas are fitted in both languages and
each package builds its own table from its own fits. R fits with
glm.control(epsilon = 1e-15), as in the other parity tests, so the
comparison measures the table and not R's convergence tolerance.

Four tables, one R run:

* A: ns(), factor(BOEKJAAR) (latest year as base), a continuous x
  categorical interaction, a severity model on a regrouped column that only
  the severity data holds (merge_vars), and a chosen base per variable.
* B: a continuous x continuous interaction with a trimmed grid and
  base_level = "exposure".
* C: a cut() term (read on its bands), a logical predictor (a two-level
  categorical, FALSE as base, RT-14) and a categorical x categorical
  interaction, REGIO * factor(BOEKJAAR).
* D: a grouped column the caller put in 'data', for a level the severity
  data never saw (MV-05): kept, with an empty severity cell.

Structure, labels and flags are compared exactly; volumes at rtol 1e-9;
factors, uplifts, standard errors and intercepts at rtol 1e-6."""
import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo._utils import _txt
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.rating_table import make_rating_table

from ._r import as_float, requires_r, run_r

pytestmark = [pytest.mark.parity, requires_r]

RTOL = 1e-6

F_A_FREQ = ("AantalClaims ~ ns(LEEFTIJD, 3) + factor(BOEKJAAR) + GEWICHT * REGIO"
            " + Stedelijkheid + offset(log(Exposure))")
F_A_SEV = "AvgLoss ~ LEEFTIJD + REGIO + Sted_groep"
F_B_FREQ = ("AantalClaims ~ LEEFTIJD * log(GEWICHT) + REGIO"
            " + offset(log(Exposure))")
F_C_FREQ = ("AantalClaims ~ cut(LEEFTIJD, c(17, 30, 50, 85)) + GARAGE"
            " + REGIO * factor(BOEKJAAR) + offset(log(Exposure))")
F_C_SEV = "AvgLoss ~ REGIO + GARAGE"
F_D_FREQ = "AantalClaims ~ Stedelijkheid + LEEFTIJD + offset(log(Exposure))"
F_D_SEV = "AvgLoss ~ Sted_groep"

R_CODE = """
ctl <- glm.control(epsilon = 1e-15, maxit = 100)
tab <- function(tb) list(tb = tb, names = names(tb),
                         ifreq = attr(tb, "intercept_frequency"),
                         isev = attr(tb, "intercept_severity"),
                         iprem = attr(tb, "intercept_premium"),
                         base = lapply(attr(tb, "base_values"), as.character))
mfa <- glm(%s, family = poisson(), data = d, control = ctl)
msa <- glm(%s, family = Gamma(link = "log"), data = s, weights = AantalClaims,
           control = ctl)
out$a <- tab(suppressWarnings(make_rating_table(
  mfa, msa, data = d, merge_vars = list(Stedelijkheid = "Sted_groep"),
  base = list(REGIO = "Zuid", LEEFTIJD = 40))))
mfb <- glm(%s, family = poisson(), data = d, control = ctl)
out$b <- tab(suppressWarnings(make_rating_table(
  mfb, NULL, data = d, trim = c(0.01, 0.99), base_level = "exposure")))
mfc <- glm(%s, family = poisson(), data = d, control = ctl)
msc <- glm(%s, family = Gamma(link = "log"), data = s, weights = AantalClaims,
           control = ctl)
out$c <- tab(suppressWarnings(make_rating_table(mfc, msc, data = d)))
mfd <- glm(%s, family = poisson(), data = dd, control = ctl)
msd <- glm(%s, family = Gamma(link = "log"), data = sd, weights = AantalClaims,
           control = ctl)
out$d <- tab(suppressWarnings(make_rating_table(
  mfd, msd, data = dd, merge_vars = list(Stedelijkheid = "Sted_groep"),
  base = list(Stedelijkheid = "2"))))
""" % (F_A_FREQ, F_A_SEV, F_B_FREQ, F_C_FREQ, F_C_SEV, F_D_FREQ, F_D_SEV)


def _quiet(fun, *args, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fun(*args, **kw)


@pytest.fixture(scope="module")
def frames():
    rng = np.random.default_rng(2027)
    n = 12000
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "BOEKJAAR": rng.integers(2020, 2024, n),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "Stedelijkheid": pd.Categorical(rng.integers(1, 6, n).astype(str),
                                        categories=["1", "2", "3", "4", "5"]),
        "Exposure": np.round(rng.uniform(.1, 1, n), 3)})
    d["GARAGE"] = d.LEEFTIJD % 3 == 0
    sted = d.Stedelijkheid.astype(str).astype(float)
    eta = (-2.0 + .012 * (d.LEEFTIJD - 40) + .1 * sted
           + np.where(d.REGIO == "Zuid", .2, np.where(d.REGIO == "West", .35, 0))
           + .3 * np.log(d.GEWICHT / 1300) - .15 * d.GARAGE)
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta))
    sev = rng.gamma(2, 1500 * np.exp(.01 * (d.LEEFTIJD - 40))
                    * np.where(sted >= 4, 1.25, 1) / 2, n)
    d["SCHADELAST"] = np.where(d.AantalClaims > 0, d.AantalClaims * sev, 0.0)
    s = d[d.AantalClaims > 0].copy()
    s["AvgLoss"] = s.SCHADELAST / s.AantalClaims
    st = s.Stedelijkheid.astype(str)
    # the regrouping lives in the severity data only, as it usually does
    s["Sted_groep"] = pd.Categorical(np.where(st.isin(["4", "5"]), "4-5", st),
                                     categories=["1", "2", "3", "4-5"])
    return d.reset_index(drop=True), s.reset_index(drop=True)


def _d_frames(d, s):
    """Table D: the caller's own grouped column in 'data', and severity data
    without a claim at Stedelijkheid 1, so that level has no group there."""
    st = d.Stedelijkheid.astype(str)
    dd = d.assign(Sted_groep=pd.Categorical(np.where(st.isin(["4", "5"]), "4-5", st),
                                            categories=["1", "2", "3", "4-5"]))
    sd = s[s.Stedelijkheid.astype(str) != "1"].copy()
    sd["Sted_groep"] = sd.Sted_groep.cat.remove_unused_categories()
    return dd, sd.reset_index(drop=True)


@pytest.fixture(scope="module")
def r_out(frames):
    d, s = frames
    dd, sd = _d_frames(d, s)
    return run_r(R_CODE, {"d": d, "s": s, "dd": dd, "sd": sd})


@pytest.fixture(scope="module")
def py_tables(frames):
    d, s = frames
    mfa = fit_glm(F_A_FREQ, d, family="poisson")
    msa = fit_glm(F_A_SEV, s, family="gamma", weights="AantalClaims")
    a = _quiet(make_rating_table, mfa, msa, data=d,
               merge_vars={"Stedelijkheid": "Sted_groep"},
               base={"REGIO": "Zuid", "LEEFTIJD": 40})
    mfb = fit_glm(F_B_FREQ, d, family="poisson")
    b = _quiet(make_rating_table, mfb, None, data=d, trim=(0.01, 0.99),
               base_level="exposure")
    mfc = fit_glm(F_C_FREQ, d, family="poisson")
    msc = fit_glm(F_C_SEV, s, family="gamma", weights="AantalClaims")
    c = _quiet(make_rating_table, mfc, msc, data=d)
    dd, sd = _d_frames(d, s)
    mfd = fit_glm(F_D_FREQ, dd, family="poisson")
    msd = fit_glm(F_D_SEV, sd, family="gamma", weights="AantalClaims")
    dt = _quiet(make_rating_table, mfd, msd, data=dd,
                merge_vars={"Stedelijkheid": "Sted_groep"},
                base={"Stedelijkheid": "2"})
    return {"a": a, "b": b, "c": c, "d": dt}


def _strs(x):
    return [None if v is None else str(v) for v in x]


def _close(py, r, rtol, atol=1e-12):
    py = np.asarray(py, dtype=float)
    r = as_float(r)
    assert py.shape == r.shape
    np.testing.assert_array_equal(np.isnan(py), np.isnan(r))
    ok = ~np.isnan(py)
    np.testing.assert_allclose(py[ok], r[ok], rtol=rtol, atol=atol)
    return float(np.max(np.abs(py[ok] - r[ok]) / np.maximum(np.abs(r[ok]), 1e-300),
                        initial=0.0))


@pytest.mark.parametrize("key", ["a", "b", "c", "d"])
def test_rating_table_matches_r(key, py_tables, r_out):
    tb, ro = py_tables[key], r_out[key]
    rt = ro["tb"]
    # structure: the same columns in the same order, the same rows
    assert list(tb.columns) == list(ro["names"])
    assert len(tb) == len(rt["Variable"])
    for col in ("Variable", "Type", "Level", "Group", "XVar", "GroupVar"):
        assert _strs(tb[col].where(tb[col].notna(), None)) == _strs(rt[col]), col
    assert tb.IsBase.tolist() == list(rt["IsBase"])
    assert tb.IsThin.tolist() == list(rt["IsThin"])
    _close(tb.LevelNum, rt["LevelNum"], 1e-12)
    # volumes
    _close(tb.Exposure, rt["Exposure"], 1e-9)
    _close(tb.ClaimCount, rt["ClaimCount"], 1e-12)
    # factors, uplifts, standard errors
    for col in tb.columns:
        if col.startswith(("Factor_", "Uplift_", "SE_")):
            _close(tb[col], rt[col], RTOL, atol=1e-10)
    # intercepts and base values
    _close([tb.attrs["intercept_frequency"], tb.attrs["intercept_severity"],
            tb.attrs["intercept_premium"]],
           [ro["ifreq"], ro["isev"], ro["iprem"]], RTOL)
    pb = {k: _txt(v) for k, v in tb.attrs["base_values"].items()}
    assert pb == {k: str(v) for k, v in ro["base"].items()}


def test_rating_table_parity_covers_what_it_should(py_tables):
    a, b, c, d = py_tables["a"], py_tables["b"], py_tables["c"], py_tables["d"]
    # A: merged block, the latest year as base, the chosen bases, an
    # interaction with uplifts
    assert "Sted_groep" not in set(a.Variable)
    assert a.Level[(a.Variable == "BOEKJAAR") & a.IsBase].tolist() == ["2023"]
    assert a.Level[(a.Variable == "REGIO") & a.IsBase & a.Group.isna()].tolist() == ["Zuid"]
    assert "GEWICHT:REGIO" in set(a.Variable)
    # B: a trimmed grid, so less exposure than the portfolio
    assert b.Exposure[(b.Variable == "LEEFTIJD") & b.Group.isna()].sum() \
        < b.Exposure[b.Variable == "REGIO"].sum()
    # C: the bands and the logical levels
    assert set(c.Level[c.Variable == "LEEFTIJD"]) == {"(17,30]", "(30,50]", "(50,85]"}
    g = c[c.Variable == "GARAGE"]
    assert g.Level.tolist() == ["FALSE", "TRUE"]
    assert (g.Type == "categorical").all()
    assert g.Level[g.IsBase].tolist() == ["FALSE"]
    # D: the caller's group kept for the level the severity data never saw
    s = d[d.Variable == "Stedelijkheid"]
    assert len(s) == 5
    assert np.isnan(s.Factor_Severity[s.Level == "1"].iloc[0])
    assert np.isfinite(s.Factor_Frequency[s.Level == "1"].iloc[0])


def test_report_max_relative_differences(py_tables, r_out, capsys):
    """Not an assertion beyond the one above: prints the largest relative
    difference per column, for the porting report (pytest -s)."""
    lines = []
    for key in ("a", "b", "c", "d"):
        tb, rt = py_tables[key], r_out[key]["tb"]
        for col in tb.columns:
            if col.startswith(("Factor_", "Uplift_", "SE_")) or col in ("Exposure",):
                lines.append("%s %-17s %.2e" % (key, col,
                                                 _close(tb[col], rt[col], 1e-4, 1e-8)))
    with capsys.disabled():
        print("\n" + "\n".join(lines))
