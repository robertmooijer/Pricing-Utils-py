"""Tests for make_rating_table(): the port of tests/testthat/test-rating-table.R,
plus the rating-table tests of test-awkward-names.R, test-data-table-input.R
and test-review-findings.R. The data are drawn with numpy's generator, so
every test states a property, not a number copied from R."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo import PricingError, PricingWarning
from pricingtools_rmo._formula import r_cut
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.rating_table import make_rating_table


# ---- helpers ------------------------------------------------------------------------


def main(t, v):
    """Main-effect rows of one variable."""
    return t[(t.Variable == v) & t.Group.isna()]


def basis(t, v):
    return t[(t.Variable == v) & t.Group.isna() & t.IsBase]


def fac(t, v, lvl, col="Factor_Frequency"):
    s = main(t, v)
    return float(s[col][s.Level == str(lvl)].iloc[0])


def cat(values, categories=None):
    """A pandas Categorical as R's factor(): sorted levels unless given."""
    values = np.asarray(values, dtype=object)
    if categories is None:
        categories = sorted(set(values.tolist()))
    return pd.Categorical(values, categories=categories)


def collect(fn, *a, **k):
    """Every PricingWarning message raised by a call, and its result."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        res = fn(*a, **k)
    return res, [str(x.message) for x in w if issubclass(x.category, PricingWarning)]


def quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def attrs_equal(a, b):
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, float) and isinstance(y, float):
            if not (x == y or (np.isnan(x) and np.isnan(y))):
                return False
        elif x != y:
            return False
    return True


def sev_rows(d):
    s = d[d.AantalClaims > 0].copy()
    s["AvgLoss"] = s.SCHADELAST / s.AantalClaims
    return s


# ---- test-rating-table.R ------------------------------------------------------------


def test_rating_table_builds_incl_inline_factor_boekjaar(tbl):
    assert {"LEEFTIJD", "REGIO", "BOEKJAAR"} <= set(tbl.Variable)
    assert (tbl.Type[tbl.Variable == "BOEKJAAR"] == "categorical").all()


def test_every_variable_has_exactly_one_base_row_with_factor_1(tbl):
    b = tbl[tbl.IsBase & tbl.Group.isna()]
    assert len(b) == tbl.Variable[tbl.Group.isna()].nunique()
    f = np.concatenate([b.Factor_Frequency, b.Factor_Severity])
    f = f[~np.isnan(f)]
    assert np.all(np.abs(f - 1) < 1e-10)


def test_intercept_x_factors_reconstructs_the_model_prediction_exactly(tbl, m_freq, dat):
    i_f = tbl.attrs["intercept_frequency"]
    s_age = main(tbl, "LEEFTIJD")
    age = s_age.LevelNum.iloc[34]                 # arbitrary grid point
    test_row = pd.DataFrame({
        "LEEFTIJD": [age],
        "REGIO": pd.Categorical(["Zuid"], categories=dat.REGIO.cat.categories),
        "BOEKJAAR": [2022], "Exposure": [1.0]})
    pred_direct = m_freq.predict(test_row)[0]
    recon = (i_f * s_age.Factor_Frequency.iloc[34] * fac(tbl, "REGIO", "Zuid")
             * fac(tbl, "BOEKJAAR", "2022"))
    assert abs(pred_direct - recon) < 1e-10


def test_intercept_is_per_unit_of_exposure(tbl, m_freq, dat):
    i_f = tbl.attrs["intercept_frequency"]
    # the accounting year sits at its latest level by default (RT-15)
    base_pred = m_freq.predict(pd.DataFrame({
        "LEEFTIJD": [np.median(dat.LEEFTIJD)],
        "REGIO": pd.Categorical(["Noord"], categories=dat.REGIO.cat.categories),
        "BOEKJAAR": [2023], "Exposure": [1.0]}))[0]
    assert abs(i_f - base_pred) < 1e-10


def test_interaction_rows_have_a_unit_base_cell_and_unit_uplift_at_reference(tbl2):
    d_int = tbl2[tbl2.Variable == "LEEFTIJD:REGIO"]
    assert len(d_int) > 0
    assert abs(d_int.Factor_Frequency[d_int.IsBase].iloc[0] - 1) < 1e-10
    u_ref = d_int.Uplift_Frequency[d_int.Group == "Noord"]
    assert np.all(np.abs(u_ref - 1) < 1e-8)


def test_non_log_link_raises_a_warning(d_sev, dat):
    m_inv = fit_glm("AvgLoss ~ REGIO", d_sev, family="gamma", link="inverse",
                    weights="AantalClaims")
    with pytest.warns(PricingWarning, match="log link"):
        make_rating_table(None, m_inv, data=dat)


def test_base_level_exposure_picks_the_largest_level_as_reference(m_freq, dat):
    tbl3 = make_rating_table(m_freq, None, data=dat, base_level="exposure")
    largest = dat.groupby("REGIO", observed=True).Exposure.sum().idxmax()
    b3 = tbl3.Level[(tbl3.Variable == "REGIO") & tbl3.IsBase].tolist()
    assert b3 == [largest]


def test_the_thin_cell_column_is_correct(tbl, m_freq, dat):
    ok = tbl.ClaimCount.notna()
    assert (tbl.IsThin[ok] == (tbl.ClaimCount[ok] < 30)).all()

    with pytest.warns(PricingWarning, match="very little experience"):
        tbl_thin = make_rating_table(m_freq, None, data=dat, min_claims=1e6)
    assert tbl_thin.IsThin.dropna().astype(bool).all()


def test_the_thin_cell_table_still_plots(m_freq, dat):
    rp = pytest.importorskip("pricingtools_rmo.rating_plot")
    tbl_thin = quiet(make_rating_table, m_freq, None, data=dat, min_claims=1e6)
    assert rp.make_rating_plot(tbl_thin, "REGIO") is not None


def _grid_data(n, seed, claims):
    rng = np.random.default_rng(seed)
    dg = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "GEWICHT": np.round(np.clip(rng.normal(1350, 250, n), 800, 2400), -1),
        "Exposure": np.round(rng.uniform(.3, 1, n), 3)})
    dg["AantalClaims"] = rng.poisson(dg.Exposure * claims(dg))
    return dg


def test_continuous_grids_run_in_readable_steps():
    dg = _grid_data(20000, 8, lambda d: np.exp(
        -1.7 + .02 * np.maximum(0, 30 - d.LEEFTIJD)))
    mg = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + ns(GEWICHT, 3) + "
                 "offset(log(Exposure))", dg, family="poisson")
    tg = make_rating_table(mg, None, data=dg)

    # a whole-number column with few enough values is listed value by value
    age = main(tg, "LEEFTIJD").LevelNum.to_numpy()
    assert np.unique(np.diff(age)).tolist() == [1]
    assert np.all(age == np.round(age))

    # a wide column gets a rounded step instead
    wt = main(tg, "GEWICHT").LevelNum.to_numpy()
    step = np.unique(np.diff(wt))
    assert len(step) == 1
    assert np.all(np.abs(wt / step[0] - np.round(wt / step[0])) < 1e-9)

    # the base sits on the grid, so there is a row with factor exactly 1
    for v in ("LEEFTIJD", "GEWICHT"):
        s = main(tg, v)
        assert s.IsBase.sum() == 1
        assert s.Factor_Frequency[s.IsBase].iloc[0] == pytest.approx(1, abs=1e-12)


def test_grid_step_can_be_set_globally_or_per_variable():
    dg = _grid_data(15000, 8, lambda d: np.exp(-1.6))
    mg = fit_glm("AantalClaims ~ LEEFTIJD + GEWICHT + offset(log(Exposure))",
                 dg, family="poisson")
    lv = lambda t, v: t.LevelNum[t.Variable == v].to_numpy()

    t1 = make_rating_table(mg, None, data=dg,
                           grid_step=pd.Series({"LEEFTIJD": 5, "GEWICHT": 200}))
    assert np.unique(np.diff(lv(t1, "LEEFTIJD"))).tolist() == [5]
    assert np.unique(np.diff(lv(t1, "GEWICHT"))).tolist() == [200]

    t2 = make_rating_table(mg, None, data=dg, grid_step=10)
    assert np.unique(np.diff(lv(t2, "GEWICHT"))).tolist() == [10]

    # a dict works the same as a named Series (R: named list / named vector)
    t3 = make_rating_table(mg, None, data=dg,
                           grid_step={"LEEFTIJD": 5, "GEWICHT": 200})
    np.testing.assert_array_equal(t3.LevelNum, t1.LevelNum)

    # naming only one variable leaves the rest on the automatic step
    t4 = make_rating_table(mg, None, data=dg, grid_step={"LEEFTIJD": 5})
    auto = make_rating_table(mg, None, data=dg)
    assert np.unique(np.diff(lv(t4, "LEEFTIJD"))).tolist() == [5]
    np.testing.assert_array_equal(lv(t4, "GEWICHT"), lv(auto, "GEWICHT"))

    # a misspelled name is called out rather than ignored
    with pytest.warns(PricingWarning, match="no continuous variable"):
        make_rating_table(mg, None, data=dg, grid_step={"LEEFTJID": 5})


def test_a_wide_integer_column_with_few_values_is_listed_not_stepped():
    # A coded column such as a sum insured holds a handful of whole numbers
    # spread over a million. Treating "few distinct values" as "step by 1"
    # built a grid over the whole range - a million rows - instead of the
    # five values that actually occur.
    rng = np.random.default_rng(21)
    ng = 3000
    dg = pd.DataFrame({
        "VERZBEDRAG": rng.choice([0, 100000, 250000, 500000, 1000000], ng),
        "LEEFTIJD": np.round(rng.uniform(18, 80, ng)),
        "Exposure": rng.uniform(.3, 1, ng)})
    dg["AantalClaims"] = rng.poisson(dg.Exposure * .1)
    mg = fit_glm("AantalClaims ~ VERZBEDRAG + LEEFTIJD + offset(log(Exposure))",
                 dg, family="poisson")
    tg = make_rating_table(mg, None, data=dg)

    np.testing.assert_array_equal(main(tg, "VERZBEDRAG").LevelNum,
                                  np.sort(dg.VERZBEDRAG.unique()))
    assert len(tg) < 200
    # an age still runs year by year
    assert np.unique(np.diff(main(tg, "LEEFTIJD").LevelNum)).tolist() == [1]
    # and there is still exactly one base row with a factor of 1
    b = tg[(tg.Variable == "VERZBEDRAG") & tg.IsBase]
    assert len(b) == 1
    assert b.Factor_Frequency.iloc[0] == pytest.approx(1, abs=1e-12)
    assert b.LevelNum.iloc[0] in set(dg.VERZBEDRAG)


def test_round_large_levels_are_not_written_in_scientific_notation():
    # as.character(1e5) is "1e+05", which is unusable both as a tariff level
    # and as the lookup key the export builds from it
    rng = np.random.default_rng(22)
    ng = 2000
    dg = pd.DataFrame({"BEDRAG": rng.choice([100000, 200000, 1000000], ng),
                       "Exposure": rng.uniform(.3, 1, ng)})
    dg["AantalClaims"] = rng.poisson(dg.Exposure * .1)
    mg = fit_glm("AantalClaims ~ BEDRAG + offset(log(Exposure))", dg,
                 family="poisson")
    tg = make_rating_table(mg, None, data=dg)
    lvl = tg.Level[tg.Variable == "BEDRAG"].tolist()
    assert not any("e" in x for x in lvl)
    assert lvl == ["100000", "200000", "1000000"]
    # Level and LevelNum must still describe the same value
    np.testing.assert_array_equal([float(x) for x in lvl],
                                  tg.LevelNum[tg.Variable == "BEDRAG"])


def test_interaction_cells_carry_exposure_and_claims_for_any_variable_type():
    # Volume was filled only for categorical x categorical, so a continuous
    # rating factor crossed with a region left Exposure, ClaimCount and
    # IsThin empty - in the very cells where a portfolio runs thin
    rng = np.random.default_rng(23)
    ng = 8000
    dg = pd.DataFrame({"LEEFTIJD": np.round(rng.uniform(18, 70, ng)),
                       "REGIO": cat(rng.choice(["Noord", "Zuid"], ng)),
                       "Exposure": rng.uniform(.4, 1, ng)})
    dg["AantalClaims"] = rng.poisson(dg.Exposure * .15)
    mg = fit_glm("AantalClaims ~ LEEFTIJD * REGIO + offset(log(Exposure))", dg,
                 family="poisson")
    tg = quiet(make_rating_table, mg, None, data=dg)
    ig = tg[tg.Group.notna()]

    assert len(ig) > 0
    assert not ig.Exposure.isna().any()
    assert not ig.ClaimCount.isna().any()
    assert not ig.IsThin.isna().any()
    # every policy is counted exactly once across the cells
    assert ig.Exposure.sum() == pytest.approx(dg.Exposure.sum(), rel=1e-8)
    assert ig.ClaimCount.sum() == dg.AantalClaims.sum()

    # and a cell matches a hand count on the same grid
    g = np.unique(ig.LevelNum)
    brk = np.concatenate([[-np.inf], g[1:] - np.diff(g) / 2, [np.inf]])
    nz = (dg.REGIO == "Noord").to_numpy()
    b = np.searchsorted(brk, dg.LEEFTIJD[nz], side="right") - 1
    hand = np.bincount(b, weights=dg.Exposure[nz], minlength=len(g))
    cel = ig[ig.Group == "Noord"].sort_values("LevelNum")
    np.testing.assert_allclose(cel.Exposure.to_numpy(), hand, rtol=1e-8)


def test_the_rating_table_carries_no_credibility_columns(tbl):
    assert not any(c.startswith("Credibility") for c in tbl.columns)
    assert "credibility" not in tbl.attrs


def test_the_uplift_columns_are_absent_without_an_interaction(tbl, tbl2):
    # An uplift only exists relative to the two main effects it sits between,
    # so on a tariff with no interaction term at all it would be NA in every
    # row. Better not to ship the column than to ship it empty.
    assert not any(c.startswith("Uplift") for c in tbl.columns)
    assert {"Group", "GroupVar"} <= set(tbl.columns)           # structure stays

    # with an interaction they are back, and filled on exactly those rows
    assert {"Uplift_Frequency", "Uplift_Severity", "Uplift_Premium"} <= set(tbl2.columns)
    assert not tbl2.Uplift_Frequency[tbl2.Group.notna()].isna().any()
    assert tbl2.Uplift_Frequency[tbl2.Group.isna()].isna().all()

    # a variable without an interaction term gets no interaction rows at all
    assert not tbl2.Variable[tbl2.Group.notna()].str.contains("BOEKJAAR").any()


def test_the_export_copes_with_a_table_that_has_no_uplift_columns(tbl, tmp_path):
    ex = pytest.importorskip("pricingtools_rmo.export")
    openpyxl = pytest.importorskip("openpyxl")
    f = str(tmp_path / "no_uplift.xlsx")
    quiet(ex.export_rating_table, tbl, f)
    wb = openpyxl.load_workbook(f)
    assert "Tariff" in wb.sheetnames
    d1 = pd.read_excel(f, sheet_name="REGIO") if "REGIO" in wb.sheetnames else None
    assert d1 is not None
    assert not any(str(c).startswith("Uplift") for c in d1.columns)


def _base_book(n, seed, regions, lin):
    rng = np.random.default_rng(seed)
    db = pd.DataFrame({"LEEFTIJD": np.round(rng.uniform(18, 80, n)),
                       "REGIO": cat(rng.choice(regions, n)),
                       "Exposure": np.round(rng.uniform(.2, 1, n), 3)})
    db["AantalClaims"] = rng.poisson(db.Exposure * lin(db))
    return db


def test_the_base_can_be_chosen_per_variable_default_unchanged():
    db = _base_book(20000, 91, ["Noord", "Zuid", "Randstad"], lambda d: np.exp(
        -2 + .01 * d.LEEFTIJD + .3 * (d.REGIO == "Randstad")))
    mb = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", db,
                 family="poisson")

    t0 = make_rating_table(mb, None, data=db)
    t0b = make_rating_table(mb, None, data=db, base=None)
    pd.testing.assert_frame_equal(t0, t0b)
    assert attrs_equal(t0.attrs, t0b.attrs)
    assert basis(t0, "REGIO").Level.tolist() == ["Noord"]          # first level

    t1 = make_rating_table(mb, None, data=db, base={"REGIO": "Zuid", "LEEFTIJD": 40})
    assert basis(t1, "REGIO").Level.tolist() == ["Zuid"]
    assert basis(t1, "LEEFTIJD").LevelNum.tolist() == [40]
    assert basis(t1, "REGIO").Factor_Frequency.iloc[0] == pytest.approx(1)
    assert basis(t1, "LEEFTIJD").Factor_Frequency.iloc[0] == pytest.approx(1)
    # exactly one base row per variable, and no duplicate grid point at 40
    assert len(basis(t1, "LEEFTIJD")) == 1
    assert ((t1.Variable == "LEEFTIJD") & (t1.LevelNum == 40)).sum() == 1

    # a named Series works too, and overrides base_level for that variable
    t2 = make_rating_table(mb, None, data=db, base_level="exposure",
                           base=pd.Series({"REGIO": "Zuid"}))
    assert basis(t2, "REGIO").Level.tolist() == ["Zuid"]

    # a base between grid points is placed exactly, not snapped
    t3 = make_rating_table(mb, None, data=db, base={"LEEFTIJD": 40.5})
    assert basis(t3, "LEEFTIJD").LevelNum.tolist() == [40.5]

    # premiums do not move: same policy, both tables
    def prem(t, age, reg):
        return (t.attrs["intercept_frequency"] * fac(t, "LEEFTIJD", age)
                * fac(t, "REGIO", reg))
    assert prem(t0, "60", "Randstad") == pytest.approx(prem(t1, "60", "Randstad"),
                                                       rel=1e-10)
    nd = pd.DataFrame({"LEEFTIJD": [60.0],
                       "REGIO": pd.Categorical(["Randstad"],
                                               categories=db.REGIO.cat.categories),
                       "Exposure": [1.0]})
    assert prem(t1, "60", "Randstad") == pytest.approx(mb.predict(nd)[0], rel=1e-10)


def test_a_wrong_base_is_refused_or_reported():
    db = _base_book(5000, 92, ["Noord", "Zuid"], lambda d: .12)
    mb = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", db,
                 family="poisson")
    with pytest.raises(PricingError, match="not a level"):
        make_rating_table(mb, None, data=db, base={"REGIO": "Oost"})
    with pytest.raises(PricingError, match="must be a number"):
        make_rating_table(mb, None, data=db, base={"LEEFTIJD": "oud"})
    with pytest.raises(PricingError, match="offset or exposure"):
        make_rating_table(mb, None, data=db, base={"Exposure": 2})
    with pytest.raises(PricingError, match="must be named"):
        make_rating_table(mb, None, data=db, base=["Zuid"])
    with pytest.warns(PricingWarning, match="ignored: REGOI"):
        make_rating_table(mb, None, data=db, base={"REGOI": "Zuid"})
    with pytest.warns(PricingWarning, match="outside the data"):
        make_rating_table(mb, None, data=db, base={"LEEFTIJD": 120})


def _sted_book(seed):
    rng = np.random.default_rng(seed)
    nm = 30000
    dm = pd.DataFrame({
        "Stedelijkheid": cat(rng.integers(1, 6, nm).astype(str)),
        "LEEFTIJD": np.round(rng.uniform(18, 80, nm)),
        "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    s = dm.Stedelijkheid.astype(str)
    dm["Sted_groep"] = cat(np.where(s.isin(["4", "5"]), "4-5", s))
    dm["AantalClaims"] = rng.poisson(dm.Exposure * np.exp(-2.2 + .15 * s.astype(float)))
    sev = rng.gamma(3, 600 * np.where(dm.Sted_groep == "4-5", 1.25, 1), nm)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0, dm.AantalClaims * sev, 0.0)
    return dm


def _nd(dm, ds=None, **vals):
    """One prediction row, the categorical columns on the data's levels."""
    out = {}
    for k, v in vals.items():
        src = dm if k in dm.columns else ds
        if src is not None and k in src.columns \
                and isinstance(src[k].dtype, pd.CategoricalDtype):
            out[k] = pd.Categorical([v], categories=src[k].cat.categories)
        else:
            out[k] = [v]
    return pd.DataFrame(out)


def test_two_columns_for_one_rating_factor_can_be_merged_into_one_block():
    dm = _sted_book(93)
    mf = fit_glm("AantalClaims ~ Stedelijkheid + LEEFTIJD + offset(log(Exposure))",
                 dm, family="poisson")
    ds = sev_rows(dm)
    ms = fit_glm("AvgLoss ~ Sted_groep", ds, family="gamma", weights="AantalClaims")
    mv = {"Stedelijkheid": "Sted_groep"}

    # undeclared: two blocks, each padded with neutral factors, and a hint
    with pytest.warns(PricingWarning, match="regrouping of another"):
        t0 = make_rating_table(mf, ms, data=dm)
    assert {"Stedelijkheid", "Sted_groep"} <= set(t0.Variable)
    np.testing.assert_array_equal(t0.Factor_Severity[t0.Variable == "Stedelijkheid"],
                                  np.ones(5))

    t1 = make_rating_table(mf, ms, data=dm, merge_vars=mv)
    assert "Sted_groep" not in set(t1.Variable)
    s1 = t1[t1.Variable == "Stedelijkheid"]
    assert len(s1) == 5
    # each level carries the severity factor of the group it falls in
    sv = lambda lv: s1.Factor_Severity[s1.Level == lv].iloc[0]
    assert sv("4") == sv("5")
    g45 = t0.Factor_Severity[(t0.Variable == "Sted_groep") & (t0.Level == "4-5")].iloc[0]
    assert sv("4") == pytest.approx(g45, rel=1e-12)
    # and the frequency factor is untouched
    np.testing.assert_allclose(s1.Factor_Frequency,
                               t0.Factor_Frequency[t0.Variable == "Stedelijkheid"],
                               rtol=1e-12)
    # volumes stay on the levels of the column a policy carries
    np.testing.assert_allclose(
        s1.Exposure, dm.groupby("Stedelijkheid", observed=True).Exposure.sum(),
        rtol=1e-10)

    # one block, one lookup, and the premium still reconstructs exactly
    nd = _nd(dm, Stedelijkheid="5", Sted_groep="4-5", LEEFTIJD=40.0,
             Exposure=1.0, AantalClaims=1)

    def prem(t, sted="5"):
        return (t.attrs["intercept_premium"]
                * fac(t, "Stedelijkheid", sted, "Factor_Premium")
                * fac(t, "LEEFTIJD", "40", "Factor_Premium"))
    direct = mf.predict(nd)[0] * ms.predict(nd)[0]
    assert prem(t1) == pytest.approx(direct, rel=1e-10)
    # the undeclared table needs both blocks to get there, the merged one does not
    assert prem(t0) * fac(t0, "Sted_groep", "4-5", "Factor_Premium") == \
        pytest.approx(direct, rel=1e-10)

    # a chosen base moves both columns, so the base row stays a real policy
    t2 = make_rating_table(mf, ms, data=dm, merge_vars=mv,
                           base={"Stedelijkheid": "5"})
    b2 = t2[(t2.Variable == "Stedelijkheid") & t2.IsBase]
    assert b2.Level.tolist() == ["5"]
    np.testing.assert_allclose([b2.Factor_Frequency.iloc[0],
                                b2.Factor_Severity.iloc[0]], [1, 1])
    # the intercept is that base policy: urbanisation 5 at the base age
    nd_b = nd.copy()
    nd_b["LEEFTIJD"] = t2.attrs["base_values"]["LEEFTIJD"]
    assert t2.attrs["intercept_premium"] == pytest.approx(
        mf.predict(nd_b)[0] * ms.predict(nd_b)[0], rel=1e-10)
    # and every premium is unchanged by that choice
    nd2 = _nd(dm, Stedelijkheid="2", Sted_groep="2", LEEFTIJD=40.0,
              Exposure=1.0, AantalClaims=1)
    assert prem(t1, "2") == pytest.approx(prem(t2, "2"), rel=1e-10)
    assert prem(t2, "2") == pytest.approx(mf.predict(nd2)[0] * ms.predict(nd2)[0],
                                          rel=1e-10)

    # a named Series is accepted as well
    t1s = make_rating_table(mf, ms, data=dm,
                            merge_vars=pd.Series({"Stedelijkheid": "Sted_groep"}))
    pd.testing.assert_frame_equal(t1s, t1)


def test_a_merged_variable_exports_to_one_sheet(tmp_path):
    ex = pytest.importorskip("pricingtools_rmo.export")
    openpyxl = pytest.importorskip("openpyxl")
    dm = _sted_book(93)
    mf = fit_glm("AantalClaims ~ Stedelijkheid + LEEFTIJD + offset(log(Exposure))",
                 dm, family="poisson")
    ms = fit_glm("AvgLoss ~ Sted_groep", sev_rows(dm), family="gamma",
                 weights="AantalClaims")
    t1 = make_rating_table(mf, ms, data=dm, merge_vars={"Stedelijkheid": "Sted_groep"})
    fp = str(tmp_path / "merged.xlsx")
    quiet(ex.export_rating_table, t1, fp)
    sn = openpyxl.load_workbook(fp).sheetnames
    assert "Stedelijkheid" in sn
    assert "Sted_groep" not in sn


def _band_book(seed, sev_factor, claims=.2, band_in_data=True):
    rng = np.random.default_rng(seed)
    nm = 20000
    dm = pd.DataFrame({"LEEFTIJD": np.round(rng.uniform(18, 80, nm)),
                       "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    kl = r_cut(dm.LEEFTIJD, [17, 30, 50, 80], labels=["18-30", "31-50", "51-80"])
    if band_in_data:
        dm["LEEFTIJD_KL"] = kl
    dm["AantalClaims"] = rng.poisson(dm.Exposure * claims)
    sev = rng.gamma(3, 600 * sev_factor(dm, kl), nm)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0, dm.AantalClaims * sev, 0.0)
    return dm


def test_a_band_of_a_continuous_variable_can_be_merged_into_it():
    dm = _band_book(94, lambda d, kl: np.where(np.asarray(kl) == "18-30", 1.3, 1))
    mf = fit_glm("AantalClaims ~ LEEFTIJD + offset(log(Exposure))", dm,
                 family="poisson")
    ds = sev_rows(dm)
    ms = fit_glm("AvgLoss ~ LEEFTIJD_KL", ds, family="gamma", weights="AantalClaims")

    t1 = make_rating_table(mf, ms, data=dm, merge_vars={"LEEFTIJD": "LEEFTIJD_KL"},
                           grid_step={"LEEFTIJD": 5})
    s = t1[t1.Variable == "LEEFTIJD"]
    assert "LEEFTIJD_KL" not in set(t1.Variable)
    # a grid point takes the band it falls in, including the band edges
    band = lambda age: s.Factor_Severity[s.LevelNum == age].iloc[0]
    assert band(20) == pytest.approx(band(30))             # both in 18-30
    assert band(35) == pytest.approx(band(50))             # both in 31-50
    assert band(55) == pytest.approx(band(80))             # both in 51-80
    assert band(30) != pytest.approx(band(35))
    nd = _nd(dm, LEEFTIJD=55.0, LEEFTIJD_KL="51-80", Exposure=1.0, AantalClaims=1)
    assert t1.attrs["intercept_premium"] * fac(t1, "LEEFTIJD", "55", "Factor_Premium") \
        == pytest.approx(mf.predict(nd)[0] * ms.predict(nd)[0], rel=1e-10)


def test_an_interaction_on_a_merged_column_is_reported_under_one_name():
    rng = np.random.default_rng(95)
    nm = 25000
    dm = pd.DataFrame({"Sted": cat(rng.integers(1, 5, nm).astype(str)),
                       "REGIO": cat(rng.choice(["Noord", "Zuid"], nm)),
                       "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    s = dm.Sted.astype(str)
    dm["Sted_groep"] = cat(np.where(s.isin(["3", "4"]), "3-4", s))
    dm["AantalClaims"] = rng.poisson(dm.Exposure * .2)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0,
                                dm.AantalClaims * rng.gamma(3, 600, nm), 0.0)
    mf = fit_glm("AantalClaims ~ Sted + REGIO + offset(log(Exposure))", dm,
                 family="poisson")
    ms = fit_glm("AvgLoss ~ Sted_groep * REGIO", sev_rows(dm), family="gamma",
                 weights="AantalClaims")

    t1 = make_rating_table(mf, ms, data=dm, merge_vars={"Sted": "Sted_groep"})
    assert not t1.Variable.str.contains("Sted_groep").any()
    it = t1[t1.Group.notna()]
    assert len(it) > 0
    assert it.Variable.unique().tolist() == ["Sted:REGIO"]
    # the interaction runs over the four levels of the finer column
    assert set(it.Level) == set(dm.Sted.cat.categories)
    assert it.Factor_Frequency[it.IsBase].iloc[0] == pytest.approx(1)
    # the two grouped levels share the severity relativity
    sv = lambda lv, g: it.Factor_Severity[(it.Level == lv) & (it.Group == g)].iloc[0]
    assert sv("3", "Zuid") == pytest.approx(sv("4", "Zuid"), rel=1e-12)


def test_a_merge_declaration_that_cannot_hold_is_refused():
    rng = np.random.default_rng(96)
    nm = 5000
    dm = pd.DataFrame({"A": cat(rng.integers(1, 5, nm).astype(str)),
                       "B": cat(rng.choice(["x", "y"], nm)),
                       "Exposure": rng.uniform(.3, 1, nm)})
    dm["AantalClaims"] = rng.poisson(dm.Exposure * .2)
    a = dm.A.astype(str)
    dm["A_groep"] = cat(np.where(a.isin(["3", "4"]), "3-4", a))
    mf = fit_glm("AantalClaims ~ A + offset(log(Exposure))", dm, family="poisson")

    with pytest.raises(PricingError, match="not a regrouping"):
        make_rating_table(mf, None, data=dm, merge_vars={"A": "B"})
    with pytest.raises(PricingError, match="onto itself"):
        make_rating_table(mf, None, data=dm, merge_vars={"A": "A"})
    # a misspelled column is in neither the data nor a model, so there is
    # nothing to derive it from either
    with pytest.raises(PricingError, match="no model uses it"):
        make_rating_table(mf, None, data=dm, merge_vars={"A": "A_group"})
    with pytest.raises(PricingError, match="regrouping of both"):
        make_rating_table(mf, None, data=dm,
                          merge_vars={"A": "A_groep", "B": "A_groep"})
    with pytest.raises(PricingError, match="must be named"):
        make_rating_table(mf, None, data=dm, merge_vars=["A_groep"])
    # a valid relation on columns no model uses changes nothing, and says so
    dm["B_alles"] = cat(["alles"] * nm)
    with pytest.warns(PricingWarning, match="neither model uses"):
        make_rating_table(mf, None, data=dm, merge_vars={"B": "B_alles"})


def test_a_regrouped_column_missing_from_data_is_taken_from_the_model():
    rng = np.random.default_rng(97)
    nm = 25000
    dm = pd.DataFrame({"Stedelijkheid": cat(rng.integers(1, 6, nm).astype(str)),
                       "LEEFTIJD": np.round(rng.uniform(18, 80, nm)),
                       "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    s = dm.Stedelijkheid.astype(str)
    dm["AantalClaims"] = rng.poisson(dm.Exposure * np.exp(-2 + .15 * s.astype(float)))
    sev = rng.gamma(3, 600 * np.where(s.isin(["4", "5"]), 1.25, 1), nm)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0, dm.AantalClaims * sev, 0.0)
    ds = sev_rows(dm)
    # the grouping is made next to the average claim, so only the severity
    # dataset has it
    ss = ds.Stedelijkheid.astype(str)
    ds["Sted_groep"] = cat(np.where(ss.isin(["4", "5"]), "4-5", ss))
    mf = fit_glm("AantalClaims ~ Stedelijkheid + LEEFTIJD + offset(log(Exposure))",
                 dm, family="poisson")
    ms = fit_glm("AvgLoss ~ Sted_groep", ds, family="gamma", weights="AantalClaims")
    mv = {"Stedelijkheid": "Sted_groep"}
    assert "Sted_groep" not in dm.columns

    t1 = make_rating_table(mf, ms, data=dm, merge_vars=mv)
    s1 = t1[t1.Variable == "Stedelijkheid"]
    assert len(s1) == 5
    assert s1.Factor_Severity[s1.Level == "4"].iloc[0] == \
        s1.Factor_Severity[s1.Level == "5"].iloc[0]

    # exactly what adding the column by hand would have given
    mp = ms.data[["Stedelijkheid", "Sted_groep"]].drop_duplicates()
    dm2 = dm.merge(mp, on="Stedelijkheid", how="left")
    t2 = make_rating_table(mf, ms, data=dm2, merge_vars=mv)
    pd.testing.assert_frame_equal(t1, t2, rtol=1e-10)

    # and the caller's data frame is left alone
    assert "Sted_groep" not in dm.columns

    nd = _nd(dm, ds, Stedelijkheid="5", Sted_groep="4-5", LEEFTIJD=40.0,
             Exposure=1.0, AantalClaims=1)
    assert (t1.attrs["intercept_premium"]
            * fac(t1, "Stedelijkheid", "5", "Factor_Premium")
            * fac(t1, "LEEFTIJD", "40", "Factor_Premium")) == \
        pytest.approx(mf.predict(nd)[0] * ms.predict(nd)[0], rel=1e-10)

    # a level with no claims at all has no group to derive, and is named
    ds3 = ds[ds.Stedelijkheid != "1"].copy()
    ds3["Sted_groep"] = ds3.Sted_groep.cat.remove_unused_categories()
    ms3 = fit_glm("AvgLoss ~ Sted_groep", ds3, family="gamma", weights="AantalClaims")
    with pytest.raises(PricingError, match="do not occur in the data the severity model"):
        make_rating_table(mf, ms3, data=dm, merge_vars=mv)

    # (R also tests a glm.nb model without model$data, read through the data
    # its call names (DATA-05); a Python GLMFit always carries its data, so
    # there is nothing to port.)

    # a column that exists nowhere is not a merge at all
    with pytest.raises(PricingError, match="no model uses it"):
        make_rating_table(mf, ms, data=dm, merge_vars={"Stedelijkheid": "Sted_xyz"})


def test_a_band_missing_from_data_is_derived_per_nearest_observed_value():
    rng = np.random.default_rng(98)
    nm = 20000
    dm = pd.DataFrame({"LEEFTIJD": np.round(rng.uniform(18, 80, nm)),
                       "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    dm["AantalClaims"] = rng.poisson(dm.Exposure * .25)
    sev = rng.gamma(3, 600 * np.where(dm.LEEFTIJD <= 30, 1.3, 1), nm)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0, dm.AantalClaims * sev, 0.0)
    ds = sev_rows(dm)
    ds["LEEFTIJD_KL"] = r_cut(ds.LEEFTIJD, [17, 30, 50, 80],
                              labels=["18-30", "31-50", "51-80"])
    mf = fit_glm("AantalClaims ~ LEEFTIJD + offset(log(Exposure))", dm,
                 family="poisson")
    ms = fit_glm("AvgLoss ~ LEEFTIJD_KL", ds, family="gamma", weights="AantalClaims")

    t1 = make_rating_table(mf, ms, data=dm, merge_vars={"LEEFTIJD": "LEEFTIJD_KL"},
                           grid_step={"LEEFTIJD": 5})
    s = t1[t1.Variable == "LEEFTIJD"]
    band = lambda age: s.Factor_Severity[s.LevelNum == age].iloc[0]
    assert band(20) == pytest.approx(band(30))
    assert band(35) == pytest.approx(band(50))
    assert band(55) == pytest.approx(band(80))
    assert band(30) != pytest.approx(band(35))
    assert "LEEFTIJD_KL" not in dm.columns


def test_one_column_name_with_two_groupings_is_refused_not_averaged_over():
    rng = np.random.default_rng(99)
    nm = 20000
    dm = pd.DataFrame({"MERK": cat(rng.choice(["A", "B", "C", "D"], nm)),
                       "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    dm["AantalClaims"] = rng.poisson(dm.Exposure * np.exp(
        -2 + .3 * dm.MERK.isin(["C", "D"])))
    sev = rng.gamma(3, 600 * np.where(dm.MERK == "D", 1.4, 1), nm)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0, dm.AantalClaims * sev, 0.0)
    merk = lambda d: d.MERK.astype(str)

    # the same grouping on both sides is fine and stays fine
    df = dm.copy()
    df["MERK_GROEP"] = cat(np.where(merk(df).isin(["A", "B"]), "G1", "G2"))
    ds = sev_rows(df)
    mf = fit_glm("AantalClaims ~ MERK_GROEP + offset(log(Exposure))", df,
                 family="poisson")
    ms = fit_glm("AvgLoss ~ MERK_GROEP", ds, family="gamma", weights="AantalClaims")
    t_ok = make_rating_table(mf, ms, data=dm, merge_vars={"MERK": "MERK_GROEP"})
    nd = pd.DataFrame({"MERK_GROEP": pd.Categorical(["G2"], categories=["G1", "G2"]),
                       "Exposure": [1.0], "AantalClaims": [1]})
    assert t_ok.attrs["intercept_premium"] * fac(t_ok, "MERK", "C", "Factor_Premium") \
        == pytest.approx(mf.predict(nd)[0] * ms.predict(nd)[0], rel=1e-10)

    # same labels, different membership: this used to produce a table whose
    # severity factors came from the frequency grouping, silently
    ds2 = sev_rows(dm)
    ds2["MERK_GROEP"] = cat(np.where(merk(ds2).isin(["A", "B", "C"]), "G1", "G2"))
    ms2 = fit_glm("AvgLoss ~ MERK_GROEP", ds2, family="gamma", weights="AantalClaims")
    with pytest.raises(PricingError, match="does not mean the same thing everywhere"):
        make_rating_table(mf, ms2, data=dm, merge_vars={"MERK": "MERK_GROEP"})
    with pytest.raises(PricingError, match="'C' of 'MERK'"):
        make_rating_table(mf, ms2, data=dm, merge_vars={"MERK": "MERK_GROEP"})

    # different labels: refused with the same message rather than a bare
    # prediction failure further down
    ds3 = sev_rows(dm)
    ds3["MERK_GROEP"] = cat(np.where(merk(ds3).isin(["B", "C"]), "B-C", merk(ds3)))
    ms3 = fit_glm("AvgLoss ~ MERK_GROEP", ds3, family="gamma", weights="AantalClaims")
    with pytest.raises(PricingError, match="does not mean the same thing everywhere"):
        make_rating_table(mf, ms3, data=dm, merge_vars={"MERK": "MERK_GROEP"})

    # the way out: two names, both declared
    df4 = dm.copy()
    df4["MERK_GR_F"] = cat(np.where(merk(df4).isin(["A", "B"]), "A-B", merk(df4)))
    ds4 = sev_rows(dm)
    ds4["MERK_GR_S"] = cat(np.where(merk(ds4).isin(["B", "C"]), "B-C", merk(ds4)))
    mf4 = fit_glm("AantalClaims ~ MERK_GR_F + offset(log(Exposure))", df4,
                  family="poisson")
    ms4 = fit_glm("AvgLoss ~ MERK_GR_S", ds4, family="gamma", weights="AantalClaims")
    t4 = make_rating_table(mf4, ms4, data=dm,
                           merge_vars={"MERK": ["MERK_GR_F", "MERK_GR_S"]})
    s4 = t4[t4.Variable == "MERK"]
    assert len(s4) == 4
    for m in dm.MERK.cat.categories:
        nd4 = pd.DataFrame({
            "MERK_GR_F": pd.Categorical(["A-B" if m in ("A", "B") else m],
                                        categories=df4.MERK_GR_F.cat.categories),
            "MERK_GR_S": pd.Categorical(["B-C" if m in ("B", "C") else m],
                                        categories=ds4.MERK_GR_S.cat.categories),
            "Exposure": [1.0], "AantalClaims": [1]})
        assert t4.attrs["intercept_premium"] \
            * s4.Factor_Premium[s4.Level == m].iloc[0] == \
            pytest.approx(mf4.predict(nd4)[0] * ms4.predict(nd4)[0], rel=1e-10)


def test_two_bandings_of_one_continuous_variable_are_kept_apart():
    rng = np.random.default_rng(100)
    nm = 20000
    dm = pd.DataFrame({"LEEFTIJD": np.round(rng.uniform(18, 80, nm)),
                       "Exposure": np.round(rng.uniform(.3, 1, nm), 3)})
    dm["AantalClaims"] = rng.poisson(dm.Exposure * .25)
    sev = rng.gamma(3, 600 * np.where(dm.LEEFTIJD < 35, 1.3, 1), nm)
    dm["SCHADELAST"] = np.where(dm.AantalClaims > 0, dm.AantalClaims * sev, 0.0)
    band = lambda x, c: r_cut(np.atleast_1d(np.asarray(x, dtype=float)),
                              [17, c, 80], labels=["LAAG", "HOOG"])

    # one name, same labels, different cut points: the strip between the two
    # cuts would get the wrong severity class
    df = dm.copy(); df["LFT_KL"] = band(df.LEEFTIJD, 40)
    ds = sev_rows(dm); ds["LFT_KL"] = band(ds.LEEFTIJD, 50)
    mf = fit_glm("AantalClaims ~ LFT_KL + offset(log(Exposure))", df, family="poisson")
    ms = fit_glm("AvgLoss ~ LFT_KL", ds, family="gamma", weights="AantalClaims")
    with pytest.raises(PricingError, match="does not mean the same thing everywhere"):
        make_rating_table(mf, ms, data=dm, merge_vars={"LEEFTIJD": "LFT_KL"},
                          grid_step={"LEEFTIJD": 5})

    # two names: both bandings land in one block, on the real ages
    df2 = dm.copy(); df2["LFT_F"] = band(df2.LEEFTIJD, 40)
    ds2 = sev_rows(dm); ds2["LFT_S"] = band(ds2.LEEFTIJD, 50)
    mf2 = fit_glm("AantalClaims ~ LFT_F + offset(log(Exposure))", df2, family="poisson")
    ms2 = fit_glm("AvgLoss ~ LFT_S", ds2, family="gamma", weights="AantalClaims")
    t2 = make_rating_table(mf2, ms2, data=dm,
                           merge_vars={"LEEFTIJD": ["LFT_F", "LFT_S"]},
                           grid_step={"LEEFTIJD": 5})
    s2 = t2[t2.Variable == "LEEFTIJD"]
    for a in (20, 45, 60):
        nd = pd.DataFrame({"LFT_F": band(a, 40), "LFT_S": band(a, 50),
                           "Exposure": [1.0], "AantalClaims": [1]})
        assert t2.attrs["intercept_premium"] \
            * s2.Factor_Premium[s2.LevelNum == a].iloc[0] == \
            pytest.approx(mf2.predict(nd)[0] * ms2.predict(nd)[0], rel=1e-10)
    # the age between the two cut points is where the two bandings disagree,
    # and it now carries the severity of its own band
    at = lambda col, a: s2[col][s2.LevelNum == a].iloc[0]
    assert at("Factor_Severity", 45) == pytest.approx(at("Factor_Severity", 20))
    assert at("Factor_Frequency", 45) != pytest.approx(at("Factor_Frequency", 20))


# ---- Python-specific behaviour ----------------------------------------------------


def test_columns_come_in_the_r_order(tbl, tbl2):
    assert list(tbl.columns) == [
        "Variable", "Type", "Level", "LevelNum", "Group", "IsBase", "Exposure",
        "ClaimCount", "Factor_Frequency", "Factor_Severity", "Factor_Premium",
        "SE_Frequency", "SE_Severity", "XVar", "GroupVar", "IsThin"]
    assert list(tbl2.columns) == [
        "Variable", "Type", "Level", "LevelNum", "Group", "IsBase", "Exposure",
        "ClaimCount", "Factor_Frequency", "Factor_Severity", "Factor_Premium",
        "SE_Frequency", "SE_Severity", "Uplift_Frequency", "Uplift_Severity",
        "Uplift_Premium", "XVar", "GroupVar", "IsThin"]
    # the variables in R's term order, not patsy's (which puts the spline last)
    assert list(dict.fromkeys(tbl.Variable)) == ["LEEFTIJD", "REGIO", "BOEKJAAR"]
    assert set(tbl.attrs) == {"intercept_frequency", "intercept_severity",
                              "intercept_premium", "base_values"}
    assert tbl.attrs["intercept_premium"] == pytest.approx(
        tbl.attrs["intercept_frequency"] * tbl.attrs["intercept_severity"])
    assert np.isnan(tbl2.attrs["intercept_premium"])


def test_a_grouped_column_in_data_keeps_its_value_for_a_level_the_model_never_saw():
    # [MV-05] U.grouping_column keeps a grouped column the caller put in
    # 'data' for source values the model's data never saw. The level then
    # gets an empty severity cell (RT-10), as any level the severity model
    # was not fitted on does. (Python had this first; R was fixed to match.)
    dm = _sted_book(93)
    mf = fit_glm("AantalClaims ~ Stedelijkheid + LEEFTIJD + offset(log(Exposure))",
                 dm, family="poisson")
    ds = sev_rows(dm)
    ds = ds[ds.Stedelijkheid != "1"].copy()
    ds["Sted_groep"] = ds.Sted_groep.cat.remove_unused_categories()
    ms = fit_glm("AvgLoss ~ Sted_groep", ds, family="gamma", weights="AantalClaims")
    # (the base row has to be a level both models know, RT-10)
    with pytest.raises(PricingError, match="the base row has 'Sted_groep' = 1"):
        make_rating_table(mf, ms, data=dm, merge_vars={"Stedelijkheid": "Sted_groep"})
    tb, w = collect(make_rating_table, mf, ms, data=dm,
                    merge_vars={"Stedelijkheid": "Sted_groep"},
                    base={"Stedelijkheid": "2"})
    s = tb[tb.Variable == "Stedelijkheid"]
    assert len(s) == 5
    assert np.isnan(s.Factor_Severity[s.Level == "1"].iloc[0])
    assert np.isfinite(s.Factor_Frequency[s.Level == "1"].iloc[0])
    assert np.isfinite(s.Factor_Severity[s.Level == "2"].iloc[0])
    assert any("no severity factor for Stedelijkheid = 1" in x for x in w)
    # without the column in 'data' the level has no group, as in R
    with pytest.raises(PricingError, match="do not occur in the data the severity"):
        make_rating_table(mf, ms, data=dm.drop(columns="Sted_groep"),
                          merge_vars={"Stedelijkheid": "Sted_groep"},
                          base={"Stedelijkheid": "2"})


def test_the_rating_table_does_not_depend_on_the_index_of_data(m_freq, m_sev, dat, tbl):
    d2 = dat.copy()
    d2.index = np.arange(len(d2))[::-1] * 7 + 3
    t2 = make_rating_table(m_freq, m_sev, data=d2)
    pd.testing.assert_frame_equal(t2, tbl, rtol=1e-10)


def test_arguments_are_checked(m_freq, dat):
    with pytest.raises(PricingError, match="at least one model"):
        make_rating_table(None, None, data=dat)
    with pytest.raises(PricingError, match="'trim' must be two quantiles"):
        make_rating_table(m_freq, None, data=dat, trim=(.9, .1))
    with pytest.raises(PricingError, match="'min_claims' must not be negative"):
        make_rating_table(m_freq, None, data=dat, min_claims=-1)
    with pytest.raises(PricingError, match="'year_col' must be one column name"):
        make_rating_table(m_freq, None, data=dat, year_col=["A", "B"])
    with pytest.raises(PricingError, match="column\\(s\\) not found"):
        make_rating_table(m_freq, None, data=dat, claims_col="Schades")


# ---- test-awkward-names.R ------------------------------------------------------------


@pytest.fixture(scope="module")
def daw():
    rng = np.random.default_rng(41)
    naw = 15000
    d = pd.DataFrame({"REGIO": cat(rng.choice(["Stad", "Dorp"], naw)),
                      "LEEFTIJD": np.round(rng.uniform(18, 80, naw)),
                      "POLISJAREN": np.round(rng.uniform(.3, 1, naw), 3)})
    d["AUTO GEWICHT"] = np.round(rng.uniform(800, 2000, naw))
    d["Exposure"] = d.POLISJAREN
    d["AantalClaims"] = rng.poisson(d.POLISJAREN * np.exp(
        -1.7 + 0.3 * (d.REGIO == "Stad") + 0.0002 * (d["AUTO GEWICHT"] - 1400)))
    d["SCHADELAST"] = d.AantalClaims * 1500.0
    return d


def test_a_column_name_with_a_space_survives_the_whole_rating_table(daw):
    m = fit_glm("AantalClaims ~ REGIO + `AUTO GEWICHT` + offset(log(Exposure))",
                daw, family="poisson")
    tb = make_rating_table(m, None, data=daw)
    # the awkward variable gets its own rows, and so do the others
    assert {"REGIO", "AUTO GEWICHT"} <= set(tb.Variable)
    assert (tb.Variable == "AUTO GEWICHT").sum() > 1
    # and the factors are real numbers, not NA
    assert not tb.Factor_Frequency.isna().any()
    # (the R test goes on to make_rating_plot(), make_pdp() and
    # detect_interactions() on the same name; those live in their own
    # modules and test files)


def test_i_terms_resolve_to_their_underlying_column(daw):
    m = fit_glm("AantalClaims ~ REGIO + LEEFTIJD + I(LEEFTIJD^2) + "
                "offset(log(Exposure))", daw, family="poisson")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tb = make_rating_table(m, None, data=daw)
    assert "LEEFTIJD" in set(tb.Variable)
    # LEEFTIJD appears once, not once per term it occurs in
    assert tb.Variable.nunique() == 2


def test_the_intercept_is_per_unit_of_exposure_whatever_the_offset_is_called(daw):
    # exposure_col stays at its default while the model offsets on POLISJAREN
    m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(POLISJAREN))", daw,
                family="poisson")
    with pytest.warns(PricingWarning, match="offsets on"):
        tb = make_rating_table(m, None, data=daw)
    i_f = tb.attrs["intercept_frequency"]
    base_row = pd.DataFrame({
        "LEEFTIJD": [np.median(daw.LEEFTIJD)],
        "REGIO": pd.Categorical([daw.REGIO.cat.categories[0]],
                                categories=daw.REGIO.cat.categories),
        "POLISJAREN": [1.0]})
    assert i_f == pytest.approx(m.predict(base_row)[0], rel=1e-10)
    # and specifically NOT the median-exposure version that used to come out
    base_row["POLISJAREN"] = np.median(daw.POLISJAREN)
    assert i_f != pytest.approx(m.predict(base_row)[0], rel=1e-6)


def test_a_trimmed_grid_excludes_the_policies_it_does_not_cover(daw):
    m = fit_glm("AantalClaims ~ `AUTO GEWICHT` + REGIO + offset(log(Exposure))",
                daw, family="poisson")
    x = daw["AUTO GEWICHT"]

    # untrimmed: every policy is counted somewhere
    tb_full = make_rating_table(m, None, data=daw)
    s_full = tb_full[tb_full.Variable == "AUTO GEWICHT"]
    assert s_full.Exposure.sum() == pytest.approx(daw.Exposure.sum(), rel=1e-8)

    # trimmed: the bars cover exactly the policies inside the grid range,
    # rather than sweeping the tails into the outer bins
    tb_trim = make_rating_table(m, None, data=daw, trim=(.01, .99))
    s_trim = tb_trim[tb_trim.Variable == "AUTO GEWICHT"]
    inside = (x >= s_trim.LevelNum.min()) & (x <= s_trim.LevelNum.max())
    assert s_trim.Exposure.sum() == pytest.approx(daw.Exposure[inside].sum(), rel=1e-8)
    assert s_trim.Exposure.sum() < daw.Exposure.sum()


# ---- test-data-table-input.R -----------------------------------------------------------


def test_make_rating_table_accepts_non_pandas_input():
    # R: a data.table. Python: anything U.as_df() turns into a DataFrame,
    # here a dict of columns.
    rng = np.random.default_rng(23)
    ndt = 20000
    d = pd.DataFrame({"LEEFTIJD": np.round(rng.uniform(18, 80, ndt)),
                      "REGIO": cat(rng.choice(["Stad", "Dorp"], ndt)),
                      "Exposure": np.round(rng.uniform(.3, 1, ndt), 3)})
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(
        -1.6 + 0.02 * np.maximum(0, 30 - d.LEEFTIJD) + 0.3 * (d.REGIO == "Stad")))
    m = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", d,
                family="poisson")
    tb = make_rating_table(m, None, data={c: d[c] for c in d.columns})
    assert {"LEEFTIJD", "REGIO"} <= set(tb.Variable)
    assert tb.IsBase.any()
    pd.testing.assert_frame_equal(tb, make_rating_table(m, None, data=d))


# ---- test-review-findings.R --------------------------------------------------------------


def rv_book(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 85, n)),
        "REGIO": cat(rng.choice(["Noord", "Zuid", "West"], n, p=[.5, .3, .2])),
        "BRANDSTOF": cat(rng.choice(["Benzine", "Diesel", "Waterstof"], n,
                                    p=[.7, .295, .005])),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "BOEKJAAR": rng.integers(2021, 2025, n),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    reg = d.REGIO.astype(str).map({"Noord": 0, "Zuid": .2, "West": .35}).astype(float)
    eta = (np.log(0.08) + 0.6 * np.exp(-(d.LEEFTIJD - 18) / 8) + reg
           + 0.3 * np.log(d.GEWICHT / 1300))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta))
    d["SCHADELAST"] = np.where(d.AantalClaims > 0,
                               d.AantalClaims * rng.gamma(1.5, 2500 / 1.5, n), 0.0)
    d["PREMIE_OUD"] = d.Exposure * 300 * np.exp(0.2 * (d.REGIO == "West"))
    return d


@pytest.fixture(scope="module")
def rv():
    return rv_book()


def test_a_level_without_claims_does_not_drop_the_variable_from_the_table(rv):
    d1 = rv.copy()
    z = d1.BRANDSTOF == "Waterstof"
    d1.loc[z, "AantalClaims"] = 0
    d1.loc[z, "SCHADELAST"] = 0.0
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + BRANDSTOF + offset(log(Exposure))",
                 d1, family="poisson")
    ms = fit_glm("AvgLoss ~ REGIO + BRANDSTOF", sev_rows(d1), family="gamma",
                 weights="AantalClaims")
    tb, w = collect(make_rating_table, mf, ms, data=d1)
    assert "BRANDSTOF" in set(tb.Variable)
    # the severity model never saw the level: an NA cell, said in a warning
    r = tb[(tb.Variable == "BRANDSTOF") & (tb.Level == "Waterstof")]
    assert np.isnan(r.Factor_Severity.iloc[0])
    assert np.isnan(r.Factor_Premium.iloc[0])
    assert np.isfinite(r.Factor_Frequency.iloc[0])
    assert any("no severity factor for BRANDSTOF = Waterstof" in x for x in w)
    # and the level without claims is named among the thin ones
    assert any("no claims at all (BRANDSTOF = Waterstof)" in x for x in w)


def test_cut_in_the_formula_does_not_give_a_table_full_of_na(rv):
    mf = fit_glm("AantalClaims ~ cut(LEEFTIJD, c(17, 24, 40, 65, 99)) + REGIO + "
                 "offset(log(Exposure))", rv, family="poisson")
    tb = quiet(make_rating_table, mf, None, data=rv)
    assert not np.isnan(tb.attrs["intercept_frequency"])
    assert not tb.Factor_Frequency[tb.Variable == "REGIO"].isna().any()
    # the bands are the levels, and intercept x factors is the prediction
    s = tb[tb.Variable == "LEEFTIJD"]
    assert set(s.Level) == {"(17,24]", "(24,40]", "(40,65]", "(65,99]"}
    nd = pd.DataFrame({"LEEFTIJD": [30.0],
                       "REGIO": pd.Categorical(["West"],
                                               categories=rv.REGIO.cat.categories),
                       "Exposure": [1.0]})
    f = lambda v, lv: tb.Factor_Frequency[(tb.Variable == v) & (tb.Level == lv)].iloc[0]
    assert tb.attrs["intercept_frequency"] * f("LEEFTIJD", "(24,40]") * f("REGIO", "West") \
        == pytest.approx(mf.predict(nd)[0], rel=1e-10)
    # the volumes are counted on the bands
    band = r_cut(rv.LEEFTIJD, [17, 24, 40, 65, 99])
    hand = rv.Exposure.groupby(band, observed=False).sum()
    np.testing.assert_allclose(s.Exposure, hand[s.Level].to_numpy(), rtol=1e-10)


def test_a_logical_predictor_does_not_break_the_table(rv):
    d1 = rv.copy()
    d1["GARAGE"] = d1.LEEFTIJD % 3 == 0
    mf = fit_glm("AantalClaims ~ REGIO + GARAGE + offset(log(Exposure))", d1,
                 family="poisson")
    tb = make_rating_table(mf, None, data=d1)
    g = tb[tb.Variable == "GARAGE"]
    assert set(g.Level) == {"FALSE", "TRUE"}
    assert not g.Factor_Frequency.isna().any()


def test_a_logical_predictor_is_a_two_level_rating_factor_with_false_as_base(rv,
                                                                             tmp_path):
    # [RT-14] It used to be read as continuous (in R too, fixed with it):
    # the base was the most frequent value rather than the model's
    # reference, the export wrote the levels as numbers and the rating plot
    # drew a line between FALSE and TRUE.
    from pricingtools_rmo.export import export_rating_table
    from pricingtools_rmo.rating_plot import make_rating_plot
    d1 = rv.copy()
    d1["GARAGE"] = d1.LEEFTIJD % 3 != 0                  # TRUE is the majority
    mf = fit_glm("AantalClaims ~ REGIO + GARAGE + offset(log(Exposure))", d1,
                 family="poisson")
    tb = make_rating_table(mf, None, data=d1)
    g = tb[tb.Variable == "GARAGE"]
    assert g.Type.tolist() == ["categorical", "categorical"]
    assert g.Level.tolist() == ["FALSE", "TRUE"]
    assert g.Level[g.IsBase].tolist() == ["FALSE"]
    assert g.Factor_Frequency[g.Level == "TRUE"].iloc[0] == pytest.approx(
        np.exp(mf.coef["GARAGE[T.True]"]), rel=1e-10)
    # the largest level under base_level = "exposure", or any chosen one
    tx = make_rating_table(mf, None, data=d1, base_level="exposure")
    assert tx.Level[(tx.Variable == "GARAGE") & tx.IsBase].tolist() == ["TRUE"]
    tb2 = make_rating_table(mf, None, data=d1, base={"GARAGE": True})
    assert tb2.Level[(tb2.Variable == "GARAGE") & tb2.IsBase].tolist() == ["TRUE"]
    with pytest.raises(PricingError, match="not a level of 'GARAGE'"):
        make_rating_table(mf, None, data=d1, base={"GARAGE": "ja"})
    # the premium rebuilds from the table
    nd = pd.DataFrame({"REGIO": pd.Categorical(["West"],
                                               categories=rv.REGIO.cat.categories),
                       "GARAGE": [True], "Exposure": [1.0]})
    f = lambda v, lv: tb.Factor_Frequency[(tb.Variable == v) & (tb.Level == lv)].iloc[0]
    assert tb.attrs["intercept_frequency"] * f("REGIO", "West") * f("GARAGE", "TRUE") \
        == pytest.approx(mf.predict(nd)[0], rel=1e-10)
    # the rating plot draws bars, the export writes the levels as text
    assert "bar" in [t.type for t in make_rating_plot(tb, "GARAGE").data]
    openpyxl = pytest.importorskip("openpyxl")
    fx = str(tmp_path / "garage.xlsx")
    export_rating_table(tb, fx)
    ws = openpyxl.load_workbook(fx)["GARAGE"]
    assert [ws.cell(row=r, column=1).value for r in (2, 3)] == ["FALSE", "TRUE"]


def test_trim_does_not_take_exposure_out_of_the_slices_of_a_group_variable(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD * log(GEWICHT) + REGIO + "
                 "offset(log(Exposure))", rv, family="poisson")
    tb = quiet(make_rating_table, mf, None, data=rv, trim=(0.005, 0.995))
    main_e = tb.Exposure[(tb.Variable == "LEEFTIJD") & tb.Group.isna()].sum()
    int_e = tb.Exposure[tb.Group.notna()].sum()
    assert int_e == pytest.approx(main_e, rel=0.01)


def test_the_rating_table_carries_standard_errors_of_its_factors(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", rv,
                 family="poisson")
    tb = make_rating_table(mf, None, data=rv)
    se = np.sqrt(np.diag(mf.vcov.to_numpy()))
    se = dict(zip(mf.vcov.index, se))
    r = tb[tb.Variable == "REGIO"]
    # a categorical level against the reference: the coefficient's own SE
    assert r.SE_Frequency[r.Level == "West"].iloc[0] == \
        pytest.approx(se["REGIO[T.West]"], rel=1e-8)
    assert r.SE_Frequency[r.IsBase].iloc[0] == 0
    # a linear term: |x - base| times the slope's SE
    a = tb[tb.Variable == "LEEFTIJD"]
    i = a.LevelNum == 60
    assert a.SE_Frequency[i].iloc[0] == pytest.approx(
        abs(60 - a.LevelNum[a.IsBase].iloc[0]) * se["LEEFTIJD"], rel=1e-8)
    # no severity model, no severity SE
    assert tb.SE_Severity.isna().all()


def test_the_accounting_year_sits_at_its_latest_level_by_default(rv):
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + factor(BOEKJAAR) + "
                 "offset(log(Exposure))", rv, family="poisson")
    tb = make_rating_table(mf, None, data=rv)
    assert tb.Level[(tb.Variable == "BOEKJAAR") & tb.IsBase].tolist() == ["2024"]
    tb0 = make_rating_table(mf, None, data=rv, year_col=None)
    assert tb0.Level[(tb0.Variable == "BOEKJAAR") & tb0.IsBase].tolist() == ["2021"]
    tb1 = make_rating_table(mf, None, data=rv, base={"BOEKJAAR": "2022"})
    assert tb1.Level[(tb1.Variable == "BOEKJAAR") & tb1.IsBase].tolist() == ["2022"]
    # a year given as a number works the same
    tb2 = make_rating_table(mf, None, data=rv, base={"BOEKJAAR": 2022})
    assert tb2.Level[(tb2.Variable == "BOEKJAAR") & tb2.IsBase].tolist() == ["2022"]


def test_claims_without_payment_are_pointed_out(rv):
    d1 = rv.copy()
    rng = np.random.default_rng(12)
    nil = (d1.AantalClaims > 0) & (rng.uniform(size=len(d1)) < 0.15)
    d1.loc[nil, "SCHADELAST"] = 0.0
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", d1,
                 family="poisson")
    s = sev_rows(d1)
    ms = fit_glm("AvgLoss ~ REGIO", s[s.SCHADELAST > 0], family="gamma",
                 weights="AantalClaims")
    with pytest.warns(PricingWarning, match="without payment"):
        make_rating_table(mf, ms, data=d1)
