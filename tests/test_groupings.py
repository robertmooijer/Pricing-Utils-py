"""Tests for add_groupings(): the port of tests/testthat/test-groupings.R.

add_groupings() adds the grouped columns a model was fitted on to a dataset
that only has the source columns, reading the relation off the data each
model carries. It shares its code with merge_vars in make_rating_table(),
so the table and a predict() on new data agree."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from pricingtools_rmo import PricingError, PricingWarning
from pricingtools_rmo._formula import r_cut
from pricingtools_rmo.groupings import add_groupings
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.rating_table import make_rating_table

LADDER = np.array([1, .9, .8, .7, .6])


def cat(values, categories=None):
    values = np.asarray(values, dtype=object)
    if categories is None:
        categories = sorted(set(values.tolist()))
    return pd.Categorical(values, categories=categories)


@pytest.fixture(scope="module")
def g():
    rng = np.random.default_rng(88)
    ng = 30000
    dg = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, ng)),
        "REGIO": cat(rng.choice(["Noord", "Zuid", "Randstad"], ng)),
        "Stedelijkheid": cat(rng.integers(1, 6, ng).astype(str)),
        "MERK": cat(rng.choice(["BMW", "MERCEDES", "NISSAN", "TOYOTA"], ng)),
        "BM_KLASSE": rng.integers(1, 6, ng),
        "Exposure": np.round(rng.uniform(.1, 1, ng), 3)})
    dg["BM_KORTING"] = LADDER[dg.BM_KLASSE - 1]
    dg["AantalClaims"] = rng.poisson(dg.Exposure * dg.BM_KORTING * np.exp(
        -2.1 + .012 * (dg.LEEFTIJD - 40) + .2 * (dg.MERK == "NISSAN")))
    sev = rng.gamma(3, 700 * np.where(dg.MERK == "MERCEDES", 1.4, 1), ng)
    dg["SCHADELAST"] = np.where(dg.AantalClaims > 0, dg.AantalClaims * sev, 0.0)

    merk = lambda d: d.MERK.astype(str)
    gf = dg.copy()
    gf["MERK_GR_F"] = cat(np.where(merk(gf).isin(["BMW", "MERCEDES"]),
                                   "BMW_MERCEDES", merk(gf)))
    g_freq = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + MERK_GR_F + REGIO + "
                     "offset(log(Exposure)) + offset(log(BM_KORTING))", gf,
                     family="poisson")
    gs = dg[dg.AantalClaims > 0].copy()
    gs["AvgLoss"] = gs.SCHADELAST / gs.AantalClaims
    gs["MERK_GR_S"] = cat(np.where(merk(gs).isin(["BMW", "NISSAN"]),
                                   "BMW_NISSAN", merk(gs)))
    st = gs.Stedelijkheid.astype(str)
    gs["Stedelijkheid_groep"] = cat(np.where(st.isin(["4", "5"]), "4-5", st))
    gs["LFT_KL"] = r_cut(gs.LEEFTIJD, [17, 30, 50, 80],
                         labels=["18-30", "31-50", "51-80"])
    g_sev = fit_glm("AvgLoss ~ MERK_GR_S + Stedelijkheid_groep + LFT_KL", gs,
                    family="gamma", weights="AantalClaims")
    gmv = {"Stedelijkheid": "Stedelijkheid_groep",
           "MERK": ["MERK_GR_F", "MERK_GR_S"],
           "LEEFTIJD": "LFT_KL"}
    return dict(dg=dg, gs=gs, g_freq=g_freq, g_sev=g_sev, gmv=gmv)


def new_pf():
    d = pd.DataFrame({"LEEFTIJD": [22.0, 30, 31, 50, 64],
                      "REGIO": ["Noord", "Zuid", "Randstad", "Noord", "Zuid"],
                      "Stedelijkheid": [1, 3, 4, 5, 2],
                      "MERK": ["BMW", "MERCEDES", "NISSAN", "TOYOTA", "BMW"],
                      "BM_KLASSE": [1, 3, 5, 2, 4]})
    d["Exposure"] = 1.0
    d["BM_KORTING"] = LADDER[d.BM_KLASSE - 1]
    return d


def as_str(x):
    return [str(v) for v in x]


def test_the_grouped_columns_are_derived_from_the_source_columns(g):
    nd = add_groupings(new_pf(), g["g_freq"], g["g_sev"], merge_vars=g["gmv"])
    assert as_str(nd.Stedelijkheid_groep) == ["1", "3", "4-5", "4-5", "2"]
    assert as_str(nd.MERK_GR_F) == ["BMW_MERCEDES", "BMW_MERCEDES", "NISSAN",
                                    "TOYOTA", "BMW_MERCEDES"]
    assert as_str(nd.MERK_GR_S) == ["BMW_NISSAN", "MERCEDES", "BMW_NISSAN",
                                    "TOYOTA", "BMW_NISSAN"]
    # the band edges land on the right side
    assert as_str(nd.LFT_KL) == ["18-30", "18-30", "31-50", "31-50", "51-80"]
    # with the types and levels the models were fitted on
    assert list(nd.MERK_GR_S.cat.categories) == list(g["gs"].MERK_GR_S.cat.categories)
    assert list(nd.LFT_KL.cat.categories) == list(g["gs"].LFT_KL.cat.categories)

    # and predict() runs on it, matching the rating table built from the
    # same declaration, BM applied on top
    prem = g["g_freq"].predict(nd) * g["g_sev"].predict(nd)
    tb = make_rating_table(g["g_freq"], g["g_sev"], data=g["dg"],
                           merge_vars=g["gmv"], grid_step={"LEEFTIJD": 1})

    def fac(v, lvl):
        s = tb[(tb.Variable == v) & tb.Group.isna()]
        return s.Factor_Premium[s.Level == str(lvl)].iloc[0]

    def lv(x):
        return str(int(x)) if float(x).is_integer() else str(x)

    recon = [tb.attrs["intercept_premium"] * fac("LEEFTIJD", lv(r.LEEFTIJD))
             * fac("REGIO", r.REGIO) * fac("Stedelijkheid", r.Stedelijkheid)
             * fac("MERK", r.MERK) * r.BM_KORTING for r in nd.itertuples()]
    np.testing.assert_allclose(recon, prem, rtol=1e-10)


def test_anything_that_would_be_a_guess_is_refused(g):
    gf, gsv, gmv = g["g_freq"], g["g_sev"], g["gmv"]
    b = new_pf(); b.loc[1, "Stedelijkheid"] = 6
    with pytest.raises(PricingError,
                       match="6 of 'Stedelijkheid' do not occur in the data the severity"):
        add_groupings(b, gf, gsv, merge_vars=gmv)
    b = new_pf(); b.loc[2, "MERK"] = "TESLA"
    with pytest.raises(PricingError,
                       match="TESLA of 'MERK' do not occur in the data the frequency"):
        add_groupings(b, gf, gsv, merge_vars=gmv)
    # a column already present is checked, never overwritten
    b = new_pf()
    b["MERK_GR_F"] = ["BMW_MERCEDES", "MERCEDES", "NISSAN", "TOYOTA", "BMW_MERCEDES"]
    # rows are counted from 0, as everywhere in the Python package (R: row 2)
    with pytest.raises(PricingError, match=r"row 1 \(0-based\) has 'MERK' = MERCEDES"):
        add_groupings(b, gf, gsv, merge_vars=gmv)
    b.loc[1, "MERK_GR_F"] = "BMW_MERCEDES"
    ok = add_groupings(b, gf, gsv, merge_vars=gmv)
    pd.testing.assert_series_equal(ok.MERK_GR_F, b.MERK_GR_F)

    # a numeric code is looked up exactly, not given a neighbour's group
    gs2 = g["gs"].copy()
    gs2["Stedelijkheid"] = gs2.Stedelijkheid.astype(str).astype(float)
    s2 = fit_glm("AvgLoss ~ MERK_GR_S + Stedelijkheid_groep + LFT_KL", gs2,
                 family="gamma", weights="AantalClaims")
    b = new_pf(); b.loc[1, "Stedelijkheid"] = 6
    with pytest.raises(PricingError, match="do not occur"):
        add_groupings(b, gf, s2, merge_vars=gmv)

    # the arguments themselves
    with pytest.raises(PricingError, match="at least one model"):
        add_groupings(new_pf(), merge_vars=gmv)
    with pytest.raises(PricingError, match="'merge_vars' is required"):
        add_groupings(new_pf(), gf, gsv)
    with pytest.raises(PricingError, match="must be named"):
        add_groupings(new_pf(), gf, gsv, merge_vars=["LFT_KL"])
    with pytest.raises(PricingError, match="onto itself"):
        add_groupings(new_pf(), gf, gsv, merge_vars={"LEEFTIJD": "LEEFTIJD"})
    with pytest.raises(PricingError, match="no model uses it"):
        add_groupings(new_pf(), gf, gsv, merge_vars={"LEEFTIJD": "LFT_XX"})
    with pytest.raises(PricingError, match="must be glm objects"):
        add_groupings(new_pf(), "a model", merge_vars=gmv)
    with pytest.raises(PricingError, match="declared more than once"):
        add_groupings(new_pf(), gf, gsv,
                      merge_vars={"LEEFTIJD": "LFT_KL", "MERK": "LFT_KL"})


def test_one_name_grouped_two_ways_by_the_two_models_is_refused(g):
    dg = g["dg"]
    merk = lambda d: d.MERK.astype(str)
    f2 = dg.copy()
    f2["MG"] = cat(np.where(merk(f2).isin(["BMW", "MERCEDES"]), "G1", "G2"))
    s2 = dg[dg.AantalClaims > 0].copy()
    s2["AvgLoss"] = s2.SCHADELAST / s2.AantalClaims
    s2["MG"] = cat(np.where(merk(s2).isin(["BMW", "MERCEDES", "NISSAN"]), "G1", "G2"))
    mf2 = fit_glm("AantalClaims ~ MG + offset(log(Exposure))", f2, family="poisson")
    ms2 = fit_glm("AvgLoss ~ MG", s2, family="gamma", weights="AantalClaims")
    with pytest.raises(PricingError, match="does not mean the same thing "
                                           "everywhere: 'NISSAN' of 'MERK'"):
        add_groupings(new_pf(), mf2, ms2, merge_vars={"MERK": "MG"})
    # the same grouping on both sides is fine
    s3 = dg[dg.AantalClaims > 0].copy()
    s3["AvgLoss"] = s3.SCHADELAST / s3.AantalClaims
    s3["MG"] = cat(np.where(merk(s3).isin(["BMW", "MERCEDES"]), "G1", "G2"))
    ms3 = fit_glm("AvgLoss ~ MG", s3, family="gamma", weights="AantalClaims")
    nd = add_groupings(new_pf(), mf2, ms3, merge_vars={"MERK": "MG"})
    assert as_str(nd.MG) == ["G1", "G1", "G2", "G2", "G1"]


def test_a_continuous_source_outside_the_fitted_range_takes_the_outer_band(g):
    gf, gsv, gmv = g["g_freq"], g["g_sev"], g["gmv"]
    b = new_pf()
    b["LEEFTIJD"] = [15.0, 30, 31, 50, 85]
    with pytest.warns(PricingWarning, match="1 row above 80 got '51-80' and 1 row "
                                            "below 18 got '18-30'"):
        nd = add_groupings(b, gf, gsv, merge_vars=gmv)
    assert as_str(nd.LFT_KL) == ["18-30", "18-30", "31-50", "31-50", "51-80"]
    # and the spline that also uses the age is named, with the remedy
    with pytest.warns(PricingWarning, match="continuous term of the frequency model"):
        add_groupings(b, gf, gsv, merge_vars=gmv)
    with pytest.warns(PricingWarning, match=r"pmin\(pmax\(LEEFTIJD, 18\), 80\)"):
        add_groupings(b, gf, gsv, merge_vars=gmv)
    # inside the range nothing is said
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        add_groupings(new_pf(), gf, gsv, merge_vars=gmv)


def test_the_input_is_left_alone(g):
    # R: "a data.table comes back as one, and the input is left alone". A
    # pandas frame comes back as a new frame; the caller's is not touched.
    d = new_pf()
    before = d.copy()
    out = add_groupings(d, g["g_freq"], g["g_sev"], merge_vars=g["gmv"])
    assert isinstance(out, pd.DataFrame)
    assert {"LFT_KL", "MERK_GR_F", "MERK_GR_S", "Stedelijkheid_groep"} <= set(out.columns)
    assert "LFT_KL" not in d.columns
    pd.testing.assert_frame_equal(d, before)
    # a dict of columns is read as a data frame, as everywhere in the package
    out2 = add_groupings({c: d[c].tolist() for c in d.columns}, g["g_freq"],
                         g["g_sev"], merge_vars=g["gmv"])
    pd.testing.assert_frame_equal(out2.reset_index(drop=True),
                                  out.reset_index(drop=True), check_dtype=False)


# ---- Python-specific behaviour ----------------------------------------------------


def test_a_grouped_column_already_in_data_is_kept_for_a_value_the_model_never_saw(g):
    # Deliberate deviation from R: a grouped column the caller supplies keeps
    # its value where the model's data cannot check it (R stops there). The
    # value is then the caller's responsibility; predict() will name it.
    b = new_pf()
    b.loc[2, "MERK"] = "TESLA"
    b["MERK_GR_F"] = ["BMW_MERCEDES", "BMW_MERCEDES", "TESLA", "TOYOTA",
                      "BMW_MERCEDES"]
    out = add_groupings(b, g["g_freq"], None, merge_vars={"MERK": "MERK_GR_F"})
    assert out.MERK_GR_F.tolist() == b.MERK_GR_F.tolist()
    # a value the model's data does know is still checked
    b.loc[0, "MERK_GR_F"] = "NISSAN"
    with pytest.raises(PricingError, match=r"row 0 \(0-based\) has 'MERK' = BMW"):
        add_groupings(b, g["g_freq"], None, merge_vars={"MERK": "MERK_GR_F"})


def test_a_column_no_model_uses_is_left_as_it_is(g):
    b = new_pf()
    b["EXTRA"] = ["a", "b", "c", "d", "e"]
    with pytest.warns(PricingWarning, match="no model uses 'EXTRA', so it is left"):
        out = add_groupings(b, g["g_freq"], g["g_sev"], merge_vars={"MERK": "EXTRA"})
    assert out.EXTRA.tolist() == b.EXTRA.tolist()
    # and a named Series works as a dict, as R's named vector does
    out = add_groupings(new_pf(), g["g_freq"], g["g_sev"],
                        merge_vars=pd.Series({"LEEFTIJD": "LFT_KL"}))
    assert as_str(out.LFT_KL) == ["18-30", "18-30", "31-50", "31-50", "51-80"]
