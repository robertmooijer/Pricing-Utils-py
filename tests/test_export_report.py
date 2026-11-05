"""export_rating_table() and pricing_report(): the port of
tests/testthat/test-export-report.R, with the export tests of
test-awkward-names.R and the export/report tests of test-review-findings.R."""
import os
import warnings

import numpy as np
import pandas as pd
import pytest

openpyxl = pytest.importorskip("openpyxl")

from pricingtools_rmo._core import PricingMessage, PricingWarning
from pricingtools_rmo.export import export_rating_table
from pricingtools_rmo.model import fit_glm
from pricingtools_rmo.rating_table import make_rating_table
from pricingtools_rmo.report import pricing_report


def quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def read_sheet(path, sheet, header_row=1, n_rows=None):
    """A sheet as a DataFrame, header on `header_row`, cell types kept."""
    ws = openpyxl.load_workbook(path, data_only=True)[sheet]
    rows = list(ws.iter_rows(min_row=header_row, values_only=True))
    head, body = rows[0], rows[1:]
    if n_rows is not None:
        body = body[:n_rows]
    return pd.DataFrame(body, columns=head)


def rv_book(n=20000, seed=1):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 85, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "GEWICHT": np.round(rng.uniform(800, 2200, n), -1),
        "Exposure": np.round(rng.uniform(0.1, 1, n), 3)})
    eta = (np.log(0.08) + 0.6 * np.exp(-(d.LEEFTIJD - 18) / 8)
           + d.REGIO.map({"Noord": 0, "Zuid": .2, "West": .35}).astype(float))
    d["AantalClaims"] = rng.poisson(d.Exposure * np.exp(eta))
    d["SCHADELAST"] = np.where(d.AantalClaims > 0,
                               d.AantalClaims * rng.gamma(1.5, 2500 / 1.5, n), 0.0)
    return d


# ---- test-export-report.R ---------------------------------------------------------


def test_export_rating_table_writes_a_structured_workbook(tbl, tbl2, tmp_path):
    f = str(tmp_path / "rating.xlsx")
    export_rating_table(tbl, f)
    assert os.path.exists(f)
    sn = openpyxl.load_workbook(f).sheetnames
    assert {"Overview", "REGIO", "LEEFTIJD", "BOEKJAAR"} <= set(sn)
    f2 = str(tmp_path / "rating_int.xlsx")
    export_rating_table(tbl2, f2)
    assert any("LEEFTIJD" in s and "REGIO" in s
               for s in openpyxl.load_workbook(f2).sheetnames)


def test_the_tariff_sheet_rebuilds_a_premium_by_lookup(tbl, tmp_path):
    f = str(tmp_path / "tariff.xlsx")
    export_rating_table(tbl, f)
    assert "Tariff" in openpyxl.load_workbook(f).sheetnames
    # row 1 holds the how-to note, so the header starts on row 2
    tf = read_sheet(f, "Tariff", header_row=2)
    assert {"Key", "Variable", "Level", "Factor_Premium"} <= set(tf.columns)
    assert len(tf) == len(tbl)
    # every main-effect key is unique, which is what VLOOKUP needs
    main = tf[tf.Group.isna()]
    assert not main.Key.duplicated().any()
    # and the factors multiply back to the model, exactly as on the
    # per-variable sheets
    s = tbl[(tbl.Variable == "LEEFTIJD") & tbl.Group.isna()].reset_index(drop=True)
    age = s.Level[9]
    got = tf.Factor_Frequency[tf.Key == "LEEFTIJD|" + str(age)].iloc[0]
    assert got == pytest.approx(s.Factor_Frequency[9], rel=1e-12)


def test_continuous_levels_are_written_as_numbers_not_text(tbl, tbl2, tmp_path):
    # Level is text in the table because a categorical level is text, but
    # for a continuous variable Excel would then flag every cell as
    # "number stored as text"
    f = str(tmp_path / "levels.xlsx")
    export_rating_table(tbl, f)

    def block(sheet, var):
        nr = int(((tbl.Variable == var) & tbl.Group.isna()).sum())
        return read_sheet(f, sheet, n_rows=nr)
    assert all(isinstance(v, (int, float)) for v in block("LEEFTIJD", "LEEFTIJD").Level)
    assert all(isinstance(v, str) for v in block("REGIO", "REGIO").Level)
    # a factor() term keeps its levels as text, which is correct
    assert all(isinstance(v, str) for v in block("BOEKJAAR", "BOEKJAAR").Level)

    # interaction sheet: the X axis is the continuous one here
    f2 = str(tmp_path / "levels_int.xlsx")
    export_rating_table(tbl2, f2)
    iv = tbl2.Variable[tbl2.Group.notna()].iloc[0]
    sn = [s for s in openpyxl.load_workbook(f2).sheetnames
          if "LEEFTIJD" in s and "REGIO" in s][0]
    nr = int(((tbl2.Variable == iv) & tbl2.Group.notna()).sum())
    xi = read_sheet(f2, sn, n_rows=nr)
    assert all(isinstance(v, (int, float)) for v in xi.Level)
    assert all(isinstance(v, str) for v in xi.Group)


def test_the_matrix_under_an_interaction_keeps_numeric_levels_too(tbl2, tmp_path):
    f = str(tmp_path / "pivot.xlsx")
    quiet(export_rating_table, tbl2, f)
    iv = tbl2.Variable[tbl2.Group.notna()].iloc[0]
    import re
    sn = re.sub(r"[^A-Za-z0-9 _.-]", "_", iv)
    nr = int(((tbl2.Variable == iv) & tbl2.Group.notna()).sum())
    piv = read_sheet(f, sn, header_row=nr + 4)
    piv = piv[piv.Level.notna()]
    assert len(piv) > 0
    assert all(isinstance(v, (int, float)) for v in piv.Level)


def test_pricing_report_writes_a_complete_html_report(m_freq, m_sev, dat, tmp_path):
    f = str(tmp_path / "report.html")
    quiet(pricing_report, m_freq, m_sev, dat, file=f, grid_res=20)
    assert os.path.exists(f)
    assert os.path.getsize(f) > 50000
    html = _read(f)
    assert "Model diagnostics" in html
    assert "var-REGIO" in html
    assert "Rating factors" in html
    assert "plotly-graph-div" in html
    assert os.path.exists(str(tmp_path / "report_files" / "plotly.min.js"))


def test_the_one_way_block_covers_severity_not_just_frequency(m_freq, m_sev, dat,
                                                             tmp_path):
    f = str(tmp_path / "oneway.html")
    quiet(pricing_report, m_freq, m_sev, dat, file=f, include="oneway")
    html = _read(f)
    assert "One-way observed – frequency" in html
    assert "One-way observed – severity" in html
    # without a severity model there is nothing to read it against, so the
    # heaviest block is not doubled for no reason
    f2 = str(tmp_path / "oneway_freq.html")
    quiet(pricing_report, m_freq, None, dat, file=f2, include="oneway")
    html2 = _read(f2)
    assert "One-way observed – frequency" in html2
    assert "One-way observed – severity" not in html2


# ---- test-awkward-names.R ------------------------------------------------------------


def test_export_sheet_names_are_deduplicated_case_insensitively(dat, tmp_path):
    m = fit_glm("AantalClaims ~ REGIO + offset(log(Exposure))", dat)
    tb = make_rating_table(m, None, data=dat)
    tb2 = pd.concat([tb, tb.assign(Variable="Regio")], ignore_index=True)
    tb2.attrs = tb.attrs
    f = str(tmp_path / "case.xlsx")
    export_rating_table(tb2, f)
    sn = openpyxl.load_workbook(f).sheetnames
    assert len(sn) == len({s.lower() for s in sn})


def test_export_rating_table_accepts_digits_0(dat, tmp_path):
    m = fit_glm("AantalClaims ~ REGIO + offset(log(Exposure))", dat)
    tb = make_rating_table(m, None, data=dat)
    f = str(tmp_path / "digits0.xlsx")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        export_rating_table(tb, f, digits=0)
    assert os.path.exists(f)


# ---- test-review-findings.R --------------------------------------------------------------


def test_the_tariff_sheet_instruction_rebuilds_the_premium_with_an_interaction(tmp_path):
    rv = rv_book()
    mf = fit_glm("AantalClaims ~ LEEFTIJD * REGIO + offset(log(Exposure))", rv)
    tb = make_rating_table(mf, None, data=rv)
    f = str(tmp_path / "t.xlsx")
    export_rating_table(tb, f)
    tf = read_sheet(f, "Tariff", header_row=2)
    p = rv.iloc[:200]

    def look(keys):
        m = dict(zip(tf.Key, tf.Factor_Frequency))
        return np.array([m[k] for k in keys], dtype=float)
    iv = tb.Variable[tb.Group.notna()].iloc[0]
    lv = [str(int(a)) for a in p.LEEFTIJD]
    lit = (tb.attrs["intercept_frequency"]
           * look(["LEEFTIJD|" + a for a in lv])
           * look(["REGIO|" + str(r) for r in p.REGIO])
           * look([iv + "|" + a + "|" + str(r) for a, r in zip(lv, p.REGIO)]))
    np.testing.assert_allclose(lit, mf.predict(p.assign(Exposure=1.0)), rtol=1e-6)


def test_the_intercepts_on_the_overview_sheet_are_numbers_not_text(tmp_path):
    rv = rv_book()
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", rv)
    f = str(tmp_path / "o.xlsx")
    export_rating_table(make_rating_table(mf, None, data=rv), f)
    v = openpyxl.load_workbook(f)["Overview"].cell(row=3, column=2).value
    assert isinstance(v, float)


def test_pricing_report_writes_a_report_for_a_small_book_too(tmp_path):
    d1 = rv_book(n=400, seed=5)
    mf = fit_glm("AantalClaims ~ LEEFTIJD + REGIO + offset(log(Exposure))", d1)
    f = str(tmp_path / "small.html")
    quiet(pricing_report, mf, None, d1, file=f, grid_res=10)
    assert os.path.exists(f)


def test_the_report_passes_merge_vars_and_base_to_its_rating_table(tmp_path):
    d1 = rv_book()
    d1["REGIO_GRP"] = np.where(d1.REGIO == "Noord", "Noord", "Rest")
    mf = fit_glm("AantalClaims ~ REGIO + LEEFTIJD + offset(log(Exposure))", d1)
    s1 = d1[d1.AantalClaims > 0].copy()
    s1["AvgLoss"] = s1.SCHADELAST / s1.AantalClaims
    ms = fit_glm("AvgLoss ~ REGIO_GRP", s1, family="gamma", weights="AantalClaims")
    f = str(tmp_path / "merge.html")

    def rep_warn(**kw):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pricing_report(mf, ms, d1, file=f, grid_res=10, include="rating", **kw)
        return [str(x.message) for x in w if issubclass(x.category, PricingWarning)]
    # without the declaration the report's table holds two blocks, and says so
    assert any("looks like a regrouping" in m for m in rep_warn())
    # with it, one block, exactly as make_rating_table() builds it
    assert not any("looks like a regrouping" in m
                   for m in rep_warn(merge_vars={"REGIO": "REGIO_GRP"}))
    assert os.path.exists(f)
