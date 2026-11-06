"""Regenerate the static figures used in README.md.

The plots are interactive plotly figures, which GitHub cannot render in a
README. This script writes each figure to an HTML page, takes a screenshot
of it with headless Chrome (or Edge) and saves a PNG to docs/figures/. Tables
are rendered as a styled HTML table, as the Excel export styles them.
The counterpart of demo/make_readme_figures.R in the R package; the
portfolio and the figures are the same, drawn by the Python package.

Requires: Pillow (pip install -e ".[figures]") and a Chrome or Edge
installation; set CHROME to its path if it is not found.

Run from the project root:  python demo/make_readme_figures.py
"""
from __future__ import annotations

import html as _html
import os
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

import numpy as np
import pandas as pd
import plotly
from PIL import Image, ImageChops

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pricingtools_rmo import (agg_all, detect_interactions, double_lift,  # noqa: E402
                              fit_glm, glm_collinearity, glm_diagnostics,
                              make_pdp, make_plot, make_rating_plot,
                              make_rating_table, model_lift, plot_glm_influence,
                              plot_glm_predictor, plot_glm_qq, plot_glm_residuals,
                              plot_residual_heatmap, premium_impact,
                              pricing_report, screen_features, ta_blue)

OUT_DIR = os.path.join("docs", "figures")
os.makedirs(OUT_DIR, exist_ok=True)
TMP = tempfile.mkdtemp(prefix="readme_figs_")
warnings.simplefilter("ignore")


# ---- the browser -------------------------------------------------------------------

def _find_chrome():
    cands = [os.environ.get("CHROME"),
             r"C:\Program Files\Google\Chrome\Application\chrome.exe",
             r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
             os.path.expandvars(r"%LOCALAPPDATA%\Google\Chrome\Application\chrome.exe"),
             r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
             r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
             "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"]
    cands += [shutil.which(n) for n in ("google-chrome", "chromium", "chromium-browser",
                                         "chrome", "msedge")]
    for c in cands:
        if c and os.path.exists(c):
            return c
    sys.exit("No Chrome or Edge found; set the CHROME environment variable.")


CHROME = _find_chrome()


def _screenshot(page, png, width, height, scale):
    """Headless screenshot of a local page. A profile of its own keeps the
    call away from a Chrome window that is already open."""
    if os.path.exists(png):
        os.remove(png)
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                    "--no-first-run", "--no-default-browser-check",
                    "--user-data-dir=" + os.path.join(TMP, "profile"),
                    "--force-device-scale-factor=%s" % scale,
                    "--window-size=%d,%d" % (width, height),
                    "--virtual-time-budget=5000", "--screenshot=" + png,
                    "file:///" + os.path.abspath(page).replace("\\", "/")],
                   capture_output=True, timeout=300)
    if not os.path.exists(png):
        sys.exit("The browser wrote no screenshot of " + page)
    return Image.open(png).convert("RGB")


def _report(png):
    print("written: %s - %d KB" % (png, round(os.path.getsize(png) / 1024)))


# The browser's window is a little larger than the page it shows, so the
# page is drawn at a fixed size in a larger window and cut out exactly.
_PAD = (60, 160)

with open(os.path.join(TMP, "plotly.min.js"), "w", encoding="utf-8") as fh:
    fh.write(plotly.offline.get_plotlyjs())


def snap(fig, name, width=900, height=460):
    """One plotly figure as a PNG, the mode bar left out, at 1.5x."""
    fig.update_layout(width=width, height=height, autosize=False)
    div = fig.to_html(full_html=False, include_plotlyjs=False,
                      config={"displayModeBar": False, "staticPlot": True})
    page = os.path.join(TMP, name + ".html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write('<!doctype html><html><head><meta charset="utf-8">'
                 '<script src="plotly.min.js"></script>'
                 '<style>body{margin:0;background:#fff}</style></head>'
                 '<body>' + div + '</body></html>')
    s = 1.5
    img = _screenshot(page, os.path.join(TMP, name + ".png"),
                      width + _PAD[0], height + _PAD[1], s)
    png = os.path.join(OUT_DIR, "README-%s.png" % name)
    img.crop((0, 0, round(width * s), round(height * s))).save(png, optimize=True)
    _report(png)


def _fmt_cols(df, digits):
    """Per column, as export_rating_table() formats: counts with a thousands
    separator and no decimals, other numbers with decimals."""
    out = {}
    for nm in df.columns:
        x = df[nm]
        if pd.api.types.is_bool_dtype(x) or not pd.api.types.is_numeric_dtype(x):
            out[nm] = "chr"
            continue
        v = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
        fin = v[np.isfinite(v)]
        if not len(fin):
            out[nm] = "chr"
        elif np.all(np.abs(fin - np.round(fin)) < 1e-9):
            out[nm] = "int"
        else:
            out[nm] = "big" if np.max(np.abs(fin)) >= 1000 else "dec"
    return out


def _cell(v, kind, digits):
    if v is None or (isinstance(v, float) and np.isnan(v)) or v is pd.NA:
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if kind == "int":
        return "{:,.0f}".format(float(v))
    if kind == "big":
        return "{:,.1f}".format(float(v))
    if kind == "dec":
        return "{:.{d}f}".format(float(v), d=digits)
    return str(v)


def snap_table(df, name, digits=3, vwidth=1800):
    """A data frame excerpt as a styled table image, highlighted as
    export_rating_table() does: base row light blue, thin rows grey italic."""
    df = df.reset_index(drop=True)
    kinds = _fmt_cols(df, digits)
    rows = []
    def flag(col, i):
        if col not in df.columns:
            return False
        v = df[col].iloc[i]
        return False if v is None or pd.isna(v) else bool(v)

    for i in range(len(df)):
        cls = "base" if flag("IsBase", i) else "thin" if flag("IsThin", i) else ""
        tds = "".join("<td>%s</td>" % _html.escape(_cell(df[c].iloc[i], kinds[c], digits))
                      for c in df.columns)
        rows.append('<tr class="%s">%s</tr>' % (cls, tds))
    head = "".join("<th>%s</th>" % _html.escape(str(c)) for c in df.columns)
    page = os.path.join(TMP, name + ".html")
    with open(page, "w", encoding="utf-8") as fh:
        fh.write("""<!doctype html><html><head><meta charset="utf-8"><style>
body{margin:0;background:#fff;font-family:'Segoe UI',Arial,sans-serif;}
.wrap{display:inline-block;padding:14px;}
table{border-collapse:collapse;font-size:13px;}
th{background:#00365E;color:#fff;padding:7px 11px;text-align:left;white-space:nowrap;}
td{border-bottom:1px solid #D0D8E0;padding:6px 11px;white-space:nowrap;}
tr.base td{background:#A8C8E0;font-weight:bold;}
tr.thin td{color:#999;font-style:italic;}
</style></head><body><div class="wrap"><table><thead><tr>%s</tr></thead>
<tbody>%s</tbody></table></div></body></html>""" % (head, "".join(rows)))
    s = 2
    img = _screenshot(page, os.path.join(TMP, name + ".png"), vwidth,
                      (len(df) + 2) * 34 + 200, s)
    # cut out the table with the 14px padding of .wrap around it
    box = ImageChops.difference(img, Image.new("RGB", img.size, "white")).getbbox()
    pad = 14 * s
    box = (max(box[0] - pad, 0), max(box[1] - pad, 0),
           min(box[2] + pad, img.size[0]), min(box[3] + pad, img.size[1]))
    png = os.path.join(OUT_DIR, "README-%s.png" % name)
    img.crop(box).save(png, optimize=True)
    _report(png)


def snap_page(page, name, width=1100, height=850):
    """Screenshot of an HTML page, viewport only, for the report."""
    s = 1.5
    img = _screenshot(page, os.path.join(TMP, name + ".png"), width + 14,
                      height + 93, s)
    png = os.path.join(OUT_DIR, "README-%s.png" % name)
    img.crop((0, 0, min(round(width * s), img.size[0]),
              min(round(height * s), img.size[1]))).save(png, optimize=True)
    _report(png)


# ---- portfolio (the structure of the R script, numpy's own random draws) ---------

rng = np.random.default_rng(2026)
n = 40000
regio_lv = ["Noord", "Oost", "Zuid", "West", "Randstad"]
fuel_lv = ["Benzine", "Diesel", "Elektrisch", "Waterstof"]
dat = pd.DataFrame({
    "LEEFTIJD": np.clip(np.round(rng.normal(46, 15, n)), 18, 85),
    "REGIO": pd.Categorical(rng.choice(regio_lv, n, p=[.15, .18, .22, .20, .25]),
                            categories=sorted(regio_lv)),
    "GEWICHT": np.round(np.clip(rng.normal(1350, 250, n), 800, 2400), -1),
    "BRANDSTOF": pd.Categorical(rng.choice(fuel_lv, n, p=[.623, .25, .125, .002]),
                                categories=fuel_lv),
    "BOEKJAAR": rng.integers(2020, 2025, n),
    "Exposure": np.round(rng.uniform(0.05, 1, n), 3)})
reg = dat.REGIO.astype(str)
fuel = dat.BRANDSTOF.astype(str)
L = dat.LEEFTIJD.to_numpy()
reg_f = reg.map({"Noord": 0, "Oost": .05, "Zuid": .12, "West": .08, "Randstad": .28})
fuel_f = fuel.map({"Benzine": 0, "Diesel": .10, "Elektrisch": -.05, "Waterstof": 0})
lin_f = (-2.1 + 0.035 * (30 - np.minimum(L, 30)) + 0.012 * (np.maximum(L, 65) - 65)
         + reg_f + 0.00025 * (dat.GEWICHT - 1350) + fuel_f
         + 0.012 * (30 - np.minimum(L, 30)) * (reg == "Randstad"))
dat["AantalClaims"] = rng.poisson(dat.Exposure * np.exp(lin_f))
fuel_s = fuel.map({"Benzine": 0, "Diesel": .05, "Elektrisch": .35, "Waterstof": .45})
mu_s = 1800 * np.exp(0.0004 * (dat.GEWICHT - 1350) + fuel_s)
dat["SCHADELAST"] = np.where(dat.AantalClaims > 0,
                             rng.gamma(1.6, mu_s / 1.6) * dat.AantalClaims, 0.0)
reg_old = reg.map({"Noord": .90, "Oost": .95, "Zuid": 1.05, "West": 1.00,
                   "Randstad": 1.15})
dat["HUIDIGE_PREMIE"] = dat.Exposure * 260 * reg_old

# ---- models --------------------------------------------------------------------------

F_FREQ = ("AantalClaims ~ ns(LEEFTIJD, 5) + REGIO + ns(GEWICHT, 3) + BRANDSTOF"
          " + factor(BOEKJAAR) + LEEFTIJD:REGIO + offset(log(Exposure))")
m_freq = fit_glm(F_FREQ, dat, family="poisson")
d_sev = dat[dat.AantalClaims > 0].copy()
d_sev["AvgLoss"] = d_sev.SCHADELAST / d_sev.AantalClaims
m_sev = fit_glm("AvgLoss ~ ns(GEWICHT, 3) + BRANDSTOF", d_sev, family="gamma",
                weights="AantalClaims")
tbl = make_rating_table(m_freq, m_sev, data=dat, base_level="exposure",
                        trim=(0.005, 0.995))

# ---- figures -----------------------------------------------------------------------

snap(make_plot(dat, "REGIO", "Frequency", ta_blue, y_label="Frequency",
               display="color", by_year=True), "oneway")
snap(plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=40), "actual-vs-expected")
snap(make_pdp(m_freq, dat, "LEEFTIJD", metric="Frequency", grid_res=40), "pdp")
snap(make_rating_plot(tbl, "LEEFTIJD"), "rating-factors")
# BRANDSTOF has a deliberately rare level (Waterstof), dimmed as a thin cell
snap(make_rating_plot(tbl, "BRANDSTOF"), "rating-plot-categorical", width=800)
snap_table(tbl.loc[(tbl.Variable == "BRANDSTOF") & tbl.Group.isna(),
                   ["Level", "IsBase", "Exposure", "ClaimCount", "IsThin",
                    "Factor_Frequency", "Factor_Severity", "Factor_Premium"]],
           "rating-table")
ivar = tbl.Variable[tbl.Group.notna()].iloc[0]
snap(make_rating_plot(tbl, ivar, metric="Premium"), "interaction")
snap(plot_glm_residuals(m_freq), "residuals")
# seeded: the count case is randomised, and the figure should not move
snap(plot_glm_qq(m_freq, seed=42), "qq")

# Influence: a keying error in a predictor (a 20-tonne car) is what this plot
# finds: high leverage, so a residual that looks innocent
d_inf = dat.copy()
d_inf.loc[0, "GEWICHT"] = 20000
d_inf.loc[0, "AantalClaims"] = 3
m_inf = fit_glm(F_FREQ, d_inf, family="poisson")
snap(plot_glm_influence(m_inf, n_label=6), "influence")

# The interaction scan needs volume, so a bigger portfolio, with the model
# deliberately fitted without the interaction that is in the data
rng2 = np.random.default_rng(7)
n2 = 400000
regio2 = ["Stad", "Rand", "Dorp", "Platteland"]
d2 = pd.DataFrame({
    "LEEFTIJD": np.round(rng2.uniform(18, 80, n2)),
    "REGIO": pd.Categorical(rng2.choice(regio2, n2, p=[.30, .25, .25, .20]),
                            categories=sorted(regio2)),
    "Exposure": np.round(rng2.uniform(0.1, 1, n2), 3)})
jong2 = np.maximum(0, 30 - d2.LEEFTIJD)
lin2 = (-1.9 + 0.025 * jong2 + 0.010 * np.maximum(0, d2.LEEFTIJD - 65)
        + d2.REGIO.astype(str).map({"Stad": .25, "Rand": .10, "Dorp": .05,
                                    "Platteland": 0})
        + 0.050 * jong2 * (d2.REGIO == "Stad"))            # the missing interaction
d2["AantalClaims"] = rng2.poisson(d2.Exposure * np.exp(lin2))
m_gap = fit_glm("AantalClaims ~ ns(LEEFTIJD, 5) + REGIO + offset(log(Exposure))", d2)
snap(plot_residual_heatmap(m_gap, "LEEFTIJD", "REGIO", n_bins=10),
     "residual-heatmap", width=800)

# Portfolio comparison: two frequency models against claim counts
m_thin = fit_glm("AantalClaims ~ REGIO + offset(log(Exposure))", dat)
lift = model_lift(m_freq, data=dat, actual_col="AantalClaims")
snap(lift.plot, "lift", width=800)
snap(double_lift(dat, model_freq_new=m_freq, model_freq_old=m_thin,
                 actual_col="AantalClaims").plot, "double-lift", width=800)
imp = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                     old_premium_col="HUIDIGE_PREMIE", by=["REGIO", "BRANDSTOF"])
snap(imp.plot, "premium-impact")

# Two predictors on one fixed y-axis
rng_y = (0, 0.3)
snap(plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=40, y_range=rng_y),
     "yrange-age", width=640, height=380)
snap(plot_glm_predictor(m_freq, "REGIO", y_range=rng_y),
     "yrange-regio", width=640, height=380)

# ---- the functions that return a table or a file -----------------------------------

snap_table(agg_all(dat, "REGIO", by_year=True).head(8), "agg-all")
snap_table(glm_diagnostics(m_freq, m_sev), "diagnostics")

# a deliberately duplicated predictor, so the pair lights up
dat["GEWICHT_PROXY"] = dat.GEWICHT + np.round(rng.normal(0, 30, n))
m_col = fit_glm("AantalClaims ~ LEEFTIJD + GEWICHT + GEWICHT_PROXY + REGIO"
                " + BRANDSTOF + offset(log(Exposure))", dat)
snap_table(glm_collinearity(m_col), "collinearity")

snap_table(detect_interactions(m_gap, n_bins=8, n_sim=200, seed=1)[
    ["VarX", "VarY", "Claims", "Deviance", "DF", "Z", "P", "MaxAE",
     "MaxAE_ExposureShare"]], "detect-interactions")

# baseline without GEWICHT, plus a pure-noise column
dat["RUIS"] = np.round(rng.normal(size=n), 2)
m_base_scr = fit_glm("AantalClaims ~ ns(LEEFTIJD, 5) + REGIO + offset(log(Exposure))",
                     dat)
scr = screen_features(m_base_scr, seed=1, n_shap=0,
                      features=["LEEFTIJD", "REGIO", "GEWICHT", "GEWICHT_PROXY",
                                "BRANDSTOF", "RUIS"])
snap(scr.plot, "screen-features", width=760, height=420)

snap(lift.plot_lorenz, "lorenz", width=700, height=460)

imp_s = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                       old_premium_col="HUIDIGE_PREMIE",
                       spotlight=(dat.REGIO == "Randstad") & (dat.LEEFTIJD < 25))
snap(imp_s.plot, "premium-impact-spotlight", width=800)

# what the Overview sheet of export_rating_table() holds
bv = tbl.attrs["base_values"]
ov = pd.DataFrame({
    "Item": (["Generated", "Intercept frequency (per exposure unit)",
              "Intercept severity", "Intercept premium"]
             + ["Base value " + k for k in bv]),
    "Value": ([time.strftime("%Y-%m-%d %H:%M"),
               "%.5f" % tbl.attrs["intercept_frequency"],
               "%.1f" % tbl.attrs["intercept_severity"],
               "%.2f" % tbl.attrs["intercept_premium"]]
              + [("%g" % v) if isinstance(v, (float, np.floating)) else str(v)
                 for v in bv.values()])})
snap_table(ov, "export-overview", vwidth=900)

rep_file = os.path.join(TMP, "report.html")
pricing_report(m_freq, m_sev, dat, file=rep_file,
               title="Motor portfolio \u2013 GLM pricing",
               variables=["LEEFTIJD", "REGIO"],
               include=["diagnostics", "oneway", "rating"])
snap_page(rep_file, "pricing-report", height=1150)

print("\nAll figures written to", OUT_DIR)
shutil.rmtree(TMP, ignore_errors=True)
