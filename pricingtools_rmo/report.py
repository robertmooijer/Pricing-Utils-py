"""One-call HTML pricing report (the port of R/report.R)."""
from __future__ import annotations

import html
import os
import re
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

from . import _utils as U
from ._core import PricingError, message, stop, warn
from .colors import ta_blue, ta_gold

_INCLUDE = ("diagnostics", "oneway", "ae", "pdp", "rating", "interactions")

_CSS = """
    body{font-family:'Segoe UI',Arial,sans-serif;color:#222;margin:0;background:#fff;}
    .ta-wrap{max-width:1100px;margin:0 auto;padding:24px;}
    h1{color:#00365E;}
    h2{color:#00365E;border-bottom:2px solid #A8C8E0;padding-top:18px;}
    h3{color:#0073AB;margin-bottom:4px;}
    .ta-table{border-collapse:collapse;margin:10px 0;font-size:13px;}
    .ta-table th{background:#00365E;color:#fff;padding:6px 10px;text-align:left;}
    .ta-table td{border-bottom:1px solid #D0D8E0;padding:5px 10px;}
    .ta-error{color:#C44536;}
    .ta-meta{color:#6B7A8D;font-size:13px;margin:2px 0;}
    .ta-toc{margin:14px 0;}
    .ta-toc a{color:#0073AB;text-decoration:none;margin-right:14px;}
"""


def _esc(x):
    return html.escape(str(x))


# [REP-03] formatC(format = "d") goes through an integer in R, so anything
# beyond 2^31 printed as "NA": a large GVIF did
def _fmt_num(x, digits=4):
    if x is None or (isinstance(x, (float, np.floating)) and np.isnan(x)):
        return ""
    x = float(x)
    if abs(x) >= 1e9:
        return "%.3g" % x
    if abs(x) >= 1000:
        return "{:,.0f}".format(round(x))
    return np.format_float_positional(x, precision=digits, unique=False,
                                      fractional=False, trim="-")


def _html_table(df, digits=4):
    head = "".join("<th>%s</th>" % _esc(c) for c in df.columns)
    rows = []
    for row in df.itertuples(index=False):
        cells = []
        for x in row:
            if isinstance(x, (bool, np.bool_)):
                txt = "TRUE" if x else "FALSE"
            elif isinstance(x, (int, float, np.integer, np.floating)):
                txt = _fmt_num(x, digits)
            elif x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x)):
                txt = ""
            else:
                txt = _esc(x)
            cells.append("<td>%s</td>" % txt)
        rows.append("<tr>%s</tr>" % "".join(cells))
    return ('<table class="ta-table"><thead><tr>%s</tr></thead>'
            "<tbody>%s</tbody></table>" % (head, "".join(rows)))


def pricing_report(model_freq=None, model_sev=None, data=None,
                   file="pricing_report.html", title="GLM Pricing Report",
                   variables=None, include=_INCLUDE, by_year=False,
                   top_interactions=3, exposure_col="Exposure",
                   claims_col="AantalClaims", loss_col="SCHADELAST",
                   year_col="BOEKJAAR", grid_res=50, base_level="first",
                   trim=(0, 1), merge_vars=None, base=None, grid_step=None):
    """One-call HTML pricing report.

    Bundles the whole analysis into a single self-browsable HTML report:
    model diagnostics (fit table + binned residuals) and, per variable, the
    one-way observed plots, actual vs expected, partial dependence and the
    rating-factor plot, plus a section with the interaction plots. Frequency
    and severity are shown side by side throughout, wherever the
    corresponding model is supplied. Individual plot failures are shown as
    a note instead of aborting the report.

    The plots are regular interactive plotly charts. Next to the ``.html``
    file a ``<name>_files/`` folder is written with plotly.js; keep the two
    together when sharing the report.

    Parameters
    ----------
    model_freq, model_sev : GLMFit, optional
        Fitted models (at least one).
    data : pandas.DataFrame
        Dataset with raw rows.
    file : str
        Output path of the ``.html`` file.
    title : str
        Report title.
    variables : list of str, optional
        Which variables get a section (default: all base variables of both
        models).
    include : sequence of str
        Which blocks to render: any of "diagnostics", "oneway", "ae", "pdp",
        "rating", "interactions" (default: all). The "interactions" block
        runs detect_interactions() on the frequency model and plots the
        strongest pairs as A/E heatmaps. The "oneway" block shows observed
        frequency, and observed severity as well when `model_sev` is
        supplied. Note that the observed severity reads `loss_col` as a
        **total** loss amount, not an average per claim: it is aggregated
        as ``sum(loss) / sum(claims)``.
    by_year : bool
        Split the one-way plots by accounting year.
    top_interactions : int
        Number of pairs from the interaction scan to plot as heatmaps.
    exposure_col, claims_col, loss_col, year_col : str
        Column names, see agg_all().
    grid_res, base_level, trim
        Passed to make_pdp() / make_rating_table().
    merge_vars, base, grid_step
        Passed to make_rating_table(), so the report shows the same tariff
        as a table built with them. `year_col` and `loss_col` are passed on
        too.

    Returns
    -------
    str
        The normalised path of the written file.
    """
    import plotly.offline

    from .ae import plot_glm_predictor
    from .diagnostics import glm_collinearity, glm_diagnostics, plot_glm_residuals
    from .interactions import detect_interactions, plot_residual_heatmap
    from .pdp import make_pdp
    from .plot_oneway import make_plot
    from .rating_plot import make_rating_plot
    from .rating_table import make_rating_table

    include = [include] if isinstance(include, str) else list(include)
    bad = [i for i in include if i not in _INCLUDE]
    if bad:
        stop("pricing_report: unknown block(s) in 'include': ", ", ".join(bad))
    if base_level not in ("first", "exposure"):
        stop("pricing_report: 'base_level' must be \"first\" or \"exposure\".")
    if model_freq is None and model_sev is None:
        stop("Provide at least one model.")
    U.check_models("pricing_report", model_freq=model_freq, model_sev=model_sev)
    data = U.as_df(data)
    U.check_cols(data, [exposure_col, claims_col, loss_col]
                 + ([year_col] if by_year else []), "pricing_report")

    if variables is None:
        variables = []
        for m in (model_freq, model_sev):
            for v in U.model_base_vars(m, data):
                if v not in variables:
                    variables.append(v)
    unknown = [v for v in variables if v not in data.columns]
    if unknown:
        warn("pricing_report: variable(s) skipped, not in 'data': ",
             ", ".join(unknown))
        variables = [v for v in variables if v in data.columns]
    variables = [v for v in variables if v != exposure_col]
    if not variables:
        stop("pricing_report: no variables to report on.")

    def anchor(v):
        return re.sub(r"[^A-Za-z0-9_-]", "_", v)

    # [CONV-04] [REP-02] [REP-03]
    # Every plot is built inside the guard, so one that cannot be built (a
    # heatmap with every cell thin, on a small book) becomes a note instead
    # of taking the whole report down.
    def safe(build):
        try:
            fig = build()
            cfg = getattr(fig, "pricing_config", None)
            return fig.to_html(full_html=False, include_plotlyjs=False,
                               config=cfg)
        except Exception as e:  # noqa: BLE001 - a note, not a crash
            return '<p class="ta-error">Could not render: %s</p>' % _esc(e)

    def fml(model, name):
        if model is None:
            return ""
        return '<p class="ta-meta"><b>%s</b>: %s</p>' % (_esc(name),
                                                        _esc(model.formula))

    def in_model_data(model, v):
        return model is not None and v in model.data.columns

    # [REP-01]
    # Rating table (once; feeds the per-variable and interaction plots)
    tbl = None
    if "rating" in include:
        try:
            tbl = make_rating_table(model_freq, model_sev, data=data,
                                    grid_res=grid_res,
                                    exposure_col=exposure_col,
                                    claims_col=claims_col,
                                    base_level=base_level, trim=trim,
                                    grid_step=grid_step, base=base,
                                    merge_vars=merge_vars, year_col=year_col,
                                    loss_col=loss_col)
        except PricingError as e:
            warn("pricing_report: rating table failed: ", str(e))
            tbl = None

    parts = []
    diag_sec = ""
    if "diagnostics" in include:
        dg = glm_diagnostics(model_freq, model_sev)
        col_tbl = ""
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cl = glm_collinearity(model_freq if model_freq is not None
                                      else model_sev)
            if len(cl):
                col_tbl = ("<h3>Collinearity between terms</h3>"
                           '<p class="ta-meta">Generalised VIF per term. Read '
                           "GVIF_scaled like a VIF on the square-root scale: 3 "
                           "corresponds to a VIF of about 9. A high value means "
                           "that term's factors cannot be read on their own. "
                           "Terms in an interaction are not flagged: their GVIF "
                           "depends on how the variables are coded.</p>"
                           + _html_table(cl))
        except Exception:  # noqa: BLE001
            col_tbl = ""
        diag_sec = ('<h2 id="diagnostics">Model diagnostics</h2>'
                    + _html_table(dg) + col_tbl)
        if model_freq is not None:
            diag_sec += ("<h3>Binned residuals – frequency</h3>"
                         + safe(lambda: plot_glm_residuals(model_freq)))
        if model_sev is not None:
            diag_sec += ("<h3>Binned residuals – severity</h3>"
                         + safe(lambda: plot_glm_residuals(model_sev)))

    var_secs = []
    for v in variables:
        sec = '<h2 id="var-%s">%s</h2>' % (anchor(v), _esc(v))

        def oneway(metric, colr, v=v):
            return safe(lambda: make_plot(data, v, metric, colr, metric,
                                          display="color", by_year=by_year,
                                          exposure_col=exposure_col,
                                          claims_col=claims_col,
                                          loss_col=loss_col,
                                          year_col=year_col))
        if "oneway" in include:
            sec += "<h3>One-way observed – frequency</h3>" + oneway("Frequency", ta_blue)
        # [REP-02]
        # Severity only alongside a severity model, the way the ae and pdp
        # blocks gate theirs.
        if "oneway" in include and model_sev is not None:
            sec += "<h3>One-way observed – severity</h3>" + oneway("Severity", ta_gold)
        if "ae" in include and in_model_data(model_freq, v):
            sec += ("<h3>Actual vs expected – frequency</h3>"
                    + safe(lambda v=v: plot_glm_predictor(model_freq, v)))
        if "ae" in include and in_model_data(model_sev, v):
            sec += ("<h3>Actual vs expected – severity</h3>"
                    + safe(lambda v=v: plot_glm_predictor(model_sev, v)))
        if "pdp" in include and model_freq is not None and v in model_freq.rhs_vars():
            sec += ("<h3>Partial dependence – frequency</h3>"
                    + safe(lambda v=v: make_pdp(model_freq, data, v,
                                                metric="Frequency",
                                                grid_res=grid_res,
                                                exposure_col=exposure_col,
                                                claims_col=claims_col,
                                                loss_col=loss_col)))
        if "pdp" in include and model_sev is not None and v in model_sev.rhs_vars():
            sec += ("<h3>Partial dependence – severity</h3>"
                    + safe(lambda v=v: make_pdp(model_sev, data, v,
                                                metric="Severity",
                                                grid_res=grid_res,
                                                exposure_col=exposure_col,
                                                claims_col=claims_col,
                                                loss_col=loss_col)))
        if tbl is not None and v in set(tbl["Variable"]):
            sec += "<h3>Rating factors</h3>" + safe(lambda v=v: make_rating_plot(tbl, v))
        var_secs.append(sec)

    # The rating-table interaction plots belong to the interaction block, so
    # leaving "interactions" out of `include` removes them along with the scan
    int_sec = ""
    if tbl is not None and "interactions" in include:
        ivars = list(pd.unique(tbl.loc[tbl["Group"].notna(), "Variable"]))
        if ivars:
            int_sec = '<h2 id="interactions">Interactions</h2>' + "".join(
                "<h3>%s</h3>" % _esc(iv) + safe(lambda iv=iv: make_rating_plot(tbl, iv))
                for iv in ivars)

    # Interaction scan: cell-level structure the one-way checks cannot see
    scan_sec = ""
    if "interactions" in include:
        scan_model = model_freq if model_freq is not None else model_sev
        try:
            scan = detect_interactions(scan_model, n_bins=10, n_sim=100)
        except PricingError as e:
            warn("pricing_report: interaction scan failed: ", str(e))
            scan = None
        if scan is not None and len(scan):
            top = scan.head(top_interactions)
            show = scan[["VarX", "VarY", "Claims", "Z", "P", "MaxAE",
                         "MaxAE_ExposureShare"]]
            scan_sec = ('<h2 id="interaction-scan">Interaction scan</h2>'
                        '<p class="ta-meta">Ranked by signal (Z) against a '
                        "simulated no-interaction null. MaxAE is the A/E of "
                        "the worst cell with enough claims, and is the column "
                        "to judge materiality on.</p>" + _html_table(show))
            for _, r in top.iterrows():
                scan_sec += ("<h3>%s × %s</h3>" % (_esc(r.VarX), _esc(r.VarY))
                             + safe(lambda r=r: plot_residual_heatmap(
                                 scan_model, r.VarX, r.VarY)))

    toc = '<p class="ta-toc">'
    if "diagnostics" in include:
        toc += '<a href="#diagnostics">Diagnostics</a>'
    toc += "".join('<a href="#var-%s">%s</a>' % (anchor(v), _esc(v))
                   for v in variables)
    if scan_sec:
        toc += '<a href="#interaction-scan">Interaction scan</a>'
    if int_sec:
        toc += '<a href="#interactions">Interactions</a>'
    toc += "</p>"

    libdir = re.sub(r"\.html?$", "", os.path.basename(file)) + "_files"
    out_dir = os.path.dirname(os.path.abspath(file))
    os.makedirs(os.path.join(out_dir, libdir), exist_ok=True)
    with open(os.path.join(out_dir, libdir, "plotly.min.js"), "w",
              encoding="utf-8") as fh:
        fh.write(plotly.offline.get_plotlyjs())

    page = (
        "<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\"/>"
        "<title>%s</title>"
        '<script src="%s/plotly.min.js"></script>'
        "<style>%s</style></head><body>"
        '<div class="ta-wrap"><h1>%s</h1>'
        '<p class="ta-meta">Generated %s – %s rows</p>%s%s%s%s%s%s%s'
        "</div></body></html>"
        % (_esc(title), libdir, _CSS, _esc(title),
           datetime.now().strftime("%Y-%m-%d %H:%M"),
           "{:,}".format(len(data)), fml(model_freq, "Frequency model"),
           fml(model_sev, "Severity model"), toc, diag_sec,
           "".join(var_secs), scan_sec, int_sec))
    with open(file, "w", encoding="utf-8") as fh:
        fh.write(page)
    path = os.path.abspath(file)
    message("pricing_report: written to ", path, " (dependencies in ",
            libdir, "/)")
    return path
