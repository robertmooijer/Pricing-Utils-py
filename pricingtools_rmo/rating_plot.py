"""make_rating_plot(): rating factors from a rating table; the port of
R/rating-plot.R."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _utils as U
from ._core import set_config, stop
from ._plot_common import (category_axis, collate_key, fill_axis, match_arg,
                           plot_x)
from .colors import (ta_blue, ta_gold, ta_lightblue, ta_muted, ta_navy,
                     ta_year_palette)

_THIN = "<br><i>⚠ low claim volume</i>"


def _thin(d):
    # Thin cells (IsThin from make_rating_table): dim the markers and add a
    # hover note, so levels backed by few claims stand out visually
    if "IsThin" in d.columns:
        return np.array([v is True or (isinstance(v, (bool, np.bool_)) and bool(v))
                         for v in d["IsThin"].astype(object)], dtype=bool)
    return np.zeros(len(d), dtype=bool)


def _has(d, col):
    return col in d.columns and bool(pd.to_numeric(d[col], errors="coerce").notna().any())


def _num(d, col):
    return pd.to_numeric(d[col], errors="coerce").to_numpy(dtype=float)


def make_rating_plot(rating_tbl, var, metric_fmt=4, metric=None, y_range=None):
    """Plot rating factors from a rating table.

    Plots one variable (or interaction) from a :func:`make_rating_table`
    result.

    Main effects show the factor curves for frequency, severity and premium
    on the primary axis (with a dotted reference line at 1.0) and exposure
    bars on the secondary axis. Interactions (``var="A:B"``) show one series
    per level of the group variable for a single metric. Levels flagged
    ``IsThin`` are drawn with dimmed markers and a "low claim volume" note
    in the tooltip.

    Parameters
    ----------
    rating_tbl : pandas.DataFrame
        Output of :func:`make_rating_table`.
    var : str
        Variable name, or ``"A:B"`` for an interaction.
    metric_fmt : int
        Number of decimals in the tooltip (default 4).
    metric : {"Frequency", "Severity", "Premium"}, optional
        For interaction plots (default: Premium when both models are
        present).
    y_range : (lo, hi), optional
        Fixes the factor (primary) y-axis range, so factor plots for
        different variables become directly comparable. None (default)
        auto-scales to this plot's own values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    y_range = U.check_range(y_range, "make_rating_plot")
    rating_tbl = U.as_df(rating_tbl)
    d = rating_tbl[(rating_tbl["Variable"] == var).fillna(False).to_numpy(dtype=bool)]
    d = d.reset_index(drop=True)
    if len(d) == 0:
        stop("Variable '", var, "' not found.")

    is_interaction = bool(d["Group"].notna().any()) if "Group" in d.columns else False

    # Regular main-effect plot ---------------------------------------------------
    if not is_interaction:
        is_cat = d["Type"].iloc[0] == "categorical"
        if is_cat:
            lev = [U._txt(v) for v in d["Level"]]
            xs = pd.Series(pd.Categorical(lev, categories=lev))
        else:
            xs = pd.Series(_num(d, "LevelNum"))
        x = plot_x(xs)

        mode_line = "markers" if is_cat else "lines+markers"

        thin = _thin(d)
        thin_label = [_THIN if t else "" for t in thin]
        op = np.where(thin, 0.4, 1.0)

        def ht(series):
            return ("<b>" + var + ":</b> %{x}"
                    "<br><b>" + series + ":</b> %{y:." + str(metric_fmt)
                    + "f}%{text}<extra></extra>")

        has_freq = _has(d, "Factor_Frequency")
        has_sev = _has(d, "Factor_Severity")
        has_prem = has_freq and has_sev

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=x, y=_num(d, "Exposure"),
            name="Exposure", yaxis="y2",
            marker=dict(color=ta_lightblue, opacity=0.5, line=dict(width=0)),
            hovertemplate=("<b>" + var + ":</b> %{x}"
                           "<br><b>Exposure:</b> %{y:,.0f}"
                           "<extra></extra>")))

        # With a categorical axis show markers only: do not pass a line
        # (plotly would otherwise reset the mode to "markers+lines" and
        # connect categories).
        def add(col, name, colr, width, size_cat, dash=None):
            line = None if is_cat else dict(color=colr, width=width,
                                            **({"dash": dash} if dash else {}))
            fig.add_trace(go.Scatter(
                x=x, y=_num(d, col), mode=mode_line, name=name, yaxis="y",
                line=line,
                marker=dict(color=colr, size=size_cat if is_cat else 5,
                            opacity=op),
                text=thin_label, hovertemplate=ht(name)))

        if has_freq:
            add("Factor_Frequency", "Factor Frequency", ta_blue, 2.2, 8)
        if has_sev:
            add("Factor_Severity", "Factor Severity", ta_gold, 2.2, 8)
        if has_prem:
            add("Factor_Premium", "Factor Premium", ta_navy, 2.6, 9, "dash")

        fig.update_layout(
            title=dict(text="Rating factors – " + var,
                       font=dict(color=ta_navy, size=14)),
            xaxis=fill_axis(dict(title=var, tickangle=-45,
                                 tickfont=dict(size=9), showgrid=False),
                            category_axis(xs)),
            yaxis=dict(title="Factor (base = 1.0)", gridcolor="#D0D8E0",
                       zeroline=False, range=y_range,
                       autorange=y_range is None),
            yaxis2=dict(title=dict(text="Exposure", font=dict(color=ta_muted)),
                        overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color=ta_muted)),
            shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=1, y1=1,
                         line=dict(color=ta_muted, width=1, dash="dot"))],
            legend=dict(orientation="h", y=-0.2),
            hoverlabel=dict(bgcolor="white", font=dict(size=12)),
            barmode="overlay",
            plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(b=80, r=80))
        return set_config(fig, "rating")

    # Interaction plot: one series per level of the group variable -----------------
    xvar = d["XVar"].iloc[0]
    gvar = d["GroupVar"].iloc[0]
    is_cat_x = d["Type"].iloc[0] == "categorical"
    mode_line = "markers" if is_cat_x else "lines+markers"

    # Too busy to show all three factors at once -> pick one.
    has_freq = _has(d, "Factor_Frequency")
    has_sev = _has(d, "Factor_Severity")
    if metric is not None:
        metric = match_arg(metric, ["Frequency", "Severity", "Premium"], None,
                           "metric")
    elif has_freq and has_sev:
        metric = "Premium"
    elif has_freq:
        metric = "Frequency"
    else:
        metric = "Severity"
    ycol = "Factor_" + metric

    groups = list(pd.unique(d["Group"].astype(object)))
    colors = ta_year_palette(len(groups))
    lev_order = list(dict.fromkeys(U._txt(v) for v in d["Level"]))

    fig = go.Figure()
    xs_all = []

    # Exposure bars per X level (summed over the groups), only available for
    # categorical x categorical (elsewhere Exposure is NA in the table)
    expo = _num(d, "Exposure")
    has_expo = bool(np.any(~np.isnan(expo)) and np.any(expo[~np.isnan(expo)] > 0))
    if has_expo:
        ok = ~np.isnan(expo) & d["Level"].notna().to_numpy()
        lv = [U._txt(v) for v in d["Level"][ok]]
        ex = (pd.DataFrame({"Level": lv, "Exposure": expo[ok]})
              .groupby("Level", sort=False)["Exposure"].sum().reset_index())
        # aggregate() returns the levels sorted
        ex = ex.iloc[sorted(range(len(ex)), key=lambda i: collate_key(ex["Level"][i]))]
        if is_cat_x:
            ex["_x"] = pd.Categorical(ex["Level"], categories=lev_order)
            ex = ex.iloc[np.argsort(ex["_x"].cat.codes.to_numpy(), kind="stable")]
        else:
            ex["_x"] = pd.to_numeric(ex["Level"], errors="coerce")
            xv = ex["_x"].to_numpy(dtype=float)
            ex = ex.iloc[np.argsort(np.where(np.isnan(xv), np.inf, xv), kind="stable")]
        fig.add_trace(go.Bar(
            x=plot_x(ex["_x"]), y=ex["Exposure"].to_numpy(dtype=float),
            name="Exposure", yaxis="y2",
            marker=dict(color=ta_lightblue, opacity=0.5, line=dict(width=0)),
            hovertemplate=("<b>" + xvar + ":</b> %{x}"
                           "<br><b>Exposure:</b> %{y:,.0f}"
                           "<extra></extra>")))
        xs_all.append(ex["_x"])

    for i, g in enumerate(groups):
        dg = d[(d["Group"].astype(object) == g).to_numpy(dtype=bool)].copy()
        if is_cat_x:
            dg["_x"] = pd.Categorical([U._txt(v) for v in dg["Level"]],
                                      categories=lev_order)
        else:
            dg["_x"] = _num(dg, "LevelNum")
            xv = dg["_x"].to_numpy(dtype=float)
            dg = dg.iloc[np.argsort(np.where(np.isnan(xv), np.inf, xv), kind="stable")]
        thin_g = _thin(dg)
        fig.add_trace(go.Scatter(
            x=plot_x(dg["_x"]), y=_num(dg, ycol),
            mode=mode_line,
            name=str(gvar) + " = " + U._txt(g), yaxis="y",
            line=None if is_cat_x else dict(color=colors[i], width=2.2),
            marker=dict(color=colors[i], size=9 if is_cat_x else 5,
                        opacity=np.where(thin_g, 0.4, 1.0)),
            text=[_THIN if t else "" for t in thin_g],
            hovertemplate=("<b>" + xvar + ":</b> %{x}"
                           "<br><b>" + str(gvar) + ":</b> " + U._txt(g)
                           + "<br><b>Factor " + metric + ":</b> %{y:."
                           + str(metric_fmt) + "f}%{text}<extra></extra>")))
        xs_all.append(dg["_x"])

    fig.update_layout(
        title=dict(text=("Interaction – " + xvar + " × " + str(gvar)
                         + "  (Factor " + metric + ")"),
                   font=dict(color=ta_navy, size=14)),
        xaxis=fill_axis(dict(title=xvar, tickangle=-45, tickfont=dict(size=9),
                             showgrid=False), category_axis(*xs_all)),
        yaxis=dict(title="Factor " + metric + " (base cell = 1.0)",
                   gridcolor="#D0D8E0", zeroline=False,
                   range=y_range, autorange=y_range is None),
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=1, y1=1,
                     line=dict(color=ta_muted, width=1, dash="dot"))],
        legend=dict(orientation="h", y=-0.2, title=dict(text=str(gvar))),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        barmode="overlay",
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=80, r=80 if has_expo else 40))
    if has_expo:
        fig.update_layout(
            yaxis2=dict(title=dict(text="Exposure", font=dict(color=ta_muted)),
                        overlaying="y", side="right", showgrid=False,
                        tickfont=dict(color=ta_muted)))
    return set_config(fig, "interaction")
