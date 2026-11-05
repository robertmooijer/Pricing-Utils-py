"""make_plot(): one-way plot of frequency or severity with volume bars; the
port of R/plot-oneway.R."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import _utils as U
from ._core import set_config, stop
from ._plot_common import (arrange, category_axis, fill_axis, fmt_big,
                           is_num, match_arg, n2mfrow, plot_x)
from .agg import agg_all
from .colors import ta_lightblue, ta_muted, ta_navy, ta_year_palette

# ggplot2 sizes are in mm; ggplotly turns them into pixels this way
_GG_PX = 72.27 / 25.4 * 96 / 72


def make_plot(data, col, metric="Frequency", color_single=None, y_label=None,
              display="color", by_year=None,
              metric_fmt=4,
              exposure_col="Exposure",
              claims_col="AantalClaims",
              loss_col="SCHADELAST",
              year_col="BOEKJAAR",
              discrete_cutoff=25,
              y_range=None):
    """One-way plot of frequency or severity with exposure bars.

    Takes raw policy rows, aggregates them internally via :func:`agg_all`
    and plots the chosen metric with volume bars on a secondary axis.

    The bars show the denominator the metric is built on, so they say how
    solid each point is: exposure for a frequency, the claim count for a
    severity (which is a mean per claim). Those two can run in opposite
    directions - the level with the most exposure is often the one with
    the fewest claims - so exposure bars under a severity line would point
    the reader the wrong way.

    With a character/categorical X-axis (or a numeric one with at most
    `discrete_cutoff` unique values) only markers are shown, so categories
    are never connected by a line. Note: in color mode with
    ``by_year=True`` the exposure bars show the sum over all years, while
    the lines are per year.

    Parameters
    ----------
    data : pandas.DataFrame
        Raw rows.
    col : str
        Name of the X-axis column.
    metric : {"Frequency", "Severity"}
        Computed on the internal aggregate.
    color_single : str
        Colour for the metric line when ``by_year=False`` (and in facet
        mode).
    y_label : str
        Label for the Y-axis.
    display : {"color", "facet"}
        Coloured lines per year, or one panel per year.
    by_year : bool
        True = split by accounting year.
    metric_fmt : int
        Number of decimals in the tooltip (default 4).
    exposure_col, claims_col, loss_col, year_col : str
        Column names, see :func:`agg_all`.
    discrete_cutoff : int
        Maximum number of unique numeric values that is still treated as
        discrete (markers only, default 25).
    y_range : (lo, hi), optional
        Fixes the primary y-axis range, so plots for different variables
        become directly comparable. None (default) auto-scales to this
        plot's own values. In facet mode a fixed range applies to every
        facet, overriding the per-year free scales.

    Returns
    -------
    plotly.graph_objects.Figure

    Examples
    --------
    >>> make_plot(d, "REGIO", "Frequency", ta_blue, "Frequency",
    ...           display="color", by_year=True)
    """
    display = match_arg(display, ["color", "facet"], "make_plot", "display")
    metric = match_arg(metric, ["Frequency", "Severity"], "make_plot", "metric")
    y_range = U.check_range(y_range, "make_plot")
    if by_year is None:
        stop('make_plot: argument "by_year" is missing, with no default')
    data = U.as_df(data)
    U.check_cols(data, [col, exposure_col, claims_col, loss_col,
                        year_col if by_year else None], "make_plot")

    # Aggregate the raw rows; agg_all returns standardised column names
    # (Exposure, ClaimCount, Loss, Frequency, Severity and, when split, Year)
    agg = agg_all(data, col, by_year,
                  exposure_col=exposure_col, claims_col=claims_col,
                  loss_col=loss_col, year_col=year_col)
    agg = agg.rename(columns={col: ".x"})

    # [DATA-04]
    # With a character/factor X-axis show markers only (no line connecting
    # categories). Numeric columns with few unique values (e.g. integer-coded
    # categories) are also treated as discrete.
    is_discrete = (not is_num(agg[".x"])
                   or agg[".x"].nunique(dropna=False) <= discrete_cutoff)

    # Sort by X (and Year when split) so the line runs cleanly
    agg = arrange(agg, ["Year", ".x"] if by_year else [".x"])

    # [DATA-03]
    # Volume bars: the denominator the metric is actually built on, so the
    # bar says how solid each point is. A severity is a mean per claim, so
    # its bar is the claim count; exposure would be the wrong denominator
    # and can run the other way entirely (the level with the most exposure
    # is often the one with the fewest claims).
    vol_col = "ClaimCount" if metric == "Severity" else "Exposure"
    vol_base = "Number of claims" if metric == "Severity" else "Exposure"

    if by_year:
        # dplyr's group_by(): sorted groups, NA last, empty levels dropped
        exp_agg = (agg.groupby(".x", sort=True, dropna=False, observed=True)
                   [vol_col].sum().reset_index())
    else:
        exp_agg = agg
    exp_name = vol_base + " (all years)" if by_year else vol_base

    if y_label is None:
        stop('make_plot: argument "y_label" is missing, with no default')

    # Facet mode: the panels ggplot2 + ggplotly draw in R -----------------------
    if by_year and display == "facet":
        if color_single is None:
            stop('make_plot: argument "color_single" is missing, with no default')
        return _facet_plot(agg, col, metric, color_single, y_label, vol_col,
                           vol_base, is_discrete, metric_fmt, y_range)

    # Color mode or no-year mode: native plotly ----------------------------------
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_x(exp_agg[".x"]),
        y=exp_agg[vol_col].to_numpy(dtype=float),
        name=exp_name,
        yaxis="y2",
        marker=dict(color=ta_lightblue, opacity=0.5, line=dict(width=0)),
        hovertemplate=("<b>" + col + ":</b> %{x}"
                       "<br><b>" + exp_name + ":</b> %{y:,.0f}"
                       "<extra></extra>")))
    xs = [exp_agg[".x"]]

    def add_metric_trace(d, year_label, colr, marker_size, line_width,
                         legend_name, show_legend=True):
        if year_label is None:
            ht = ("<b>" + col + ":</b> %{x}"
                  "<br><b>" + metric + ":</b> %{y:." + str(metric_fmt)
                  + "f}<extra></extra>")
        else:
            ht = ("<b>" + col + ":</b> %{x}"
                  "<br><b>Year:</b> " + year_label
                  + "<br><b>" + metric + ":</b> %{y:." + str(metric_fmt)
                  + "f}<extra></extra>")
        x = plot_x(d[".x"])
        y = d[metric].to_numpy(dtype=float)
        if is_discrete:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="markers",
                name=legend_name, legendgroup=legend_name, yaxis="y",
                marker=dict(color=colr, size=marker_size),
                showlegend=show_legend, hovertemplate=ht))
        else:
            fig.add_trace(go.Scatter(
                x=x, y=y, mode="lines+markers",
                name=legend_name, legendgroup=legend_name, yaxis="y",
                line=dict(color=colr, width=line_width),
                marker=dict(color=colr, size=marker_size),
                showlegend=show_legend, hovertemplate=ht))
        xs.append(d[".x"])

    if not by_year:
        if color_single is None:
            stop('make_plot: argument "color_single" is missing, with no default')
        add_metric_trace(agg, None, color_single, marker_size=7,
                         line_width=2.2, legend_name=metric)
    else:
        years = list(agg["Year"].cat.categories)
        colors = ta_year_palette(len(years))
        for i, yr in enumerate(years):
            d_year = agg[agg["Year"] == yr]
            add_metric_trace(d_year, yr, colors[i], marker_size=6,
                             line_width=2, legend_name=str(yr))

    xaxis = fill_axis(dict(title=col, tickangle=-45, tickfont=dict(size=9),
                           showgrid=False), category_axis(*xs))
    fig.update_layout(
        title=dict(text=metric + " – " + col,
                   font=dict(color=ta_navy, size=14)),
        xaxis=xaxis,
        yaxis=dict(title=y_label, gridcolor="#D0D8E0", zeroline=False,
                   range=y_range, autorange=y_range is None),
        yaxis2=dict(title=dict(text=vol_base, font=dict(color=ta_muted)),
                    overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color=ta_muted)),
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=80, r=80))
    return set_config(fig, "plot")


def _facet_plot(agg, col, metric, color_single, y_label, vol_col, vol_base,
                is_discrete, metric_fmt, y_range):
    """One panel per year, as the ggplot2 + ggplotly chart of the R package:
    bars rescaled per panel to the range of the metric, the metric as
    points (and a line for a continuous axis), free y scales unless
    `y_range` pins them."""
    # Rescale exposure per facet to the range of the metric, so that bars
    # and line visually share the same y-axis.
    parts = []
    for yr, g in agg.groupby("Year", sort=True, observed=True):
        g = g.copy()
        mv = g[metric].to_numpy(dtype=float)
        vv = g[vol_col].to_numpy(dtype=float)
        metric_max = np.nanmax(mv) if np.any(~np.isnan(mv)) else -np.inf
        exp_max = np.nanmax(vv) if np.any(~np.isnan(vv)) else -np.inf
        ok = (np.isfinite(exp_max) and exp_max > 0
              and np.isfinite(metric_max) and metric_max > 0)
        g["_exp_scaled"] = vv / exp_max * metric_max if ok else 0.0
        g["_exp_label"] = [vol_base + ": " + fmt_big(np.round(v)) for v in vv]
        g["_metric_label"] = [metric + ": " + fmt_big(v, metric_fmt) for v in mv]
        parts.append(g)
    years = list(agg["Year"].cat.categories)
    ncol, nrow = n2mfrow(len(years))
    specs = [[({} if r * ncol + c < len(years) else None) for c in range(ncol)]
             for r in range(nrow)]
    fig = make_subplots(rows=nrow, cols=ncol, specs=specs,
                        subplot_titles=["<b>" + str(y) + "</b>" for y in years],
                        x_title=col, y_title=y_label,
                        horizontal_spacing=0.05, vertical_spacing=0.12)
    cat_ax = category_axis(agg[".x"])
    for i, g in enumerate(parts):
        r, c = i // ncol + 1, i % ncol + 1
        x = plot_x(g[".x"])
        xt = ["" if v is None else (U._txt(v) if not isinstance(v, str) else v)
              for v in x]
        fig.add_trace(go.Bar(
            x=x, y=g["_exp_scaled"].to_numpy(dtype=float), width=0.6,
            name=vol_base, legendgroup=vol_base, showlegend=i == 0,
            marker=dict(color="rgba(168,200,224,0.38)",
                        line=dict(width=0)),
            text=[col + ": " + a + "<br />" + b
                  for a, b in zip(xt, g["_exp_label"])],
            textposition="none", hoverinfo="text"), row=r, col=c)
        if not is_discrete:
            fig.add_trace(go.Scatter(
                x=x, y=g[metric].to_numpy(dtype=float), mode="lines",
                name=metric, legendgroup=metric, showlegend=i == 0,
                line=dict(color=color_single, width=0.8 * _GG_PX),
                text=[col + ": " + a for a in xt], hoverinfo="text"),
                row=r, col=c)
        fig.add_trace(go.Scatter(
            x=x, y=g[metric].to_numpy(dtype=float), mode="markers",
            name=metric, legendgroup=metric,
            showlegend=bool(i == 0 and is_discrete),
            marker=dict(color=color_single, size=2 * _GG_PX),
            text=[col + ": " + a + "<br />" + b
                  for a, b in zip(xt, g["_metric_label"])],
            hoverinfo="text"), row=r, col=c)
    # A fixed range pins every facet to the same scale (overrides free_y)
    fig.update_yaxes(gridcolor="#EBEBEB", zeroline=False,
                     range=y_range, autorange=y_range is None)
    fig.update_xaxes(tickangle=-45, tickfont=dict(size=8 * 96 / 72),
                     gridcolor="#EBEBEB", **(cat_ax or {}))
    n_strip = len(years)
    for k, a in enumerate(fig.layout.annotations):
        if k < n_strip:
            a.font = dict(color=ta_navy, size=9 * 96 / 72)
    fig.update_layout(
        title=dict(text="<b>" + metric + " - " + col + "</b>", x=0,
                   xref="paper", font=dict(color=ta_navy, size=12 * 96 / 72)),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(b=60),
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="closest")
    return set_config(fig, "plot")
