"""plot_glm_predictor(): actual vs expected per predictor; the port of
R/ae.R."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import _utils as U
from ._core import set_config, stop, warn
from ._formula import r_cut, translate
from ._plot_common import (as_r_factor, category_axis, fill_axis, is_num,
                           match_arg, wmean)
from .colors import ta_gold, ta_lightblue, ta_muted, ta_navy, ta_year_palette
from .model import GLMFit


def _norm(code):
    return re.sub(r"\s+", "", translate(code))


def _model_frame_column(model, tr, predictor):
    """The column of R's model.frame() called `predictor`, when there is one
    that is not simply a column of the data: the response, a term such as
    factor(BOEKJAAR) or ns(LEEFTIJD, 3), an offset(...) or (weights). A
    plain data column that the model treats as a factor comes back as one,
    with the levels the model was fitted on, as model.frame() holds it.
    Returns None when the name is not a model-frame column."""
    d = tr.data
    if predictor == model.parsed.response or (
            predictor not in d.columns
            and _norm(predictor) == _norm(model.parsed.response)):
        return tr.y
    for t in model.terms:
        for code in t.factors:
            if _norm(code) != _norm(predictor) and code != predictor:
                continue
            if predictor in d.columns and code not in model.xlevels:
                return None
            val = U._eval_factor_values(code, d)
            if val.ndim == 2:
                return val
            if code in model.xlevels:
                return as_r_factor(val, [U._txt(v) for v in model.xlevels[code]])
            return pd.to_numeric(pd.Series(val), errors="coerce").to_numpy(dtype=float)
    for e in model.offset_exprs:
        if _norm("offset(" + e + ")") == _norm(predictor) and predictor not in d.columns:
            return np.asarray(model.offset_value(d, [model.offset_exprs.index(e)]),
                              dtype=float)
    if predictor == "(weights)" and predictor not in d.columns:
        return np.asarray(tr.weights, dtype=float)
    return None


def plot_glm_predictor(model, predictor,
                       data=None,
                       n_bins=150,
                       weight_var=None,
                       weight_label=None,
                       color=ta_year_palette(1)[0],
                       color_pred=ta_gold,
                       title=None, ylab=None, xlab=None,
                       metric_fmt=4,
                       bin_type="quantile",
                       exposure_col="Exposure",
                       ci=True, ci_level=0.95,
                       y_range=None, ratio=False):
    """Actual vs expected per predictor.

    Compares observed and predicted values of a fitted GLM per (binned)
    level of a predictor, computed on the model's own training data.

    The mode is detected automatically. For an offset model with log link
    (the typical frequency model) the response and predictions are counts;
    both are divided by the exposure recovered from ``exp(offset)``, giving
    observed and predicted frequency per bin. Otherwise (the typical
    severity model) the response is already an average and
    prior-weight-weighted means are shown per bin.

    A numeric predictor with at most `n_bins` distinct values is **not
    binned at all**: every value gets its own point, on its exact position.
    Only when there are more distinct values than `n_bins` is binning
    applied: with quantile bins by default (each bin holds roughly the same
    number of observations, avoiding noisy thin tails), or equal-width bins
    with ``bin_type="width"``. A binned point sits at the weight-weighted
    mean of the values in its bin, not at the bin edge.

    Parameters
    ----------
    model : GLMFit
        A fitted model (:func:`fit_glm`).
    predictor : str
        Name of the predictor.
    data : pandas.DataFrame, optional
        Data to evaluate on. None (default) uses the model's own rows,
        which is in-sample. Pass a later accounting year, a holdout or a
        renewal month to see whether the tariff still matches the
        experience: that is the technical monitor this plot is usually
        wanted for.

        The data needs the predictors, the response, and any weight or
        offset column, since the whole point is to hold the realised figure
        against the prediction. Rows with a missing value in one of the
        model's variables are dropped, as the fit dropped them, and a
        factor level the model never saw is an error rather than a silently
        omitted row. The weights are re-evaluated on the new rows, so a
        severity model weighted by the claim count keeps weighting by that
        period's claim counts.

        One thing changes in how the chart reads. In-sample, a categorical
        term that is in the model is pinned to the observed level under a
        canonical link (see `ci` below), so there is nothing to see there by
        construction. On other data nothing is pinned, and a gap between the
        two lines is real evidence that the level has moved.

        The title does not say which data was used, so set `title` yourself
        when you put several periods next to each other.
    n_bins : int
        Maximum number of points for a numeric predictor (default 150).
        Predictors with at most this many distinct values are shown
        unbinned, one point per value.
    weight_var : str, optional
        Weight/exposure column in the model data (override; an unknown name
        is an error). Takes precedence over `exposure_col`.
    weight_label : str, optional
        Axis title for the bars.
    color, color_pred : str
        Colours for the observed and predicted lines.
    title, ylab, xlab : str, optional
        Labels.
    metric_fmt : int
        Number of decimals in the tooltip (default 4).
    bin_type : {"quantile", "width"}
    exposure_col : str
        Exposure column in the model's training data (default
        ``"Exposure"``), used as the denominator for a count model and as
        the height of the volume bars.

        It matters when a model carries **more than one offset**: a known
        bonus-malus scale alongside the exposure, say. ``exp(offset)`` is
        then ``Exposure * BM``, and dividing by it would put the axis in
        claims per BM-adjusted policy-year, which is not a quantity anyone
        reads off a chart. Dividing both series by the exposure itself
        leaves the A/E untouched and puts the axis back in claims per
        policy-year. With a single ``offset(log(Exposure))`` the two are the
        same number. When the column is absent the function falls back to
        ``exp(offset)``.
    ci : bool
        Draw error bars on the observed points (default True). The bars
        answer the question the plot is really asking: is the gap between
        observed and predicted bigger than the noise this bin carries? A bin
        whose bar comfortably spans the predicted line is not evidence of
        anything, however far apart the two markers look.
    ci_level : float
        Confidence level for those bars (default 0.95).

        The interval is on the **observed** point only. The predicted line
        is the model, and its parameter uncertainty is a different and
        usually much smaller quantity. The standard error comes from the
        fitted family's own variance function, scaled by the Pearson
        dispersion for families that estimate one: for counts the bin total
        has variance ``phi * sum(V(mu))``, so the rate divides that by the
        exposure, and for a weighted mean it is the usual
        ``phi * sum(w V(mu)) / (sum w)^2``. Poisson and binomial hold the
        dispersion at 1, since that is the assumption being examined. With
        `data`, the dispersion still comes from the fitted model: it is a
        property of the model, not of the period being monitored.

        Measured coverage on a correctly specified Poisson is 0.96 for a
        binned continuous predictor and 0.98 for a categorical one outside
        the model, so the bars are honest to mildly conservative.

        One case where they cannot be read as a test: for a **categorical
        term that is in the model** under a canonical link, the score
        equations force observed to equal predicted at every level exactly,
        so the bar contains the predicted point by construction and coverage
        is 1 by identity rather than by fit. There the bar length is still
        worth reading - it says how much evidence the level carries - but
        the absence of a gap is arithmetic, not agreement. See
        ``detect_interactions()`` for what does look inside those levels.
    y_range : (lo, hi), optional
        Fixes the primary y-axis range, e.g. to make multiple
        ``plot_glm_predictor()`` calls for different predictors visually
        comparable. None by default, in which case the axis auto-scales to
        the observed/predicted values of this plot only.
    ratio : bool
        False (default) draws the observed and predicted series as two
        lines. True draws their ratio instead: one A/E line with a dashed
        reference line at 1, which is the same quantity
        ``plot_residual_heatmap()`` colours per cell and ``double_lift()``
        plots per bin. Counts give ``sum(y) / sum(mu)`` per bin, a weighted
        model the ratio of the two weighted means.

        Two lines show the level, which is what you want when the question
        is how the risk runs across a variable. The ratio shows the error,
        which is what you want when the question is where the tariff is
        off: a level sitting 10% low is obvious at 0.90, and not obvious at
        all when two lines nearly touch on an axis that spans the whole
        book. The error bars are the ones on the observed point divided by
        the prediction, so a bin whose band covers 1 is not evidence of
        anything.

        It only says something where observed and predicted are free to
        differ. In-sample, a categorical term that is in the model is
        pinned to 1 at every level under a canonical link (see `ci`), so
        use it on `data` the model never saw, on a variable outside the
        model, or per bin of a continuous variable.

    Returns
    -------
    plotly.graph_objects.Figure
        Weight bars and, with ``ratio=False``, an "Observed" and a
        "Predicted" line; with ``ratio=True`` a single "A/E" line and a
        reference line at 1.
    """
    bin_type = match_arg(bin_type, ["quantile", "width"], None, "bin_type")
    if not isinstance(model, GLMFit):
        stop("'model' must be a glm object.")
    # [DATA-06]
    model = U.as_na_omit(model)
    if not isinstance(ratio, (bool, np.bool_)):
        stop("plot_glm_predictor: 'ratio' must be TRUE or FALSE.")
    if n_bins < 2:
        stop("plot_glm_predictor: 'n_bins' must be at least 2.")
    if not isinstance(ci_level, (int, float, np.integer, np.floating)) \
            or isinstance(ci_level, bool) or not ci_level > 0 or not ci_level < 1:
        stop("plot_glm_predictor: 'ci_level' must be a single value strictly "
             "between 0 and 1.")
    y_range = U.check_range(y_range, "plot_glm_predictor")

    tr = (U.glm_training_data(model, "plot_glm_predictor") if data is None
          else U.glm_newdata_parts(model, data, "plot_glm_predictor"))
    response_name = model.parsed.response

    # Retrieve the predictor
    x_values = _model_frame_column(model, tr, predictor)
    if x_values is None:
        if tr.data is not None and predictor in tr.data.columns:
            x_values = tr.data[predictor].reset_index(drop=True)
        else:
            stop("Predictor '" + predictor + "' not found.")

    # [AE-07]
    # model.frame() holds a spline or poly term as a matrix column, which
    # would fail several frames further down with "replacement has 0 rows".
    # Name the column to ask for instead.
    if isinstance(x_values, np.ndarray) and x_values.ndim == 2:
        stop("plot_glm_predictor: '", predictor, "' is a multi-column model "
             "term, not a single predictor. Plot the underlying variable "
             "instead, e.g. '", U.term_base_var(predictor, tr.data), "'.")

    # Determine weight/exposure vector + mode
    link = model.link
    has_offset = tr.offset is not None and bool(np.any(np.asarray(tr.offset) != 0))

    # [AE-01] [OFF-04]
    # exp(offset) recovers an exposure only when the link is log AND the
    # offset is itself a logarithm; with offset(Exposure) it is exp(Exposure),
    # which is not a volume. Both conditions are read off the model.
    off_is_log = U.offset_is_log(model) is True
    if has_offset and link == "log" and off_is_log:
        # [OFF-08]
        # Prefer the exposure column itself over exp(offset). With one offset
        # the two are the same number, but a model may carry a second known
        # relativity as an offset - a bonus-malus scale, say - and then
        # exp(offset) is Exposure x BM. Dividing both series by that is still
        # a like-for-like comparison, but the axis then reads "per BM-adjusted
        # policy-year", which is not a quantity anyone wants to read off a
        # chart. Dividing both by the real exposure leaves the A/E identical
        # and puts the axis back in claims per policy-year.
        if tr.data is not None and exposure_col in tr.data.columns:
            w = tr.data[exposure_col].to_numpy(dtype=float)
        else:
            w = np.exp(np.asarray(tr.offset, dtype=float))  # offset(log(Exposure)) -> Exposure
        use_rate = True        # response & predict are COUNTS -> /exposure
        w_title = "Exposure"
    else:
        if has_offset and link != "log":
            warn("plot_glm_predictor: an offset is present but the link is "
                 "not 'log'; exp(offset) is then not an exposure. The prior "
                 "weights are used instead.")
        if has_offset and link == "log" and not off_is_log:
            warn("plot_glm_predictor: the model's offset is not a logarithm, "
                 "so exp(offset) is not an exposure. The prior weights are "
                 "used instead.")
        w = np.asarray(tr.weights, dtype=float)   # GLM weights (e.g. claim counts for severity)
        use_rate = False       # response is already an average -> do not divide
        w_title = "Weight"
    # Manual override of the weight column
    if weight_var is not None:
        if tr.data is None or weight_var not in tr.data.columns:
            stop("plot_glm_predictor: weight_var '", weight_var, "' not found in ",
                 "model$data." if data is None else "'data'.")
        w = tr.data[weight_var].to_numpy(dtype=float)
    # Show bars as soon as there is a real weight concept (offset, weights or override)
    has_weight = (has_offset or weight_var is not None
                  or bool(np.any((w != 1) & ~np.isnan(w))))
    if weight_label is not None:
        w_title = weight_label

    # [AE-05]
    # On other data the prediction has to be asked for row by row; the
    # offset rides along with it, so both series stay counts for a count
    # model exactly as they are in-sample.
    mu = (np.asarray(model.fitted, dtype=float) if data is None
          else np.asarray(model.predict(tr.data, type="response"), dtype=float))

    observed = np.asarray(tr.y, dtype=float)
    # [AE-03]
    # Sampling variance of the response, from the family's own variance
    # function, so the error bars follow whatever family was fitted rather
    # than assume Poisson. Multiplied by the prior weight because a GLM
    # states Var(y_i) = phi * V(mu_i) / w_i.
    var_unit = np.asarray(model.variance(mu), dtype=float)
    prior_w = np.asarray(tr.weights, dtype=float)
    phi = 1.0 if model.family in ("poisson", "binomial") else \
        float(np.sum(model.residuals("pearson") ** 2) / model.df_residual)

    # Grouping
    if not isinstance(x_values, (pd.Series, pd.Categorical)):
        x_values = pd.Series(x_values)
    x_values = pd.Series(x_values).reset_index(drop=True)
    if is_num(x_values):
        x_var = x_values.to_numpy(dtype=float, na_value=np.nan)
        # [AE-02]
        x_binned = len(np.unique(x_var[~np.isnan(x_var)])) > n_bins
        if not x_binned:
            # Fewer distinct values than requested bins: do not bin at all.
            # One point per observed value, positioned on that exact value.
            # Binning here would merge neighbouring values (e.g. vehicle ages
            # 0 and 1) into a single point at their weighted mean, and
            # because cut() spans the observed range, a single outlier would
            # silently shift every point.
            bin_group = pd.Categorical(x_var, categories=np.unique(x_var[~np.isnan(x_var)]))
        elif bin_type == "quantile":
            # Quantile bins: each bin holds ~equally many observations, so
            # thin tails do not produce a noisy "observed" line.
            xv = x_var[~np.isnan(x_var)]
            brks = np.unique(np.quantile(xv, np.linspace(0, 1, n_bins + 1)))
            if len(brks) < 2:
                brks = np.array([xv.min() - 0.5, xv.max() + 0.5])
            bin_group = r_cut(x_var, brks, include_lowest=True, dig_lab=4)
        else:
            # cut(x, breaks = n_bins): n_bins equal-width intervals
            bin_group = r_cut(x_var, [n_bins], include_lowest=True, dig_lab=4)
        x_is_numeric = True
    else:
        x_fac = as_r_factor(x_values)
        bin_group = x_fac
        x_var = np.asarray(x_fac.astype(object))
        x_is_numeric = False
        x_binned = False

    # Aggregation:
    # frequency:  sum(counts) / sum(exposure)
    # severity etc.: weighted mean of the response with the GLM weights
    codes = np.asarray(pd.Categorical(bin_group).codes)
    present = [c for c in np.unique(codes) if c >= 0] + \
        ([-1] if np.any(codes < 0) else [])     # dplyr: empty levels dropped, NA last
    cats = list(pd.Categorical(bin_group).categories)
    rows = []
    for c in present:
        i = codes == c
        wi = w[i]
        if not x_is_numeric:
            x_plot = x_var[i][0]
        elif x_binned:
            # Binned: the weighted mean position within the bin
            x_plot = wmean(x_var[i], wi)
        else:
            # Unbinned: the value itself (first() rather than a mean, so a
            # group with zero total weight still gets its exact position
            # instead of NaN)
            x_plot = x_var[i][0]
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_obs = (np.sum(observed[i]) / np.sum(wi) if use_rate
                       else wmean(observed[i], wi))
            avg_pred = (np.sum(mu[i]) / np.sum(wi) if use_rate
                        else wmean(mu[i], wi))
            # Standard error of the OBSERVED point, which is the one that
            # carries sampling noise; the predicted line is the model.
            # Counts: the bin total has variance phi * sum(V(mu_i)), so the
            # rate divides that by the exposure. Weighted means: Var(ybar) =
            # phi * sum(w_i V(mu_i)) / (sum w_i)^2, the usual weighted-mean
            # form.
            se = (np.sqrt(np.sum(var_unit[i])) / np.sum(wi) if use_rate
                  else np.sqrt(np.sum(prior_w[i] * var_unit[i])) / np.sum(prior_w[i]))
        rows.append(dict(bin_group=cats[c] if c >= 0 else None, x_plot=x_plot,
                         avg_observed=avg_obs, avg_predicted=avg_pred,
                         weight_sum=float(np.sum(wi)), se_observed=se,
                         n=int(np.sum(i))))
    agg = pd.DataFrame(rows, columns=["bin_group", "x_plot", "avg_observed",
                                      "avg_predicted", "weight_sum",
                                      "se_observed", "n"])
    agg["se_observed"] = np.sqrt(phi) * agg["se_observed"].to_numpy(dtype=float)

    if x_is_numeric:
        xp = agg["x_plot"].to_numpy(dtype=float)
        agg = agg.iloc[np.lexsort([np.where(np.isnan(xp), np.inf, xp)])].reset_index(drop=True)
        agg["x_plot"] = agg["x_plot"].astype(float)
    else:
        # already in level order; the label is the level itself
        agg["x_plot"] = agg["bin_group"]

    scatter_mode = "lines+markers" if x_is_numeric else "markers"

    # [AE-06]
    # The ratio of the two series, which is the same A/E the heatmap and the
    # lift report: counts give sum(y) / sum(mu), a weighted model the ratio
    # of the two weighted means, which is sum(w y) / sum(w mu). A bin whose
    # prediction is zero has no ratio to show rather than an infinite one.
    if ratio:
        ap = agg["avg_predicted"].to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            agg["ae"] = np.where(ap > 0, agg["avg_observed"].to_numpy(dtype=float) / ap,
                                 np.nan)
            # The interval is the one on the observed point, carried through
            # the division: the predicted value is the model, so it scales
            # the bar rather than widening it. The band therefore says
            # whether this bin's distance from 1 is bigger than the noise it
            # carries.
            agg["se_ae"] = np.where(ap > 0, agg["se_observed"].to_numpy(dtype=float) / ap,
                                    np.nan)

    # Labels
    default_metric = ("A/E" if ratio else "Frequency" if use_rate
                      else "Avg. " + response_name)
    if title is None:
        title = default_metric + " – " + predictor
    if xlab is None:
        xlab = predictor
    if ylab is None:
        ylab = default_metric

    xaxis_cfg = dict(title=xlab, tickangle=-45, tickfont=dict(size=9),
                     showgrid=False)
    if not x_is_numeric:
        xaxis_cfg["categoryorder"] = "array"
        xaxis_cfg["categoryarray"] = list(agg["x_plot"])

    x = (agg["x_plot"].to_numpy(dtype=float) if x_is_numeric
         else list(agg["x_plot"]))
    fig = go.Figure()

    if has_weight:
        fig.add_trace(go.Bar(
            x=x, y=agg["weight_sum"].to_numpy(dtype=float), name=w_title,
            yaxis="y2",
            marker=dict(color=ta_lightblue, opacity=0.5, line=dict(width=0)),
            hovertemplate=("<b>" + xlab + ":</b> %{x}<br><b>" + w_title
                           + ":</b> %{y:,.0f}<extra></extra>")))

    # Error bars on the observed points, so a gap between the two lines can
    # be read against the noise the bin actually carries. Thin bins get long
    # bars, which is the honest picture: they were never evidence.
    se_plot = agg["se_ae" if ratio else "se_observed"].to_numpy(dtype=float)
    err_cfg = (dict(type="data",
                    array=se_plot * stats.norm.ppf(1 - (1 - ci_level) / 2),
                    color=color, thickness=1.2, width=3)
               if ci is True or (isinstance(ci, np.bool_) and bool(ci)) else None)

    if ratio:
        fig.add_trace(go.Scatter(
            x=x, y=agg["ae"].to_numpy(dtype=float), name="A/E", yaxis="y",
            mode=scatter_mode, error_y=err_cfg,
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=7),
            hovertemplate=("<b>" + xlab + ":</b> %{x}"
                           "<br><b>A/E:</b> %{y:." + str(metric_fmt)
                           + "f}<extra></extra>")))
    else:
        fig.add_trace(go.Scatter(
            x=x, y=agg["avg_observed"].to_numpy(dtype=float), name="Observed",
            yaxis="y", mode=scatter_mode, error_y=err_cfg,
            line=dict(color=color, width=2.2),
            marker=dict(color=color, size=7),
            hovertemplate=("<b>" + xlab + ":</b> %{x}"
                           "<br><b>Observed:</b> %{y:." + str(metric_fmt)
                           + "f}<extra></extra>")))
        fig.add_trace(go.Scatter(
            x=x, y=agg["avg_predicted"].to_numpy(dtype=float), name="Predicted",
            yaxis="y", mode=scatter_mode,
            line=dict(color=color_pred, width=2, dash="dash"),
            marker=dict(color=color_pred, size=6, symbol="diamond"),
            hovertemplate=("<b>" + xlab + ":</b> %{x}"
                           "<br><b>Predicted:</b> %{y:." + str(metric_fmt)
                           + "f}<extra></extra>")))

    # The line the whole chart is read against. Drawn across the plotting
    # area rather than as a trace, so it needs no x values of its own and
    # stays put for a categorical axis as well.
    ref_line = ([dict(type="line", xref="paper", x0=0, x1=1, yref="y",
                      y0=1, y1=1,
                      line=dict(color=ta_gold, width=2, dash="dash"),
                      layer="below")] if ratio else None)

    xaxis_cfg = fill_axis(xaxis_cfg, None if x_is_numeric
                          else category_axis(agg["x_plot"]))
    fig.update_layout(
        shapes=ref_line,
        title=dict(text=title, font=dict(color=ta_navy, size=14)),
        xaxis=xaxis_cfg,
        yaxis=dict(title=ylab, gridcolor="#D0D8E0", zeroline=False,
                   range=y_range, autorange=y_range is None),
        yaxis2=dict(title=dict(text=w_title, font=dict(color=ta_muted)),
                    overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color=ta_muted)),
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=80, r=80))
    return set_config(fig, "plot")
