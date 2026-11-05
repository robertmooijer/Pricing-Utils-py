"""make_pdp(): partial dependence plot with observed overlay; the port of
R/pdp.R."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _utils as U
from ._core import set_config, stop, warn
from ._formula import expr_vars
from ._plot_common import (category_axis, fill_axis, is_categorical, is_num,
                           match_arg, plot_x, r_sort_unique, wmean)
from .colors import ta_blue, ta_gold, ta_lightblue, ta_muted, ta_navy
from .model import GLMFit


def make_pdp(model,
             raw_data,
             pred_var,
             metric="Frequency",
             transform=None,
             grid_res=50,
             y_label=None,
             metric_fmt=4,
             exposure_col="Exposure",
             claims_col="AantalClaims",
             loss_col="SCHADELAST",
             discrete_cutoff=25,
             y_range=None,
             keep_offsets=True):
    """Partial dependence plot with observed overlay.

    Computes the PDP on the response scale: for each grid value of
    `pred_var` the whole training data is set to that value, predicted on
    the response scale, and the weighted arithmetic mean is taken
    (frequency: exposure-weighted; severity: claim-weighted). The exposure
    offset is subtracted on the link scale, so the PDP is a frequency per
    unit of exposure and directly comparable to the observed line.
    (``exp(mean(link))`` would be a geometric mean, including the offset,
    and systematically biased low.)

    Any other offset, a bonus-malus scale carried as ``offset(log(BM))``,
    stays in at each policy's own value by default, because the observed
    line is at each policy's own BM as well. Taking it out put the PDP at
    the class without discount: on a book with an average BM of 0.74 it ran
    35% above the observed line at every level of a variable the model had
    exactly right. The rating table is the place for factors at BM = 1.

    The "Observed" line is a one-way marginal (including correlations with
    other rating factors); the PDP is a partial effect. Differences between
    the two therefore do not necessarily indicate misfit.

    Performance: the training data is first collapsed to the unique
    profiles of the model's other predictors (with summed weights), so the
    cost is `grid_res` x (number of unique profiles) predictions instead of
    `grid_res` x n rows, with an exactly identical result.

    Parameters
    ----------
    model : GLMFit
        A fitted model (:func:`fit_glm`).
    raw_data : pandas.DataFrame
        Raw data for the observed line and the bars.
    pred_var : str
        Name of the predictor.
    metric : {"Frequency", "Severity"}
    transform : optional
        Deprecated and ignored (response scale is the default).
    grid_res : int
        Number of grid points for continuous predictors (default 50).
    y_label : str, optional
        Y-axis label (default automatic).
    metric_fmt : int
        Number of decimals in the tooltip (default 4).
    exposure_col, claims_col, loss_col : str
        Column names, see :func:`agg_all`.
    discrete_cutoff : int
        See :func:`make_plot`.
    y_range : (lo, hi), optional
        Fixes the primary y-axis range, so PDPs for different predictors
        become directly comparable. None (default) auto-scales to this
        plot's own values.
    keep_offsets : bool
        True (default): only the exposure offset is taken out, and any
        other offset stays in at each policy's own value, as it does in
        ``premium_impact()`` and ``model_lift()``. False takes every offset
        out, which puts the PDP at BM = 1 and no longer on the level of the
        observed line. The default y-axis title names the offsets kept.

    Returns
    -------
    plotly.graph_objects.Figure
        Exposure/claim bars, the observed one-way line and the PDP line.
    """
    metric = match_arg(metric, ["Frequency", "Severity"], "make_pdp", "metric")
    y_range = U.check_range(y_range, "make_pdp")
    if not isinstance(model, GLMFit):
        stop("make_pdp: 'model' must be a glm object.")
    if not isinstance(keep_offsets, (bool, np.bool_)):
        stop("make_pdp: 'keep_offsets' must be TRUE or FALSE.")
    keep_offsets = bool(keep_offsets)
    # [DATA-06]
    model = U.as_na_omit(model)
    if transform is not None:
        warn("make_pdp: 'transform' is deprecated and ignored; the PDP is "
             "computed directly on the response scale.")
    raw_data = U.as_df(raw_data)
    U.check_cols(raw_data, [pred_var, exposure_col, claims_col,
                            loss_col if metric == "Severity" else None],
                 "make_pdp")

    # Observed aggregation depending on metric ---------------------------------
    if metric == "Frequency":
        obs = raw_data
        num, den = claims_col, exposure_col
    else:
        cl = pd.to_numeric(raw_data[claims_col], errors="coerce")
        obs = raw_data.loc[(cl > 0).fillna(False).to_numpy(dtype=bool)]
        num, den = loss_col, claims_col
    g = obs.groupby(obs[pred_var], sort=True, dropna=False, observed=True)
    s_num = g[num].sum()
    s_den = g[den].sum()
    obs_agg = pd.DataFrame({".x": s_num.index,
                            ".obs": s_num.to_numpy(dtype=float)
                            / s_den.to_numpy(dtype=float)
                            if len(s_num) else np.array([], dtype=float),
                            ".exposure": s_den.to_numpy(dtype=float)})
    if is_categorical(obs[pred_var]):
        obs_agg[".x"] = pd.Categorical(s_num.index, categories=obs[pred_var].cat.categories)
    v = obs_agg[".obs"].to_numpy(dtype=float)
    obs_agg[".obs"] = np.where(np.isfinite(v), v, np.nan)
    # data.table's order() puts NA first
    na = pd.isna(obs_agg[".x"]).to_numpy()
    obs_agg = pd.concat([obs_agg[na], obs_agg[~na]]).reset_index(drop=True)

    kept = U.kept_offsets([model], exposure_col, keep_offsets)
    if y_label is None:
        y_label = {"Frequency": "Frequency (claims / exposure)",
                   "Severity": "Severity (loss / claim)"}[metric]
        if kept:
            y_label = y_label + ", incl. " + ", ".join(kept)
    exposure_label = {"Frequency": "Exposure",
                      "Severity": "Number of claims"}[metric]

    # Training data, weights and offset -----------------------------------------
    tr = U.glm_training_data(model, "make_pdp")
    if tr.data is None:
        stop("make_pdp: training data cannot be recovered; fit the model with "
             "a 'data =' argument.")
    train_df = tr.data
    if pred_var not in train_df.columns:
        stop("make_pdp: predictor '", pred_var,
             "' not found in the model's training data.")

    has_offset = tr.offset is not None and bool(np.any(np.asarray(tr.offset) != 0))

    if metric == "Frequency":
        if has_offset:
            if model.link != "log":
                warn("make_pdp: an offset is present but the link is not "
                     "'log'; exp(offset) is then not an exposure and the "
                     "weighting is unreliable.")
            elif U.offset_is_log(model) is False:
                warn("make_pdp: the model's offset is not a logarithm, so "
                     "exp(offset) is not an exposure and the weighting is "
                     "unreliable.")
            # [PDP-02] [OFF-08]
            # The observed line and the bars are already built from
            # `exposure_col`, so the PDP average is weighted by the same
            # column rather than by exp(offset). The two differ once a model
            # carries a second offset, a known bonus-malus scale for
            # instance, and weighting the two halves of one chart differently
            # is not something a reader can be expected to notice.
            w_avg = (train_df[exposure_col].to_numpy(dtype=float)
                     if exposure_col in train_df.columns
                     else np.exp(np.asarray(tr.offset, dtype=float)))
        elif not np.all(np.asarray(tr.weights) == 1):
            w_avg = np.asarray(tr.weights, dtype=float)   # rate model with weights = exposure
        else:
            warn("make_pdp: no offset or weights found; the PDP is averaged "
                 "unweighted.")
            w_avg = np.ones(len(train_df))
    else:
        w_avg = np.asarray(tr.weights, dtype=float)       # severity: prior weights = claim counts

    # [OFF-06] [PDP-05]
    # Take the exposure offset out, so the PDP is a frequency per unit of
    # exposure, and leave any other offset in at each policy's own value: the
    # observed line it is drawn against is at each policy's own BM too.
    # Taking a BM offset out as well put the PDP at the class without
    # discount, 1.35 times the observed line at every level (measured with an
    # average BM of 0.74). offset_split() decides which expressions go, as
    # it does for premium_impact() and the lifts.
    #
    # The variables of the offsets that go are pinned to a constant, because
    # that is what lets the data collapse to unique predictor profiles below;
    # those offsets are then subtracted on the link scale by
    # predict_no_offset(), which is what makes this correct for
    # offset(Exposure) and offset(log(Months / 12)) as well as for the usual
    # offset(log(Exposure)). A kept offset's variables stay as they are and
    # become part of the profile.
    sp = U.offset_split(model, exposure_col, keep_offsets)
    nd0 = train_df.copy()
    if has_offset:
        ov = U.offset_vars(model)
        pin = []
        for i in sp.drop:
            for v in expr_vars(model.offset_exprs[i]):
                if v not in pin and v not in sp.kept:
                    pin.append(v)
        for v in pin:
            if v in nd0.columns:
                nd0[v] = 1.0
        if sp.mixed:
            warn("make_pdp: the exposure offset also holds '",
                 ", ".join(sp.mixed), "' in the same expression, so the two "
                 "cannot be separated and both are taken out. Write them as "
                 "separate terms, offset(log(", exposure_col, ")) + "
                 "offset(log(", sp.mixed[0], ")), to keep the second in.")
        absent = [v for v in ov if v not in nd0.columns]
        if absent:
            warn("make_pdp: offset variable(s) ", ", ".join(absent),
                 " are not in the training data; the offset cannot be "
                 "neutralised and the PDP is then NOT a frequency per unit "
                 "of exposure.")
        elif exposure_col not in ov:
            warn("make_pdp: the model offsets on '", ", ".join(ov),
                 "' rather than on '", exposure_col, "'; the PDP is per unit "
                 "of that variable while the observed line and the bars use '",
                 exposure_col, "'.")

    # [PDP-04]
    # Grid ------------------------------------------------------------------------
    x_train = train_df[pred_var].reset_index(drop=True)
    is_discrete = (not is_num(x_train)
                   or x_train.nunique(dropna=False) <= discrete_cutoff)
    if is_categorical(x_train):
        grid_vals = list(x_train.cat.categories)
    elif not is_num(x_train):
        grid_vals = r_sort_unique(x_train)
    elif x_train.nunique(dropna=False) <= grid_res:
        grid_vals = r_sort_unique(x_train)
    else:
        xv = x_train.to_numpy(dtype=float, na_value=np.nan)
        grid_vals = list(np.linspace(np.nanmin(xv), np.nanmax(xv), grid_res))

    # [PDP-03]
    # Collapse to unique predictor profiles (performance) -------------------------
    # predict() only depends on the model's predictor columns. Rows that
    # share the same values for all predictors other than pred_var receive
    # the same prediction on every grid point, so they are collapsed into
    # one profile with summed weight. On typical pricing data (categorical
    # rating factors, integer ages, exposure already neutralised to 1) this
    # reduces the work from grid_res x n rows to grid_res x (profiles),
    # while leaving the weighted mean exactly unchanged.
    # The offset variables count too: an offset that is not in the formula's
    # terms, and a kept one, changes the prediction per row
    vars_needed = list(dict.fromkeys(
        U.rhs_vars(model) + [v for v in U.offset_vars(model) if v in nd0.columns]))
    missing_vars = [v for v in vars_needed if v not in nd0.columns]
    if missing_vars:
        stop("make_pdp: column(s) required for prediction not found in the "
             "training data: ", ", ".join(missing_vars))
    key_vars = [v for v in vars_needed if v != pred_var]
    if key_vars:
        dtc = nd0[key_vars].reset_index(drop=True).copy()
        dtc[".w"] = w_avg
        prof = (dtc.groupby(key_vars, sort=False, dropna=False, observed=True)
                [".w"].sum().reset_index())
    else:
        # pred_var is the only predictor: a single profile suffices
        prof = pd.DataFrame({".w": [float(np.sum(w_avg))]})
    w_prof = prof[".w"].to_numpy(dtype=float)
    prof = prof.drop(columns=".w")

    # PDP: weighted mean response prediction per grid value -----------------------
    # [PDP-01]
    # Deliberately an arithmetic mean on the response scale; exp(mean(link))
    # would be a geometric mean that is systematically biased low.
    yhat = np.empty(len(grid_vals))
    for i, gv in enumerate(grid_vals):
        nd = prof.copy()
        nd[pred_var] = U.coerce_like([gv] * len(nd), x_train)
        yhat[i] = wmean(U.predict_no_offset(model, nd, sp.drop), w_prof,
                        na_rm=True)

    pdx = (pd.Categorical(grid_vals, categories=x_train.cat.categories)
           if is_categorical(x_train) else pd.Series(grid_vals))
    pdd = pd.DataFrame({".x": pdx, ".yhat": yhat})

    scatter_mode = "markers" if is_discrete else "lines+markers"

    # Plot ------------------------------------------------------------------------
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=plot_x(obs_agg[".x"]),
        y=obs_agg[".exposure"].to_numpy(dtype=float),
        name=exposure_label,
        yaxis="y2",
        marker=dict(color=ta_lightblue, opacity=0.5, line=dict(width=0)),
        hovertemplate=("<b>" + pred_var + ":</b> %{x}"
                       "<br><b>" + exposure_label + ":</b> %{y:,.0f}"
                       "<extra></extra>")))
    fig.add_trace(go.Scatter(
        x=plot_x(obs_agg[".x"]),
        y=obs_agg[".obs"].to_numpy(dtype=float),
        mode=scatter_mode,
        name="Observed",
        yaxis="y",
        line=dict(color=ta_gold, width=1.8, dash="dot"),
        marker=dict(color=ta_gold, size=6),
        hovertemplate=("<b>" + pred_var + ":</b> %{x}"
                       "<br><b>Observed:</b> %{y:." + str(metric_fmt) + "f}"
                       "<extra></extra>")))
    fig.add_trace(go.Scatter(
        x=plot_x(pdd[".x"]),
        y=pdd[".yhat"].to_numpy(dtype=float),
        mode=scatter_mode,
        name="PDP (model)",
        yaxis="y",
        line=dict(color=ta_blue, width=2.2),
        marker=dict(color=ta_blue, size=6),
        hovertemplate=("<b>" + pred_var + ":</b> %{x}"
                       "<br><b>PDP:</b> %{y:." + str(metric_fmt) + "f}"
                       "<extra></extra>")))
    fig.update_layout(
        title=dict(text="Partial Dependence – " + pred_var + " – " + metric,
                   font=dict(color=ta_navy, size=14)),
        xaxis=fill_axis(dict(title=pred_var, tickangle=-45,
                             tickfont=dict(size=9), showgrid=False),
                        category_axis(obs_agg[".x"], obs_agg[".x"], pdd[".x"])),
        yaxis=dict(title=y_label, gridcolor="#D0D8E0", zeroline=False,
                   range=y_range, autorange=y_range is None),
        yaxis2=dict(title=dict(text=exposure_label, font=dict(color=ta_muted)),
                    overlaying="y", side="right", showgrid=False,
                    tickfont=dict(color=ta_muted)),
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        barmode="overlay",
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=80, r=80))
    return set_config(fig, "pdp")
