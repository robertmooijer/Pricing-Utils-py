"""Portfolio-level model comparison; the port of R/lift.R.

Every other diagnostic in the package looks at one variable or one pair of
variables. These two answer the question those cannot: does the model
separate risk across the book, and is the candidate better than what it
replaces.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _utils as U
from ._core import Result, set_config, stop, warn
from .colors import ta_blue, ta_gold, ta_muted, ta_navy


def model_lift(model_freq=None, model_sev=None, data=None,
               actual_col="SCHADELAST",
               exposure_col="Exposure",
               claims_col="AantalClaims",
               weight_col=None,
               n_bins=10, y_range=None,
               keep_offsets=True,
               n_boot=200, conf_level=0.90, seed=None,
               at=None):
    """Lift chart and Gini for a risk premium model.

    Sorts the portfolio by predicted risk premium, splits it into bins of
    equal exposure, and compares the actual with the predicted rate in each.
    A model that separates risk produces a rising actual line that tracks
    the predicted one; a flat actual line means the model orders nothing.

    Bins hold equal **exposure** rather than equal policy counts, so each
    point carries the same weight in the comparison, and both series are on
    the same scale so they share one axis.

    Parameters
    ----------
    model_freq, model_sev : GLMFit, optional
        Fitted models. Supply both for a risk premium, or one for a
        frequency- or severity-only lift. The exposure offset is taken out,
        so the prediction is a rate per unit of exposure; any other offset
        stays in by default (see `keep_offsets`).
    data : pandas.DataFrame, optional
        Data to evaluate on. None (default) uses the model's own rows, which
        is in-sample and is labelled as such on the plot; pass a holdout for
        an honest out-of-sample lift.
    actual_col : str
        Column with the realised amount to compare against: the loss amount
        for a risk premium or severity model, the claim count for a
        frequency model.
    exposure_col : str
        Exposure column (default ``"Exposure"``). This is also the column
        held at 1 to neutralise a model's offset.
    claims_col : str
        Claim count column (default ``"AantalClaims"``), used as the weight
        for a severity-only lift.
    weight_col : str, optional
        Column to bin and weight on. None (default) picks it from the
        models: `claims_col` for a severity-only lift, otherwise
        `exposure_col`. A severity model predicts a mean **per claim**, so
        weighting it by exposure multiplies a per-claim amount by a per-year
        one and the totals do not reconcile; frequency and risk premium are
        rates per exposure unit and keep the exposure. Set this explicitly
        only to override that choice.
    n_bins : int
        Number of equal-weight bins (default 10).
    y_range : (lo, hi), optional
        Fix the rate axis.
    keep_offsets : bool
        True (default): only the exposure offset is taken out, and any other
        offset - a bonus-malus scale carried as ``offset(log(BM))`` - stays
        in the prediction. The claims being compared against happened at
        each policy's own BM, so the prediction has to include it: taking it
        out as well put every policy at the class without discount, and a
        correct model then read about as much too expensive as the average
        discount, in every bin. False takes every offset out.
        ``stats.offsets_kept`` and the axis title say which stayed in.
    n_boot : int
        Bootstrap replicates for an interval around the Gini (default 200;
        0 switches it off). A Gini on raw losses is dominated by a few large
        claims: on a 16,000-policy holdout the 90% interval ran from 0.14 to
        0.33 around 0.23, wider than the difference between two models one
        might want to choose between. Policies are resampled with Poisson(1)
        counts, so the book is sorted once and a replicate is one pass over
        it. The interval is in the title and in ``gini_ci``. When it is
        wider than half the Gini and wider than 0.05 **and** the ten largest
        records carry 10% or more of `actual_col`, a warning says so, since
        capping large losses is then the usual remedy; a wide interval on a
        small book has no such remedy and only shows in the title.
    conf_level : float
        Coverage of that interval (default 0.90).
    seed : int, optional
        Seed for the bootstrap, for ``numpy.random.default_rng(seed)``.
    at : dict, optional
        Values to price every policy at, e.g. ``{"BOEKJAAR": 2024}``; see
        :func:`premium_impact`. For a lift against the claims of the data
        itself this is normally not wanted: those claims happened in their
        own year.

    Returns
    -------
    Result
        ``table`` (one row per bin), ``gini``, ``gini_ci``, ``stats``,
        ``plot`` (the lift chart) and ``plot_lorenz`` (the curve behind the
        Gini). The ``Exposure`` and ``ExposureShare`` columns hold whatever
        `weight_col` resolved to, so on a severity-only lift they are claim
        counts; the axis titles name it.

    Notes
    -----
    ``gini`` is computed on the exposure-weighted Lorenz curve: cumulative
    share of actual losses against cumulative share of exposure, ordered by
    the prediction. Zero means the model does not discriminate at all;
    higher means it separates risk better. It measures **ordering only** -
    a model can have an excellent Gini and still be badly calibrated, which
    is what the lift chart itself shows. Note this is not the classical
    income Gini, which orders by the variable being measured. A window with
    no losses at all has no Lorenz curve to measure, and ``gini`` is then
    NaN (R's NA), printed as "NA" in the titles.

    Bins are cut on the ordered predictions, so with fewer distinct
    predicted values than bins the policies sharing a prediction are split
    across bins in arbitrary order. That is reported with a warning rather
    than drawn as if the model had that resolution.

    See Also
    --------
    double_lift : to compare two candidate tariffs.
    """
    fn = "model_lift"
    if model_freq is None and model_sev is None:
        stop("model_lift: provide at least one model.")
    U.check_models(fn, model_freq=model_freq, model_sev=model_sev)
    if not isinstance(keep_offsets, (bool, np.bool_)):
        stop("model_lift: 'keep_offsets' must be TRUE or FALSE.")
    keep_offsets = bool(keep_offsets)
    U.check_boot(n_boot, conf_level, fn)
    n_boot = int(n_boot)
    # [CONV-08]
    # [PY-05] a Generator per call, never the global numpy random state
    rng = np.random.default_rng(seed)

    # [RATE-07]
    # The weight has to be the denominator the prediction is a rate over. A
    # severity model predicts a mean per claim, so weighting it by exposure
    # multiplies a per-claim amount by a per-year one and the totals do not
    # reconcile at all. Frequency and risk premium are rates per exposure
    # unit, so those keep the exposure.
    sev_only = model_freq is None and model_sev is not None
    if weight_col is None:
        weight_col = claims_col if sev_only else exposure_col
    y_range = U.check_range(y_range, fn)
    if n_bins < 2:
        stop("model_lift: 'n_bins' must be at least 2.")

    in_sample = data is None
    if in_sample:
        src = U.glm_training_data(model_freq if model_freq is not None
                                  else model_sev, fn)
        if src.data is None:
            stop("model_lift: training data cannot be recovered; fit the "
                 "model with a 'data =' argument or pass 'data' explicitly.")
        data = src.data
    else:
        data = U.as_df(data)
    U.check_cols(data, [actual_col, exposure_col, weight_col], fn)

    # [RT-13]
    U.nil_claims_warning(model_freq, model_sev, data, claims_col, actual_col,
                         fn)
    # [OFF-05] [RATE-10]
    pd_ = U.apply_at(data, at, fn)
    rate = (np.asarray(U.model_rate(model_freq, pd_, exposure_col, fn,
                                    keep_offsets, "frequency"), dtype=float)
            * np.asarray(U.model_rate(model_sev, pd_, exposure_col, fn,
                                      keep_offsets, "severity"), dtype=float))
    kept = U.kept_offsets([model_freq, model_sev], exposure_col, keep_offsets)
    w = _num(data[weight_col])
    act = _num(data[actual_col])

    with np.errstate(invalid="ignore"):
        ok_rate = np.isfinite(rate)
        ok_w = np.isfinite(w) & (w > 0)
        ok_act = np.isfinite(act)
    keep = ok_rate & ok_w & ok_act
    n_drop = int(np.sum(~keep))
    # [RATE-05]
    if n_drop > 0:
        warn("model_lift: ", n_drop, " row(s) dropped (",
             U.drop_reasons({"no usable prediction": ~ok_rate,
                             "no positive '%s'" % weight_col: ~ok_w,
                             "missing '%s'" % actual_col: ~ok_act}),
             ").")
    if not keep.any():
        stop("model_lift: no usable rows left.")
    rate = rate[keep]
    w = w[keep]
    act = act[keep]

    U.check_ae_scale(float(np.sum(act)), float(np.sum(rate * w)), fn,
                     actual_col)

    b = U.exposure_bins(rate, w, n_bins, fn)
    bins = np.unique(b)
    ex = np.bincount(b, weights=w)[bins]
    ac = np.bincount(b, weights=act)[bins]
    pr = np.bincount(b, weights=rate * w)[bins]
    with np.errstate(divide="ignore", invalid="ignore"):
        tbl = pd.DataFrame({
            "Bin": bins,
            "Exposure": ex,
            "ExposureShare": ex / np.sum(ex),
            "ActualRate": ac / ex,
            "PredictedRate": pr / ex,
            "AE": np.where(pr > 0, ac / pr, np.nan)})

    gini = U.gini_exposure(rate, act, w)
    # [RATE-11]
    # A point Gini on raw losses moves with a handful of large claims, so it
    # comes with an interval. A wide one is worth a warning when large losses
    # are the reason, because capping them is then the remedy; a wide one on
    # a small book is plain sampling noise and shows in the title only.
    gini_ci = np.asarray(U.gini_boot(rate, act, w, n_boot, conf_level, rng),
                         dtype=float)
    note = U.large_loss_note(act, actual_col)
    if np.all(np.isfinite(np.r_[gini, gini_ci])) and note \
            and gini_ci[1] - gini_ci[0] > max(0.5 * abs(gini), 0.05):
        warn("model_lift: the %.0f%% interval for the Gini runs from %.3f to "
             "%.3f around %.3f, so the point value says little on its own."
             % (100 * conf_level, gini_ci[0], gini_ci[1], gini), note)
    lbl = " (in-sample)" if in_sample else " (out-of-sample)"
    ci_lab = (" (%.0f%%: %.3f to %.3f)" % (100 * conf_level, gini_ci[0],
                                           gini_ci[1])
              if np.all(np.isfinite(gini_ci)) else "")
    # Every axis names the weight actually in force, so a severity lift does
    # not claim to be per unit of exposure
    w_lab = "exposure" if weight_col == exposure_col else weight_col
    rate_lab = ("Rate per unit of " + w_lab
                + (", incl. " + ", ".join(kept) if kept else ""))

    ht = "<b>Bin %{x}</b><br>%{data.name}: %{y:.2f}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tbl["Bin"], y=tbl["PredictedRate"], name="Predicted",
        mode="lines+markers",
        line=dict(color=ta_gold, width=2, dash="dash"),
        marker=dict(color=ta_gold, size=8, symbol="diamond"),
        hovertemplate=ht))
    fig.add_trace(go.Scatter(
        x=tbl["Bin"], y=tbl["ActualRate"], name="Actual",
        mode="lines+markers",
        line=dict(color=ta_blue, width=2),
        marker=dict(color=ta_blue, size=8),
        hovertemplate=ht))
    fig.update_layout(
        title=dict(text=("Lift – " + U._txt(n_bins) + " bins of equal "
                         + w_lab + lbl + "  ·  Gini " + _f3(gini)
                         + ci_lab),
                   font=dict(color=ta_navy, size=14)),
        xaxis=dict(title="Bin, ordered by predicted rate",
                   dtick=1, showgrid=False),
        yaxis=dict(title=rate_lab,
                   gridcolor="#D0D8E0", zeroline=False,
                   range=y_range, autorange=y_range is None),
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=70, r=40))
    set_config(fig, "lift")

    o = np.argsort(rate, kind="stable")
    with np.errstate(divide="ignore", invalid="ignore"):
        cx = np.r_[0.0, np.cumsum(w[o]) / np.sum(w)]
        cy = np.r_[0.0, np.cumsum(act[o]) / np.sum(act)]
    # unique(round(seq(1, length(cx), length.out = 400))), 0-based
    step = np.unique(np.round(np.linspace(1, len(cx), 400))).astype(int) - 1
    p_lor = go.Figure()
    p_lor.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], name="No discrimination", mode="lines",
        line=dict(color=ta_muted, width=1, dash="dot"),
        hoverinfo="skip"))
    p_lor.add_trace(go.Scatter(
        x=cx[step], y=cy[step], name="Lorenz", mode="lines",
        line=dict(color=ta_navy, width=2),
        hovertemplate=(w_lab + ": %{x:.1%}"
                       "<br>Losses: %{y:.1%}<extra></extra>")))
    p_lor.update_layout(
        title=dict(text="Lorenz curve%s  ·  Gini %s%s"
                        % (lbl, _f3(gini), ci_lab),
                   font=dict(color=ta_navy, size=14)),
        xaxis=dict(title="Cumulative " + w_lab, tickformat=".0%",
                   showgrid=False),
        yaxis=dict(title="Cumulative losses", tickformat=".0%",
                   gridcolor="#D0D8E0", zeroline=False),
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=70, r=40))
    set_config(p_lor, "lorenz")

    return Result(table=tbl, gini=gini, gini_ci=gini_ci,
                  stats=Result(gini=gini, gini_ci=gini_ci,
                               conf_level=conf_level, n_boot=n_boot,
                               in_sample=in_sample, n_bins=n_bins,
                               n_rows=len(rate), n_dropped=n_drop,
                               total_actual=float(np.sum(act)),
                               total_predicted=float(np.sum(rate * w)),
                               offsets_kept=list(kept)),
                  plot=fig, plot_lorenz=p_lor)


def double_lift(data,
                model_freq_new=None, model_sev_new=None,
                model_freq_old=None, model_sev_old=None,
                old_premium_col=None,
                old_premium_basis="amount",
                actual_col="SCHADELAST",
                exposure_col="Exposure",
                claims_col="AantalClaims",
                weight_col=None,
                n_bins=10, rebase=True, y_range=None,
                keep_offsets=True,
                n_boot=200, conf_level=0.90, seed=None,
                at=None):
    """Double lift chart: which of two tariffs is right where they disagree.

    Sorts the portfolio by the ratio of the two predicted rates, bins it by
    equal exposure, and shows the actual-over-expected of **each** model per
    bin. The bins at the ends are the policies where the two tariffs
    disagree most, and that is where the comparison is decided: the model
    whose line stays closer to 1.0 across the range is the better one.

    This is the test a single lift chart cannot do. Both models can look
    convincing on their own lift chart while still disagreeing sharply about
    individual policies, and only their disagreement reveals which one is
    right.

    By default both sides are rebased to the same total, so the chart is
    about differentiation rather than rate level.

    Parameters
    ----------
    data : pandas.DataFrame
        Data to evaluate on.
    model_freq_new, model_sev_new : GLMFit, optional
        The candidate models.
    model_freq_old, model_sev_old : GLMFit, optional
        The incumbent models, or:
    old_premium_col : str, optional
        Column holding the current premium.
    old_premium_basis : {"amount", "rate"}
        ``"amount"`` (default; divided by the weight column, normally the
        exposure, internally) or ``"rate"`` (already per unit of it).
    actual_col : str
        Column with the realised amount (default ``"SCHADELAST"``).
    exposure_col : str
        Exposure column (default ``"Exposure"``).
    claims_col : str
        Claim count column (default ``"AantalClaims"``), the weight when
        both sides are severity models.
    weight_col : str, optional
        Column to bin and weight on. None (default) picks it as
        :func:`model_lift` does: `claims_col` when both sides are severity
        models, otherwise `exposure_col`. Set it only to override that
        choice.
    n_bins : int
        Number of equal-weight bins (default 10).
    rebase : bool
        True (default) scales the new rates so the weighted totals match,
        isolating differentiation from level.
    y_range : (lo, hi), optional
        Fix the A/E axis.
    keep_offsets : bool
        True (default): only the exposure offset is taken out of both rates,
        and any other offset - a bonus-malus scale - stays in, as it must
        for an A/E against claims that happened at each policy's own BM. An
        old premium column must then include the discount too. False takes
        every offset out. See :func:`premium_impact`.
    n_boot : int
        Bootstrap replicates behind ``winner`` (default 200). 0 switches the
        bootstrap off, and ``winner`` then compares the two point values
        alone.
    conf_level : float
        Coverage of the interval (default 0.90).
    seed : int, optional
        Seed for the bootstrap, for ``numpy.random.default_rng(seed)``.
    at : dict, optional
        Values to price every policy at, e.g. ``{"BOEKJAAR": 2024}``; see
        :func:`premium_impact`. The models price the data with these values;
        an old premium column is read as it is.

    Returns
    -------
    Result
        ``table`` (per bin: weight, the mean rate ratio, and the A/E of each
        model; the ``Exposure`` column holds whatever `weight_col` resolved
        to), ``stats``, and ``plot``. ``stats.rate_level_change`` is
        measured before any rebasing, so it reports the difference in rate
        level whether or not `rebase` is in force. ``stats.mad_diff`` is
        ``mad_old - mad_new``, positive when the new model is closer to 1,
        ``stats.mad_diff_ci`` its interval, and ``stats.winner`` is
        ``"new"``, ``"old"``, ``"inconclusive"`` or ``"tie"``.

    Notes
    -----
    The verdict compares the mean absolute distance of each A/E series from
    1 across the bins. On raw losses a few large claims decide those means:
    on a 16,000-policy holdout the verdict was "old" on the losses as
    recorded and "new" on the same losses capped. The difference
    ``mad_old - mad_new`` therefore gets a bootstrap interval, and
    ``winner`` names a model only when that interval excludes 0; otherwise
    it is ``"inconclusive"``, with a warning when the ten largest records
    carry 10% or more of `actual_col`. Policies are resampled with
    Poisson(1) counts within the bins cut on the full data, and with
    ``rebase=True`` every replicate is rebased again.

    Bins hold equal **weight**. That is the exposure, unless both sides are
    severity models: a severity model predicts a mean per claim, so it is
    weighted by claims, as in :func:`model_lift`. A severity model on one
    side and a frequency or risk premium on the other is refused, because
    the ratio of a rate per claim to a rate per exposure unit compares
    nothing.

    See Also
    --------
    model_lift, premium_impact
    """
    fn = "double_lift"
    U.check_models(fn, model_freq_new=model_freq_new, model_sev_new=model_sev_new,
                   model_freq_old=model_freq_old, model_sev_old=model_sev_old)
    if old_premium_basis not in ("amount", "rate"):
        stop(fn, ": 'old_premium_basis' should be one of \"amount\", "
                 "\"rate\".")
    if not isinstance(keep_offsets, (bool, np.bool_)):
        stop("double_lift: 'keep_offsets' must be TRUE or FALSE.")
    keep_offsets = bool(keep_offsets)
    U.check_boot(n_boot, conf_level, fn)
    n_boot = int(n_boot)
    # [CONV-08]
    # [PY-05] a Generator per call, never the global numpy random state
    rng = np.random.default_rng(seed)
    y_range = U.check_range(y_range, fn)
    if n_bins < 2:
        stop("double_lift: 'n_bins' must be at least 2.")

    # [RATE-12]
    # The weight follows the models, as in model_lift(). Both sides must be
    # rates over the same thing: a severity is per claim, a frequency or a
    # risk premium per unit of exposure, and the ratio of the two orders the
    # book by something that is neither tariff.
    sev_new = model_freq_new is None and model_sev_new is not None
    has_old = model_freq_old is not None or model_sev_old is not None
    sev_old = model_freq_old is None and model_sev_old is not None
    if has_old and sev_new != sev_old:
        stop("double_lift: one side is a severity model and the other is "
             "not, so one rate is per claim and the other per unit of "
             "exposure. Compare severity with severity, or frequency x "
             "severity with frequency x severity.")
    if weight_col is None:
        weight_col = claims_col if sev_new else exposure_col
    data = U.as_df(data)
    U.check_cols(data, [exposure_col, weight_col, actual_col,
                        old_premium_col], fn)

    # An old premium amount is divided by the weight, so that rate x weight
    # gives the amount back, as it does for the models' side
    r = U.old_new_rates(data, model_freq_new, model_sev_new,
                        model_freq_old, model_sev_old, old_premium_col,
                        old_premium_basis, exposure_col, fn,
                        keep_offsets, at=at, divisor_col=weight_col)
    w = _num(data[weight_col])
    act = _num(data[actual_col])

    with np.errstate(invalid="ignore"):
        ok_new = np.isfinite(r.new)
        ok_old = np.isfinite(r.old) & (r.old > 0)
        ok_w = np.isfinite(w) & (w > 0)
        ok_act = np.isfinite(act)
    keep = ok_new & ok_old & ok_w & ok_act
    n_drop = int(np.sum(~keep))
    # [RATE-05]
    if n_drop > 0:
        warn("double_lift: ", n_drop, " row(s) dropped (",
             U.drop_reasons({"no usable new rate": ~ok_new,
                             "no usable old rate": ~ok_old,
                             "no positive '%s'" % weight_col: ~ok_w,
                             "missing '%s'" % actual_col: ~ok_act}),
             ").")
    if not keep.any():
        stop("double_lift: no usable rows left.")
    new = r.new[keep]
    old = r.old[keep]
    w = w[keep]
    act = act[keep]

    U.check_ae_scale(float(np.sum(act)), float(np.sum(old * w)), fn,
                     actual_col)

    # [RATE-03]
    # The rate-level difference is a property of the two tariffs, not of the
    # decision to rebase, so it is measured before any scaling. Deriving it
    # from scale_f would report exactly 0 whenever rebase = FALSE, which is
    # the one case where the level difference is still in the chart.
    rate_level_change = float(np.sum(new * w) / np.sum(old * w) - 1)
    scale_f = np.sum(old * w) / np.sum(new * w) if rebase else 1.0
    new_raw = new
    new = new * scale_f
    ratio = new / old

    b = U.exposure_bins(ratio, w, n_bins, fn)
    bins = np.unique(b)
    ex = np.bincount(b, weights=w)[bins]
    ac = np.bincount(b, weights=act)[bins]
    pnew = np.bincount(b, weights=new * w)[bins]
    pold = np.bincount(b, weights=old * w)[bins]
    rr = np.bincount(b, weights=ratio * w)[bins]
    with np.errstate(divide="ignore", invalid="ignore"):
        tbl = pd.DataFrame({
            "Bin": bins,
            "Exposure": ex,
            "RateRatio": rr / ex,
            "ActualRate": ac / ex,
            "AE_New": np.where(pnew > 0, ac / pnew, np.nan),
            "AE_Old": np.where(pold > 0, ac / pold, np.nan)})

    # [RATE-09]
    # Mean absolute distance from 1: the smaller, the better calibrated the
    # model is across the range where the two disagree
    mad_new = _nanmean(np.abs(tbl["AE_New"].to_numpy() - 1))
    mad_old = _nanmean(np.abs(tbl["AE_Old"].to_numpy() - 1))
    mad_diff = mad_old - mad_new

    # [RATE-11]
    # How much of that difference survives resampling the policies. The bins
    # stay as cut on the full data; every replicate reweights the policies
    # within them and, with rebase, rebases again. The interval is the spread
    # of the replicates laid around the point value, not their quantiles as
    # such: mean |A/E - 1| grows with noise, and more for the model whose A/E
    # sits near 1, so every replicate is pulled towards the old model and a
    # percentile interval can sit beside its own point value.
    diff_ci = np.array([np.nan, np.nan])
    if n_boot > 0:
        def mad_of(a, p):
            with np.errstate(divide="ignore", invalid="ignore"):
                return _nanmean(np.abs(np.where(p > 0, a / p, np.nan) - 1))
        pn0 = new_raw * w
        po0 = old * w
        nb = int(b.max()) + 1
        sims = np.empty(n_boot)
        for i in range(n_boot):
            k = rng.poisson(1.0, len(act))
            pn = pn0 * k
            po = po0 * k
            if rebase and np.sum(pn) > 0:
                pn = pn * np.sum(po) / np.sum(pn)
            sa = np.bincount(b, weights=act * k, minlength=nb)[bins]
            sn = np.bincount(b, weights=pn, minlength=nb)[bins]
            so = np.bincount(b, weights=po, minlength=nb)[bins]
            sims[i] = mad_of(sa, so) - mad_of(sa, sn)
        sims = sims[~np.isnan(sims)]
        if len(sims):
            diff_ci = mad_diff + np.quantile(sims - sims.mean(),
                                             [(1 - conf_level) / 2,
                                              1 - (1 - conf_level) / 2])

    # [RATE-09] [RATE-11]
    # A winner only where the interval excludes 0. Without a bootstrap the
    # point values decide, as they did before the interval existed.
    boot_ok = n_boot > 0 and bool(np.all(np.isfinite(diff_ci)))
    if _all_equal(mad_new, mad_old):
        winner = "tie"
    elif boot_ok:
        winner = ("new" if diff_ci[0] > 0 else "old" if diff_ci[1] < 0
                  else "inconclusive")
    else:
        winner = "new" if mad_new < mad_old else "old"
    # Said out loud when large losses decide it, the case where the point
    # values flip with capping; otherwise the verdict and the title say it
    note = U.large_loss_note(act, actual_col)
    if winner == "inconclusive" and note:
        warn("double_lift: the point values favour the %s model (mean "
             "|A/E - 1| new %.3f, old %.3f), but the %.0f%% interval for the "
             "difference runs from %.3f to %.3f and includes 0, so 'winner' "
             "is \"inconclusive\"."
             % ("new" if mad_new < mad_old else "old", mad_new, mad_old,
                100 * conf_level, diff_ci[0], diff_ci[1]), note)
    verdict = ("<br><sup>old - new %.3f (%.0f%%: %.3f to %.3f): %s</sup>"
               % (mad_diff, 100 * conf_level, diff_ci[0], diff_ci[1], winner)
               if boot_ok else "")

    def ht(nm):
        return ("<b>Bin %{x}</b><br>" + nm + ": %{y:.3f}"
                "<br>Rate ratio: %{text:.3f}<extra></extra>")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=tbl["Bin"], y=tbl["AE_Old"], name="A/E old",
        mode="lines+markers",
        line=dict(color=ta_muted, width=2, dash="dash"),
        marker=dict(color=ta_muted, size=8, symbol="diamond"),
        text=tbl["RateRatio"], hovertemplate=ht("A/E old")))
    fig.add_trace(go.Scatter(
        x=tbl["Bin"], y=tbl["AE_New"], name="A/E new",
        mode="lines+markers",
        line=dict(color=ta_blue, width=2),
        marker=dict(color=ta_blue, size=8),
        text=tbl["RateRatio"], hovertemplate=ht("A/E new")))
    fig.update_layout(
        title=dict(
            text=("Double lift – mean |A/E - 1|: new %s vs old %s%s"
                  % (_f3(mad_new), _f3(mad_old), verdict)),
            font=dict(color=ta_navy, size=14)),
        xaxis=dict(title=("Bin, ordered by new/old rate ratio "
                          "(left = new is cheaper)"),
                   dtick=1, showgrid=False),
        yaxis=dict(title="Actual / Expected", gridcolor="#D0D8E0",
                   zeroline=False, range=y_range,
                   autorange=y_range is None),
        shapes=[dict(type="line", xref="paper", x0=0, x1=1,
                     y0=1, y1=1,
                     line=dict(color=ta_muted, width=1, dash="dot"))],
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=80, r=40))
    set_config(fig, "double-lift")

    return Result(table=tbl,
                  stats=Result(mad_new=mad_new, mad_old=mad_old,
                               mad_diff=mad_diff, mad_diff_ci=diff_ci,
                               winner=winner, conf_level=conf_level,
                               n_boot=n_boot, rebase=bool(rebase),
                               rate_level_change=rate_level_change,
                               n_bins=n_bins, n_rows=len(new),
                               n_dropped=n_drop, weight_col=weight_col,
                               offsets_kept=list(r.kept)),
                  plot=fig)


# ---- helpers ---------------------------------------------------------------


def _num(x):
    """as.numeric() of a column: NaN where a value is not a number."""
    return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)


def _nanmean(x):
    """mean(x, na.rm = TRUE): NaN when nothing is left, without a warning."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(np.mean(x)) if len(x) else float("nan")


def _f3(x):
    """sprintf("%.3f", x), which writes NA for a missing value."""
    return "NA" if x is None or not np.isfinite(x) else "%.3f" % x


def _all_equal(a, b, tol=np.sqrt(np.finfo(float).eps)):
    """isTRUE(all.equal(a, b)) for two numbers: equal, both NA, or within a
    relative (absolute near 0) tolerance of 1.5e-8."""
    na_a, na_b = bool(np.isnan(a)), bool(np.isnan(b))
    if na_a or na_b:
        return na_a and na_b
    if a == b:
        return True
    xy = abs(a - b)
    xn = abs(a)
    if np.isfinite(xn) and xn > tol:
        xy = xy / xn
    return bool(xy <= tol)
