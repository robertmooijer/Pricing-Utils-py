"""Premium impact (dislocation) analysis; the port of R/impact.R."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _utils as U
from ._core import Result, set_config, stop, warn
from ._formula import eval_expr
from .colors import ta_blue, ta_gold, ta_muted, ta_navy


def premium_impact(data,
                   model_freq_new=None, model_sev_new=None,
                   model_freq_old=None, model_sev_old=None,
                   old_premium_col=None,
                   old_premium_basis="amount",
                   rebase=True,
                   by=None,
                   n_show=10,
                   exposure_col="Exposure",
                   spotlight=None,
                   x_range=None,
                   y_range=None,
                   keep_offsets=True,
                   at=None,
                   year_col="BOEKJAAR"):
    """Premium impact (dislocation) analysis.

    Compares premiums under a new model set against an old model set (or an
    existing premium column) on a per-policy basis. Premiums are computed as
    rates per unit of exposure: the exposure offset is taken out, so every
    policy is priced for a full year, while any other offset - a bonus-malus
    scale - stays in by default (see `keep_offsets`). The premium is the
    product of the models supplied (frequency x severity, or a single model
    when only one side is modelled).

    By default the new premiums are rebased so that the exposure-weighted
    totals match the old ones. That isolates the redistribution
    (dislocation) from the overall rate-level change, which is reported
    separately in the summary.

    Parameters
    ----------
    data : pandas.DataFrame
        Dataset with raw rows.
    model_freq_new, model_sev_new : GLMFit, optional
        The new (candidate) models.
    model_freq_old, model_sev_old : GLMFit, optional
        The old models, or:
    old_premium_col : str, optional
        Column in `data` with the current premium. It has to be on the same
        footing as the new rate: with the default ``keep_offsets=True`` that
        is the premium **including** the bonus-malus discount, which is what
        a policy record normally holds.
    old_premium_basis : {"amount", "rate"}
        ``"amount"`` (default; premium for the record, divided by exposure
        internally) or ``"rate"`` (already per unit of exposure).
    rebase : bool
        True = scale new premiums to the old total (default).
    by : str or list of str, optional
        Columns for a per-level impact breakdown. Per level it gives the
        exposure-weighted mean change of the policies (``MeanChangePct``,
        what a customer in that level sees on average), the change of the
        level's total premium (``TotalChangePct``), its share of the old
        premium (``PremiumShare``) and
        ``Contribution = PremiumShare x TotalChangePct``: the level's part
        of the change in the total premium. The contributions of one
        variable sum to that change exactly - to 0 after a rebase - and so
        answer which segment drives it.
    n_show : int
        Number of rows in the winners/losers tables (default 10).
    exposure_col : str
        Exposure column (default ``"Exposure"``).
    spotlight : array-like of bool or str, optional
        Optional subset to report separately: either a boolean array or
        Series with one value per row of `data` (taken in row order, not
        aligned on the index), or a string with an R-style expression
        evaluated against the columns of `data`, e.g.
        ``"REGIO == 'Stad' & LEEFTIJD < 25"``. The expression is read with
        R's operator precedence (``&``, ``|`` and ``!`` bind more loosely
        than a comparison, as in R; ``&&`` and ``||`` are read as ``&`` and
        ``|``), ``%in%`` works (``"REGIO %in% c('Stad', 'Rand')"``), and it
        can refer only to columns of `data` and the R functions a formula
        knows (``log``, ``abs``, ``pmin``, ...). Where R falls back to the
        caller's environment for other names, put the value in the string
        or pass a boolean array. Missing values count as False. It is a
        spotlight rather than a filter: the whole book is still analysed,
        and the subset appears as a second series on the histogram and as
        ``spotlight`` in the result. Its statistics are computed after the
        book-level rebase, so they show how the subset moves relative to
        the portfolio.
    x_range, y_range : (lo, hi), optional
        Fix the axes of the returned histogram, so impact plots for
        different scenarios become directly comparable. `x_range` bounds the
        premium change (%) - the axis you normally want to align - and
        `y_range` the exposure axis. Both None (default) auto-scale. Note
        that `x_range` only sets the visible window; the bins are always
        computed over the full range of changes, so the summary statistics
        are unaffected.
    keep_offsets : bool
        True (default): only the exposure offset is taken out of the rate,
        and any other offset - a bonus-malus scale carried as
        ``offset(log(BM))`` - stays in, so the rate is what the policy pays
        for a year. False takes every offset out, which puts every policy at
        the class without discount.

        Keeping it in is what makes a comparison with a current premium
        fair. That premium includes the discount, and taking the discount
        out of the new side only makes an unchanged tariff look like a large
        shift onto the discounted classes: +33% for a 40% discount and -20%
        for none, on a tariff that changed nothing. Against old models with
        the same offset the choice does not matter for the changes, which
        are identical either way; it moves ``OldRate`` and ``NewRate``
        themselves.

        Two cases fall back to taking everything out: a model whose offset
        is not on `exposure_col` at all, and an offset that holds the
        exposure and something else in one expression,
        ``offset(log(Exposure * BM))``, which is reported in a warning. The
        ``summary`` names the offsets that stayed in.
    at : dict, optional
        Values to price every policy at, e.g. ``{"BOEKJAAR": 2024}``.
        Applied to the data the models score, not to an old premium column.
        With the accounting year as a rating factor each policy is otherwise
        priced at the level of its own historic year, which against a
        current premium mixes inflation into the dislocation: measured, a
        rate-level change of +11.1% on the historic years against +34.7% at
        the tariff year.
    year_col : str or None
        Accounting-year column (default ``"BOEKJAAR"``). When the new models
        use it against an old premium column, or only one of the two model
        sets uses it, and `at` does not set it, a warning says so. Two model
        sets that both use it price every policy at the same year on both
        sides, which is like for like. None switches the warning off.

    Returns
    -------
    Result
        ``summary`` (display table with the headline numbers), ``stats``
        (the same numbers as a Result; ``quantiles`` is a dict p5 .. p95),
        ``policy`` (per-row ``OldRate``, ``NewRate``, ``NewRateRebased``,
        ``ChangePct``; ``Row`` is the 0-based position of the row in `data`,
        for ``data.iloc``), ``by_level`` (per level: exposure and its share,
        old premium and its share, mean and total change, and contribution,
        when `by` is given), ``spotlight`` (the same headline numbers for
        the subset, when `spotlight` is given), ``largest_increases`` /
        ``largest_decreases`` (top-`n_show` dislocations) and ``plot``
        (exposure-weighted histogram of the premium changes, with the
        spotlight overlaid).

    Notes
    -----
    What is not decomposed: old and new premiums are compared on the *same*
    policies, so the portfolio mix is identical on both sides and there is
    no mix effect to isolate. ``Contribution`` therefore answers "which
    segment drives the overall change", not "how much of the change is
    mix". A genuine mix-shift analysis needs two portfolio snapshots and is
    a different calculation.
    """
    fn = "premium_impact"
    U.check_models(fn, model_freq_new=model_freq_new, model_sev_new=model_sev_new,
                   model_freq_old=model_freq_old, model_sev_old=model_sev_old)
    if old_premium_basis not in ("amount", "rate"):
        stop(fn, ": 'old_premium_basis' should be one of \"amount\", "
                 "\"rate\".")
    if not isinstance(keep_offsets, (bool, np.bool_)):
        stop("premium_impact: 'keep_offsets' must be TRUE or FALSE.")
    keep_offsets = bool(keep_offsets)
    x_range = U.check_range(x_range, fn, "x_range")
    y_range = U.check_range(y_range, fn)
    if isinstance(by, str):
        by = [by]
    by = list(by) if by is not None else None
    data = U.as_df(data)
    U.check_cols(data, [exposure_col] + (by or []) + [old_premium_col], fn)

    # The spotlight is evaluated against the full data, before any rows are
    # dropped, so the mask lines up with the rows it was written for.
    # A string is evaluated as an R expression against the columns of
    # `data`; anything else is taken as the boolean mask itself.
    spot = None
    if spotlight is not None:
        if isinstance(spotlight, str):
            try:
                s = eval_expr(_r_logical(spotlight), data)
            except Exception as e:    # noqa: BLE001 - reported with its text
                stop("premium_impact: 'spotlight' could not be evaluated "
                     "against 'data': ", e)
        else:
            s = spotlight
        spot = _as_mask(s, len(data))
        if spot is None:
            stop("premium_impact: 'spotlight' must give one TRUE/FALSE per "
                 "row of 'data'.")

    # [RATE-10]
    has_old_models = model_freq_old is not None or model_sev_old is not None
    U.year_warning([model_freq_new, model_sev_new],
                   [model_freq_old, model_sev_old] if has_old_models else None,
                   year_col, at, fn)
    # [OFF-05]
    r = U.old_new_rates(data, model_freq_new, model_sev_new,
                        model_freq_old, model_sev_old, old_premium_col,
                        old_premium_basis, exposure_col, fn,
                        keep_offsets, at=at)
    new_rate = r.new
    old_rate = r.old

    expo = pd.to_numeric(data[exposure_col], errors="coerce").to_numpy(
        dtype=float, na_value=np.nan)
    with np.errstate(invalid="ignore"):
        ok_new = np.isfinite(new_rate)
        ok_old = np.isfinite(old_rate) & (old_rate > 0)
        ok_exp = np.isfinite(expo) & (expo > 0)
    keep = ok_new & ok_old & ok_exp
    n_drop = int(np.sum(~keep))
    # [RATE-05]
    if n_drop > 0:
        warn("premium_impact: ", n_drop, " row(s) dropped (",
             U.drop_reasons({"no usable new premium": ~ok_new,
                             "no usable old premium": ~ok_old,
                             "no positive exposure": ~ok_exp}),
             ").")
    if not keep.any():
        stop("premium_impact: no usable rows left.")

    idx = np.flatnonzero(keep)
    new_rate = new_rate[keep]
    old_rate = old_rate[keep]
    expo = expo[keep]

    # [RATE-03]
    old_total = float(np.sum(old_rate * expo))
    new_total = float(np.sum(new_rate * expo))
    rate_change = new_total / old_total - 1
    scale_f = old_total / new_total if rebase else 1.0
    new_adj = new_rate * scale_f
    change = (new_adj / old_rate - 1) * 100

    qs = _wq(change, expo, [0.05, 0.25, 0.50, 0.75, 0.95])
    tot_w = np.sum(expo)
    share10 = float(np.sum(expo[np.abs(change) > 10]) / tot_w)
    share25 = float(np.sum(expo[np.abs(change) > 25]) / tot_w)

    policy = pd.DataFrame({"Row": idx})
    if by is not None:
        for v in by:
            policy[v] = data[v].iloc[idx].reset_index(drop=True)
    policy["Exposure"] = expo
    policy["OldRate"] = old_rate
    policy["NewRate"] = new_rate
    policy["NewRateRebased"] = new_adj
    policy["ChangePct"] = change

    # [RATE-04]
    # Contribution decomposition. The portfolio is the same on both sides, so
    # there is no mix effect to isolate; what can be decomposed is who causes
    # the overall change. A level's contribution is its part of the change in
    # the total premium, (new - old) / old total, so the levels of one
    # variable sum to that change exactly, and to 0 after a rebase.
    #
    # It used to be ExposureShare x the exposure-weighted mean change, which
    # sums to the mean change of a policy rather than of the premium: -4.0%
    # on a rebased book whose total had not moved, because a percentage on a
    # cheap policy weighs as much as one on an expensive policy. The mean
    # change per level is still reported, as what a customer in it sees.
    by_level = None
    if by is not None:
        po = old_rate * expo
        pn = new_adj * expo
        parts = []
        for v in by:
            codes, levels = _level_codes(data[v].iloc[idx])
            k = len(levels)
            ex = np.bincount(codes, weights=expo, minlength=k)
            chw = np.bincount(codes, weights=change * expo, minlength=k)
            op = np.bincount(codes, weights=po, minlength=k)
            np_ = np.bincount(codes, weights=pn, minlength=k)
            with np.errstate(divide="ignore", invalid="ignore"):
                parts.append(pd.DataFrame({
                    "Variable": v,
                    "Level": pd.Series(levels, dtype=object),
                    "Exposure": ex,
                    "MeanChangePct": chw / ex,
                    "OldPremium": op,
                    "NewPremium": np_,
                    "ExposureShare": ex / tot_w,
                    "PremiumShare": op / old_total,
                    "TotalChangePct": 100 * (np_ / op - 1),
                    "Contribution": 100 * (np_ - op) / old_total}))
        by_level = pd.concat(parts, ignore_index=True)
        # R's order(Variable, -abs(Contribution)): variables alphabetically,
        # within one the largest contribution first; stable, NA last
        ctb = by_level["Contribution"].to_numpy(dtype=float)
        o = sorted(range(len(by_level)),
                   key=lambda i: (_collate(by_level["Variable"].iat[i]),
                                  bool(np.isnan(ctb[i])),
                                  0.0 if np.isnan(ctb[i]) else -abs(ctb[i])))
        by_level = by_level.iloc[o].reset_index(drop=True)

    # [RATE-05]
    # Spotlight: the same headline numbers for a subset, computed AFTER the
    # book-level rebase, so it shows how that subset moves relative to the
    # book rather than relative to itself.
    spotlight_res = None
    sm = None
    if spot is not None:
        sm = spot[idx]
        if not sm.any():
            warn("premium_impact: 'spotlight' selects no usable rows.")
        else:
            sq = _wq(change[sm], expo[sm], [0.05, 0.50, 0.95])
            e_sm = expo[sm]
            spotlight_res = Result(
                n_rows=int(np.sum(sm)),
                exposure=float(np.sum(e_sm)),
                exposure_share=float(np.sum(e_sm) / tot_w),
                old_total=float(np.sum(old_rate[sm] * e_sm)),
                new_total=float(np.sum(new_adj[sm] * e_sm)),
                mean_change=float(np.sum(change[sm] * e_sm) / np.sum(e_sm)),
                median_change=float(sq[1]),
                p5_change=float(sq[0]),
                p95_change=float(sq[2]),
                share_gt_10=float(np.sum(e_sm[np.abs(change[sm]) > 10])
                                  / np.sum(e_sm)),
                contribution=float(100 * (np.sum(new_adj[sm] * e_sm)
                                          - np.sum(old_rate[sm] * e_sm))
                                   / old_total))
            spotlight_res.summary = pd.DataFrame({
                "Metric": ["Rows in spotlight", "Exposure share of the book",
                           "Mean change (exposure-weighted)", "Median change",
                           "P5 .. P95 change", "Exposure share |change| > 10%",
                           "Contribution to the portfolio change"],
                "Value": ["{:,}".format(spotlight_res.n_rows),
                          "%.1f%%" % (100 * spotlight_res.exposure_share),
                          "%+.2f%%" % spotlight_res.mean_change,
                          "%+.2f%%" % spotlight_res.median_change,
                          "%+.1f%% .. %+.1f%%" % (sq[0], sq[2]),
                          "%.1f%%" % (100 * spotlight_res.share_gt_10),
                          "%+.2f%%" % spotlight_res.contribution]})

    # R's order(ChangePct, decreasing = TRUE) is stable: ties keep their row
    # order, and the decreases are that order reversed
    ord_ = np.argsort(-change, kind="stable")
    n_top = int(min(n_show, len(policy)))
    largest_increases = policy.iloc[ord_[:n_top]].reset_index(drop=True)
    largest_decreases = policy.iloc[ord_[::-1][:n_top]].reset_index(drop=True)

    summary_df = pd.DataFrame({
        "Metric": ["Old total premium (rate x exposure)",
                   "New total premium (before rebase)",
                   "Overall rate-level change",
                   "Rebased to old total",
                   "Median change (exposure-weighted)",
                   "P5 .. P95 change (exposure-weighted)",
                   "Exposure share |change| > 10%",
                   "Exposure share |change| > 25%",
                   "Rows used / dropped",
                   "Offsets kept in the rate"],
        "Value": [_big(old_total),
                  _big(new_total),
                  "%+.1f%%" % (100 * rate_change),
                  "yes" if rebase else "no",
                  "%+.2f%%" % qs[2],
                  "%+.1f%% .. %+.1f%%" % (qs[0], qs[4]),
                  "%.1f%%" % (100 * share10),
                  "%.1f%%" % (100 * share25),
                  "%d / %d" % (len(idx), n_drop),
                  U.kept_label(r.kept)]})

    med = float(qs[2])

    # [RATE-05]
    # Pre-bin the histogram: a plotly histogram trace would embed all raw
    # per-policy values in the widget (tens of MB on large portfolios); 60
    # pre-computed bars keep the file small regardless of n.
    lo, hi = float(np.min(change)), float(np.max(change))
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    brks = np.linspace(lo, hi, 61)
    # cut(change, brks, include.lowest = TRUE): right-closed bins, the
    # lowest break included in the first
    bin_id = np.clip(np.searchsorted(brks, change, side="left") - 1, 0, 59)
    h_exp = np.bincount(bin_id, weights=expo, minlength=60)
    mids = (brks[:-1] + brks[1:]) / 2
    widths = np.diff(brks)

    bar_name = "Exposure" if spotlight_res is None else "Whole portfolio"
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mids, y=h_exp, width=widths, name=bar_name,
        marker=dict(color=ta_blue, opacity=0.75,
                    line=dict(color="white", width=0.5)),
        hovertemplate=("<b>Change:</b> %{x:.1f}%"
                       "<br><b>Exposure:</b> %{y:,.0f}"
                       "<extra></extra>")))

    if spotlight_res is not None:
        hs = np.bincount(bin_id[sm], weights=expo[sm], minlength=60)
        fig.add_trace(go.Bar(
            x=mids, y=hs, width=widths, name="Spotlight",
            marker=dict(color=ta_gold, opacity=0.85,
                        line=dict(color="white", width=0.5)),
            hovertemplate=("<b>Change:</b> %{x:.1f}%"
                           "<br><b>Exposure (spotlight):</b> %{y:,.0f}"
                           "<extra></extra>")))

    fig.update_layout(
        title=dict(text=("Premium impact"
                         + (" (rebased to old total)" if rebase else "")
                         + " – rate-level change %+.1f%%"
                         % (100 * rate_change)),
                   font=dict(color=ta_navy, size=14)),
        xaxis=dict(title="Premium change (%)", showgrid=False,
                   zeroline=False,
                   range=x_range, autorange=x_range is None),
        yaxis=dict(title="Exposure", gridcolor="#D0D8E0",
                   zeroline=False,
                   range=y_range, autorange=y_range is None),
        shapes=[
            dict(type="line", xref="x", yref="paper",
                 x0=0, x1=0, y0=0, y1=1,
                 line=dict(color=ta_muted, width=1, dash="dot")),
            dict(type="line", xref="x", yref="paper",
                 x0=med, x1=med, y0=0, y1=1,
                 line=dict(color=ta_gold, width=1.5, dash="dash"))],
        annotations=[dict(x=med, y=1, xref="x", yref="paper",
                          text="median %+.1f%%" % med,
                          showarrow=False, yanchor="bottom",
                          font=dict(color=ta_gold, size=11))],
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        barmode="overlay",
        legend=dict(orientation="h", y=-0.2),
        showlegend=spotlight_res is not None,
        plot_bgcolor="white", paper_bgcolor="white",
        # room above the plot for the median label, which otherwise runs
        # into the title
        margin=dict(t=60, b=60, r=40))
    set_config(fig, "impact")

    return Result(
        summary=summary_df,
        stats=Result(old_total=old_total, new_total=new_total,
                     rate_change=float(rate_change), rebase=bool(rebase),
                     quantiles=dict(zip(["p5", "p25", "p50", "p75", "p95"],
                                        [float(q) for q in qs])),
                     share_gt_10=share10, share_gt_25=share25,
                     n_rows=len(idx), n_dropped=n_drop,
                     offsets_kept=list(r.kept)),
        policy=policy,
        by_level=by_level,
        spotlight=spotlight_res,
        largest_increases=largest_increases,
        largest_decreases=largest_decreases,
        plot=fig)


# ---- helpers ---------------------------------------------------------------


def _wq(x, w, probs):
    """Exposure-weighted quantiles, as R's
    approx(cumsum(w[o]) / sum(w), x[o], xout = probs, rule = 2,
    ties = "ordered")."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o]) / np.sum(w)
    xs = x[o]
    return np.array([_approx1(p, cw, xs) for p in probs])


def _approx1(v, x, y):
    """R's C routine approx1() for linear interpolation with rule = 2: a
    bisection on x, which is taken as ordered."""
    n = len(x)
    if v < x[0]:
        return float(y[0])
    if v > x[n - 1]:
        return float(y[n - 1])
    i, j = 0, n - 1
    while i < j - 1:
        ij = (i + j) // 2
        if v < x[ij]:
            j = ij
        else:
            i = ij
    if v == x[j]:
        return float(y[j])
    if v == x[i]:
        return float(y[i])
    return float(y[i] + (y[j] - y[i]) * ((v - x[i]) / (x[j] - x[i])))


def _big(x):
    """format(round(x), big.mark = ",")."""
    return "{:,.0f}".format(float(np.round(x)) + 0.0)


def _collate(s):
    """Sort key for a column name close to R's locale collation: case
    first ignored, lower case before upper case on a tie."""
    s = str(s)
    return (s.casefold(), s.swapcase())


def _level_codes(x):
    """Group codes of a column in order of first appearance, keyed on the
    text R's as.character() gives (2024, not 2024.0); NA is a level of its
    own, None."""
    codes, uniq = pd.factorize(pd.Series(x).reset_index(drop=True),
                               use_na_sentinel=False)
    txt = [None if _isna(u) else U._txt(u) for u in uniq]
    # two values with one text (1 and 1.0 in an object column) are one level
    c2, lev = pd.factorize(pd.Series(txt, dtype=object),
                           use_na_sentinel=False)
    lev = [None if _isna(v) else v for v in lev]
    return np.asarray(c2)[codes], lev


def _isna(v):
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _as_mask(s, n):
    """A spotlight value as a boolean mask of length n (NA -> False), or
    None when it is not one TRUE/FALSE per row."""
    if isinstance(s, (bool, np.bool_)) or np.isscalar(s) or s is None:
        return None
    if isinstance(s, pd.DataFrame):
        return None
    if isinstance(s, (pd.Series, pd.Index, pd.Categorical)) \
            or hasattr(s, "dtype"):
        dt = getattr(s, "dtype", None)
        if not pd.api.types.is_bool_dtype(dt):
            if dt == object:
                s = list(np.asarray(s, dtype=object))
            else:
                return None
        else:
            arr = pd.array(s, dtype="boolean")
            if len(arr) != n:
                return None
            return np.asarray(arr.fillna(False), dtype=bool)
    vals = list(s)
    if len(vals) != n:
        return None
    out = np.zeros(n, dtype=bool)
    for i, v in enumerate(vals):
        if isinstance(v, (bool, np.bool_)):
            out[i] = bool(v)
        elif v is None or v is pd.NA or (isinstance(v, float) and np.isnan(v)):
            out[i] = False
        else:
            return None
    return out


def _r_logical(expr):
    """An R logical expression rewritten for Python's operator precedence.

    In R, ``&``, ``|`` and ``!`` bind more loosely than a comparison, so
    ``REGIO == "Stad" & LEEFTIJD < 25`` is two comparisons joined. Python
    binds ``&`` and ``|`` more tightly, so every operand is put in brackets;
    ``!`` becomes ``~``, ``&&`` / ``||`` become ``&`` / ``|``, and
    ``x %in% y`` becomes ``np.isin(x, y)``."""
    s = expr.strip()
    if not s:
        return s
    if s[0] == "(" and _close_of(s, 0) == len(s) - 1:
        return "(" + _r_logical(s[1:-1]) + ")"
    for op in ("|", "&"):
        parts = _split_op(s, op)
        if len(parts) > 1:
            return (" %s " % op).join("(" + _r_logical(p) + ")" for p in parts)
    if s.startswith("!") and not s.startswith("!="):
        return "~(" + _r_logical(s[1:]) + ")"
    parts = _split_op(s, "%in%")
    if len(parts) == 2:
        return "np.isin(%s, %s)" % (parts[0].strip(), parts[1].strip())
    return s


def _close_of(s, i):
    """Position of the bracket closing the one at s[i]."""
    depth, quote = 0, None
    for k in range(i, len(s)):
        ch = s[k]
        if quote:
            if ch == quote:
                quote = None
            continue
        if ch in "\"'`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                return k
    return -1


def _split_op(s, op):
    """Split at top-level occurrences of op (outside brackets and quotes);
    a doubled & or | counts as one."""
    parts, cur, depth, quote, i = [], [], 0, None, 0
    while i < len(s):
        ch = s[i]
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in "\"'`":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif depth == 0 and s.startswith(op, i):
            parts.append("".join(cur))
            cur = []
            i += len(op)
            if len(op) == 1 and s.startswith(op, i):
                i += 1
            continue
        cur.append(ch)
        i += 1
    parts.append("".join(cur))
    return parts
