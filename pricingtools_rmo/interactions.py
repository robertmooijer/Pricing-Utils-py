"""Interaction detection; the port of R/interactions.R.

[INT-01]
One-way A/E checks cannot reveal a missing interaction.

The score equations weight each observation by (dmu/deta)/V(mu), which is
exactly 1 for a canonical link. What remains is sum(x*(y - mu)) = 0 for
every column of the design matrix, so a categorical variable in the model
has an A/E of exactly 1 at every level - whatever happens inside the
cells. Poisson with a log link is canonical, so a frequency model is
blind by identity.

Gamma with a log link is NOT canonical (the canonical link for Gamma is
the inverse), so a severity model is not blind by identity. It is blind
in practice. The log link puts the score weight at mu/mu^2 = 1/mu, which
pins the weighted mean of the RATIO y/mu to 1 rather than the ratio of
the weighted totals: measured on 40,000 rows, sum(w*y/mu)/sum(w) sits at
3e-09 from 1 while the A/E itself sits at 1.2e-04. The residue that
leaves in the A/E is far too small to work with - an omitted interaction
of 35% moves the one-way A/E by 0.26%, invisible on an axis in the
thousands.

Either way, the two functions here look at the cells.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import _utils as U
from ._core import attach, message, set_config, stop, warn
from .colors import ta_muted, ta_navy
from .diagnostics import _r_format
from .model import GLMFit

# Diverging colour scale for A/E, centred on 1.0: two hues (the house navy
# for overpriced, red for underpriced) with a neutral grey midpoint. Never
# a hue at the midpoint and never a rainbow, so the reader cannot mistake a
# colour change for a change in direction.
_ta_diverging = [
    [0.00, "#00365E"], [0.15, "#0073AB"], [0.32, "#6FA3C7"],
    [0.44, "#C9DCEA"], [0.50, "#F2F2F1"], [0.56, "#EFD3CE"],
    [0.68, "#DC9E94"], [0.85, "#C44536"], [1.00, "#8E2A1F"]]


def _levels(g):
    """The levels of a grouped variable as R prints them."""
    return [U._txt(c) for c in g.categories]


def plot_residual_heatmap(model, var_x, var_y, n_bins=20, min_claims=30,
                          z_range=None, title=None):
    """Actual-vs-expected heatmap for a pair of variables.

    Shows the exposure-weighted A/E ratio of a fitted GLM per cell of
    ``var_x`` x ``var_y``. This is the check that a one-way plot cannot do:
    a missing interaction averages out in the margins while individual
    cells can be badly mispriced. Exactly so for a frequency model -
    Poisson with a log link is canonical, so the fitted total equals the
    observed total at every level of a categorical term in the model - and
    near enough for a Gamma severity, where an omitted interaction of 35%
    moves the one-way A/E by a quarter of a percent.

    Both variables are grouped with the same rule used elsewhere in the
    package: categorical as is, a numeric variable with at most ``n_bins``
    distinct values on its exact values, otherwise quantile bins.

    Cells backed by fewer than ``min_claims`` claims are left blank rather
    than coloured, and marked with a grey cross: in a thin cell the A/E is
    dominated by noise, and colouring it would be the loudest thing on the
    plot. Their volume is still shown in the tooltip.

    Parameters
    ----------
    model : GLMFit
    var_x, var_y : str
        Names of the two variables. They must be columns of the data the
        model was fitted on; the values are taken from the model's own
        rows, so they always align with the fitted values.
    n_bins : int
        Maximum number of groups per variable (default 20).
    min_claims : float
        Cells with fewer claims are shown as blank (default 30).
    z_range : (lo, hi), optional
        Fix the colour scale, so several heatmaps can be compared. ``None``
        (default) scales to this plot, still centred on 1.
    title : str, optional

    Returns
    -------
    plotly.graph_objects.Figure
        The underlying cell table (``gx``, ``gy``, ``Actual``,
        ``Expected``, ``Exposure``, ``Claims``, ``AE``, ``IsThin``) is
        attached as ``fig.cells``.

    See Also
    --------
    detect_interactions : rank all pairs before plotting one.
    """
    fn = "plot_residual_heatmap"
    if not isinstance(model, GLMFit):
        stop("plot_residual_heatmap: 'model' must be a glm object.")
    # [DATA-06] with na.exclude the expected values were padded and shifted
    # against the actuals, so the cells were silently wrong
    model = U.as_na_omit(model)
    z_range = U.check_range(z_range, fn, "z_range")

    parts = U.glm_ae_parts(model, fn)
    gx = U.group_values(U.glm_var_values(parts, var_x, fn), n_bins)
    gy = U.group_values(U.glm_var_values(parts, var_y, fn), n_bins)
    lx, ly = _levels(gx), _levels(gy)

    cx = np.asarray(gx.codes)
    cy = np.asarray(gy.codes)
    keep = (cx >= 0) & (cy >= 0)
    a = np.asarray(parts.actual, dtype=float)[keep]
    e = np.asarray(parts.expected, dtype=float)[keep]
    expo = np.asarray(parts.exposure, dtype=float)[keep]
    clm = np.asarray(parts.claims, dtype=float)[keep]
    cx, cy = cx[keep], cy[keep]
    # data.table's by = .(gx, gy): the cells in order of first appearance
    key = cx.astype(np.int64) * len(ly) + cy
    cell_id, uk = pd.factorize(key)
    k = len(uk)
    cells = pd.DataFrame({
        "gx": pd.Categorical.from_codes(np.asarray(uk) // len(ly), categories=lx),
        "gy": pd.Categorical.from_codes(np.asarray(uk) % len(ly), categories=ly),
        "Actual": np.bincount(cell_id, a, k),
        "Expected": np.bincount(cell_id, e, k),
        "Exposure": np.bincount(cell_id, expo, k),
        "Claims": np.bincount(cell_id, clm, k),
    })
    with np.errstate(divide="ignore", invalid="ignore"):
        cells["AE"] = np.where(cells["Expected"] > 0,
                               cells["Actual"] / cells["Expected"], np.nan)
    cells["IsThin"] = (cells["Claims"] < min_claims).to_numpy(dtype=bool)

    ix = np.asarray(cells["gx"].cat.codes)
    iy = np.asarray(cells["gy"].cat.codes)
    z = np.full((len(ly), len(lx)), np.nan)
    txt = np.full((len(ly), len(lx)), None, dtype=object)

    thin = cells["IsThin"].to_numpy(dtype=bool)
    # [INT-02]
    # Thin cells: no colour, only a marker and their volume in the tooltip
    z[iy, ix] = np.where(thin, np.nan, cells["AE"].to_numpy(dtype=float))
    ae_txt = ["NA" if np.isnan(v) else "%.3f" % v for v in cells["AE"]]
    clm_txt = _r_format(np.round(cells["Claims"].to_numpy(dtype=float)), ",")
    exp_txt = _r_format(np.round(cells["Exposure"].to_numpy(dtype=float)), ",")
    gxs = cells["gx"].astype(str).tolist()
    gys = cells["gy"].astype(str).tolist()
    cell_txt = [
        "<b>" + var_x + ":</b> " + gxs[i]
        + "<br><b>" + var_y + ":</b> " + gys[i]
        + "<br><b>A/E:</b> " + ae_txt[i]
        + "<br>Claims: " + clm_txt[i]
        + " · " + parts.exposure_label + ": " + exp_txt[i]
        + ("<br><i>too few claims - not coloured</i>" if thin[i] else "")
        for i in range(len(cells))]
    for i in range(len(cells)):
        txt[iy[i], ix[i]] = cell_txt[i]

    # [INT-02]
    # With every cell thin there is nothing to colour, and plotly cannot build
    # a heatmap whose values are all NA, whatever its colour range: it failed
    # with "Wasn't able to determine range of domain", which took a whole
    # report down on a small book. The colour layer is then left out and the
    # crosses, with their tooltips, are the plot.
    all_thin = bool(np.all(np.isnan(z)))
    if all_thin:
        warn("plot_residual_heatmap: every cell of ", var_x, " x ", var_y,
             " has fewer than ", U._txt(min_claims), " claims, so none is ",
             "coloured. Lower 'min_claims' or 'n_bins' to see more.")

    fig = go.Figure()
    if not all_thin:
        zl = [[None if np.isnan(v) else float(v) for v in row] for row in z]
        hm = dict(
            x=lx, y=ly, z=zl, colorscale=_ta_diverging, zmid=1,
            xgap=2, ygap=2,                     # surface gap between cells
            text=txt.tolist(), hoverinfo="text",
            colorbar=dict(title=dict(text="A/E", side="right"),
                          thickness=12, outlinewidth=0,
                          tickfont=dict(size=10, color=ta_muted)))
        if z_range is not None:
            hm["zmin"], hm["zmax"] = z_range[0], z_range[1]
        fig.add_trace(go.Heatmap(**hm))

    if thin.any():
        ti = np.flatnonzero(thin)
        fig.add_trace(go.Scatter(
            x=[gxs[i] for i in ti], y=[gys[i] for i in ti],
            mode="markers", showlegend=False,
            marker=dict(symbol="x", size=7, color=ta_muted, opacity=0.55),
            text=[txt[iy[i], ix[i]] for i in ti], hoverinfo="text"))

    if title is None:
        title = "A/E per cell – " + var_x + " × " + var_y

    fig.update_layout(
        title=dict(text=title, font=dict(color=ta_navy, size=14)),
        xaxis=dict(title=var_x, tickangle=-45, tickfont=dict(size=9),
                   showgrid=False, type="category", categoryorder="array",
                   categoryarray=lx),
        yaxis=dict(title=var_y, tickfont=dict(size=9), showgrid=False,
                   type="category", categoryorder="array", categoryarray=ly),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=80, l=110, r=40))
    set_config(fig, "heatmap")
    attach(fig, "cells", cells)
    return fig


# [INT-03]
# Match E to A's row and column margins (iterative proportional fitting).
# What is left is the interaction, not the main effects. Works on one
# table (2-d) or a stack of them (3-d, the first axis the stack), each
# table stopping on its own as R's loop does.
def _rake(A, E, iter=50, tol=1e-9):
    A = np.asarray(A, dtype=float)
    E = np.array(E, dtype=float)
    single = A.ndim == 2
    if single:
        A, E = A[None], E[None]
    E = np.broadcast_to(E, A.shape).copy()
    ra = A.sum(axis=2)
    ca = A.sum(axis=1)
    active = np.ones(A.shape[0], dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for _ in range(iter):
            if not active.any():
                break
            Ea = E[active]
            r = ra[active] / Ea.sum(axis=2)
            r[~np.isfinite(r)] = 1
            Ea = Ea * r[:, :, None]
            cs = ca[active] / Ea.sum(axis=1)
            cs[~np.isfinite(cs)] = 1
            Ea = Ea * cs[:, None, :]
            E[active] = Ea
            done = np.max(np.abs(Ea.sum(axis=2) - ra[active]), axis=1) < tol
            idx = np.flatnonzero(active)
            active[idx[done]] = False
    return E[0] if single else E


def _dev_of(A, E):
    """Likelihood-ratio deviance of A against E, over the cells with E > 0;
    per table for a stack."""
    A = np.asarray(A, dtype=float)
    E = np.asarray(E, dtype=float)
    ok = E > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(A > 0, A * np.log(A / E), 0.0) - (A - E)
    t = np.where(ok, t, 0.0)
    if A.ndim == 2:
        return float(2 * t.sum())
    return 2 * t.sum(axis=(1, 2))


def detect_interactions(model, vars=None, n_bins=10, min_claims=30, n_sim=200,
                        top_n=None, seed=None):
    """Rank variable pairs by unexplained interaction structure.

    Scans every two-way combination of ``vars`` and ranks them by how much
    cell-level lack of fit remains after the row and column margins have
    been matched. The margins are matched by iterative proportional
    fitting, so main-effect misfit (a spline that needs another knot, a
    level that is simply mispriced) is scaled away first and what remains
    is genuine interaction structure.

    The statistic is the likelihood-ratio deviance between the additive and
    the saturated model on the two-way table of actual versus expected
    claims. Because that deviance is unreliable in sparse tables, the
    reference distribution is obtained by simulation rather than from the
    chi-square asymptotics whenever the model is a Poisson count model:
    claims are resampled from the raked expected values under the additive
    null and the statistic is recomputed, ``n_sim`` times. For other models
    the dispersion-scaled chi-square is used instead (reported in the
    ``Method`` column).

    Claim counts are nearly always **overdispersed**, and a Poisson null
    reads that extra variance as interaction: on ten books with a
    dispersion of 1.9 and no interaction at all, 33 of 60 pairs came out at
    ``P`` < 0.05 where 3 would be expected. The deviance is therefore
    divided by the dispersion of a cell total,
    ``phi = sum((y - mu)^2) / sum(mu)`` over the rows, and the simulation
    draws from a Poisson with the expected values divided by ``phi``, which
    is the quasi-Poisson scaling of a cell total. That is not the row-level
    Pearson dispersion of :func:`glm_diagnostics`: a cell total adds rows by
    their mean, and under a gamma frailty the rows with a larger mean carry
    more of the extra variance (measured 1.91 against 1.72). A dispersion
    below 1 is left at 1. The value used is in the ``Dispersion`` column.

    The ranking is by statistical signal (``Z``), which is not the same
    thing as materiality: an effect can be highly significant and
    commercially irrelevant, or the reverse. ``MaxAE`` and
    ``MaxAE_ExposureShare`` are reported alongside so the two can be judged
    separately.

    Parameters
    ----------
    model : GLMFit
    vars : list of str, optional
        Variables to scan. ``None`` (default) uses the model's own
        predictors (excluding the offset). Variables that are *not* in the
        model may be passed too, which is how an entirely omitted rating
        factor shows up.
    n_bins : int
        Maximum number of groups per variable (default 10).
    min_claims : float
        Cells with fewer claims are ignored when determining ``MaxAE``
        (default 30).
    n_sim : int
        Simulations for the reference distribution (default 200).
    top_n : int, optional
        Return only the strongest ``top_n`` pairs (default all).
    seed : int, optional
        Seed for a reproducible reference distribution.

    Returns
    -------
    pandas.DataFrame
        Strongest first, one row per pair: ``VarX``, ``VarY``, ``Cells``,
        ``Claims``, ``Deviance`` (before the dispersion scaling), ``DF``,
        ``Z``, ``P``, ``Method``, ``Dispersion``, ``MaxAE`` (A/E of the
        worst non-thin cell), ``MaxAE_Claims`` and ``MaxAE_ExposureShare``.

    See Also
    --------
    plot_residual_heatmap : inspect a pair.
    """
    fn = "detect_interactions"
    if not isinstance(model, GLMFit):
        stop("detect_interactions: 'model' must be a glm object.")
    # [DATA-06]
    model = U.as_na_omit(model)
    # [CONV-08]
    # [PY-05] a Generator per call, never the global numpy random state
    rng = np.random.default_rng(seed)

    parts = U.glm_ae_parts(model, fn)
    pool = parts.data
    off = U.offset_vars(model)

    if vars is None:
        vars = [v for v in U.model_base_vars(model, pool) if v not in off]
        vars = [v for v in vars if v in pool.columns]
    elif isinstance(vars, str):
        vars = [vars]
    vars = list(dict.fromkeys(v for v in vars if v not in off))
    if len(vars) < 2:
        stop("detect_interactions: need at least two variables to pair up.")

    grp = {v: U.group_values(U.glm_var_values(parts, v, fn), n_bins)
           for v in vars}

    # [INT-03]
    # The resampling null needs the aggregated actuals to be Poisson counts.
    # That depends on the family and the response, not on whether the model
    # carries an exposure offset.
    fam = model.family
    boot = bool(parts.poisson_counts)
    phi = float(np.sum(model.residuals("pearson") ** 2) / model.df_residual)
    if not boot:
        message("detect_interactions: the response is not Poisson counts (",
                fam, "), so the reference distribution comes from the ",
                "dispersion-scaled chi-square rather than simulation; it is ",
                "less reliable in sparse tables.")

    # [INT-05]
    # Overdispersion is variance the Poisson null does not have, and it read
    # as interaction: with a dispersion of 1.9 and no interaction, 33 of 60
    # pairs came out at P < 0.05. With a cell total of variance phi x its
    # mean, D / phi is compared with a Poisson null on means E / phi; the
    # deviance is homogeneous of degree one, so that is the same test on
    # rescaled counts.
    #
    # The phi of a cell total is not the row-level Pearson one. A total adds
    # rows by their mean, so it is sum((y - mu)^2) / sum(mu), a mean-weighted
    # Pearson. Under a gamma frailty, Var(y) = mu + mu^2 / theta, that is
    # larger whenever the means vary across rows: measured 1.72 for the
    # row-level Pearson against 1.91 for this one and 1.91 for the totals of
    # random partitions of the same rows, and with 1.72 the scan still found
    # 8 of 60 pairs at P < 0.05. Under a constant phi the two agree. Below 1
    # it stays 1.
    phi_s = phi
    if boot:
        a = np.asarray(parts.actual, dtype=float)
        e = np.asarray(parts.expected, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            phi_s = (np.sum((a - e) ** 2) / np.sum(e) * len(a)
                     / max(1, model.df_residual))
        phi_s = max(1.0, float(phi_s)) if np.isfinite(phi_s) else 1.0
        if phi_s >= 1.2:
            message("detect_interactions: the claim counts are overdispersed ",
                    "(dispersion of a cell total ", U._txt(round(phi_s, 2)),
                    "), so the simulation is scaled by it; without that, "
                    "overdispersion reads as interaction.")

    actual = np.asarray(parts.actual, dtype=float)
    expected = np.asarray(parts.expected, dtype=float)
    exposure = np.asarray(parts.exposure, dtype=float)
    claims = np.asarray(parts.claims, dtype=float)

    def tab(vals, cx, cy, shape):
        # tapply(..., list(gx, gy), sum) with the empty cells at 0
        out = np.zeros(shape)
        np.add.at(out, (cx, cy), vals)
        return out

    def one_pair(pr):
        gx, gy = grp[pr[0]], grp[pr[1]]
        cx = np.asarray(gx.codes)
        cy = np.asarray(gy.codes)
        keep = (cx >= 0) & (cy >= 0)
        shape = (len(gx.categories), len(gy.categories))
        cx, cy = cx[keep], cy[keep]
        A = tab(actual[keep], cx, cy, shape)
        E = tab(expected[keep], cx, cy, shape)
        W = tab(exposure[keep], cx, cy, shape)
        C = tab(claims[keep], cx, cy, shape)
        if shape[0] < 2 or shape[1] < 2:
            return None

        Er = _rake(A, E)
        D = _dev_of(A, Er)
        dfr = int((np.sum(A.sum(axis=1) > 0) - 1) * (np.sum(A.sum(axis=0) > 0) - 1))
        if dfr < 1:
            return None

        if boot:
            Es = Er / phi_s
            Ds = D / phi_s
            As = rng.poisson(np.broadcast_to(Es, (int(n_sim),) + Es.shape))
            sim = _dev_of(As, _rake(As, Es))
            with np.errstate(divide="ignore", invalid="ignore"):
                z = (Ds - np.mean(sim)) / np.std(sim, ddof=1)
            p = (1 + np.sum(sim >= Ds)) / (n_sim + 1)
            meth = "simulation"
        else:
            z = (D / phi - dfr) / np.sqrt(2 * dfr)
            p = stats.chi2.sf(D / phi, dfr)
            meth = "chisq"

        with np.errstate(divide="ignore", invalid="ignore"):
            ae = np.where(E > 0, A / E, np.nan)
        ae[C < min_claims] = np.nan
        # [INT-04]
        # R's which.max() runs down the columns of the matrix
        dev = np.abs(ae - 1).ravel(order="F")
        worst = None if np.all(np.isnan(dev)) else int(np.nanargmax(dev))
        aef, cf, wf = (ae.ravel(order="F"), C.ravel(order="F"),
                       W.ravel(order="F"))

        return {
            "VarX": pr[0], "VarY": pr[1],
            "Cells": int(np.sum(C > 0)), "Claims": float(np.sum(C)),
            "Deviance": D, "DF": dfr, "Z": float(z), "P": float(p),
            "Method": meth, "Dispersion": phi_s,
            "MaxAE": np.nan if worst is None else float(aef[worst]),
            "MaxAE_Claims": np.nan if worst is None else float(cf[worst]),
            "MaxAE_ExposureShare": (np.nan if worst is None
                                    else float(wf[worst] / np.sum(W))),
        }

    rows = []
    for pr in itertools.combinations(vars, 2):
        try:
            r = one_pair(pr)
        except Exception as err:      # R's tryCatch(): skip the pair, say why
            warn("detect_interactions: pair ", " x ".join(pr), " skipped: ",
                 str(err))
            r = None
        if r is not None:
            rows.append(r)

    if not rows:
        stop("detect_interactions: no usable pairs found.")
    out = pd.DataFrame(rows)
    o = np.argsort(-out["Z"].to_numpy(dtype=float), kind="stable")
    out = out.iloc[o].reset_index(drop=True)
    if top_n is not None:
        out = out.head(int(top_n)).reset_index(drop=True)
    return out
