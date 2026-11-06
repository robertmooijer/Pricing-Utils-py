"""Distributional fit and influence; the port of R/qq-influence.R.

Both functions here take a single model, so they serve a frequency and a
severity model alike; the family decides what is computed.

A normal Q-Q of the classical residuals is close to useless for a
frequency model, because none of them are normal for counts. Measured on
a correctly specified Poisson with 20,000 rows (87% of them claim-free),
share outside a nominal 5% band and a KS test against N(0,1):

  base R plot(glm), std. Pearson    8.34%    p < 2e-16
  boot::glm.diag,   std. deviance   1.31%    p < 2e-16
  quantile residual                 5.10%    p = 0.86

The two classical ones fail in opposite directions - Pearson too heavy,
deviance too light - and both condemn a model that is exactly right. The
randomised quantile residual of Dunn and Smyth (1996) is exactly standard
normal instead, discreteness included, which is what makes a Q-Q plot
readable for counts at all. Verified identical to statmod::qres.pois().
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

from . import _utils as U
from ._core import attach, message, set_config, stop, warn
from .colors import ta_blue, ta_gold, ta_lightblue, ta_muted, ta_navy
from .diagnostics import _r_format, _r_format_int, _r_format_value
from .model import GLMFit


# [DIAG-04]
# Dispersion the reference distribution should use. Poisson and binomial
# fix it at 1 by assumption, and that assumption is exactly what the plot
# is testing, so estimating it here would hide the very thing being looked
# for. Gamma and gaussian carry a free dispersion, estimated from Pearson.
def _qr_dispersion(model):
    if model.family in ("poisson", "binomial"):
        return 1.0
    return float(np.sum(model.residuals("pearson") ** 2) / model.df_residual)


# [DIAG-04]
# Randomised quantile residuals. Returns None when the family has no
# closed-form CDF here, so the caller can fall back and say so.
def _quantile_residuals(model, fn, rng=None):
    """Randomised quantile residuals of a fitted model, or None when its
    family has no closed-form CDF here. `rng` is a numpy Generator, used
    for a discrete response only."""
    if rng is None:
        rng = np.random.default_rng()
    tr = U.glm_training_data(model, fn)
    y = np.asarray(model.y, dtype=float)
    mu = np.asarray(model.fitted, dtype=float)
    w = np.asarray(tr.weights, dtype=float)
    phi = _qr_dispersion(model)
    fam = model.family

    u = None
    if fam in ("poisson", "quasipoisson"):
        a = stats.poisson.cdf(y - 1, mu)
        b = stats.poisson.cdf(y, mu)
        u = rng.uniform(a, b)                   # randomised: y is discrete
    elif fam == "binomial":
        m = np.maximum(np.round(w), 1)
        k = np.round(y * m)
        a = stats.binom.cdf(k - 1, m, mu)
        b = stats.binom.cdf(k, m, mu)
        u = rng.uniform(a, b)
    elif fam == "gamma":
        # y | mu ~ Gamma(shape = w/phi, scale = phi*mu/w): mean mu and
        # variance phi*mu^2/w, which is the GLM's own assumption
        u = stats.gamma.cdf(y, a=w / phi, scale=phi * mu / w)
    elif fam == "gaussian":
        u = stats.norm.cdf(y, mu, np.sqrt(phi / w))

    # [DIAG-08]
    # A negative binomial has a closed-form CDF too. With prior weights the
    # likelihood is a weighted one and no longer a distribution per row, so
    # that case falls back.
    th = U.nb_theta(model)
    if u is None and th is not None and np.all(np.abs(w - 1) < 1e-12):
        p = th / (th + mu)
        a = stats.nbinom.cdf(y - 1, th, p)
        b = stats.nbinom.cdf(y, th, p)
        u = rng.uniform(a, b)

    if u is None:
        return None
    # Guard the tails: a u of exactly 0 or 1 would become -Inf / Inf and drop
    # the very observation the plot is about
    eps = np.finfo(float).eps
    return stats.norm.ppf(np.minimum(np.maximum(u, eps), 1 - eps))


def plot_glm_qq(model, n_max=5000, band=True, seed=None, title=None,
                y_range=None):
    """Normal Q-Q plot of quantile residuals.

    Checks the distributional assumption of a fitted GLM: not whether the
    mean is right (that is what ``plot_glm_predictor()`` and
    :func:`plot_glm_residuals` are for) but whether the spread and shape of
    the response match the family. Takes a single model, so it serves a
    frequency and a severity model alike.

    **Why not the classical residuals.** R's ``plot(model)`` draws its Q-Q
    panel from ``rstandard(type = "pearson")``, and
    ``boot::glm.diag.plots()`` from standardised deviance residuals. For
    count data neither is normal. On a correctly specified Poisson with
    20,000 rows, 87% of them claim-free:

    ======================  =========================  =================
    residual                outside a nominal 5% band  KS test vs N(0,1)
    ======================  =========================  =================
    base R, std. Pearson    8.34%                      p < 2e-16
    boot, std. deviance     1.31%                      p < 2e-16
    quantile residual       5.10%                      p = 0.86
    ======================  =========================  =================

    The two classical ones fail in opposite directions - Pearson too heavy
    in the tails, deviance too light - and both reject a model that is
    exactly right, so neither can say anything about one that is not. Where
    the fitted mean varies little they also collapse onto visible bands,
    because the response takes only a handful of values.

    This function instead uses the **randomised quantile residual** of
    Dunn and Smyth (1996): each observation is mapped through its own
    fitted CDF, uniformly at random within the jump for a discrete
    response, and then through the normal quantile function. Under a
    correctly specified model these are exactly standard normal,
    discreteness and all, so the reference line is the identity and a
    departure means something. The R implementation is verified against
    ``statmod::qresid()``, the reference by one of the authors; this port
    is checked against the R package.

    The other principled route is a simulated envelope, as ``DHARMa`` and
    ``hnp`` use: simulate from the fitted model and locate each observation
    in that distribution. It generalises further (mixed models,
    zero-inflation) but costs ``n_sim`` simulations per row, so the
    closed-form residual is the better fit for a large book.

    Because the count case is randomised, two calls give slightly different
    points. Pass ``seed`` when you need the same picture twice.

    **Reading it.** Points on the line means the family and its variance
    function fit. A curved tail on a Gamma severity usually means the tail
    is heavier than Gamma allows (consider a log-normal, or capping large
    losses). An S-shape on a frequency model points at over- or
    underdispersion; a Poisson that is really overdispersed shows heavy
    tails on both sides, which is the same message :func:`glm_diagnostics`
    gives numerically.

    The band is a pointwise 95% interval, from the Beta distribution of the
    order statistics of a standard normal sample. Two caveats. It is
    pointwise, not simultaneous, so with several thousand points a handful
    will fall outside it even when the model is right - read the shape, do
    not count the exceptions. And it treats the fitted coefficients as
    known, while a simulated envelope would account for their estimation,
    so it is marginally too narrow. Neither matters for reading curvature,
    which is what the plot is for.

    Parameters
    ----------
    model : GLMFit
        Poisson, quasipoisson, binomial, gamma and gaussian families are
        supported, and the negative binomial (``family="negbin"``) without
        prior weights; for anything else the function falls back to
        deviance residuals and says so.
    n_max : int
        Plot at most this many points, evenly spaced through the sorted
        residuals with both extremes always kept (default 5000). The
        residuals are computed on every row regardless; this only thins
        what is drawn, so the widget stays small on a large portfolio.
    band : bool
        Draw the pointwise 95% envelope (default True).
    seed : int, optional
        Seed for the randomisation used on a discrete response, for a
        reproducible plot.
    title : str, optional
        Plot title.
    y_range : (lo, hi), optional
        Fix the sample-quantile axis, so Q-Q plots of different models
        become directly comparable.

    Returns
    -------
    plotly.graph_objects.Figure
        The residuals themselves are attached as ``fig.residuals``, and
        ``fig.type`` records whether they are quantile or deviance
        residuals.

    See Also
    --------
    glm_diagnostics : the dispersion behind an S-shape.
    plot_glm_residuals : mean structure.
    """
    if not isinstance(model, GLMFit):
        stop("plot_glm_qq: 'model' must be a glm object.")
    # [DATA-06] with na.exclude the fitted values were padded with NA and
    # recycled against the response, so the plot was silently wrong
    model = U.as_na_omit(model)
    if n_max < 10:
        stop("plot_glm_qq: 'n_max' must be at least 10.")
    y_range = U.check_range(y_range, "plot_glm_qq")
    # [CONV-08]
    # [PY-05] a Generator per call, never the global numpy random state
    rng = np.random.default_rng(seed)

    r = _quantile_residuals(model, "plot_glm_qq", rng)
    rtype = "quantile"
    if r is None:
        warn("plot_glm_qq: no quantile residuals are available for the '",
             model.family, "' family, so deviance residuals are shown. Those ",
             "are only approximately normal, and not at all so for a ",
             "discrete response.")
        r = np.asarray(model.residuals("deviance"), dtype=float)
        rtype = "deviance"
    if model.family == "quasipoisson":
        warn("plot_glm_qq: a quasipoisson has no likelihood, so the Poisson ",
             "CDF is used and the estimated overdispersion is not reflected. ",
             "Expect tails that look heavier than they are.")

    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        stop("plot_glm_qq: too few usable residuals (", n, ").")

    ordr = np.sort(r)
    pp = (np.arange(1, n + 1) - 0.5) / n            # Hazen plotting positions
    theo = stats.norm.ppf(pp)

    # Thin for drawing only: keep both extremes, spread the rest evenly
    if n > n_max:
        idx = np.concatenate([[1], np.round(np.linspace(1, n, int(n_max))), [n]])
        idx = pd.unique(idx.astype(np.int64))
    else:
        idx = np.arange(1, n + 1)
    i0 = idx - 1

    vals = np.concatenate([theo[i0], ordr[i0]])
    vals = vals[np.isfinite(vals)]
    lim = [float(vals.min()), float(vals.max())]
    fig = go.Figure()

    if isinstance(band, (bool, np.bool_)) and bool(band):    # isTRUE(band)
        # The i-th order statistic of n standard normals has a Beta(i, n-i+1)
        # CDF value, so its pointwise interval comes straight from qbeta
        lo = stats.norm.ppf(stats.beta.ppf(0.025, idx, n - idx + 1))
        hi = stats.norm.ppf(stats.beta.ppf(0.975, idx, n - idx + 1))
        fig.add_trace(go.Scatter(
            x=theo[i0], y=hi, mode="lines",
            line=dict(color="rgba(107,122,141,0.4)", width=1, dash="dot"),
            name="95% pointwise", legendgroup="band", hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=theo[i0], y=lo, mode="lines",
            line=dict(color="rgba(107,122,141,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(168,200,224,0.25)",
            name="95% pointwise", legendgroup="band", showlegend=False,
            hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines", name="Perfect fit",
        line=dict(color=ta_muted, width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=theo[i0], y=ordr[i0], mode="markers", name=rtype + " residuals",
        marker=dict(color=ta_blue, size=5, opacity=0.75),
        hovertemplate=("<b>Theoretical:</b> %{x:.3f}"
                       "<br><b>Sample:</b> %{y:.3f}<extra></extra>")))

    if title is None:
        title = ("Normal Q-Q – " + rtype + " residuals – "
                 + model.family_label + "/" + model.link)

    yaxis = dict(title="Sample quantile", gridcolor="#D0D8E0", zeroline=False,
                 autorange=y_range is None)
    if y_range is not None:
        yaxis["range"] = y_range
    fig.update_layout(
        title=dict(text=title, font=dict(color=ta_navy, size=14)),
        xaxis=dict(title="Theoretical quantile", gridcolor="#D0D8E0",
                   zeroline=False),
        yaxis=yaxis,
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=70, r=40))
    set_config(fig, "qq")
    attach(fig, "residuals", r)
    attach(fig, "type", rtype)
    return fig


def _fmt_c(x, fmt):
    """formatC(x, format =, digits =) for a vector: C's printf, NA as "NA"."""
    return ["NA" if np.isnan(v) else ("Inf" if v == np.inf else
                                      "-Inf" if v == -np.inf else fmt % v)
            for v in np.asarray(x, dtype=float)]


def plot_glm_influence(model, n_label=10, data_cols=None, n_max=5000,
                       max_rows=2e5, seed=None, title=None, y_range=None):
    """Influence plot: leverage against residual, sized by Cook's distance.

    Finds the rows that move the fit, and shows why they do. Leverage on
    the x-axis, standardised deviance residual on the y-axis, and Cook's
    distance as the marker size: a point is influential because it is
    unusual in the predictors (far right), because it is badly fitted (far
    up or down), or both.

    **On a large portfolio.** Cook's distance was designed for regressions
    with tens of observations, where a single point can genuinely swing a
    coefficient. On a book of 100,000 policies no single row does, and
    every absolute rule of thumb breaks down: ``D > 1`` never triggers, and
    ``D > 4/n`` flags thousands of perfectly ordinary rows. So this plot is
    deliberately **relative**: it ranks rows against each other and labels
    the worst ``n_label``, rather than testing them against a threshold
    that does not apply.

    What that ranking is good for is data quality: a vehicle weight of 20
    tonnes or an age of 200 lands at the top and is worth opening. It is a
    much weaker tool for model structure, where ``detect_interactions()``
    and :func:`plot_glm_residuals` have far more power.

    **What it cannot see.** Influence is not the same as misfit, and on a
    Poisson frequency model the two can be opposites. The IRLS weight of a
    row is its fitted mean, so a policy with a tiny exposure carries almost
    no weight in the design matrix and has almost no leverage - whatever
    its response. Implant the classic keying error, an exposure of 0.001
    with two claims on it, and it becomes the **worst-fitted row in the
    book** (Pearson residual 169, rank 1 of 20,000) while ranking around
    30th on Cook's distance, with a leverage of 5e-07. Cook's distance
    structurally cannot see it.

    The mirror image is just as awkward: a row with extreme predictors gets
    leverage above 0.9, tops the Cook ranking, and has an *unremarkable*
    residual - because at that leverage the fit bends to meet it, so the
    point masks itself.

    That is exactly why this is a two-axis plot rather than an index plot
    of ``D``. Neither axis alone finds both, and a residual check such as
    :func:`plot_glm_qq` or :func:`plot_glm_residuals` is the complement
    that catches the low-exposure case.

    **Cost.** The leverages come from a QR decomposition of the weighted
    design matrix, which is the expensive part and grows with rows times
    coefficients. ``max_rows`` caps it with a message rather than letting a
    large model exhaust memory.

    Parameters
    ----------
    model : GLMFit
    n_label : int
        Number of most influential rows to highlight and label (default 10).
    data_cols : list of str, optional
        Extra columns from the model's training data to carry into the
        tooltip and the returned table. ``None`` (default) uses the model's
        own base variables.
    n_max : int
        Plot at most this many ordinary points (default 5000); the labelled
        rows are always drawn on top of them.
    max_rows : int, optional
        Compute on a random sample of at most this many rows (default 2e5);
        ``None`` uses all of them.
    seed : int, optional
        Seed for that sample.
    title : str, optional
    y_range : (lo, hi), optional
        Fix the residual axis.

    Returns
    -------
    plotly.graph_objects.Figure
        The ``n_label`` most influential rows are attached as
        ``fig.influential``: ``Row`` (index label in the training data),
        ``CooksD``, ``Leverage``, ``StdResid``, ``Actual``, ``Fitted`` and
        the ``data_cols``.

    See Also
    --------
    plot_glm_residuals, glm_diagnostics
    """
    if not isinstance(model, GLMFit):
        stop("plot_glm_influence: 'model' must be a glm object.")
    # [DATA-06]
    model = U.as_na_omit(model)
    if n_label < 1:
        stop("plot_glm_influence: 'n_label' must be at least 1.")
    y_range = U.check_range(y_range, "plot_glm_influence")
    # [CONV-08]
    # [PY-05] a Generator per call, never the global numpy random state
    rng = np.random.default_rng(seed)

    tr = U.glm_training_data(model, "plot_glm_influence")
    n_full = len(tr.data)

    # [DIAG-05]
    # Sub-sample before the QR, which is what actually costs
    if max_rows is not None and n_full > max_rows:
        message("plot_glm_influence: refitting on a sample of ",
                _r_format([max_rows], big_mark=",")[0], " of ",
                _r_format_int(n_full), " rows (max_rows); leverage ",
                "depends on the whole design matrix, so the values are those ",
                "of the sample. Pass max_rows = None to use every row.")
        keep = np.sort(rng.choice(n_full, int(max_rows), replace=False))
        d_s = tr.data.iloc[keep] if tr.data is not None else None
        if d_s is None:
            stop("plot_glm_influence: training data cannot be recovered, so the ",
                 "model cannot be refitted on a sample; fit it with a 'data =' ",
                 "argument or pass max_rows = None.")
        model = U.as_na_omit(model.refit(d_s))
        tr = U.glm_training_data(model, "plot_glm_influence")

    cd = np.asarray(model.cooks_distance(), dtype=float)
    hv = np.asarray(model.hatvalues(), dtype=float)
    rs = np.asarray(model.rstandard("deviance"), dtype=float)
    y = np.asarray(model.y, dtype=float)
    mu = np.asarray(model.fitted, dtype=float)
    rn = np.asarray(tr.data.index)

    ok = np.isfinite(cd) & np.isfinite(hv) & np.isfinite(rs)
    if not ok.any():
        stop("plot_glm_influence: no usable influence measures; the model may ",
             "be rank deficient.")
    n = int(ok.sum())

    if data_cols is None:
        data_cols = U.model_base_vars(model, tr.data)
    elif isinstance(data_cols, str):
        data_cols = [data_cols]
    data_cols = [c for c in dict.fromkeys(data_cols) if c in tr.data.columns]

    df = pd.DataFrame({"Row": rn, "CooksD": cd, "Leverage": hv, "StdResid": rs,
                       "Actual": y, "Fitted": mu})[ok]
    if data_cols:
        extra = tr.data.loc[:, data_cols][ok]
        df = pd.concat([df.reset_index(drop=True),
                        extra.reset_index(drop=True)], axis=1)
    df = df.reset_index(drop=True)

    ordr = np.argsort(-df["CooksD"].to_numpy(dtype=float), kind="stable")
    top_i = ordr[:min(n_label, n)]
    top = df.iloc[top_i].reset_index(drop=True)
    rest = df.drop(index=df.index[top_i])
    if len(rest) > n_max:
        rest = rest.iloc[np.sort(rng.choice(len(rest), int(n_max),
                                            replace=False))]
    rest = rest.reset_index(drop=True)

    def lbl(d):
        if data_cols:
            extra = ["<br>" + "<br>".join(
                "%s: %s" % (cc, _r_format_value(d[cc].iloc[i]))
                for cc in data_cols) for i in range(len(d))]
        else:
            extra = [""] * len(d)
        cdf = _fmt_c(d["CooksD"], "%.3g")
        lev = _fmt_c(d["Leverage"], "%.3g")
        sr = _fmt_c(d["StdResid"], "%.2f")
        ac = _fmt_c(d["Actual"], "%.4g")
        ft = _fmt_c(d["Fitted"], "%.4g")
        return ["<b>Row %s</b>" % U._txt(d["Row"].iloc[i])
                + "<br>Cook's D: " + cdf[i]
                + "<br>Leverage: " + lev[i]
                + "<br>Std. residual: " + sr[i]
                + "<br>Actual: " + ac[i]
                + " · Fitted: " + ft[i]
                + extra[i] for i in range(len(d))]

    # The on-plot label has to be short: `text` is what plotly draws next to
    # a marker in "markers+text" mode, so putting the tooltip there paints
    # the whole thing across the panel. The detail goes in `hovertext`.
    def tag(d):
        return ["Row " + U._txt(v) for v in d["Row"]]

    # Area, not radius, follows Cook's D, so a point that is twice as
    # influential does not look four times as big
    def size_of(x, mx):
        return 5 + 20 * np.sqrt(np.maximum(np.asarray(x, dtype=float), 0) / mx)
    mx = float(np.nanmax(df["CooksD"].to_numpy(dtype=float)))
    if not np.isfinite(mx) or mx <= 0:
        mx = 1.0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rest["Leverage"], y=rest["StdResid"], mode="markers", name="Rows",
        marker=dict(color=ta_lightblue, opacity=0.55,
                    size=size_of(rest["CooksD"], mx), line=dict(width=0)),
        hovertext=lbl(rest), hoverinfo="text"))
    fig.add_trace(go.Scatter(
        x=top["Leverage"], y=top["StdResid"], mode="markers+text",
        name="Top %d" % len(top),
        marker=dict(color=ta_gold, size=size_of(top["CooksD"], mx),
                    line=dict(color=ta_navy, width=1)),
        text=tag(top), textposition="top center",
        textfont=dict(size=9, color=ta_navy),
        hovertext=lbl(top), hoverinfo="text"))

    # [DIAG-05]
    # The classic high-leverage rule, 2p/n: it still means something as a
    # reference for leverage even where Cook's own thresholds do not
    p_coef = int(np.sum(~np.isnan(model.coef.to_numpy(dtype=float))))
    hcut = 2 * p_coef / n

    if title is None:
        title = ("Influence – leverage vs residual, sized by Cook's D"
                 + " (" + _r_format_int(n) + " rows)")

    yaxis = dict(title="Standardised deviance residual", gridcolor="#D0D8E0",
                 zeroline=False, autorange=y_range is None)
    if y_range is not None:
        yaxis["range"] = y_range
    fig.update_layout(
        title=dict(text=title, font=dict(color=ta_navy, size=14)),
        xaxis=dict(title="Leverage", gridcolor="#D0D8E0", zeroline=False),
        yaxis=yaxis,
        shapes=[
            dict(type="line", xref="paper", x0=0, x1=1, y0=0, y1=0,
                 line=dict(color=ta_muted, width=1, dash="dot")),
            dict(type="line", xref="x", yref="paper", x0=hcut, x1=hcut,
                 y0=0, y1=1, line=dict(color=ta_muted, width=1, dash="dash"))],
        annotations=[dict(x=hcut, y=1, xref="x", yref="paper", text="2p/n",
                          showarrow=False, xanchor="left", yanchor="bottom",
                          font=dict(color=ta_muted, size=10))],
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(b=70, r=40))
    set_config(fig, "influence")
    attach(fig, "influential", top)
    return fig
