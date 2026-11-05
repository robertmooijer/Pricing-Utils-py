"""Fit summary, collinearity and binned residuals; the port of
R/diagnostics.R."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _utils as U
from ._core import message, set_config, stop, warn
from ._formula import r_cut
from .colors import ta_blue, ta_muted, ta_navy
from .model import GLMFit


# ---- R's number formatting, for messages and tooltips -----------------------------


def _r_sci_parts(x, digits=7):
    """Exponent and number of significant digits R's formatReal() needs for
    one finite number (at most `digits`)."""
    ax = abs(x)
    if ax == 0:
        return 0, 1
    kp = int(math.floor(math.log10(ax)))
    alpha = ax / 10.0 ** kp
    a = round(alpha * 10 ** (digits - 1))
    if a >= 10 ** digits:            # rounding carried into the next power
        kp += 1
        a = round(a / 10)
    nsig = digits
    while nsig > 1 and a % 10 == 0:
        a //= 10
        nsig -= 1
    return kp, nsig


def _r_format(x, big_mark="", digits=7):
    """R's format(x, big.mark =, trim = TRUE) of a numeric vector: one common
    format for all elements, fixed unless scientific is narrower."""
    x = np.atleast_1d(np.asarray(x, dtype=float))
    fin = np.isfinite(x)
    if not fin.any():
        return ["NA" if np.isnan(v) else ("Inf" if v > 0 else "-Inf") for v in x]
    neg = 0
    mxsl = rgt = mxns = -10 ** 9
    mxe = 1
    for v in x[fin]:
        kp, nsig = _r_sci_parts(v, digits)
        left = kp + 1
        sleft = (v < 0) + (1 if left <= 0 else left)
        neg = max(neg, int(v < 0))
        rgt = max(rgt, nsig - left)
        mxsl = max(mxsl, sleft)
        mxns = max(mxns, nsig)
        if abs(kp) >= 100:
            mxe = 2
    rgt = max(rgt, 0)
    w_f = mxsl + rgt + (rgt != 0)
    d = mxns - 1
    w_e = neg + (d > 0) + d + 4 + mxe
    fixed = w_f <= w_e

    def one(v):
        if np.isnan(v):
            return "NA"
        if np.isinf(v):
            return "Inf" if v > 0 else "-Inf"
        if not fixed:
            return "%.*e" % (d, v)
        s = "%.*f" % (rgt, v)
        if big_mark:
            sign = "-" if s.startswith("-") else ""
            s = s.lstrip("-")
            ip, _, fp = s.partition(".")
            ip = "{:,}".format(int(ip)).replace(",", big_mark)
            s = sign + ip + ("." + fp if fp else "")
        return s
    return [one(v) for v in x]


def _r_format_int(n, big_mark=","):
    """format() of an R integer: never scientific."""
    return "{:,}".format(int(n)).replace(",", big_mark)


def _r_format_value(v):
    """format() of a single value of a data column, as R prints it."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        return _r_format([float(v)])[0]
    return str(v)


# ---- glm_diagnostics ----------------------------------------------------------------


def glm_diagnostics(model_freq=None, model_sev=None):
    """Compact fit summary for a frequency and/or severity model.

    Flags overdispersion for Poisson/binomial families (Pearson dispersion
    above 1.2), in which case standard errors are understated and a quasi-
    or negative binomial family should be considered.

    Parameters
    ----------
    model_freq, model_sev : GLMFit, optional
        Fitted models (at least one).

    Returns
    -------
    pandas.DataFrame
        One row per model: ``Model``, ``Family``, ``Link``, ``N``,
        ``Deviance``, ``DFResidual``, ``AIC``, ``Dispersion`` (Pearson
        chi-squared over df) and ``DevianceExplained`` (1 - deviance / null
        deviance).
    """
    if model_freq is None and model_sev is None:
        stop("Provide at least one model.")

    def one(model, model_name):
        if model is None:
            return None
        if not isinstance(model, GLMFit):
            stop("glm_diagnostics: '", model_name, "' must be a glm object.")
        # [DATA-06] na.exclude pads the residuals with NA, and the dispersion
        # then came out NA
        model = U.as_na_omit(model)
        # [DIAG-01]
        disp = float(np.sum(model.residuals("pearson") ** 2) / model.df_residual)

        if model.family in ("poisson", "binomial") and disp > 1.2:
            warn("glm_diagnostics: model '", model_name, "' shows ",
                 "overdispersion (Pearson dispersion = ", U._txt(round(disp, 2)),
                 " > 1); standard errors are understated. Consider a quasi-",
                 model.family, " or negative binomial family.")

        try:
            aic = float(model.aic)
        except Exception:
            aic = float("nan")
        return {
            "Model": model_name,
            "Family": model.family_label,
            "Link": model.link,
            # stats::nobs() of a glm counts the rows with a non-zero weight
            "N": int(np.sum(np.asarray(model.prior_weights) != 0)),
            "Deviance": float(model.deviance),
            "DFResidual": int(model.df_residual),
            "AIC": aic,
            "Dispersion": disp,
            "DevianceExplained": 1 - model.deviance / model.null_deviance,
        }

    rows = [r for r in (one(model_freq, "frequency"), one(model_sev, "severity"))
            if r is not None]
    return pd.DataFrame(rows).reset_index(drop=True)


# ---- glm_collinearity --------------------------------------------------------------


def _empty_collinearity():
    return pd.DataFrame({
        "Term": pd.Series([], dtype=object),
        "DF": pd.Series([], dtype="int64"),
        "GVIF": pd.Series([], dtype=float),
        "GVIF_scaled": pd.Series([], dtype=float),
        "InInteraction": pd.Series([], dtype=bool),
        "Flag": pd.Series([], dtype="boolean"),
    })


def glm_collinearity(model, threshold=3):
    """Collinearity between model terms.

    Reports the generalised variance inflation factor per **term**, not per
    coefficient. A plain VIF is meaningless for a pricing model, where a
    spline or a factor spans several columns: the generalised form of Fox
    and Monette handles multi-column terms, and ``GVIF^(1/(2*DF))`` puts
    them back on a scale comparable across terms.

    Read ``GVIF_scaled`` like a plain VIF on the square-root scale: the
    default threshold of 3 corresponds to a VIF of about 9. A high value
    means that term is largely explained by the others, so its coefficient
    is unstable and its rating factors are not interpretable on their own,
    even though the model's overall predictions may be perfectly fine.

    This is the model-level counterpart of the near-duplicate warning in
    ``screen_features()``: that one catches correlated *candidates* before
    they go in, this one catches them once they are both in the model.

    A term that is part of an **interaction**, either the interaction
    itself or a main effect it contains, is reported but not flagged. Its
    GVIF depends on how the variables are coded rather than on the data: in
    a model with the interaction ``pmin(LEEFTIJD, 30):STEDELIJKHEID``,
    writing the age part as ``pmin(LEEFTIJD, 30) - 30`` moved the scaled
    GVIF of ``STEDELIJKHEID`` from 14.12 to 1.11, with the fitted values
    identical to 2e-14. ``Flag`` is ``NA`` for those terms and
    ``InInteraction`` says which they are.

    Parameters
    ----------
    model : GLMFit
        A fitted model with at least two terms.
    threshold : float
        Flag terms whose ``GVIF_scaled`` reaches this value (default 3).

    Returns
    -------
    pandas.DataFrame
        ``Term``, ``DF``, ``GVIF``, ``GVIF_scaled``, ``InInteraction`` and
        ``Flag`` (a nullable boolean, ``<NA>`` for a term in an
        interaction), ordered by ``GVIF_scaled``. Zero rows with a message
        when the model has fewer than two terms.

    See Also
    --------
    glm_diagnostics
    """
    if not isinstance(model, GLMFit):
        stop("glm_collinearity: 'model' must be a glm object.")

    empty = _empty_collinearity()
    terms = model.terms
    labs = [t.label for t in terms]
    if len(labs) < 2:
        message("glm_collinearity: fewer than two terms, nothing to compare.")
        return empty

    # [DIAG-07]
    # Terms in an interaction: the interaction itself and every term whose
    # variables it contains. Their GVIF changes with the coding (centring a
    # variable, another contrast) while the fit stays the same.
    hi = [t for t in terms if t.order > 1]

    def _in_int(t):
        if t.order > 1:
            return True
        if not hi:
            return False
        return any(v in h.factors for v in t.factors for h in hi)
    in_int = [_in_int(t) for t in terms]

    # R's assign: 0 for the intercept, i for the columns of term i
    nm = list(model.coef.index)
    as_ = np.zeros(len(nm), dtype=int)
    for i, t in enumerate(terms):
        as_[t.cols] = i + 1
    cf = model.coef.to_numpy(dtype=float)

    # A rank-deficient fit leaves aliased coefficients as NA, and vcov may
    # keep them with a zero or NA variance; cov2cor() then quietly turns the
    # whole matrix into NA. Drop them on the coefficients themselves.
    est = ~np.isnan(cf) & np.asarray(model.keep, dtype=bool)
    n_alias = int(np.sum((as_ != 0) & ~est))
    if n_alias > 0:
        warn("glm_collinearity: ", n_alias, " coefficient(s) could not be ",
             "estimated and were dropped; the model is rank deficient, ",
             "which is itself exact collinearity.")
    keep = (as_ != 0) & est
    # position of each estimated column in vcov (which holds those only)
    vpos = np.cumsum(np.asarray(model.keep, dtype=bool)) - 1
    sel = np.flatnonzero(keep)
    as_ = as_[sel]
    v = model.vcov.to_numpy(dtype=float)[np.ix_(vpos[sel], vpos[sel])]
    dg = np.diag(v)
    good = np.isfinite(dg) & (dg > 0)
    if not good.all():
        warn("glm_collinearity: ", int(np.sum(~good)), " coefficient(s) have no ",
             "usable variance and were dropped; the model is rank ",
             "deficient.")
        as_ = as_[good]
        v = v[np.ix_(good, good)]
    if not len(as_) or len(np.unique(as_)) < 2:
        message("glm_collinearity: fewer than two estimable terms.")
        return empty

    # stats::cov2cor()
    with np.errstate(divide="ignore", invalid="ignore"):
        s = 1 / np.sqrt(np.diag(v))
        R = v * s[:, None] * s[None, :]
    np.fill_diagonal(R, 1.0)
    if np.isnan(R).any():
        warn("glm_collinearity: the coefficient correlation matrix contains ",
             "missing values; GVIF cannot be computed.")
        return empty

    # [DIAG-02]
    # Log-determinants: with correlated spline bases the determinant of a
    # 20-column correlation matrix underflows to zero long before the model
    # is anywhere near singular.
    def logdet(M):
        if not M.shape[0]:
            return 0.0
        sign, mod = np.linalg.slogdet(M)
        if not np.isfinite(mod) or sign <= 0:
            return float("nan")
        return float(mod)
    ldR = logdet(R)
    if not np.isfinite(ldR):
        warn("glm_collinearity: the coefficient correlation matrix is ",
             "singular, which is itself a sign of exact collinearity; ",
             "GVIF cannot be computed.")
        return empty

    rows = []
    for i in np.unique(as_):
        sl = as_ == i
        with np.errstate(over="ignore"):
            gvif = float(np.exp(logdet(R[np.ix_(sl, sl)])
                                + logdet(R[np.ix_(~sl, ~sl)]) - ldR))
        dfi = int(sl.sum())
        rows.append({"Term": labs[i - 1], "DF": dfi, "GVIF": gvif,
                     "GVIF_scaled": gvif ** (1 / (2 * dfi)),
                     "InInteraction": bool(in_int[i - 1])})
    out = pd.DataFrame(rows)
    out = out[np.isfinite(out["GVIF_scaled"].to_numpy(dtype=float))]
    if not len(out):
        return empty
    out["Flag"] = pd.array([pd.NA if ii else bool(g >= threshold)
                            for ii, g in zip(out["InInteraction"],
                                             out["GVIF_scaled"])],
                           dtype="boolean")
    o = np.argsort(-out["GVIF_scaled"].to_numpy(dtype=float), kind="stable")
    out = out.iloc[o].reset_index(drop=True)
    out["DF"] = out["DF"].astype("int64")
    out["InInteraction"] = out["InInteraction"].astype(bool)

    flagged = out["Flag"].fillna(False).to_numpy(dtype=bool)
    if flagged.any():
        warn("glm_collinearity: ", int(flagged.sum()), " term(s) at or above a ",
             "scaled GVIF of ", U._txt(threshold), " (",
             ", ".join(out["Term"][flagged]), "); their factors are unstable "
             "and should not be read individually.")
    ii = out["InInteraction"].to_numpy(dtype=bool)
    if ii.any():
        warn("glm_collinearity: ", ", ".join(out["Term"][ii]),
             " belong to an interaction, so their GVIF depends on how the ",
             "variables are coded (centring one changes it, the fit stays ",
             "the same). They are not flagged; judge them on the stability ",
             "of the combined effect instead.")
    return out


# ---- plot_glm_residuals ------------------------------------------------------------


def _is_numeric(x):
    return (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)
            and not isinstance(getattr(x, "dtype", None), pd.CategoricalDtype))


def plot_glm_residuals(model, predictor=None, n_bins=50,
                       residual_type="pearson", y_range=None):
    """Binned residual plot.

    For individual policy rows raw residual plots are unreadable (Poisson
    residuals are dominated by the 0/1 claim pattern). This plots the mean
    residual per (quantile) bin of the fitted value, or of a predictor,
    together with a plus/minus 2 SE band: under a correctly specified model
    roughly 95 percent of the bin means should fall inside the band.
    Systematic patterns outside the band indicate missed structure
    (candidate terms/splines).

    The bin means are drawn as points rather than joined into a line. They
    are independent draws, and a connecting line invites the eye to read a
    trend into what is mostly noise; the band is what says whether a point
    means anything.

    A Pearson residual has mean 0 and variance ``phi`` under the model, so
    its band is ``2 sqrt(phi / n)``. A **deviance** residual is not centred:
    for a Poisson count with a small mean it is negative on average, since a
    claim-free row has ``-sqrt(2 mu)``. On a correct Poisson book every bin
    then sat below the band. Deviance residuals are therefore shown minus
    their expectation under the fitted model, row by row, and the band per
    bin is ``2 sqrt(sum of the row variances) / n``. Both moments are exact
    for the Poisson, negative binomial, binomial, Gamma and Gaussian
    families; for any other family the residuals are shown as they are,
    with a warning.

    Parameters
    ----------
    model : GLMFit
    predictor : str, optional
        Bin by this predictor instead of the fitted values.
    n_bins : int
        Number of quantile bins for numeric x (default 50).
    residual_type : {"pearson", "deviance"}
        Default ``"pearson"``.
    y_range : (lo, hi), optional
        Fix the residual (y) axis range, so residual plots for different
        predictors or models become directly comparable. ``None`` (default)
        auto-scales to this plot's own values.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    if isinstance(residual_type, (list, tuple)):
        residual_type = residual_type[0]
    if residual_type not in ("pearson", "deviance"):
        stop("plot_glm_residuals: 'residual_type' should be one of ",
             "\"pearson\", \"deviance\".")
    y_range = U.check_range(y_range, "plot_glm_residuals")
    if not isinstance(model, GLMFit):
        stop("'model' must be a glm object.")
    if n_bins < 2:
        stop("plot_glm_residuals: 'n_bins' must be at least 2.")
    # [DATA-06]
    model = U.as_na_omit(model)

    r = np.asarray(model.residuals(residual_type), dtype=float)
    disp = float(np.sum(model.residuals("pearson") ** 2) / model.df_residual)

    # [DIAG-06]
    # Each row's residual variance, so the band can follow the bin; for
    # deviance residuals also their mean, which is taken off
    rv = np.repeat(disp, len(r))
    centred = False
    if residual_type == "deviance":
        mom = U.dev_resid_moments(model)
        if mom is None or np.isnan(mom.mean).any() or np.isnan(mom.var).any():
            warn("plot_glm_residuals: the mean of a deviance residual has no ",
                 "closed form here for the '", model.family, "' ",
                 "family, so the residuals are not centred and the band ",
                 "assumes a variance of phi. A deviance residual is not ",
                 "centred, so the bins can sit off zero for a correct model; ",
                 "residual_type = \"pearson\" has no such bias.")
        else:
            r = r - np.asarray(mom.mean, dtype=float)
            rv = np.asarray(mom.var, dtype=float)
            centred = True

    if predictor is None:
        x = pd.Series(np.asarray(model.fitted, dtype=float))
        xlab = "Fitted value"
    else:
        tr = U.glm_training_data(model, "plot_glm_residuals")
        if tr.data is not None and predictor in tr.data.columns:
            x = tr.data[predictor].reset_index(drop=True)
        else:
            stop("Predictor '", predictor, "' not found.")
        xlab = predictor

    x_numeric = _is_numeric(x)
    if x_numeric:
        xv = x.to_numpy(dtype=float, na_value=np.nan)
        n_unique = len(pd.unique(xv))           # NA counts, as in R's unique()
        if n_unique > n_bins:
            ok = ~np.isnan(xv)
            brks = np.unique(np.quantile(xv[ok], np.linspace(0, 1, n_bins + 1)))
            if len(brks) < 2:
                brks = np.array([np.nanmin(xv) - 0.5, np.nanmax(xv) + 0.5])
            bg = r_cut(xv, brks, include_lowest=True, dig_lab=4)
        else:
            bg = pd.Categorical(xv, categories=np.unique(xv[~np.isnan(xv)]))
    else:
        if isinstance(x.dtype, pd.CategoricalDtype):
            bg = pd.Categorical(x).remove_unused_categories()
        else:
            bg = pd.Categorical(x.astype(object).where(x.notna(), None))

    codes = np.asarray(bg.codes)
    # dplyr's group_by: groups in level order, a group of NA last
    grp = np.where(codes < 0, len(bg.categories), codes)
    ug = np.unique(grp)
    recs = []
    for g in ug:
        i = grp == g
        n = int(i.sum())
        if x_numeric:
            xp = float(np.mean(xv[i]))
        else:
            first = x.iloc[int(np.flatnonzero(i)[0])]
            xp = "NA" if pd.isna(first) else U._txt(first)
        recs.append({"bin": g, "x_plot": xp, "mean_r": float(np.mean(r[i])),
                     "n": n,
                     # [DIAG-03] [DIAG-06] 2 SE of a bin mean: 2 sqrt(sum var) / n,
                     # which is 2 sqrt(phi / n) for Pearson residuals
                     "band": 2 * math.sqrt(float(np.sum(rv[i]))) / n})
    agg = pd.DataFrame(recs)
    if x_numeric:
        o = np.argsort(agg["x_plot"].to_numpy(dtype=float), kind="stable")
        agg = agg.iloc[o].reset_index(drop=True)
    else:
        agg = agg.sort_values("bin", kind="stable").reset_index(drop=True)

    ht_mean = ("<b>" + xlab + ":</b> %{x}"
               "<br><b>Mean residual:</b> %{y:.3f}"
               "<br>%{text}<extra></extra>")
    txt = ["n = %d" % n for n in agg["n"]]

    xaxis_cfg = dict(title=xlab, tickangle=-45, tickfont=dict(size=9),
                     showgrid=False)
    if not x_numeric:
        xaxis_cfg["categoryorder"] = "array"
        xaxis_cfg["categoryarray"] = list(agg["x_plot"])

    fig = go.Figure()
    if x_numeric:
        # Plus/minus 2 SE band as a shaded ribbon
        fig.add_trace(go.Scatter(
            x=agg["x_plot"], y=agg["band"], mode="lines",
            line=dict(color="rgba(107,122,141,0.4)", width=1, dash="dot"),
            name="±2·SE", legendgroup="band", showlegend=True,
            hoverinfo="skip"))
        fig.add_trace(go.Scatter(
            x=agg["x_plot"], y=-agg["band"], mode="lines",
            line=dict(color="rgba(107,122,141,0.4)", width=1, dash="dot"),
            fill="tonexty", fillcolor="rgba(168,200,224,0.25)",
            name="±2·SE", legendgroup="band", showlegend=False,
            hoverinfo="skip"))
        # [DIAG-03]
        # Markers, not a line. Bin means are independent draws, and joining
        # them invites the eye to read a trend into what is mostly noise -
        # the band is what says whether a point means anything.
        fig.add_trace(go.Scatter(
            x=agg["x_plot"], y=agg["mean_r"], mode="markers",
            name="Mean residual", marker=dict(color=ta_blue, size=6),
            text=txt, hovertemplate=ht_mean))
    else:
        # Categorical x: error bars instead of a ribbon
        fig.add_trace(go.Scatter(
            x=agg["x_plot"], y=agg["mean_r"], mode="markers",
            name="Mean residual", marker=dict(color=ta_blue, size=8),
            error_y=dict(type="data", array=list(agg["band"]),
                         color=ta_muted, thickness=1),
            text=txt, hovertemplate=ht_mean))

    yaxis = dict(title="Mean " + residual_type + " residual"
                 + (", centred" if centred else ""),
                 gridcolor="#D0D8E0", zeroline=False,
                 autorange=y_range is None)
    if y_range is not None:
        yaxis["range"] = y_range
    fig.update_layout(
        title=dict(text="Binned " + residual_type + " residuals – " + xlab,
                   font=dict(color=ta_navy, size=14)),
        xaxis=xaxis_cfg,
        yaxis=yaxis,
        shapes=[dict(type="line", xref="paper", x0=0, x1=1, y0=0, y1=0,
                     line=dict(color=ta_muted, width=1, dash="dot"))],
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(b=80, r=40))
    return set_config(fig, "residuals")
