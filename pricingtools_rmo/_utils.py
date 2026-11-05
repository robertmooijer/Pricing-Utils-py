"""Internal helpers shared by the package; the port of R/utils-internal.R.

The decision tags (``# [OFF-05]`` and so on) are the same as in the R
package, so one search finds the reasoning in TECHNICAL.md and the code in
either language.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pandas as pd
from scipy import integrate, stats

from ._core import PricingError, Result, message, stop, warn
from ._formula import R_FUNCTIONS, eval_expr, expr_vars, r_cut, translate

# ---- input handling ------------------------------------------------------------


def check_cols(d, cols, fn):
    cols = [c for c in (cols or []) if c is not None]
    seen, uniq = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    missing = [c for c in uniq if c not in d.columns]
    if missing:
        stop(fn, ": column(s) not found in the data: ", ", ".join(missing))
    return True


# [CONV-02]
def as_df(x):
    """Normalise input to a pandas DataFrame, once, at the entry point."""
    if x is None:
        return None
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "to_pandas"):          # polars, pyarrow
        return x.to_pandas()
    return pd.DataFrame(x)


def check_range(rng, fn, arg="y_range"):
    if rng is None:
        return None
    r = list(np.atleast_1d(rng))
    if len(r) != 2 or not all(np.isfinite(r)) or r[0] >= r[1]:
        stop(fn, ": '", arg, "' must be (lo, hi) with lo < hi.")
    return [float(r[0]), float(r[1])]


# [RT-14]
def coerce_like(vals, template):
    """Coerce values to the type of an existing column, so a grid value of
    '2024' finds a numeric year, a categorical level is matched by its text,
    and a logical stays logical."""
    vals = list(vals) if not np.isscalar(vals) else [vals]
    if isinstance(template, pd.Series):
        dt = template.dtype
    else:
        template = pd.Series(template)
        dt = template.dtype
    if isinstance(dt, pd.CategoricalDtype):
        cats = list(dt.categories)
        lookup = {_txt(c): c for c in cats}
        out = [lookup.get(_txt(v)) if v is not None else None for v in vals]
        return pd.Categorical(out, categories=cats)
    if pd.api.types.is_bool_dtype(dt):
        m = {"TRUE": True, "True": True, "true": True,
             "FALSE": False, "False": False, "false": False}
        out = []
        for v in vals:
            if isinstance(v, (bool, np.bool_)):
                out.append(bool(v))
            else:
                out.append(m.get(_txt(v)))
        return pd.array(out, dtype="boolean")
    if pd.api.types.is_numeric_dtype(dt):
        return pd.to_numeric(pd.Series([_txt(v) if isinstance(v, str) else v
                                         for v in vals], dtype=object),
                             errors="coerce").to_numpy(dtype=float)
    return np.array([None if v is None or (isinstance(v, float) and np.isnan(v))
                     else _txt(v) for v in vals], dtype=object)


def _txt(v):
    """A value as R would print it in a level: 2024 not 2024.0."""
    if isinstance(v, (bool, np.bool_)):
        return "TRUE" if v else "FALSE"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, (float, np.floating)):
        if np.isnan(v):
            return "NA"
        return str(int(v)) if float(v).is_integer() else level_chr([v])[0]
    return str(v)


# ---- the model's own data ------------------------------------------------------


def as_na_omit(model):
    """[DATA-06] In Python a model is always fitted as with na.omit."""
    return model


# [DATA-01] [DATA-05]
def glm_training_data(model, fn="pricingtools"):
    """Training rows of a model (original columns), with response, prior
    weights and offset, aligned row for row."""
    return Result(data=model.data, y=model.y, weights=model.prior_weights,
                  offset=model.offset)


# [AE-05]
def glm_newdata_parts(model, newdata, fn="pricingtools"):
    """The same pieces evaluated on other data, so a model can be held
    against a period it was never fitted on. Rows with a missing value in a
    model variable are dropped, as at fit time."""
    newdata = as_df(newdata)
    if not isinstance(newdata, pd.DataFrame) or not len(newdata):
        stop(fn, ": 'data' must be a data frame with at least one row.")
    need = list(dict.fromkeys(model.response_vars() + model.rhs_vars()
                              + model.offset_vars() + model.weights_vars()))
    miss = [v for v in need if v not in newdata.columns]
    if miss:
        stop(fn, ": 'data' has no column(s) ", ", ".join(miss),
             ". Besides the predictors, the response and any weight or offset "
             "column are needed as well, because the point of the plot is to "
             "hold the realised figure against the prediction.")
    # [SCORE-01]
    un = unseen_levels(model, newdata)
    if un:
        stop(unseen_message(un, "", fn, at_hint=False))
    na = np.zeros(len(newdata), dtype=bool)
    for v in need:
        na |= pd.isna(newdata[v]).to_numpy()
    d = newdata.loc[~na]
    if not len(d):
        stop(fn, ": no row of 'data' is usable; every row has a missing value "
                 "in one of the model's variables.")
    y = np.asarray(eval_expr(model.parsed.response, d), dtype=float)
    if model.weights_expr is None:
        pw = np.ones(len(d))
    else:
        pw = np.asarray(eval_expr(model.weights_expr, d), dtype=float)
        if len(pw) != len(d) or np.isnan(pw).any():
            stop(fn, ": the model's weights (", model.weights_expr, ") do not "
                 "give one usable value per row of 'data'.")
    off = model.offset_value(d) if model.offset_exprs else None
    return Result(data=d, y=y, weights=pw, offset=off)


# ---- factor levels the model never saw -------------------------------------------


def _eval_factor_values(code, data):
    """Values of a patsy factor code on `data`, without the categorical box."""
    ns_ = dict(R_FUNCTIONS)

    def _col_(name):
        return data[name].to_numpy()

    ident = lambda x, *a, **k: np.asarray(x, dtype=object)
    ns_.update({"_col_": _col_, "Q": _col_, "C": ident, "factor": ident})
    for v in expr_vars(code):
        if v in data.columns and v.isidentifier():
            ns_[v] = data[v].to_numpy()
    val = eval(compile(translate(code, colfun="_col_"), "<f>", "eval"),
               {"__builtins__": {}}, ns_)
    return np.asarray(val, dtype=object)


# [SCORE-01]
def unseen_levels(model, data):
    """Values of a factor term that a model was not fitted on, per data
    column: a dict column -> Result(values, n, rows)."""
    out = {}
    if not model.xlevels:
        return out
    for code, levels in model.xlevels.items():
        col = term_base_var(code, data)
        if col not in data.columns:
            continue
        try:
            v = _eval_factor_values(code, data)
        except Exception:
            continue
        if len(v) != len(data):
            continue
        lev = {_txt(x) for x in levels}
        txt = np.array([None if pd.isna(x) else _txt(x) for x in v], dtype=object)
        bad = np.array([t is not None and t not in lev for t in txt])
        if bad.any():
            vals = list(dict.fromkeys(t for t in txt[bad]))
            out[col] = Result(values=vals, n=int(bad.sum()), rows=bad)
    return out


def model_label(label):
    if label is None or label == "" or label == "the":
        return "model "
    return label + " model "


def unseen_message(un, label, fn, at_hint=True):
    parts = []
    for cl, u in un.items():
        parts.append("'%s' = %s (%d row%s)" % (cl, ", ".join(u.values[:5]), u.n,
                                               "" if u.n == 1 else "s"))
    return (fn + ": the data holds value(s) the " + model_label(label)
            + "was not fitted on: " + "; ".join(parts) + ". A model has no "
            "factor for a level it never saw. Map those values to a level it "
            "knows or leave the rows out"
            + ("; for an accounting year beyond the data, price at the latest "
               "known year with 'at' and index the result yourself"
               if at_hint else "") + ".")


# [SCORE-02]
def range_warning(model, data, label, fn):
    """Numeric predictors outside the range the model was fitted on."""
    tr = model.data
    fac = {term_base_var(c, tr) for c in model.xlevels}
    vars_ = [v for v in model.rhs_vars() if v not in model.offset_vars()]
    vars_ = [v for v in vars_ if v in tr.columns and v in data.columns
             and v not in fac]
    msgs = []
    for v in vars_:
        if not (pd.api.types.is_numeric_dtype(tr[v])
                and pd.api.types.is_numeric_dtype(data[v])) \
                or pd.api.types.is_bool_dtype(tr[v]):
            continue
        lo, hi = np.nanmin(tr[v]), np.nanmax(tr[v])
        x = data[v].to_numpy(dtype=float, na_value=np.nan)
        out = int(np.sum((x < lo) | (x > hi)))
        if out:
            msgs.append("'%s' (%d row%s outside %s to %s)"
                        % (v, out, "" if out == 1 else "s", _fmt(lo), _fmt(hi)))
    if msgs:
        warn(fn, ": the data goes beyond the range the ", model_label(label),
             "was fitted on for ", ", ".join(msgs), "; the model extrapolates "
             "there (a spline linearly on the log scale, a log term without "
             "limit). Cap the values to the fitted range if that is not "
             "intended.")


def _fmt(x):
    x = float(x)
    return str(int(x)) if x.is_integer() else "%g" % x


# [RATE-10]
def apply_at(data, at, fn):
    """Price every row at one value of some columns, e.g.
    at={"BOEKJAAR": 2024}."""
    if at is None:
        return data
    if not isinstance(at, dict) or not at or any(not str(k) for k in at):
        stop(fn, ": 'at' must be a dict of column names, e.g. "
                 "{'BOEKJAAR': 2024}.")
    check_cols(data, list(at), fn)
    data = data.copy()
    for v, val in at.items():
        if not np.isscalar(val) and val is not None:
            stop(fn, ": 'at' for '", v, "' must be one value.")
        new = coerce_like([val] * len(data), data[v])
        if pd.isna(pd.Series(new)).any():
            lv = ""
            if isinstance(data[v].dtype, pd.CategoricalDtype):
                lv = "; its levels are " + ", ".join(
                    _txt(c) for c in data[v].cat.categories)
            stop(fn, ": '", val, "' is not a valid value for '", v, "'", lv, ".")
        data[v] = new
    return data


# ---- actual and expected per fitted row --------------------------------------------


def glm_ae_parts(model, fn="pricingtools"):
    """Actual and expected per fitted row, on a scale where A/E means
    something: counts for a log-link offset model, weighted totals
    otherwise."""
    tr = glm_training_data(model, fn)
    y = model.y
    mu = model.fitted
    has_offset = tr.offset is not None and np.any(tr.offset != 0)
    # [OFF-04]
    if has_offset and model.link == "log" and offset_is_log(model) is False:
        warn(fn, ": the model's offset is not a logarithm, so exp(offset) "
                 "is not an exposure. The prior weights are used instead.")
    has_offset = has_offset and offset_is_log(model) is not False
    # [OFF-08] Known gap: the volume here is exp(offset). See TECHNICAL.md.
    if has_offset and model.link == "log":
        out = Result(tr, actual=y, expected=mu, exposure=np.exp(tr.offset),
                     claims=y, exposure_label="Exposure", counts=True)
    else:
        w = tr.weights
        out = Result(tr, actual=w * y, expected=w * mu, exposure=w, claims=w,
                     exposure_label="Weight", counts=False)
    out.poisson_counts = (model.family in ("poisson", "quasipoisson")
                          and np.all(np.isfinite(out.actual))
                          and np.all(np.abs(out.actual - np.round(out.actual)) < 1e-8))
    return out


def glm_var_values(parts, var, fn):
    if parts.data is not None and var in parts.data.columns:
        return parts.data[var]
    stop(fn, ": variable '", var, "' not found in the model data. It must be "
         "a column of the data the model was fitted on.")


def group_values(x, n_bins):
    """Categorical as is, numeric with at most n_bins distinct values on its
    exact values, otherwise quantile bins: a pandas Categorical."""
    x = pd.Series(x).reset_index(drop=True)
    if isinstance(x.dtype, pd.CategoricalDtype):
        return pd.Categorical(x).remove_unused_categories()
    if not pd.api.types.is_numeric_dtype(x) or pd.api.types.is_bool_dtype(x):
        return pd.Categorical(x.astype(object).where(x.notna(), None))
    xv = x.to_numpy(dtype=float)
    u = np.unique(xv[~np.isnan(xv)])
    if len(u) <= n_bins:
        return pd.Categorical(xv, categories=u)
    brks = np.unique(np.quantile(xv[~np.isnan(xv)], np.linspace(0, 1, n_bins + 1)))
    if len(brks) < 2:
        brks = np.array([np.nanmin(xv) - .5, np.nanmax(xv) + .5])
    return r_cut(xv, brks, include_lowest=True, dig_lab=4)


# ---- offsets -------------------------------------------------------------------------


# [OFF-01]
def offset_exprs(model):
    return model.offset_exprs


# [OFF-02]
def offset_value(model, data, idx=None):
    return model.offset_value(data, idx)


# [OFF-02]
def predict_no_offset(model, newdata, idx=None):
    """Prediction on the response scale with (some of) the offset taken out
    on the link scale: right for any offset expression and any link."""
    if idx is None:
        idx = list(range(len(model.offset_exprs)))
    return model.predict(newdata, type="response", drop_offsets=idx)


# [OFF-05]
def offset_split(model, exposure_col, keep_offsets=True):
    """Which offset expressions to take out of a rate, and which stay in."""
    ex = model.offset_exprs
    n = len(ex)
    if not n:
        return Result(drop=[], kept=[], mixed=None)
    vars_ = [expr_vars(e) for e in ex]
    on_expo = [exposure_col in v for v in vars_]
    if not keep_offsets or not any(on_expo):
        return Result(drop=list(range(n)), kept=[], mixed=None)
    mixed = [i for i in range(n) if on_expo[i] and len(vars_[i]) > 1]
    kept = []
    for i in range(n):
        if not on_expo[i]:
            for v in vars_[i]:
                if v not in kept:
                    kept.append(v)
    mixed_vars = None
    if mixed:
        mixed_vars = []
        for i in mixed:
            for v in vars_[i]:
                if v != exposure_col and v not in mixed_vars:
                    mixed_vars.append(v)
    return Result(drop=[i for i in range(n) if on_expo[i]], kept=kept,
                  mixed=mixed_vars)


def kept_offsets(models, exposure_col, keep_offsets=True):
    out = []
    for m in models:
        if m is None:
            continue
        for v in offset_split(m, exposure_col, keep_offsets).kept:
            if v not in out:
                out.append(v)
    return out


def kept_label(kept):
    return ", ".join(kept) if kept else "none"


# [OFF-04]
def offset_is_log(model):
    ex = model.offset_exprs
    if not ex:
        return None
    return all(e.strip().startswith("log(") and e.strip().endswith(")")
               and _balanced(e.strip()[4:-1]) for e in ex)


def _balanced(s):
    depth = 0
    for ch in s:
        depth += ch == "("
        depth -= ch == ")"
        if depth < 0:
            return False
    return depth == 0


def offset_vars(model):
    if model is None:
        return []
    return model.offset_vars()


def rhs_vars(model):
    return model.rhs_vars()


# [SCR-06]
def default_nthread():
    lim = os.environ.get("_R_CHECK_LIMIT_CORES_", "")
    if lim and lim.lower() != "false":
        return 2
    n = os.cpu_count() or 1
    n = max(1, n - 1)
    omp = os.environ.get("OMP_THREAD_LIMIT", "")
    if omp.isdigit() and int(omp) >= 1:
        n = min(n, int(omp))
    return int(n)


# ---- rates for comparison -------------------------------------------------------------


# [OFF-05] [OFF-07] [OFF-09] [RATE-01]
def model_rate(model, data, exposure_col, fn, keep_offsets=True, label=""):
    """Predicted rate per unit of exposure, the exposure offset subtracted on
    the link scale; any other offset stays in by default."""
    if model is None:
        return np.ones(len(data))
    # [SCORE-01] [SCORE-02] what the model cannot price, named before predict()
    un = unseen_levels(model, data)
    if un:
        stop(unseen_message(un, label, fn))
    range_warning(model, data, label, fn)
    if not model.offset_exprs:
        return model.predict(data, type="response")
    sp = offset_split(model, exposure_col, keep_offsets)
    if sp.mixed:
        warn(fn, ": the exposure offset also holds '", ", ".join(sp.mixed),
             "' in the same expression, so the two cannot be separated and "
             "both are taken out of the rate. Write them as separate terms, "
             "offset(log(", exposure_col, ")) + offset(log(", sp.mixed[0],
             ")), to keep the second in.")
    ov = offset_vars(model)
    if model.link != "log":
        warn(fn, ": the model has a '", model.link, "' link, so the offset is "
                 "not a multiplicative exposure. The prediction is returned "
                 "with the offset removed, which is not a rate per unit of "
                 "exposure and should not be multiplied by a second model.")
    elif ov and exposure_col not in ov:
        warn(fn, ": the model offsets on '", ", ".join(ov), "' rather than on '",
             exposure_col, "'. The rates are therefore per unit of that "
             "variable, while the weighting uses '", exposure_col, "'.")
    try:
        return predict_no_offset(model, data, sp.drop)
    except PricingError as e:
        stop(fn, ": the model has an offset that could not be evaluated on "
                 "this data, so the result would not be a rate per unit of "
                 "exposure. ", str(e))


# [RATE-01] [RATE-02]
def old_new_rates(data, model_freq_new, model_sev_new, model_freq_old,
                  model_sev_old, old_premium_col, old_premium_basis,
                  exposure_col, fn, keep_offsets=True, at=None,
                  divisor_col=None):
    if divisor_col is None:
        divisor_col = exposure_col
    if model_freq_new is None and model_sev_new is None:
        stop(fn, ": provide at least one NEW model.")
    has_old = model_freq_old is not None or model_sev_old is not None
    if not has_old and old_premium_col is None:
        stop(fn, ": provide old models or 'old_premium_col'.")
    if has_old and old_premium_col is not None:
        stop(fn, ": provide either old models or 'old_premium_col', not both.")
    pd_ = apply_at(data, at, fn)
    new_rate = (model_rate(model_freq_new, pd_, exposure_col, fn, keep_offsets,
                           "new frequency")
                * model_rate(model_sev_new, pd_, exposure_col, fn, keep_offsets,
                             "new severity"))
    if has_old:
        old_rate = (model_rate(model_freq_old, pd_, exposure_col, fn,
                               keep_offsets, "old frequency")
                    * model_rate(model_sev_old, pd_, exposure_col, fn,
                                 keep_offsets, "old severity"))
    elif old_premium_basis == "amount":
        with np.errstate(divide="ignore", invalid="ignore"):
            old_rate = (data[old_premium_col].to_numpy(dtype=float)
                        / data[divisor_col].to_numpy(dtype=float))
    else:
        old_rate = data[old_premium_col].to_numpy(dtype=float)
    return Result(new=np.asarray(new_rate, dtype=float),
                  old=np.asarray(old_rate, dtype=float),
                  kept=kept_offsets([model_freq_new, model_sev_new,
                                     model_freq_old, model_sev_old],
                                    exposure_col, keep_offsets))


# [RATE-08]
def check_ae_scale(actual, predicted, fn, actual_col):
    if predicted <= 0 or actual <= 0:
        return
    ae = actual / predicted
    if ae > 2 or ae < 0.5:
        warn(fn, ": the overall actual/expected is ", "%.3g" % ae, ", so '",
             actual_col, "' and the model predictions are probably not the "
             "same quantity. Use claim counts with a frequency model and loss "
             "amounts with a severity or risk premium model.")


# [RATE-06]
def exposure_bins(score, weight, n_bins, fn=None):
    """Bins of roughly equal weight, ordered by score (1-based)."""
    score = np.asarray(score, dtype=float)
    weight = np.asarray(weight, dtype=float)
    nd = len(np.unique(_signif(score, 12)))
    if fn is not None and nd < n_bins:
        warn(fn, ": only ", nd, " distinct predicted value(s) for ", n_bins,
             " bins, so policies with an identical prediction are split "
             "across bins in arbitrary order. Use n_bins <= ", nd,
             " for bins that mean something.")
    o = np.argsort(score, kind="stable")
    cw = np.cumsum(weight[o]) / weight.sum()
    b = np.minimum(n_bins, np.floor(cw * n_bins) + 1).astype(int)
    out = np.zeros(len(score), dtype=int)
    out[o] = b
    return out


def _signif(x, digits):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(x == 0, 0, np.floor(np.log10(np.abs(x))))
    f = 10.0 ** (digits - 1 - mag)
    return np.round(x * f) / f


# [RATE-08]
def gini_exposure(score, actual, weight):
    o = np.argsort(np.asarray(score, dtype=float), kind="stable")
    return gini_ordered(np.asarray(weight, dtype=float)[o],
                        np.asarray(actual, dtype=float)[o])


def gini_ordered(w, a):
    if not (np.sum(a) > 0) or not (np.sum(w) > 0):
        return float("nan")
    x = np.concatenate([[0.0], np.cumsum(w) / np.sum(w)])
    y = np.concatenate([[0.0], np.cumsum(a) / np.sum(a)])
    return float(1 - 2 * np.sum(np.diff(x) * (y[:-1] + y[1:]) / 2))


# [RATE-11]
def gini_boot(score, actual, weight, n_boot, conf, rng):
    if n_boot < 1:
        return np.array([np.nan, np.nan])
    o = np.argsort(np.asarray(score, dtype=float), kind="stable")
    a = np.asarray(actual, dtype=float)[o]
    w = np.asarray(weight, dtype=float)[o]
    sims = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.poisson(1.0, len(a))
        sims[b] = gini_ordered(w * k, a * k)
    sims = sims[np.isfinite(sims)]
    if not len(sims):
        return np.array([np.nan, np.nan])
    return np.quantile(sims, [(1 - conf) / 2, 1 - (1 - conf) / 2])


def top_share(actual, k=10):
    a = np.asarray(actual, dtype=float)
    tot = np.nansum(a)
    if not (tot > 0):
        return float("nan")
    return float(np.sum(np.sort(a[~np.isnan(a)])[::-1][:k]) / tot)


# [RATE-11]
def large_loss_note(actual, actual_col):
    top = top_share(actual)
    if not (top >= 0.10):
        return ""
    return (" The ten largest records carry %.0f%% of '%s'; capping large "
            "losses narrows the interval, provided the models are compared on "
            "the same capped losses." % (100 * top, actual_col))


def check_boot(n_boot, conf_level, fn):
    if not isinstance(n_boot, (int, float, np.integer, np.floating)) \
            or isinstance(n_boot, bool) or not np.isfinite(n_boot) \
            or n_boot < 0 or n_boot != round(n_boot):
        stop(fn, ": 'n_boot' must be a whole number, 0 or more.")
    if not isinstance(conf_level, (int, float, np.floating)) \
            or isinstance(conf_level, bool) or not np.isfinite(conf_level) \
            or conf_level <= 0 or conf_level >= 1:
        stop(fn, ": 'conf_level' must be a number between 0 and 1.")
    return True


# [RATE-05]
def drop_reasons(flags):
    """Count dropped rows per reason; `flags` is a dict name -> bool array."""
    parts = []
    for name, f in flags.items():
        k = int(np.sum(np.asarray(f, dtype=bool)))
        if k > 0:
            parts.append("%s: %d" % (name, k))
    return ", ".join(parts)


# ---- grids ---------------------------------------------------------------------------


# [RT-04]
def nice_step(rng, n, integer_data):
    span = rng[1] - rng[0]
    if not np.isfinite(span) or span <= 0:
        return float("nan")
    raw = span / max(1, n)
    mag = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m >= raw / mag:
            step = mag * m
            break
    if integer_data:
        step = max(1, round(step))
    return float(step)


# [RT-04]
def list_values(uv, n, integer_data):
    return bool(integer_data and len(uv) and len(uv) <= 2 * max(1, n))


# [RT-04]
def step_grid(lo, hi, step, base):
    if not np.isfinite(step) or step <= 0:
        return np.unique(np.array([lo, hi, base], dtype=float))
    k_lo = math.ceil((lo - 1e-9) / step)
    k_hi = math.floor((hi + 1e-9) / step)
    g = np.arange(k_lo, k_hi + 1) * step if k_hi >= k_lo else np.array([])
    return np.unique(np.concatenate([g, [base]]))


# [RT-09]
def level_chr(x):
    """Grid values as text without scientific notation, as R's
    format(scientific = FALSE, drop0trailing = TRUE) writes them."""
    out = []
    for v in np.atleast_1d(x):
        if isinstance(v, str):
            out.append(v)
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            out.append(str(v))
            continue
        if np.isnan(f):
            out.append("NA")
        elif f.is_integer():
            out.append(str(int(f)))
        else:
            out.append(np.format_float_positional(f, precision=7, unique=False,
                                                  fractional=False, trim="-"))
    return out


def snap_to_step(x, step, lo, hi):
    if not np.isfinite(step) or step <= 0:
        return x
    s = round(x / step) * step
    return min(max(s, lo), hi)


# [RT-08]
def term_base_var(term, data):
    """Underlying column of a term label: 'ns(AGE, 4)' -> 'AGE'."""
    t = term.replace("`", "")
    cols = data.columns if hasattr(data, "columns") else data
    if t in cols:
        return t
    try:
        v = expr_vars(term)
    except Exception:
        v = []
    hit = [x for x in v if x in cols]
    if hit:
        return hit[0]
    import re
    m = re.search(r"\(([^,\)]+)", t)
    if not m:
        return t
    return re.sub(r"[()`\"']", "", m.group(1)).strip()


def model_base_vars(model, data):
    """Base variables of a model that exist as columns in `data`."""
    if model is None:
        return []
    out = []
    for t in model.terms:
        for code in t.factors:
            bv = term_base_var(code, data)
            if bv not in out:
                out.append(bv)
    return [v for v in out if v in data.columns]


# [RT-10]
def predictable_rows(model, newd):
    un = unseen_levels(model, newd)
    if not un:
        return np.ones(len(newd), dtype=bool)
    bad = np.zeros(len(newd), dtype=bool)
    for u in un.values():
        bad |= u.rows
    return ~bad


# ---- grouped columns ---------------------------------------------------------------------


# [MV-06]
def source_kind(pv, models, src):
    for m in models:
        if any(term_base_var(c, src) == pv for c in m.xlevels):
            return "cat"
    for m in models:
        if pv in m.rhs_vars():
            md = m.data
            if pv in md.columns and pd.api.types.is_numeric_dtype(md[pv]) \
                    and not pd.api.types.is_bool_dtype(md[pv]):
                return "cont"
    x = src[pv]
    if pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x) \
            and x.nunique(dropna=True) > 25:
        return "cont"
    return "cat"


# [MV-03] [MV-10]
def grouping_column(data, pv, a, model_freq, model_sev, fn):
    """The grouped column `a` of source `pv` for the rows of `data`: derived
    from the data of the model that uses it, or checked against it."""
    models = {k: m for k, m in (("frequency", model_freq),
                                ("severity", model_sev)) if m is not None}
    owners = {k: m for k, m in models.items() if a in m.rhs_vars()}
    have = a in data.columns
    if not owners:
        if have:
            return data[a]
        stop(fn, ": '", a, "' is not a column of 'data' and no model uses it, "
                 "so there is nothing to merge.")
    maps = {}
    for nm, m in owners.items():
        src = m.data
        if not (pv in src.columns and a in src.columns):
            if have:
                continue
            stop(fn, ": '", a, "' is not a column of 'data', and the data the ",
                 nm, " model was fitted on has no column '",
                 pv if pv not in src.columns else a, "', so the relation "
                 "between the two cannot be read off anywhere. Add the column "
                 "to 'data'.")
        ok = src[pv].notna() & src[a].notna()
        pr = pd.DataFrame({"p": [_txt(x) for x in src.loc[ok, pv]],
                           "g": [_txt(x) for x in src.loc[ok, a]]}).drop_duplicates()
        # [MV-02]
        dup = pr["p"][pr["p"].duplicated()].unique().tolist()
        if dup:
            stop(fn, ": '", a, "' is not a regrouping of '", pv, "' in the data "
                 "the ", nm, " model was fitted on: the value(s) ",
                 ", ".join(dup[:5]), " occur there with more than one group.")
        maps[nm] = Result(map=dict(zip(pr["p"], pr["g"])), src=src)
    if not maps:
        return data[a]
    names = list(maps)
    # [MV-04]
    if len(maps) == 2:
        m1, m2 = maps[names[0]].map, maps[names[1]].map
        off = [k for k in m1 if k in m2 and m1[k] != m2[k]]
        if off:
            stop(fn, ": '", a, "' does not mean the same thing everywhere: '",
                 off[0], "' of '", pv, "' is '", m1[off[0]], "' in the data the ",
                 names[0], " model was fitted on and '", m2[off[0]],
                 "' in the data the ", names[1], " model was fitted on",
                 " (%d values differ)" % len(off) if len(off) > 1 else "",
                 ". Two groupings of one variable cannot share a column name: "
                 "give them separate names and name both in 'merge_vars', as "
                 "merge_vars={'", pv, "': ['", a, "_freq', '", a, "_sev']}.")
    owner_nm = names[0]
    mp = dict(maps[names[0]].map)
    if len(maps) == 2:
        for k, v in maps[names[1]].map.items():
            mp.setdefault(k, v)
    src = maps[names[0]].src
    kind = source_kind(pv, list(models.values()), src)
    x = data[pv]
    # [MV-07]
    if kind == "cont":
        keys = np.array([float(k) for k in mp])
        o = np.argsort(keys)
        keys = keys[o]
        band = np.array(list(mp.values()), dtype=object)[o]
        xv = pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(dtype=float)
        brks = np.concatenate([[-np.inf], keys[1:] - np.diff(keys) / 2, [np.inf]])
        idx = np.searchsorted(brks, xv, side="right") - 1
        idx = np.clip(idx, 0, len(band) - 1)
        vals = np.array(band[idx], dtype=object)
        vals[np.isnan(xv)] = None
        lo = ~np.isnan(xv) & (xv < keys[0])
        hi = ~np.isnan(xv) & (xv > keys[-1])
        if not have and (lo.any() or hi.any()):
            rows = lambda k: "%d %s" % (k, "row" if k == 1 else "rows")
            bits = []
            if hi.any():
                bits.append("%s above %s got '%s'" % (rows(hi.sum()), _fmt(keys[-1]), band[-1]))
            if lo.any():
                bits.append("%s below %s got '%s'" % (rows(lo.sum()), _fmt(keys[0]), band[0]))
            also = [k for k, m in models.items()
                    if pv in m.rhs_vars() and not any(
                        term_base_var(c, m.data) == pv for c in m.xlevels)]
            warn(fn, ": '", pv, "' lies outside the range the ", owner_nm,
                 " model was fitted on (", _fmt(keys[0]), " to ", _fmt(keys[-1]),
                 "); ", " and ".join(bits), ", the outer band(s) of '", a, "'.",
                 (" '" + pv + "' is also a continuous term of the "
                  + " and ".join(also) + " model, which extrapolates beyond the "
                  "range it was fitted on; capping the values, pmin(pmax("
                  + pv + ", " + _fmt(keys[0]) + "), " + _fmt(keys[-1])
                  + "), keeps it from doing so.") if also else "")
    else:
        xt = [None if pd.isna(v) else _txt(v) for v in x]
        vals = np.array([mp.get(t) if t is not None else None for t in xt],
                        dtype=object)
        miss = list(dict.fromkeys(t for t, v in zip(xt, vals)
                                  if t is not None and v is None))
        if miss:
            if have:
                # [MV-05] a column the caller supplied keeps its value where
                # the model's data cannot check it
                pass
            else:
                stop(fn, ": the value(s) ", ", ".join(miss[:5]),
                     " and %d more" % (len(miss) - 5) if len(miss) > 5 else "",
                     " of '", pv, "' do not occur in the data the ", owner_nm,
                     " model was fitted on, so their group in '", a,
                     "' is unknown. ",
                     ("A level with no claims has no group there: give it one "
                      "in that model's data, or put '" + a + "' in 'data' "
                      "yourself.") if owner_nm == "severity" else
                     ("Map them to a value the model knows first, or put '"
                      + a + "' in 'data' yourself."))
    tmpl = src[a]
    if isinstance(tmpl.dtype, pd.CategoricalDtype):
        lookup = {_txt(c): c for c in tmpl.cat.categories}
        out = pd.Categorical([lookup.get(v) if v is not None else None for v in vals],
                             categories=tmpl.cat.categories)
    elif pd.api.types.is_numeric_dtype(tmpl) and not pd.api.types.is_bool_dtype(tmpl):
        out = pd.to_numeric(pd.Series(vals, dtype=object), errors="coerce").to_numpy()
    else:
        out = vals
    if not have:
        return pd.Series(out, index=data.index)
    # [MV-05] already there: checked, never overwritten
    cur = [None if pd.isna(v) else _txt(v) for v in data[a]]
    new = [None if (v is None or (isinstance(v, float) and np.isnan(v))) else _txt(v)
           for v in vals]
    bad = [i for i, (c, n) in enumerate(zip(cur, new))
           if c is not None and n is not None and c != n]
    if bad:
        b = bad[0]
        stop(fn, ": '", a, "' does not mean the same thing everywhere: in ",
             len(bad), " row(s) of 'data' it disagrees with the data the ",
             owner_nm, " model was fitted on; row ", b + 1, " has '", pv,
             "' = ", _txt(x.iloc[b]), " and '", a, "' = ", cur[b],
             ", where the model's data says ", new[b], ".")
    return data[a]


# ---- residual moments ------------------------------------------------------------------


# [DIAG-08]
def nb_theta(model):
    if model.family != "negbin":
        return None
    th = model.theta
    if th is not None and np.isfinite(th) and th > 0:
        return float(th)
    return None


# [DIAG-06]
def dev_resid_moments(model):
    """Mean and variance of each row's deviance residual under the fitted
    model, or None for a family without a closed form here."""
    fam = model.family
    mu = model.fitted
    w = model.prior_weights
    phi = np.sum(model.residuals("pearson") ** 2) / model.df_residual

    def on_grid(par, fun, n_grid=200):
        lo, hi = np.min(par), np.max(par)
        if hi / lo < 1 + 1e-9:
            m = fun(lo)
            return Result(mean=np.repeat(m[0], len(par)), var=np.repeat(m[1], len(par)))
        g = np.exp(np.linspace(np.log(lo), np.log(hi), n_grid))
        mm = np.array([fun(v) for v in g])
        lp = np.log(par)
        return Result(mean=np.interp(lp, np.log(g), mm[:, 0]),
                      var=np.interp(lp, np.log(g), mm[:, 1]))

    def moments(d, p):
        m1 = np.sum(p * d)
        return (m1, max(np.sum(p * d ** 2) - m1 ** 2, 0.0))

    def ylogy(y, m):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(y > 0, y * np.log(y / m), 0.0)

    if fam in ("poisson", "quasipoisson"):
        lam = np.maximum(mu * w, 1e-12)

        def f(l):
            y = np.arange(0, stats.poisson.ppf(1 - 1e-12, l) + 6)
            d = np.sign(y - l) * np.sqrt(np.maximum(2 * (ylogy(y, l) - (y - l)), 0))
            return moments(d, stats.poisson.pmf(y, l))
        out = on_grid(lam, f)
        out.var = out.var * phi
        return out
    th = nb_theta(model)
    if th is not None and np.all(np.abs(w - 1) < 1e-12):
        def f(m):
            y = np.arange(0, stats.nbinom.ppf(1 - 1e-12, th, th / (th + m)) + 6)
            with np.errstate(divide="ignore", invalid="ignore"):
                d = np.sign(y - m) * np.sqrt(np.maximum(
                    2 * (ylogy(y, m) - (y + th) * np.log((y + th) / (m + th))), 0))
            return moments(d, stats.nbinom.pmf(y, th, th / (th + m)))
        out = on_grid(np.maximum(mu, 1e-12), f)
        out.var = out.var * phi
        return out
    if fam == "binomial":
        mm = np.maximum(np.round(w), 1)
        if len(np.unique(mm)) > 20:
            return None
        mean_ = np.zeros(len(mu))
        var_ = np.zeros(len(mu))
        for m in np.unique(mm):
            i = mm == m
            k = np.arange(0, m + 1)

            def f(p, m=m, k=k):
                yk = k / m
                with np.errstate(divide="ignore", invalid="ignore"):
                    t1 = np.where(yk > 0, yk * np.log(yk / p), 0.0)
                    t2 = np.where(yk < 1, (1 - yk) * np.log((1 - yk) / (1 - p)), 0.0)
                d = np.sign(yk - p) * np.sqrt(np.maximum(2 * m * (t1 + t2), 0))
                return moments(d, stats.binom.pmf(k, m, p))
            r = on_grid(np.clip(mu[i], 1e-9, 1 - 1e-9), f)
            mean_[i] = r.mean
            var_[i] = r.var
        return Result(mean=mean_, var=var_ * phi)
    if fam == "gamma":
        def gam(wt):
            k = wt / phi

            def f(r):
                return (np.sign(r - 1) * np.sqrt(max(2 * wt * (r - 1 - np.log(r)), 0))
                        * stats.gamma.pdf(r, a=k, scale=1 / k))
            try:
                e1 = (integrate.quad(f, 0, 1, limit=200)[0]
                      + integrate.quad(f, 1, np.inf, limit=200)[0])
            except Exception:
                e1 = np.nan
            from scipy.special import digamma
            e2 = -2 * wt * (digamma(k) - np.log(k))
            return (e1, max(e2 - e1 ** 2, 0.0))
        uw = np.unique(w)
        if len(uw) <= 50:
            mm = np.array([gam(v) for v in uw])
            idx = np.searchsorted(uw, w)
            return Result(mean=mm[idx, 0], var=mm[idx, 1])
        return on_grid(np.maximum(w, 1e-12), gam, n_grid=60)
    if fam == "gaussian":
        return Result(mean=np.zeros(len(mu)), var=np.repeat(phi, len(mu)))
    return None


# [RT-13]
def nil_claims_warning(model_freq, model_sev, data, claims_col, loss_col, fn):
    if model_freq is None or model_sev is None:
        return
    if claims_col not in data.columns or loss_col not in data.columns:
        return
    if claims_col not in model_freq.response_vars():
        return
    cl = pd.to_numeric(data[claims_col], errors="coerce").to_numpy(dtype=float)
    ls = pd.to_numeric(data[loss_col], errors="coerce").to_numpy(dtype=float)
    tot = np.nansum(cl)
    if not (tot > 0):
        return
    nil = ~np.isnan(cl) & ~np.isnan(ls) & (cl > 0) & (ls == 0)
    share = cl[nil].sum() / tot
    if share >= 0.01:
        warn(fn, ": at least ", "%.1f%%" % (100 * share), " of the claims in '",
             claims_col, "' sit in records without payment ('", loss_col,
             "' = 0). The frequency model counts them, while a severity model "
             "fitted on positive amounts never predicts a zero, so frequency x "
             "severity overstates the risk premium by about that share. Count "
             "only paid claims in the frequency model, or model the share of "
             "claims without payment separately.")


# [RATE-10]
def year_warning(new_models, old_models, year_col, at, fn):
    if year_col is None or (at is not None and year_col in at):
        return

    def uses(ms):
        return any(year_col in m.rhs_vars() for m in ms if m is not None)
    u_new = uses(new_models)
    u_old = old_models is not None and uses(old_models)
    against = "a current premium" if old_models is None else "a tariff without the year"
    if u_new != u_old:
        warn(fn, ": the ", "new" if u_new else "old", " models use '", year_col,
             "' as a rating factor, so every policy is priced at the level of "
             "its own year. Against ", against, " that mixes inflation into the "
             "comparison; price every policy at the tariff year with "
             "at={'", year_col, "': <year>}.")
