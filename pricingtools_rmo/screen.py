"""Feature screening with a boosted challenger; the port of R/screen.R.

[SCR-01]
The booster never becomes a pricing model here. It is fitted with the GLM
as an offset, so it can only model what the GLM leaves behind, and every
number it returns is a diagnostic about the GLM.
"""
from __future__ import annotations

import math
import numbers

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import _utils as U
from ._core import Result, message, set_config, stop, warn
from .colors import ta_blue, ta_muted, ta_navy
from .model import GLMFit

_FN = "screen_features"


# Deviance under the model's own family, so this works for Poisson counts
# and Gamma severity alike
def _fam_deviance(fam, y, mu, wt):
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    wt = np.asarray(wt, dtype=float)
    return np.float64(np.sum(fam.resid_dev(y, mu, var_weights=wt) ** 2))


# How R sees a column: "logical", "factor" (a factor or a character
# column), "numeric", or "other" (a date, a list, ...), which R's
# is.numeric() refuses.
def _kind(x):
    dt = x.dtype
    if pd.api.types.is_bool_dtype(dt):
        return "logical"
    if isinstance(dt, (pd.CategoricalDtype, pd.StringDtype)):
        return "factor"
    if pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_complex_dtype(dt):
        return "numeric"
    if dt == object:
        inf = pd.api.types.infer_dtype(x, skipna=True)
        if inf in ("string", "mixed", "mixed-integer"):
            return "factor"
        if inf == "boolean":
            return "logical"
        if inf in ("integer", "floating", "mixed-integer-float", "decimal",
                   "empty"):
            # an all-missing column is R's logical NA: numeric, constant
            return "numeric"
    return "other"


# as.numeric() of a numeric or logical column, NaN for NA
def _as_num(x):
    if x.dtype == object:
        x = pd.to_numeric(
            x.map(lambda v: float(v) if isinstance(v, (bool, np.bool_)) else v),
            errors="coerce")
    return np.asarray(x.to_numpy(dtype=float, na_value=np.nan), dtype=float)


# as.character() of a factor or character column, None for NA
def _as_chr(x):
    if isinstance(x.dtype, pd.CategoricalDtype):
        cats = [U._txt(c) for c in x.cat.categories]
        # code -1 (NA) picks the trailing None
        return np.array(cats + [None], dtype=object)[x.cat.codes.to_numpy()]
    na = pd.isna(x).to_numpy()
    v = x.to_numpy(dtype=object).copy()
    v[na] = None
    if isinstance(x.dtype, pd.StringDtype) or \
            pd.api.types.infer_dtype(x, skipna=True) in ("string", "empty"):
        return v
    # numbers mixed into a character column print as R prints them
    return np.array([s if s is None or isinstance(s, str) else U._txt(s)
                     for s in v], dtype=object)


# levels(factor(x)): the levels present, in the factor's own order for a
# factor and sorted for a character column
def _levels_of(x):
    if isinstance(x.dtype, pd.CategoricalDtype):
        codes = x.cat.codes.to_numpy()
        cats = [U._txt(c) for c in x.cat.categories]
        return [cats[i] for i in np.unique(codes[codes >= 0])]
    ch = _as_chr(x)
    return sorted(pd.unique(ch[pd.notna(ch)]))


# Level map, taken once on the full data. Deriving levels per split would
# give a character column a different width in each split.
def _screen_levels(d, features):
    features = [features] if isinstance(features, str) else list(features)
    return {f: (_levels_of(d[f]) if _kind(d[f]) == "factor" else None)
            for f in features}


# Design matrix for xgboost: numerics as is, factors one-hot. The
# indicators are built directly rather than through a contrast-coded
# design (R's model.matrix()), which refuses a factor with a single level
# ("contrasts can be applied only to factors with 2 or more levels") even
# when no contrasts are needed.
def _screen_matrix(d, features, levs):
    """Returns a Result with ``x`` (the matrix), ``owner`` (the feature
    behind each column) and ``names`` (R's colnames)."""
    features = [features] if isinstance(features, str) else list(features)
    n = len(d)
    blocks, owner, names = [], [], []
    for f in features:
        x = d[f]
        lv = levs.get(f)
        if lv is not None:
            idx = pd.Categorical(_as_chr(x), categories=lv).codes
            m = np.zeros((n, len(lv)))
            ok = idx >= 0
            if ok.any():
                m[np.flatnonzero(ok), idx[ok]] = 1
            names += [str(f) + str(lev) for lev in lv]
        else:
            m = _as_num(x).reshape(-1, 1)
            names.append(str(f))
        blocks.append(m)
        owner += [f] * m.shape[1]
    x = np.hstack(blocks) if blocks else np.zeros((n, 0))
    return Result(x=x, owner=owner, names=names)


def _single_number(x):
    """A single real number as R's is.numeric(x) && length(x) == 1 &&
    !is.na(x) reads it, or None."""
    v = x
    if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
        a = list(np.ravel(np.asarray(v, dtype=object)))
        if len(a) != 1:
            return None
        v = a[0]
    if isinstance(v, (bool, np.bool_)) or not isinstance(v, numbers.Real):
        return None
    v = float(v)
    return None if math.isnan(v) else v


# The best boosting round counted from 1, as R's best_iteration reports
# it; None when there was no early stopping
def _best_iteration(b):
    try:
        return int(b.best_iteration) + 1
    except (AttributeError, TypeError, ValueError):
        return None


# R's predict() on an early-stopped booster uses the trees up to the best
# iteration; Python's Booster.predict() uses every tree unless told
def _iter_range(b):
    bi = _best_iteration(b)
    return (0, bi) if bi is not None else (0, 0)


def screen_features(model, features=None, split=(0.6, 0.2, 0.2), max_depth=2,
                    eta=0.05, nrounds=2000, early_stopping_rounds=40,
                    n_shap=4000, cor_threshold=0.95, max_levels=50,
                    max_rows=math.inf, tree_method="hist", nthread=None,
                    seed=None, exposure_col="Exposure",
                    claims_col="AantalClaims", loss_col="SCHADELAST"):
    """Screen candidate features against a baseline GLM.

    Fits a gradient boosting model with the GLM as an offset, so the booster
    can only explain what the GLM leaves behind. Everything it returns is
    therefore a statement about the GLM, not a pricing model: no scorable
    object is returned.

    It answers two questions, depending on the baseline you pass: with a
    minimal `model` it screens candidate features *before* you choose what
    goes into the tariff ("what does this data add on top of what I already
    price on"); with your full model it asks what the finished tariff still
    misses.

    Parameters
    ----------
    model : GLMFit
        Baseline GLM from :func:`fit_glm`. May be as small as
        ``"AantalClaims ~ 1 + offset(log(Exposure))"``. Poisson/quasipoisson
        count models and Gamma log-link models are supported.
    features : list of str, optional
        Candidate columns. None (default) uses every usable column of the
        model's data except the response, the offset, the prior weights and
        the outcome columns `claims_col`, `loss_col` and `exposure_col`.
        Offered the loss amount, a booster on a frequency model finds that
        claims follow losses and ranks it first, which is leakage rather
        than risk. Naming an outcome column here keeps it, with a warning.
    split : sequence of 3 floats
        Row shares for train / validation (early stopping) / test
        (reporting). Must sum to 1.
    max_depth : int
        Maximum tree depth for the second booster (default 2, i.e. two-way
        interactions).
    eta, nrounds, early_stopping_rounds
        Boosting parameters.
    n_shap : int
        Rows sampled for the SHAP interaction ranking (default 4000); 0
        skips it.
    cor_threshold : float
        Absolute correlation above which numeric candidates are reported as
        near-duplicates (default 0.95).
    max_levels : int
        Factors with more levels than this are skipped (default 50).
    max_rows : float
        Work on a random sample of at most this many rows. ``math.inf``
        (default) uses every row, which is what you want unless the run is
        too slow; None means the same thing, for calls written against the
        old default.

        Setting a finite value is the cheapest lever there is when the wait
        gets uncomfortable. Every stage - the baseline refit, the boosting,
        the permutation pass - is linear in rows, while the *ranking*
        settles long before the last few hundred thousand are added: on a
        1.5M-row book the order of the candidates was unchanged from 100,000
        rows upwards. ``max_rows=2e5`` is a sensible first try. The
        deviances in `summary` and `PermDeviance` then refer to the sample,
        so they are comparable within a run but not across runs of
        different sizes.
    tree_method : str
        How xgboost searches for splits, passed straight through. ``"hist"``
        (default) buckets each feature into bins before looking for a
        split, which is the single biggest lever there is on a large book:
        measured on 2,000,000 rows with 6 candidates, boosting and
        permutation together took 291.6 s under xgboost's old default and
        108.4 s under ``"hist"``, a 63% saving, for a test deviance of
        130640.5 against 130640.6 and an identical feature ranking. xgboost
        made ``hist`` its default in 2.0. ``"exact"`` and ``"approx"`` are
        the alternatives, both slower.
    nthread : int, optional
        Threads for boosting. None (default) uses one fewer than the
        available cores, capped by ``OMP_THREAD_LIMIT``. Boosting dominates
        the runtime, so this is the second lever: on an eight-core machine
        the measured gain over a single-threaded run is more than a factor
        of two.
    seed : int, optional
        Seed for the sample, the split and the boosting.
    exposure_col, claims_col, loss_col : str
        The outcome and volume columns kept out of the default candidates
        (defaults ``"Exposure"``, ``"AantalClaims"``, ``"SCHADELAST"``), see
        ``agg_all()``.

    Returns
    -------
    Result
        ``summary`` (the staged deviance comparison), ``verdict`` (one
        sentence), ``features`` (ranked candidates: Feature, InModel,
        PermDeviance, Gain), ``gain_note``, ``interactions`` (SHAP ranking,
        or None), ``correlated`` (near-duplicate pairs, or None), ``stats``
        (the raw numbers) and ``plot`` (ranked incremental importance).
        Requires the 'xgboost' package (``pip install
        pricingtools-rmo[screen]``).

    Notes
    -----
    **Reading the output.** `summary` compares out-of-sample deviance for
    the baseline, a depth-1 booster (additive corrections only) and a
    depth-`max_depth` booster (which may use interactions). The verdict is
    deliberately staged: while the depth-1 step still improves the fit, main
    effects are missing and the interaction ranking cannot be trusted yet.
    Fix the main effects, then run again.

    `features` is sorted by `PermDeviance`: the increase in out-of-sample
    deviance when that feature is shuffled, with the baseline held fixed, so
    it measures the feature's *incremental* contribution.

    **Read it as a ranking, not as a test.** Zero is not a threshold. The
    figure is a difference between two deviances on one random split, so it
    carries real sampling noise, and its scale depends on how large that
    split is - a value of 5 means nothing on its own. Measured on a
    150,000-row book over three seeds, pure noise columns ranged from -3.8
    to +2.3 while a genuine but weak predictor ranged from -1.9 to +2.0. The
    strong features were never in doubt (30 to 48); the marginal ones were
    not separable at all.

    Two cheap ways to see where the noise band sits. Add a column you know
    is noise - ``data["CONTROL"] = rng.normal(size=len(data))`` - and read
    every other feature against where it lands. And re-run with two or
    three `seed` values: anything whose sign flips between runs is inside
    the band.

    `Gain` is reported alongside but is unreliable on its own: it is biased
    towards continuous and high-cardinality features, which routinely gives
    pure noise a respectable-looking share.

    Correlated candidates split their importance between them, so a pair of
    near-duplicates can both look mediocre. Pairs above `cor_threshold` are
    reported in `correlated`.

    **Sensitivity.** A booster is *less* sensitive to a two-way interaction
    between known rating factors than ``detect_interactions()`` is: the
    cell-based test concentrates the signal into one statistic with a known
    null, while the booster has to discover the split structure and pays a
    variance cost for its flexibility. Use this function to screen features
    and to get the global verdict; use ``detect_interactions()`` to hunt for
    interactions.

    See Also
    --------
    detect_interactions, plot_residual_heatmap
    """
    try:
        import xgboost as xgb
    except ImportError:
        stop("screen_features: package 'xgboost' is required ",
             "(pip install xgboost).")
    if not isinstance(model, GLMFit):
        stop("screen_features: 'model' must be a glm object (a GLMFit from "
             "fit_glm()).")
    try:
        sp = np.asarray(split, dtype=float).ravel()
    except (TypeError, ValueError):
        sp = np.array([np.nan])
    if (sp.size != 3 or not np.all(np.isfinite(sp)) or np.any(sp <= 0)
            or abs(sp.sum() - 1) > 1e-8):
        stop("screen_features: 'split' must be three positive shares summing "
             "to 1.")
    # [CONV-08]
    # [PY-05] a Generator per call, never the global numpy random state
    rng = np.random.default_rng(seed)
    if nthread is None:
        nthread = U.default_nthread()

    tr = U.glm_training_data(model, _FN)
    if tr.data is None:
        stop("screen_features: training data cannot be recovered; fit the "
             "model with a 'data =' argument.")
    d = tr.data
    y_all = np.asarray(tr.y, dtype=float)
    w_fit = np.asarray(tr.weights, dtype=float)
    off = None if tr.offset is None else np.asarray(tr.offset, dtype=float)

    # [SCR-06]
    # Sub-sample before anything else: every later stage is linear in rows
    # None still means "no cap", so calls written against the old default of
    # 1e6 keep working; inf is the same thing said in a way that does not
    # require knowing the convention.
    if max_rows is None:
        max_rows = math.inf
    mr = _single_number(max_rows)
    if mr is None or mr < 1:
        stop("screen_features: 'max_rows' must be a single number of at "
             "least 1, or Inf to use every row.")

    n0 = len(d)
    if math.isfinite(mr) and n0 > mr:
        mr = int(math.floor(mr))
        keep_rows = np.sort(rng.choice(n0, mr, replace=False))
        message("screen_features: sampling ", "{:,}".format(mr), " of ",
                "{:,}".format(n0), " rows (max_rows).")
        d = d.iloc[keep_rows]
        y_all = y_all[keep_rows]
        w_fit = w_fit[keep_rows]
        if off is not None:
            off = off[keep_rows]
    elif n0 > 5e5:
        # Every stage here is linear in rows, so a very large book is worth a
        # word rather than a silent wait. Not a warning: using every row is
        # the default and a perfectly reasonable thing to do.
        message("screen_features: screening on all ", "{:,}".format(n0),
                " rows. The ranking is usually settled long before that, so "
                "max_rows = 2e5 is the quickest way to cut the runtime if "
                "this is slow.")

    fam = model.sm_family
    counts = (off is not None and bool(np.any(off != 0))
              and model.link == "log")
    # [SCR-02]
    if model.family in ("poisson", "quasipoisson") and counts:
        objective = "count:poisson"
    elif model.family == "gamma" and model.link == "log":
        objective = "reg:gamma"
    else:
        stop("screen_features: only Poisson count models with a log-link "
             "offset and Gamma log-link models are supported; this model is ",
             model.family, "/", model.link, ".")

    # Candidate features ------------------------------------------------------
    drop = list(dict.fromkeys(model.response_vars() + U.offset_vars(model)))
    # [SCR-08]
    # Outcome and volume columns are never candidates by default. Offered the
    # loss amount, the booster on a frequency model finds that claims follow
    # losses, ranks it first, and every verdict after that is about leakage
    # instead of risk. The response and the offset were already out; the
    # claim count, the loss amount, the exposure and the prior weights are
    # out by name as well. Asked for explicitly, they stay, with a warning.
    wvars = model.weights_vars()
    outcome = [c for c in dict.fromkeys([claims_col, loss_col, exposure_col]
                                        + list(wvars))
               if c is not None and c in d.columns and c not in drop]
    if features is None:
        features = [c for c in d.columns if c not in drop and c not in outcome]
        if outcome:
            message("screen_features: outcome and volume column(s) left out "
                    "of the candidates: ", ", ".join(map(str, outcome)), ".")
    else:
        features = [features] if isinstance(features, str) else list(features)
        asked = [f for f in dict.fromkeys(features) if f in outcome]
        if asked:
            warn("screen_features: ", ", ".join(map(str, asked)), " is an "
                 "outcome or volume column; as a candidate it carries the "
                 "response into the booster, and its importance says nothing "
                 "about risk.")
    features = [f for f in dict.fromkeys(features) if f not in drop]
    missing_f = [f for f in features if f not in d.columns]
    if missing_f:
        warn("screen_features: column(s) not in the model data, skipped: ",
             ", ".join(map(str, missing_f)))
        features = [f for f in features if f in d.columns]

    # [SCR-07]
    # Classify each candidate, so a dropped column says why it was dropped
    def _status(x):
        k = _kind(x)
        if k == "factor":
            nl = len(_levels_of(x))
            if nl < 2:
                return "constant"
            if nl > max_levels:
                return "too many levels"
            return "ok"
        if k == "other":
            return "unsupported type"
        v = _as_num(x)
        if len(np.unique(v[~np.isnan(v)])) < 2:
            return "constant"
        return "ok"

    status = {f: _status(d[f]) for f in features}
    for why in dict.fromkeys(s for s in status.values() if s != "ok"):
        warn("screen_features: skipped (", why, "): ",
             ", ".join(str(f) for f in features if status[f] == why))
    features = [f for f in features if status[f] == "ok"]
    if not features:
        stop("screen_features: no usable candidate features left; every "
             "candidate was constant, unsupported or too high-cardinality.")

    levs = _screen_levels(d, features)

    rhs = U.rhs_vars(model)
    in_model = [f for f in features if f in rhs]

    # [SCR-03]
    # Split, and refit the baseline on the training part so that the
    # comparison against the booster is out of sample on both sides
    n = len(d)
    n_tr = int(np.round(sp[0] * n))
    n_va = int(np.round(sp[1] * n))
    grp = rng.permutation(np.repeat(np.array(["train", "valid", "test"]),
                                    [n_tr, n_va, n - n_tr - n_va]))
    is_tr, is_va, is_te = grp == "train", grp == "valid", grp == "test"
    d_tr = d.iloc[np.flatnonzero(is_tr)]
    d_va = d.iloc[np.flatnonzero(is_va)]
    d_te = d.iloc[np.flatnonzero(is_te)]

    def mu_of(b, nd):
        return np.asarray(b.predict(nd), dtype=float)

    # In R the refit itself fails on a factor left with one level; patsy
    # accepts that factor, and the failure only shows when the refitted
    # baseline meets the other levels in the validation or test split. The
    # predictions are therefore part of the attempt.
    try:
        # prior weights given as an array cannot follow the rows into a
        # refit; refit() refuses them, as R's update() does
        base = model.refit(d_tr)
        mu_tr, mu_va, mu_te = (mu_of(base, d_tr), mu_of(base, d_va),
                               mu_of(base, d_te))
    except Exception:
        base = None
    if base is None:
        # Nearly always a factor level in the model formula that is absent
        # from the training split, so name it rather than leave the user
        # guessing
        culprits = [v for v in rhs
                    if v in d_tr.columns and _kind(d_tr[v]) == "factor"
                    and len(_levels_of(d_tr[v])) < 2]
        warn("screen_features: the baseline could not be refitted on the "
             "training split",
             (" (no variation left in: " + ", ".join(map(str, culprits)) + ")")
             if culprits else "",
             "; using the supplied model, whose deviance is then in-sample "
             "and flatters the baseline.")
        base = model
        mu_tr, mu_va, mu_te = (mu_of(base, d_tr), mu_of(base, d_va),
                               mu_of(base, d_te))

    # Response and weights carried explicitly, so a subset of rows (the SHAP
    # sample) stays consistent with its labels
    # [SCR-02]
    w_all = np.ones(n) if counts else w_fit
    y_tr, w_tr = y_all[is_tr], w_all[is_tr]
    y_va, w_va = y_all[is_va], w_all[is_va]
    y_te, w_te = y_all[is_te], w_all[is_te]

    # The baseline margin is the expensive part of a prediction, and it does
    # not change when a feature is shuffled, so compute it once per split
    # instead of once per permutation.
    margin_tr = np.log(mu_tr)
    margin_va = np.log(mu_va)
    margin_te = np.log(mu_te)

    # Design matrices are built once as well; the permutation loop reshuffles
    # the rows of one feature's column block.
    layout_tr = _screen_matrix(d_tr, features, levs)
    layout_te = _screen_matrix(d_te, features, levs)
    cols = np.array(layout_tr.owner, dtype=object)
    nm = layout_tr.names

    def mk_from(x, y, w, margin):
        # [SCR-01] the GLM's log prediction is the booster's base_margin,
        # so it can only model what the GLM leaves behind.
        # columns go in unnamed: xgboost refuses '[', ']' and '<' in feature
        # names, which a level such as "[18,25)" contains. Gain is mapped
        # back by column position instead.
        return xgb.DMatrix(x, label=y, weight=w, base_margin=margin,
                           missing=np.nan, nthread=nthread)

    def mk(nd, y, w, margin):
        return mk_from(_screen_matrix(nd, features, levs).x, y, w, margin)

    dm_tr = mk_from(layout_tr.x, y_tr, w_tr, margin_tr)
    dm_va = mk(d_va, y_va, w_va, margin_va)
    dm_te = mk_from(layout_te.x, y_te, w_te, margin_te)

    def fit_depth(depth):
        params = {"objective": objective, "max_depth": int(depth), "eta": eta,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "nthread": nthread, "tree_method": tree_method,
                  # one seed per booster, drawn from the run's generator
                  "seed": int(rng.integers(0, 2 ** 31 - 1))}
        return xgb.train(params, dm_tr, num_boost_round=int(nrounds),
                         evals=[(dm_va, "valid")],
                         early_stopping_rounds=early_stopping_rounds,
                         verbose_eval=False)

    def pred(b, dm, **kw):
        return b.predict(dm, iteration_range=_iter_range(b), **kw)

    b1 = fit_depth(1)
    b2 = fit_depth(max_depth) if max_depth > 1 else b1

    dev0 = _fam_deviance(fam, y_te, mu_te, w_te)
    dev1 = _fam_deviance(fam, y_te, pred(b1, dm_te), w_te)
    dev2 = _fam_deviance(fam, y_te, pred(b2, dm_te), w_te)
    with np.errstate(divide="ignore", invalid="ignore"):
        p1 = 100 * (dev1 - dev0) / dev0
        p2 = 100 * (dev2 - dev1) / dev1

    summary_df = pd.DataFrame({
        "Stage": ["baseline GLM",
                  "+ booster, depth 1 (additive only)",
                  "+ booster, depth %d (may use interactions)" % max_depth],
        "Deviance": np.round(np.array([dev0, dev1, dev2], dtype=float)),
        "Change": ["", "%+.2f%%" % p1, "%+.2f%%" % p2]})

    # [SCR-04]
    if p1 < -0.1:
        verdict = ("The baseline is missing main effects (depth 1 improves the "
                   "fit by " + "%.2f%%" % -p1 + "). Add them first: the "
                   "interaction ranking below is not trustworthy while "
                   "additive structure is still unexplained.")
    elif p2 < -0.1:
        verdict = ("Main effects look adequate, but interaction structure "
                   "remains (depth " + U._txt(max_depth) + " improves the fit "
                   "by " + "%.2f%%" % -p2 + "). Use detect_interactions() to "
                   "pin down which pair.")
    else:
        verdict = ("The booster finds nothing material beyond the baseline on "
                   "the test split.")

    # [SCR-05]
    # Incremental importance: permute one feature, keep the baseline fixed --
    base_dev = dev2
    # Shuffling the rows of a feature's column block is equivalent to
    # shuffling the underlying column, and avoids rebuilding the whole
    # design matrix and re-running the GLM prediction for every candidate.
    n_te = layout_te.x.shape[0]
    perm = []
    for f in features:
        x = layout_te.x.copy()
        jj = np.flatnonzero(cols == f)
        x[:, jj] = x[np.ix_(rng.permutation(n_te), jj)]
        perm.append(_fam_deviance(fam, y_te,
                                  pred(b2, mk_from(x, y_te, w_te, margin_te)),
                                  w_te) - base_dev)

    # R's xgb.importance() cannot read the dump of a booster built on a
    # single column ("Invalid cast, from String to Array" in xgboost 1.7),
    # and it is called after every expensive stage is already done. Gain is
    # the column this function tells you not to rank on, so losing it is a
    # footnote; losing the whole run to it is not. Gain is xgb.importance's
    # definition: the feature's total split gain, as a share of the total.
    try:
        score = b2.get_score(importance_type="total_gain")
        share = np.zeros(len(nm))
        tot = float(sum(score.values()))
        if tot > 0:
            for key, v in score.items():
                share[int(str(key).lstrip("f"))] = v / tot
        gain_by = [float(share[cols == f].sum()) for f in features]
    except Exception:
        warn("screen_features: xgboost could not report Gain for this model "
             "(it fails on a single-column booster), so the Gain column is "
             "NA. PermDeviance, which is what to rank on, is unaffected.")
        gain_by = [np.nan] * len(features)

    feat_df = pd.DataFrame({
        "Feature": features,
        "InModel": [f in in_model for f in features],
        "PermDeviance": np.round(np.asarray(perm, dtype=float), 2),
        "Gain": np.round(np.asarray(gain_by, dtype=float), 4)})
    feat_df = feat_df.iloc[np.argsort(-feat_df["PermDeviance"].to_numpy(),
                                      kind="stable")].reset_index(drop=True)

    gain_note = (
        "Rank on PermDeviance, not Gain: Gain is biased towards continuous and "
        "high-cardinality features and regularly gives pure noise a sizeable "
        "share. PermDeviance is a ranking, not a test: zero is not a threshold "
        "and the figure carries sampling noise, so re-run with another seed, "
        "or add a column you know is noise, before trusting anything near the "
        "bottom of the list.")

    # [SCR-07]
    # Near-duplicate candidates split their importance ----------------------
    # (R's is.numeric() is FALSE for a logical column, so it is left out)
    num_f = [f for f in features if _kind(d[f]) == "numeric"]
    correlated = None
    if len(num_f) > 1:
        cm = pd.DataFrame({i: _as_num(d[f]) for i, f in enumerate(num_f)}) \
            .corr(method="pearson", min_periods=1).to_numpy()
        k = len(num_f)
        # A pair with no overlapping complete observations correlates to NA;
        # it is counted and said, and kept out of the pairs below.
        n_napair = int(np.isnan(cm[np.triu_indices(k, 1)]).sum())
        if n_napair:
            warn("screen_features: ", n_napair, " numeric pair(s) could not "
                 "be correlated (no overlapping complete observations) and "
                 "are not screened for near-duplication.")
        # column-major, the order of R's which(arr.ind = TRUE)
        up = [(i, j) for j in range(k) for i in range(j)
              if np.isfinite(cm[i, j]) and abs(cm[i, j]) >= cor_threshold]
        if up:
            correlated = pd.DataFrame({
                "VarX": [num_f[i] for i, _ in up],
                "VarY": [num_f[j] for _, j in up],
                "Correlation": np.round([cm[i, j] for i, j in up], 4)})
            correlated = correlated.iloc[np.argsort(
                -np.abs(correlated["Correlation"].to_numpy()),
                kind="stable")].reset_index(drop=True)
            warn("screen_features: ", len(correlated), " near-duplicate "
                 "candidate pair(s); they split their importance between them "
                 "(see $correlated).")

    # SHAP interaction ranking ----------------------------------------------
    interactions = None
    if max_depth > 1 and n_shap > 0:
        # [SCR-06]
        # The SHAP interaction array is n_shap x (k+1) x (k+1) doubles, and k
        # grows with every factor level once one-hot encoded, so cap the
        # sample at roughly 1 GB rather than let it exhaust memory.
        k_cols = len(nm)
        max_rowsz = max(500, int(math.floor(1e9 / (8 * (k_cols + 1) ** 2))))
        if n_shap > max_rowsz:
            message("screen_features: reducing n_shap from ", U._txt(n_shap),
                    " to ", max_rowsz, " to keep the SHAP interaction array "
                    "near 1 GB (", k_cols, " matrix columns).")
            n_shap = max_rowsz
        s = rng.choice(n_te, min(int(n_shap), n_te), replace=False)
        try:
            si = pred(b2, mk_from(layout_te.x[s], y_te[s], w_te[s],
                                  margin_te[s]), pred_interactions=True)
        except Exception as e:
            warn("screen_features: SHAP interaction ranking failed: ", e)
            si = None
        if si is not None:
            kk = len(nm)
            M = np.abs(si[:, :kk, :kk]).mean(axis=0)
            # sum the column-pair values into feature pairs
            G = (cols[:, None] == np.array(features, dtype=object)[None, :]) \
                .astype(float)
            agg = G.T @ M @ G
            nf = len(features)
            up = [(i, j) for j in range(nf) for i in range(j)]
            interactions = pd.DataFrame({
                "VarX": [features[i] for i, _ in up],
                "VarY": [features[j] for _, j in up],
                "Strength": np.round([agg[i, j] for i, j in up], 6)})
            interactions = interactions.iloc[np.argsort(
                -interactions["Strength"].to_numpy(),
                kind="stable")].reset_index(drop=True)

    # Plot: one bar per candidate, muted where there is no signal ----------
    fp = feat_df.iloc[np.argsort(feat_df["PermDeviance"].to_numpy(),
                                 kind="stable")]
    labels = [str(f) for f in fp["Feature"]]
    pd_vals = fp["PermDeviance"].to_numpy(dtype=float)
    text = ["Gain: " + ("NA" if np.isnan(g) else "%.3f" % g)
            + (" · already in the model" if im else "")
            for g, im in zip(fp["Gain"].to_numpy(dtype=float), fp["InModel"])]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=pd_vals, y=labels, orientation="h",
        marker=dict(color=[ta_blue if v > 0 else ta_muted for v in pd_vals]),
        text=text,
        hovertemplate=("<b>%{y}</b><br>PermDeviance: %{x:.2f}"
                       "<br>%{text}<extra></extra>")))
    fig.update_layout(
        title=dict(text="Incremental value over the baseline",
                   font=dict(color=ta_navy, size=14)),
        xaxis=dict(title=dict(text="Deviance increase when shuffled"),
                   gridcolor="#D0D8E0", zeroline=False),
        # R's factor y keeps the bars in this order
        yaxis=dict(title=dict(text=""), tickfont=dict(size=10),
                   showgrid=False, categoryorder="array",
                   categoryarray=labels),
        shapes=[dict(type="line", xref="x", yref="paper",
                     x0=0, x1=0, y0=0, y1=1,
                     line=dict(color=ta_muted, width=1, dash="dot"))],
        hoverlabel=dict(bgcolor="white", font=dict(size=12)),
        plot_bgcolor="white", paper_bgcolor="white",
        margin=dict(l=140, b=60, r=40))
    set_config(fig, "screen")

    message("screen_features: ", verdict)

    return Result(
        summary=summary_df, verdict=verdict,
        features=feat_df, gain_note=gain_note,
        interactions=interactions, correlated=correlated,
        stats=Result(deviance_baseline=float(dev0), deviance_depth1=float(dev1),
                     deviance_depth2=float(dev2), pct_depth1=float(p1),
                     pct_depth2=float(p2), objective=objective,
                     n_train=int(is_tr.sum()), n_valid=int(is_va.sum()),
                     n_test=int(is_te.sum()), n_rows_used=n, nthread=nthread,
                     best_iteration={"depth1": _best_iteration(b1),
                                     "depth2": _best_iteration(b2)}),
        plot=fig)
