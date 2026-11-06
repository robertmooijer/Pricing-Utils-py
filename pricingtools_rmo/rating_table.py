"""make_rating_table(): multiplicative rating factors from a frequency and/or
severity model; the port of R/rating-table.R.

The decision tags (``# [RT-04]`` and so on) are those of the R package, so
one search finds the reasoning in TECHNICAL.md and the code in either
language.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import patsy

from . import _utils as U
from ._core import Result, stop, warn

_FN = "make_rating_table"


# ---- small helpers ---------------------------------------------------------------


def _is_num(x):
    """R's is.numeric() on a column: numbers, not logicals."""
    return pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)


def _is_factor(x):
    """R's is.factor() on a column. A pandas string column is the Python
    counterpart of an R factor, so it counts as one too. (R tests is.factor()
    only, so a character column that reaches this point - a merge_vars
    source that no model uses directly - ends on a numeric grid there and
    is skipped with a warning.)"""
    if isinstance(x.dtype, pd.CategoricalDtype):
        return True
    return not pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)


def _is_lgl(x):
    """R's is.logical() on a column."""
    return pd.api.types.is_bool_dtype(x)


def _lvl(vals):
    """The Level text of grid values: .level_chr(), and TRUE / FALSE for
    logical values as R's as.character() writes them."""
    vals = list(vals)
    if vals and all(isinstance(v, (bool, np.bool_)) for v in vals):
        return [U._txt(v) for v in vals]
    if vals and all(isinstance(v, str) for v in vals):
        return vals
    return U.level_chr(vals)


def _is_base(vals, b):
    """grid_vals == base_raw, as R compares them: as text when either side
    is text (a logical grid against its base "FALSE"), as numbers otherwise."""
    vals = list(vals)
    if isinstance(b, str) or any(isinstance(v, (str, bool, np.bool_)) for v in vals):
        bt = U._txt(b)
        return np.array([U._txt(v) == bt for v in vals], dtype=bool)
    return np.asarray(vals, dtype=float) == float(b)


def _txt_list(x):
    return [None if _isna(v) else U._txt(v) for v in x]


def _isna(v):
    try:
        return v is None or bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _as_num(v):
    """R's as.numeric(as.character(v)): NaN where the text is no number."""
    if v is None:
        return np.nan
    if isinstance(v, (bool, np.bool_)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).strip()
    if "_" in s:
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _as_num_r(v):
    """R's as.numeric(v): a logical is 0 / 1."""
    if isinstance(v, (bool, np.bool_)):
        return float(v)
    return _as_num(v)


def _as_num_vec(vals):
    return np.array([_as_num(v) for v in vals], dtype=float)


def _num(x):
    """A number as R's paste() writes it: 30, 40.5, 1e+06."""
    x = float(x)
    if not np.isfinite(x):
        return "NA" if np.isnan(x) else ("Inf" if x > 0 else "-Inf")
    for d in range(1, 16):
        if float("%.*g" % (d, x)) == float("%.15g" % x):
            break
    sci = "%.*e" % (d - 1, x)
    mant, ex = sci.split("e")
    sci = mant + "e" + ex[0] + ex[1:].lstrip("0").rjust(2, "0")
    fixed = np.format_float_positional(float("%.*g" % (d, x)), trim="-")
    return fixed if len(fixed) <= len(sci) else sci


def _anyna(x):
    return bool(pd.Series(x, dtype=object).isna().any()) if len(x) else False


# [PY-04]
def _r_terms(model):
    """The model's terms as R's terms() orders them: by degree, in formula
    order within a degree, and the factors of an interaction in their order
    of first appearance in the formula. A list of lists of factor codes."""
    desc = patsy.ModelDesc.from_formula(model.parsed.rhs)
    terms = [t for t in desc.rhs_termlist if t.factors]
    first = {}
    for t in terms:
        for f in t.factors:
            first.setdefault(f.code, len(first))
    terms = sorted(terms, key=lambda t: len(t.factors))   # stable
    out = []
    for t in terms:
        codes = sorted([f.code for f in t.factors], key=first.get)
        if codes not in out:
            out.append(codes)
    return out


def _r_factor_codes(model):
    """Factor codes in order of first appearance in the formula."""
    out = []
    desc = patsy.ModelDesc.from_formula(model.parsed.rhs)
    for t in desc.rhs_termlist:
        for f in t.factors:
            if f.code not in out:
                out.append(f.code)
    for c in model.xlevels:
        if c not in out:
            out.append(c)
    return out


def _named(x):
    """A pandas Series with names is R's named vector: read it as a dict."""
    if isinstance(x, pd.Series):
        if len(x) and all(isinstance(k, str) for k in x.index):
            return {k: v for k, v in x.items()}
        return list(x)
    return x


def _vec(vals):
    """Values as a list, a scalar as a list of one."""
    if isinstance(vals, (str, bytes)) or np.isscalar(vals) or vals is None:
        return [vals]
    return list(vals)


# ---- the table ---------------------------------------------------------------------


def make_rating_table(model_freq=None, model_sev=None, data=None, grid_res=50,
                      exposure_col="Exposure", claims_col="AantalClaims",
                      base_level="first", trim=(0, 1), min_claims=30,
                      grid_step=None, base=None, merge_vars=None,
                      year_col="BOEKJAAR", loss_col="SCHADELAST"):
    """Build a multiplicative rating table from frequency/severity models.

    Builds a table of multiplicative rating factors from a frequency and/or
    severity model (both log link), including two-way interactions.

    Base convention: the base level is the reference level for categorical
    variables (first level the model knows, or the level with the largest
    exposure when ``base_level="exposure"``), the latest year for the
    accounting year (``year_col``), the median for continuous variables
    (added as an explicit grid point, ``IsBase = True``), and 1 for the
    exposure column. As a result, for log-link models without interactions
    the identity holds exactly: intercept x product(factors) = model
    prediction per exposure unit. A variable that is absent from one of the
    two models gets the neutral factor 1 there (so ``Factor_Premium``
    remains correct).

    A term that turns a column into a factor inside the formula, such as
    ``cut(LEEFTIJD, c(17, 24, 40, 65, 99))``, is reported on its bands; a
    logical column (``GARAGE = True``) on ``TRUE`` and ``FALSE``.

    Interaction rows contain the joint relativity relative to the base cell
    and the pure uplift: joint / (main_factor_x * main_factor_group). They
    carry exposure and claim counts per cell whatever the types of the two
    variables: a continuous variable is binned onto its own grid first, the
    same way the one-way exposure histogram is built. Interaction cells are
    where a portfolio runs out of data, so ``IsThin`` matters most there.

    Levels backed by fewer than ``min_claims`` claims get ``IsThin = True``
    (thin categorical levels also raise a warning) and are dimmed in
    ``make_rating_plot()``. That flag counts claims; the precision of a
    factor is in ``SE_Frequency`` and ``SE_Severity``, the standard error of
    its logarithm from the model's covariance matrix. A level with 31 claims
    passes the flag with a standard error of 0.18, which is a 95% interval
    of x0.70 to x1.42 around its factor.

    A level that is in ``data`` but not in the data a model was fitted on,
    typically a level without claims that the severity model never saw,
    keeps its row: the factor of that model is NA there, and so is the
    premium factor, with a warning. Only a base row the model does not know
    is an error, since every factor is a ratio against it.

    Parameters
    ----------
    model_freq, model_sev : GLMFit, optional
        Fitted models (at least one; log link expected, a warning is raised
        otherwise).
    data : pandas.DataFrame
        The original dataset, used for base values, grids and exposure.
    grid_res : int
        Number of grid points for continuous variables (default 50).
    exposure_col : str
        Exposure column (default ``"Exposure"``).
    claims_col : str
        Claim count column (default ``"AantalClaims"``).
    base_level : {"first", "exposure"}
        ``"first"`` = first level as reference (default); ``"exposure"`` =
        level with the largest exposure as reference.
    trim : (float, float)
        Quantiles for the continuous grid, e.g. ``(0.005, 0.995)`` to avoid
        outliers/extrapolation in the tails (default ``(0, 1)`` = full
        range).
    min_claims : float
        Thin-cell threshold: levels/grid points backed by fewer claims get
        ``IsThin = True`` (default 30). A GLM applies no shrinkage to a
        categorical level, so such a factor comes essentially from those
        few claims; it is dimmed by ``make_rating_plot()`` and greyed out by
        ``export_rating_table()``. Thin categorical levels also raise a
        warning, which names the levels without any claim as well.
    grid_step : float or dict, optional
        Step size for continuous grids. ``None`` (default) picks a readable
        step per variable: roughly ``grid_res`` points across the range,
        rounded to a 1 / 2 / 5 times a power of ten, and never fractional
        for a column that holds whole numbers. So an age runs 18, 19, 20 and
        a vehicle weight 800, 850, 900 rather than 19.2653 and 832.65. A
        whole-number column with at most ``2 * grid_res`` distinct values is
        listed on those values instead of on a step, which is what gives an
        age per year and keeps a coded column (five sums insured spread over
        a million) to five rows. Override it with a single number to force
        one step everywhere, or with a dict to set steps per variable:
        ``{"LEEFTIJD": 1, "GEWICHT": 100}``. Naming only some variables is
        fine, the rest keep the automatic step; a name that matches no
        continuous variable is reported rather than ignored. The base point
        is snapped onto the grid, so the row with factor 1 is a value you
        would actually quote.
    base : dict, optional
        Choose the base (factor 1) per variable, e.g.
        ``{"REGIO": "Zuid", "LEEFTIJD": 40}``. ``None`` (default) keeps the
        rules above. A categorical base must be an existing level and
        overrides ``base_level`` for that variable, without refitting the
        model. A continuous base is placed on the grid exactly as given
        rather than snapped, with a warning if it lies outside the data.
        Variables not named keep their default base; offset and exposure
        variables cannot be named, since their base of 1 is what makes the
        intercept a rate per unit of exposure. The choice only moves the
        factors and the intercept against each other: every premium stays
        the same.
    merge_vars : dict, optional
        Declare that two columns are one rating factor, keyed by the
        variable to keep, e.g. ``{"Stedelijkheid": "Stedelijkheid_groep"}``.

        This is for the case where the severity model uses a regrouped
        version of a frequency variable: urbanisation 1 to 5 in the
        frequency model, and 1, 2, 3, 4-5 in the severity model because the
        top two classes showed the same average claim. Without it the table
        is still correct, but it holds two blocks for one tariff field, each
        padded with neutral factors of 1, and both have to be applied to
        arrive at the premium. With it there is one block, on the levels of
        the variable a policy actually carries, whose severity factor is the
        one belonging to the group that level falls in.

        The regrouped column does not have to be in ``data``. It is usually
        made next to the average claim, in the dataset the severity model is
        fitted on, while ``data`` is the whole portfolio. A fitted model
        carries its own data, so the relation is read off the model that
        uses the column and the column is derived for ``data`` here. Your
        data frame is not changed. A level that occurs in ``data`` but in no
        row of that model's data has no group to derive, which is an error
        naming those levels: a level with no claims has no severity factor
        either way. ``add_groupings()`` runs the same derivation on a dataset
        you want to score, from the same declaration.

        One grouping per column name. A grouping is often made twice, once
        in each dataset, and the two need not agree: the frequency data may
        put brand C with D where the severity data puts it with A and B.
        Only one of the two can be in ``data``, so the other model would be
        read on a grouping it was not fitted on. The mapping in ``data`` is
        therefore compared with the one each model's own data defines, and a
        difference is an error naming the value it turns on. When the two
        models really do group differently, give the columns separate names
        and name both: ``{"MERK": ["MERK_GROEP_freq", "MERK_GROEP_sev"]}``,
        which puts both groupings in one block on the brands themselves.

        The regrouped column must be a function of the variable it
        regroups: every value maps to exactly one group, which is checked
        against ``data``. A band of a continuous variable also works; a grid
        point that does not occur in the data takes the band of the nearest
        value that does. The merged column keeps no block of its own, cannot
        be named in ``base``, and an interaction on it is reported under the
        name of the variable it is merged into. When the relation is there
        but not declared, and each column is used by one of the two models,
        this is pointed out in a warning rather than left to be discovered
        in a spreadsheet.
    year_col : str or None
        Accounting year column (default ``"BOEKJAAR"``), or ``None``. When a
        model uses it as a rating factor, its default base is the latest
        year rather than the first level, so the intercept is the current
        price level instead of the oldest one; the other factors do not
        change. ``base`` still overrides it, e.g. ``{"BOEKJAAR": 2023}``.
    loss_col : str
        Loss amount column (default ``"SCHADELAST"``), read only to warn
        when claims without payment would make frequency x severity
        overstate the risk premium.

    Returns
    -------
    pandas.DataFrame
        One row per level/grid point per variable, with columns
        ``Variable``, ``Type``, ``Level``, ``LevelNum``, ``Group``,
        ``IsBase``, ``Exposure``, ``ClaimCount``, ``Factor_Frequency``,
        ``Factor_Severity``, ``Factor_Premium``, ``SE_Frequency``,
        ``SE_Severity``, ``Uplift_*``, ``XVar``, ``GroupVar``, ``IsThin``.
        ``SE_*`` is the standard error of the logarithm of the factor in
        that row: 0 at the base and for a variable the model does not use,
        NaN where there is no factor. On an interaction row it belongs to
        the joint relativity against the base cell. The three ``Uplift_*``
        columns are present **only when the model contains at least one
        interaction term**; without one they would be NaN in every row, so
        they are dropped. ``Group`` and ``GroupVar`` are always there,
        because ``Group.isna()`` is how a main-effect row is told from an
        interaction row. ``tbl.attrs`` holds ``intercept_frequency``,
        ``intercept_severity``, ``intercept_premium`` (predictions at the
        base point, per unit of exposure) and ``base_values`` (dict with the
        base value per variable).
    """
    if base_level not in ("first", "exposure"):
        stop(_FN, ": 'base_level' must be \"first\" or \"exposure\".")
    if model_freq is None and model_sev is None:
        stop("Provide at least one model.")
    U.check_models(_FN, model_freq=model_freq, model_sev=model_sev)
    if data is None:
        stop(_FN, ": 'data' is required.")
    data = U.as_df(data)
    U.check_cols(data, [exposure_col, claims_col], _FN)
    trim = list(trim) if trim is not None else []
    if len(trim) != 2 or any(not np.isfinite(t) for t in trim) \
            or any(t < 0 for t in trim) or any(t > 1 for t in trim) \
            or trim[0] >= trim[1]:
        stop(_FN, ": 'trim' must be two quantiles with trim[1] < trim[2].")
    if min_claims < 0:
        stop(_FN, ": 'min_claims' must not be negative.")
    grid_step, base, merge_vars = _named(grid_step), _named(base), _named(merge_vars)

    # [OFF-07]
    # Multiplicative factors are only valid for log-link models
    def check_link(model, model_name):
        if model is None:
            return
        lk = model.link
        if lk != "log":
            warn(_FN, ": model '", model_name, "' has link '", lk,
                 "' (not a log link); the factors are then not pure ",
                 "multiplicative relativities.", stacklevel=4)
    check_link(model_freq, "frequency")
    check_link(model_sev, "severity")

    # [OFF-06] [OFF-09]
    # Every offset variable is held at 1 in the base row, so the intercept is
    # per unit of whatever the model actually offsets on. If that is not
    # `exposure_col`, the Exposure column of the table measures something
    # else than the model does, which is worth saying out loud.
    offset_vars = []
    for v in U.offset_vars(model_freq) + U.offset_vars(model_sev):
        if v not in offset_vars:
            offset_vars.append(v)
    if offset_vars and exposure_col not in offset_vars:
        warn(_FN, ": the model offsets on '", ", ".join(offset_vars),
             "' while exposure_col = '", exposure_col, "'. The intercepts are "
             "per unit of the offset variable, but the Exposure and volume "
             "columns are summed from '", exposure_col, "'.")

    # Helpers -------------------------------------------------------------------
    # Underlying column of a term label (shared helper, bound to `data`)
    def base_var(term):
        return U.term_base_var(term, data)

    # [RT-08]
    # xlevels per model, keyed by the underlying column name, so that inline
    # transformations such as factor(YEAR) are also recognised as categorical.
    # Levels are kept as text, as R's xlevels are.
    # R's xlevels leave out a term that is a logical column (GARAGE, or
    # I(LEEFTIJD > 30)), where patsy reads it as categorical; it is left out
    # here too, so such a term is reported as R reports it ([RT-14]).
    xlev_all = {}
    xterm = {}
    for m in (model_freq, model_sev):
        if m is None or not m.xlevels:
            continue
        for code in _r_factor_codes(m):
            if code not in m.xlevels:
                continue
            if all(isinstance(v, (bool, np.bool_)) for v in m.xlevels[code]) \
                    and not code.startswith(("factor(", "C(")):
                continue
            bv = base_var(code)
            if bv not in xlev_all:
                xlev_all[bv] = [U._txt(v) for v in m.xlevels[code]]
            if bv not in xterm:
                xterm[bv] = code

    def var_is_cat(bv):
        return bv in xlev_all or (bv in data.columns and _is_factor(data[bv]))

    def var_levels(bv):
        if bv in xlev_all:
            return list(xlev_all[bv])
        x = data[bv]
        if isinstance(x.dtype, pd.CategoricalDtype):
            present = set(x.dropna().tolist())
            return [U._txt(c) for c in x.cat.categories if c in present]
        u = pd.unique(x.dropna())
        if _is_num(x):
            return [U._txt(v) for v in np.sort(np.asarray(u, dtype=float))]
        return sorted({U._txt(v) for v in u})

    # [RT-11]
    # An inline term that turns a column into a factor - cut(LEEFTIJD, ...), an
    # ifelse() - has levels that are not values of the column: "(24,30]" cannot
    # be put back into LEEFTIJD. Predicting on the label gave NA in every row,
    # the intercept included, since every prediction shares the base row. Such
    # a variable is run through the model on one representative value per
    # level instead - the most frequent value of the column within the level,
    # which falls in it by definition - and its volumes are counted on the
    # evaluated term. factor(BOEKJAAR) needs none of this: "2021" is a year.
    decoder = {}
    for bv, code in xterm.items():
        if bv not in data.columns:
            continue
        lv = xlev_all[bv]
        if not _anyna(U.coerce_like(lv, data[bv])):
            continue
        try:
            ev = U._eval_factor_values(code, data)
            rows = np.array(_txt_list(ev), dtype=object)
        except Exception:
            rows = None
        if rows is None or len(rows) != len(data):
            continue
        col_txt = np.array(_txt_list(data[bv]), dtype=object)
        rep_val = {}
        for lvl in lv:
            sel = np.array([r is not None and r == lvl for r in rows])
            x = [t for t in col_txt[sel] if t is not None]
            if not x:
                rep_val[lvl] = None
                continue
            cnt = pd.Series(x).value_counts()
            top = cnt[cnt == cnt.max()].index
            rep_val[lvl] = sorted(top)[0]
        decoder[bv] = Result(rows=rows, value=rep_val)

    # The values that go into a prediction frame for labels of a variable
    def to_model(bv, vals):
        vals = _vec(vals)
        dc = decoder.get(bv)
        if dc is None:
            return U.coerce_like(vals, data[bv])
        return U.coerce_like([dc.value.get(U._txt(v)) for v in vals], data[bv])

    # The level each row of `data` is in, for the volumes
    row_level_cache = {}

    def row_level(bv):
        if bv in row_level_cache:
            return row_level_cache[bv]
        dc = decoder.get(bv)
        out = (np.array(_txt_list(data[bv]), dtype=object) if dc is None
               else dc.rows)
        row_level_cache[bv] = out
        return out

    # The most recent level of an accounting year, for its default base
    def latest_of(lv):
        num = _as_num_vec(lv)
        if len(num) and not np.isnan(num).any():
            return lv[int(np.argmax(num))]
        return lv[-1]

    if year_col is not None and (not isinstance(year_col, str) or not year_col):
        stop(_FN, ": 'year_col' must be one column name or None.")

    # [MV-01]
    # One rating factor under two column names --------------------------------
    # A variable can enter the frequency model under its own name and the
    # severity model as a regrouping of itself - urbanisation 1 to 5 against
    # 1, 2, 3, 4-5 - because the severity showed no difference between the
    # top two classes. Those are two columns but one tariff field, and
    # `merge_vars` says so, so the table reports one block keyed on the
    # column a policy actually carries.
    merge_map = {}   # alias -> the variable it is a regrouping of
    alias_of = {}    # variable -> its aliases
    level_map = {}   # alias -> lookup from the variable's value

    if merge_vars is not None:
        if not isinstance(merge_vars, dict) or not merge_vars \
                or any(not isinstance(k, str) or not k for k in merge_vars):
            stop(_FN, ": 'merge_vars' must be named by the variable to keep, "
                      "e.g. {'Stedelijkheid': 'Stedelijkheid_groep'}.")
        data = data.copy()          # the caller's frame is not changed
        for pv, als in merge_vars.items():
            als = [als] if isinstance(als, str) else [str(a) for a in als]
            U.check_cols(data, [pv], _FN)
            for a in als:
                if a == pv:
                    stop(_FN, ": 'merge_vars' maps '", pv, "' onto itself.")
                if a in merge_map:
                    stop(_FN, ": '", a, "' is given as a regrouping of both '",
                         merge_map[a], "' and '", pv, "'.")
                if a in alias_of:
                    stop(_FN, ": '", a, "' is both a regrouping of '", pv,
                         "' and the variable '", ", ".join(alias_of[a]),
                         "' is merged into; chains are not supported.")
                # [MV-03]
                # Derived from the model that uses it when `data` lacks it,
                # checked against that model when `data` has it; shared with
                # add_groupings()
                g = U.grouping_column(data, pv, a, model_freq, model_sev, _FN)
                data[a] = g if isinstance(g, pd.Series) \
                    else pd.Series(g, index=data.index)
                row_level_cache.pop(a, None)
                # [MV-02]
                # The relation has to be a function of the variable: one
                # group per value. If a value occurs with two groups there is
                # no single factor to put on that row, and silently taking
                # the first would be wrong for part of the portfolio.
                ok = (data[pv].notna() & data[a].notna()).to_numpy()
                pr = pd.DataFrame({
                    "raw": np.asarray(data[pv].to_numpy()[ok], dtype=object),
                    "p": _txt_list(data[pv].to_numpy()[ok]),
                    "a": _txt_list(data[a].to_numpy()[ok])}) \
                    .drop_duplicates(subset=["p", "a"])
                bad = pr["p"][pr["p"].duplicated()].unique().tolist()
                if bad:
                    stop(_FN, ": '", a, "' is not a regrouping of '", pv,
                         "': the value(s) ", ", ".join(bad[:5]), " of '", pv,
                         "' occur with more than one value of '", a,
                         "', so no single factor belongs to them.")
                if not var_is_cat(pv) and _is_num(data[pv]):
                    num = pr["raw"].astype(float).to_numpy()
                    o = np.argsort(num, kind="stable")
                    level_map[a] = Result(num=num[o],
                                          val=pr["a"].to_numpy()[o].tolist())
                else:
                    level_map[a] = Result(num=None,
                                          val=dict(zip(pr["p"], pr["a"])))
                merge_map[a] = pv
                alias_of.setdefault(pv, []).append(a)

    # [MV-07]
    # The alias value belonging to each value of the variable it regroups.
    # A categorical variable looks its level up. A continuous grid can hold a
    # value that does not occur in the data, so there it takes the band of
    # the nearest value that does, which is the band it falls in because a
    # band covers a run of neighbouring values.
    def alias_values(a, vals):
        vals = _vec(vals)
        lm = level_map[a]
        if lm.num is None:
            out = [lm.val.get(None if _isna(v) else U._txt(v)) for v in vals]
        else:
            v = _as_num_vec(vals)
            keys = lm.num
            brks = np.concatenate([[-np.inf], keys[1:] - np.diff(keys) / 2,
                                   [np.inf]])
            idx = np.searchsorted(brks, v, side="right") - 1
            out = [None if np.isnan(vi) else lm.val[min(max(i, 0), len(lm.val) - 1)]
                   for vi, i in zip(v, idx)]
        if any(o is None for o in out):
            miss = list(dict.fromkeys(
                "NA" if _isna(v) else U._txt(v)
                for v, o in zip(vals, out) if o is None))
            stop("the value(s) ", ", ".join(miss[:5]), " of '", merge_map[a],
                 "' do not occur in 'data', so their group in '", a,
                 "' is unknown")
        return out

    # [MV-01]
    # Put a variable into a prediction frame, together with every column that
    # is a regrouping of it: if the two models use different spellings of one
    # rating factor, both have to move or the merged block would hold one
    # model at its base value while the other moves.
    def set_var(newd, vars_, bv, vals):
        if bv in vars_:
            newd[bv] = to_model(bv, vals)
        for a in alias_of.get(bv, []):
            if a in vars_:
                newd[a] = U.coerce_like(alias_values(a, vals), data[a])
        return newd

    # The name a merged variable is reported under.
    def canon(v):
        return merge_map.get(v, v)

    # Every spelling of one rating factor, to test a model against.
    def spellings(bv):
        return [bv] + alias_of.get(bv, [])

    # [RT-02]
    # Base value per variable (raw value, not yet coerced):
    #   categorical -> reference level; continuous -> median; exposure -> 1
    def base_value_raw(bv):
        # [MV-09]
        # A regrouped column sits at the group its variable's base value falls
        # in, so the base row is a policy that could exist
        if bv in merge_map:
            return alias_values(bv, base_value_raw(merge_map[bv]))[0]
        # [OFF-06]
        # Any offset variable sits at 1, not just the one named by
        # exposure_col, so the intercept is a rate per unit of exposure
        # whatever that column happens to be called
        if bv == exposure_col or bv in offset_vars:
            return 1.0
        if var_is_cat(bv):
            lv = var_levels(bv)
            # A level named in `base` wins over base_level for that variable
            if base is not None and bv in base:
                return U._txt(base[bv])
            # [RT-15] the accounting year sits at its latest level by default,
            # so the intercept is the current price level and not the oldest
            if year_col is not None and bv == year_col:
                return latest_of(lv)
            if base_level == "exposure":
                e = pd.to_numeric(data[exposure_col], errors="coerce")
                ag = (pd.DataFrame({"lvl": row_level(bv), "e": e.to_numpy()})
                      .groupby("lvl", sort=False, dropna=False)["e"].sum())
                cand = [k for k in ag.index[np.argsort(-ag.to_numpy(),
                                                       kind="stable")]]
                cand = [k for k in cand if not _isna(k) and k in lv]
                return cand[0] if cand else lv[0]
            return lv[0]
        if _is_num(data[bv]):
            return grid_of(bv)["base"]
        cnt = pd.Series(_txt_list(data[bv])).value_counts()
        return sorted(cnt[cnt == cnt.max()].index)[0]

    # [RT-01] [RT-09]
    # Base row for a model: all predictors at their base value and every
    # offset variable at 1 (base_value_raw()). The factors are ratios against
    # this row, so the offset cancels in them; the intercept is this row's
    # prediction with the whole offset subtracted on the link scale
    # (intercept_val()), so it is per unit of exposure whatever form the
    # offset takes.
    def base_df_for(model):
        vars_ = model.rhs_vars()
        missing_vars = [v for v in vars_ if v not in data.columns]
        if missing_vars:
            stop("variable(s) from the model formula not in 'data': ",
                 ", ".join(missing_vars))
        row = {}
        for nm in vars_:
            if nm in merge_map:
                row[nm] = U.coerce_like([base_value_raw(nm)], data[nm])
            else:
                row[nm] = to_model(nm, [base_value_raw(nm)])
        # the column names are kept as they are, so a column such as
        # "AUTO GEWICHT" survives (R needs check.names = FALSE here)
        return pd.DataFrame(row)

    def rep_rows(df, n):
        return df.iloc[np.zeros(n, dtype=int)].reset_index(drop=True)

    # [RT-01] [RT-03]
    # Factors for one variable: prediction on grid / prediction on base row.
    # Both with all other predictors at their base value, hence consistent
    # with intercept_val. Variable not in the model -> neutral factor 1.
    def predict_factor(model, var_name, grid_vals):
        n = len(grid_vals)
        if model is None:
            return Result(factor=np.full(n, np.nan), se=np.full(n, np.nan))
        vars_ = model.rhs_vars()
        if not any(s in vars_ for s in spellings(var_name)):
            return Result(factor=np.ones(n), se=np.zeros(n))
        base_df = base_df_for(model)
        newd = rep_rows(base_df, n)
        newd = set_var(newd, vars_, var_name, grid_vals)
        ok = U.predictable_rows(model, newd)
        preds = predict_rows(model, newd, ok)
        ref = float(np.asarray(model.predict(base_df), dtype=float)[0])
        return Result(factor=preds / ref, se=se_of(model, newd, base_df, ok))

    # [RT-10]
    # Predictions with the rows the model cannot be asked about left as NA. A
    # level that exists in the data but not in the data a model was fitted on
    # - a fuel type with exposure but no claims, which the severity model
    # therefore never saw - has no factor in that model. That is one empty
    # cell, not a reason to lose the whole variable from the table.
    def predict_rows(model, newd, ok):
        out = np.full(len(newd), np.nan)
        if ok.any():
            out[ok] = np.asarray(model.predict(newd.loc[ok]), dtype=float)
        return out

    # [RT-12]
    # Standard error of log(Factor), from the model's own covariance matrix:
    # log(Factor) = (x_row - x_base)' beta, so its variance is d' V d with d
    # the difference of the two model-matrix rows. That holds for any term a
    # model can carry - spline, interaction, inline factor - because the
    # model matrix is built exactly as predict() builds it. A volume flag
    # cannot say this: a level with 31 claims passes min_claims = 30 with a
    # standard error of 0.18, a 95% interval of x0.70 to x1.42. Only for a log
    # link, where log(Factor) is linear in the coefficients.
    def se_of(model, newd, base_df, ok):
        out = np.full(len(newd), np.nan)
        if model is None or model.link != "log" or not ok.any():
            return out
        try:
            V = np.asarray(model.vcov, dtype=float)
            keep = np.asarray(model.keep, dtype=bool)
            if V.shape != (keep.sum(), keep.sum()):
                return out
            X = model.model_matrix(newd.loc[ok])[:, keep]
            B = model.model_matrix(base_df)[:, keep]
        except Exception:
            return out
        D = X - B[0]
        out[ok] = np.sqrt(np.maximum(np.sum((D @ V) * D, axis=1), 0))
        return out

    # [RT-06]
    # Joint relativity over a 2D grid (vx x vg), rest at base, scaled to the
    # base cell (index i_base). vx varies fastest.
    def predict_factor_2d(model, vx, gx, vg, gg, i_base):
        n = len(gx) * len(gg)
        if model is None:
            return Result(factor=np.full(n, np.nan), se=np.full(n, np.nan))
        vars_ = model.rhs_vars()
        base_df = base_df_for(model)
        newd = rep_rows(base_df, n)
        newd = set_var(newd, vars_, vx, list(gx) * len(gg))
        newd = set_var(newd, vars_, vg, [g for g in gg for _ in range(len(gx))])
        ok = U.predictable_rows(model, newd)
        preds = predict_rows(model, newd, ok)
        return Result(factor=preds / preds[i_base],
                      se=se_of(model, newd, newd.iloc[[i_base]], ok))

    # [RT-04] [RT-02]
    # Grid and base point for one continuous variable, computed once and
    # reused, so the factors, the intercept and the interaction grids all
    # agree on where the base sits.
    grid_cache = {}

    def grid_of(bv, n_points=None):
        if n_points is None:
            n_points = grid_res
        key = (bv, n_points)
        if key in grid_cache:
            return grid_cache[key]
        x = pd.to_numeric(data[bv], errors="coerce").to_numpy(dtype=float)
        xn = x[~np.isnan(x)]
        rng = np.quantile(xn, trim)
        med = float(np.median(xn))
        step = np.nan
        if grid_step is not None:
            if isinstance(grid_step, dict):
                s = grid_step.get(bv, np.nan)
            elif isinstance(grid_step, (list, tuple, np.ndarray, pd.Series)):
                s = list(grid_step)[0]
            else:
                s = grid_step
            step = _as_num(s)
        int_data = bool(np.all(np.abs(xn - np.round(xn)) < 1e-9))
        inr = (x >= rng[0]) & (x <= rng[1])
        uv = np.unique(x[inr])
        # A whole-number column with few enough distinct values is listed
        # value by value, using those values rather than a step of 1 across
        # the range: a coded column such as a sum insured can hold five
        # values spread over a million, where stepping by 1 would build a
        # grid of a million rows.
        if not np.isfinite(step) and U.list_values(uv, n_points, int_data):
            out = {"values": uv, "base": float(uv[np.argmin(np.abs(uv - med))]),
                   "step": np.nan}
        else:
            if not np.isfinite(step):
                step = U.nice_step(rng, n_points, int_data)
            # snap the base onto the same grid, so the sequence stays
            # readable and the row with factor 1 is a value someone would
            # actually quote
            b0 = float(U.snap_to_step(med, step, rng[0], rng[1]))
            out = {"values": U.step_grid(rng[0], rng[1], step, b0),
                   "base": b0, "step": step}
        # [RT-15] a numeric accounting year (a trend term) sits at its latest
        # grid value, like a factor(BOEKJAAR) at its latest level
        if year_col is not None and bv == year_col:
            out["base"] = float(np.max(out["values"]))
        # A base chosen in `base` replaces the median. It is put on the grid
        # exactly as given - the user asked for 40, not for the nearest grid
        # point - after dropping any grid value that is the same number up to
        # rounding, so there is one row with factor 1 and not two.
        if base is not None and bv in base:
            ub = _as_num_r(base[bv])
            vals = out["values"][np.abs(out["values"] - ub)
                                 > 1e-9 * max(1, abs(ub))]
            out["values"] = np.sort(np.concatenate([vals, [ub]]))
            out["base"] = ub
        # a logical column keeps logical grid values, as R's sort(unique(x))
        # does, so they read TRUE / FALSE and go into predict() as logicals
        out["values"] = (np.asarray(out["values"], dtype=float).astype(bool)
                         if _is_lgl(data[bv])
                         else np.asarray(out["values"], dtype=float))
        grid_cache[key] = out
        return out

    # [RT-05]
    # Which grid point does each policy belong to? Categorical: its level.
    # Continuous: the nearest grid value, splitting on the midpoints between
    # them, the same rule the main-effect exposure histogram uses. Returns
    # NA (-1 here, 0-based otherwise) outside a trimmed range, so the volume
    # of policies the grid does not cover is not swept into the edge cells.
    #
    # [RT-16] The trimming applies to a grid that spans the trimmed range, not
    # to the three slices of a continuous group variable in an interaction:
    # those stand for the whole distribution, and cutting at the outer slices
    # threw away everything below the 10% and above the 90% quantile, a fifth
    # of the exposure of every interaction block.
    untrimmed = abs(trim[0]) < 1.5e-8 and abs(trim[1] - 1) < 1.5e-8

    def cell_index(bv, grid_vals, is_cat, drop_outside=True):
        if is_cat:
            pos = {str(g): i for i, g in reversed(list(enumerate(grid_vals)))}
            return np.array([-1 if r is None else pos.get(r, -1)
                             for r in row_level(bv)], dtype=int)
        x = pd.to_numeric(data[bv], errors="coerce").to_numpy(dtype=float)
        g = np.asarray(grid_vals, dtype=float)
        brks = np.concatenate([[-np.inf], g[1:] - np.diff(g) / 2, [np.inf]])
        i = np.searchsorted(brks, x, side="right") - 1
        i[np.isnan(x)] = -1
        outside = ~np.isnan(x) & ((x < g.min()) | (x > g.max()))
        if drop_outside and not untrimmed:
            i[outside] = -1
        return i.astype(int)

    def volumes(idx, n_cells):
        """Exposure and claims per cell; NA cells left out, NA values as 0
        (R's sum(na.rm = TRUE))."""
        e = pd.to_numeric(data[exposure_col], errors="coerce").to_numpy(dtype=float)
        c = pd.to_numeric(data[claims_col], errors="coerce").to_numpy(dtype=float)
        ok = idx >= 0
        expo = np.bincount(idx[ok], weights=np.nan_to_num(e[ok]), minlength=n_cells)
        clm = np.bincount(idx[ok], weights=np.nan_to_num(c[ok]), minlength=n_cells)
        return expo[:n_cells].astype(float), clm[:n_cells].astype(float)

    # Base variables per model (interactions split up, deduplicated)
    def model_bases(model):
        if model is None:
            return []
        out = []
        for codes in _r_terms(model):
            for code in codes:
                v = canon(base_var(code))
                if v not in out:
                    out.append(v)
        return out

    uniq_bases = []
    for v in model_bases(model_freq) + model_bases(model_sev):
        if v not in uniq_bases:
            uniq_bases.append(v)
    unknown = [v for v in uniq_bases if v not in data.columns]
    if unknown:
        warn(_FN, ": term(s) skipped because the base variable is not in "
                  "'data': ", ", ".join(unknown))
        uniq_bases = [v for v in uniq_bases if v in data.columns]
    if not uniq_bases:
        stop(_FN, ": no usable variables found.")

    def rhs_union():
        out = []
        for m in (model_freq, model_sev):
            if m is not None:
                for v in m.rhs_vars():
                    if v not in out:
                        out.append(v)
        return out

    # [MV-09]
    # A merge declaration that matches no term is as silent a mistake as a
    # misspelled grid step: the two blocks stay, and nothing says why.
    if merge_map:
        model_terms = rhs_union()
        idle = [a for a in merge_map
                if not any(x in model_terms for x in (a, merge_map[a]))]
        if idle:
            warn(_FN, ": 'merge_vars' mentions ", ", ".join(idle), ", which "
                      "neither model uses, so the declaration changes nothing.")

    # [MV-08]
    # The same situation left undeclared. Worth one line, because the table
    # is then right but reads as two tariff fields where there is one, and
    # whoever builds the tariff from it has to know they belong together.
    # Only across the two models: two nested factors inside one model (a
    # postcode and its region) are genuinely two rating factors.
    if model_freq is not None and model_sev is not None:
        rf, rs = model_freq.rhs_vars(), model_sev.rhs_vars()
        f_only = [v for v in rf if v not in rs and v in data.columns]
        s_only = [v for v in rs if v not in rf and v in data.columns]
        cand = []
        for a in f_only:
            for b in s_only:
                if a in merge_map or b in merge_map:
                    continue
                if _is_num(data[a]) and _is_num(data[b]):
                    continue
                na_, nb_ = data[a].nunique(dropna=True), data[b].nunique(dropna=True)
                # A column with very many levels is not a tariff block, and
                # two columns of the same width are not a regrouping of one
                # another
                if na_ < 2 or nb_ < 2 or na_ == nb_ or max(na_, nb_) > 1000:
                    continue
                fine, coarse = (a, b) if na_ > nb_ else (b, a)
                okr = (data[a].notna() & data[b].notna()).to_numpy()
                if not okr.any():
                    continue
                sub = pd.DataFrame({"f": _txt_list(data[fine].to_numpy()[okr]),
                                    "c": _txt_list(data[coarse].to_numpy()[okr])})
                # One group per value, which is what makes it a regrouping
                if len(sub.drop_duplicates()) == sub["f"].nunique():
                    s = "merge_vars={'%s': '%s'}" % (fine, coarse)
                    if s not in cand:
                        cand.append(s)
        if cand:
            warn(_FN, ": one column looks like a regrouping of another, each "
                      "used by one of the two models, so the table holds two "
                      "blocks for what is one tariff field. If that is what "
                      "they are, pass ", " or ".join(cand), ".")

    # [RT-02] [MV-09]
    # A chosen base per variable. Checked up front, because a wrong one does
    # not fail later: it silently moves every factor of that variable.
    if base is not None:
        if not isinstance(base, dict) or not base \
                or any(not isinstance(k, str) or not k for k in base):
            stop(_FN, ": 'base' must be named, e.g. "
                      "{'REGIO': 'Zuid', 'LEEFTIJD': 40}.")
        base = dict(base)
        merged = [k for k in base if k in merge_map]
        if merged:
            stop(_FN, ": '", ", ".join(merged), "' is merged into '",
                 ", ".join(dict.fromkeys(merge_map[k] for k in merged)),
                 "'; set the base on that variable instead.")
        fixed = [k for k in base if k in [exposure_col] + offset_vars]
        if fixed:
            stop(_FN, ": '", ", ".join(fixed), "' is an offset or exposure "
                      "variable; its base is always 1, which is what makes "
                      "the intercept a rate per unit of exposure.")
        unused = [k for k in base if k not in uniq_bases]
        if unused:
            warn(_FN, ": 'base' names with no variable in the model, and "
                      "therefore ignored: ", ", ".join(unused), ".")
            base = {k: v for k, v in base.items() if k not in unused}
        for bv in list(base):
            val = base[bv]
            if not (isinstance(val, str) or np.isscalar(val)):
                vl = list(val) if hasattr(val, "__len__") else [val]
                if len(vl) != 1:
                    stop(_FN, ": 'base' for '", bv, "' must be one value.")
                val = vl[0]
                base[bv] = val
            if var_is_cat(bv):
                lv = var_levels(bv)
                if U._txt(val) not in lv:
                    stop(_FN, ": '", U._txt(val), "' is not a level of '", bv,
                         "'. Levels: ", ", ".join(lv), ".")
            else:
                num = _as_num_r(val)
                if not np.isfinite(num):
                    stop(_FN, ": 'base' for '", bv, "' must be a number.")
                r = pd.to_numeric(data[bv], errors="coerce").to_numpy(dtype=float)
                lo, hi = float(np.nanmin(r)), float(np.nanmax(r))
                if num < lo or num > hi:
                    warn(_FN, ": the base ", _num(num), " for '", bv,
                         "' lies outside the data (", _num(lo), " to ",
                         _num(hi), "); every factor of that variable is then "
                         "relative to an extrapolated point.")
        if not base:
            base = None

    # [RT-04]
    # A misspelled name in grid_step would otherwise be ignored in silence,
    # leaving the user convinced they had set a step they had not
    if isinstance(grid_step, dict):
        named = [k for k in grid_step if k]
        cont = [b for b in uniq_bases if not var_is_cat(b) and _is_num(data[b])]
        unused = [k for k in named if k not in cont]
        if unused:
            warn(_FN, ": 'grid_step' names with no continuous variable to "
                      "apply to, and therefore ignored: ", ", ".join(unused),
                 ". Continuous variables in this model: ",
                 ", ".join(cont) if cont else "none", ".")

    # [RT-10]
    # Every factor is a ratio against the base row, so a base row a model was
    # never fitted on would empty the whole table. Caught once, up front, with
    # the value that causes it.
    for m, nm in ((model_freq, "frequency"), (model_sev, "severity")):
        if m is None:
            continue
        un = U.unseen_levels(m, base_df_for(m))
        if un:
            stop(_FN, ": the base row has ",
                 "; ".join("'%s' = %s" % (cl, u.values[0]) for cl, u in un.items()),
                 ", which the ", nm, " model was not fitted on (a level "
                 "without claims, say). Choose a base that model knows, with "
                 "base = or base_level = \"exposure\".")

    # [RT-13] claims without payment overstate frequency x severity
    U.nil_claims_warning(model_freq, model_sev, data, claims_col, loss_col, _FN)

    # [RT-06]
    # Two-way interaction pairs (base variables) from both models, deduplicated
    def gather_pairs():
        seen, res = [], []
        for m in (model_freq, model_sev):
            if m is None:
                continue
            for codes in _r_terms(m):
                if len(codes) != 2:                       # two-way only
                    continue
                bvs = [canon(base_var(c)) for c in codes]
                # An interaction between a variable and a regrouping of itself
                # is one variable after merging, so there is no two-way table
                # to build
                if bvs[0] == bvs[1]:
                    continue
                key = ":".join(sorted(bvs))
                if key in seen:
                    continue
                seen.append(key)
                res.append(bvs)
        return res

    # [CONV-04]
    # Build the rows per base variable ------------------------------------------
    def main_block(bv):
        is_cat = var_is_cat(bv)
        base_raw = base_value_raw(bv)

        # Grid: categorical = all levels; continuous = (trimmed) range with the
        # median as an explicit extra grid point so there is a row with factor 1
        if is_cat:
            grid_vals = var_levels(bv)
        else:
            grid_vals = grid_of(bv)["values"]
        is_base = _is_base(grid_vals, base_raw)

        # Exposure / claims aggregation, per level as the model sees it (for a
        # cut() term that is the band a row falls in, [RT-11]).
        # Continuous variables: exposure histogram via binning on the grid.
        # Values outside a trimmed range are excluded rather than swept into
        # the edge bins, which would make the outer bars overstate their
        # volume: the grid does not cover those policies, so neither should
        # the bars. The trimmed exposure is therefore below the portfolio
        # total by design.
        exposure, claim_count = volumes(cell_index(bv, grid_vals, is_cat),
                                        len(grid_vals))

        # Retrieve factors per model
        p_freq = predict_factor(model_freq, bv, grid_vals)
        p_sev = predict_factor(model_sev, bv, grid_vals)

        n = len(grid_vals)
        return pd.DataFrame({
            "Variable": [bv] * n,
            "Type": ["categorical" if is_cat else "continuous"] * n,
            "Level": _lvl(grid_vals),
            "LevelNum": _as_num_vec(grid_vals),
            "Group": pd.Series([None] * n, dtype=object),
            "IsBase": np.asarray(is_base, dtype=bool),
            "Exposure": exposure,
            "ClaimCount": claim_count,
            "Factor_Frequency": p_freq.factor,
            "Factor_Severity": p_sev.factor,
            "Factor_Premium": p_freq.factor * p_sev.factor,
            "SE_Frequency": p_freq.se,
            "SE_Severity": p_sev.se,
            "Uplift_Frequency": np.full(n, np.nan),
            "Uplift_Severity": np.full(n, np.nan),
            "Uplift_Premium": np.full(n, np.nan),
            "XVar": [bv] * n,
            "GroupVar": pd.Series([None] * n, dtype=object),
        })

    rows = []
    for bv in uniq_bases:
        try:
            rows.append(main_block(bv))
        except Exception as e:
            warn(_FN, ": variable '", bv, "' skipped: ", _msg(e))
    if not rows:
        stop(_FN, ": none of the variables could be processed.")

    out = pd.concat(rows, ignore_index=True)

    # [RT-06] [CONV-04]
    # Build the (two-way) interaction rows ---------------------------------------
    # Per pair: one variable becomes the X-axis, the other the "Group". The
    # factor is the joint relativity relative to the base cell (both at
    # base); the uplift is joint / (main_factor_x * main_factor_group).
    def interaction_block(bvs):
        b1, b2 = bvs
        if not (b1 in data.columns and b2 in data.columns):
            stop("base variable not in 'data'")
        c1, c2 = var_is_cat(b1), var_is_cat(b2)

        # X = continuous when exactly one is continuous; both equal: most
        # levels = X
        if c1 != c2:
            xv, gv = (b2, b1) if c1 else (b1, b2)
        elif c1 and c2:
            if len(var_levels(b2)) > len(var_levels(b1)):
                xv, gv = b2, b1
            else:
                xv, gv = b1, b2
        else:
            xv, gv = b1, b2
        cx, cg = var_is_cat(xv), var_is_cat(gv)

        base_x = base_value_raw(xv)
        base_g = base_value_raw(gv)

        gx = var_levels(xv) if cx else grid_of(xv)["values"]
        # Continuous group variable: 3 slices = q10, median (= base), q90
        if cg:
            gg = var_levels(gv)
        else:
            xg = pd.to_numeric(data[gv], errors="coerce").to_numpy(dtype=float)
            q = np.round(np.quantile(xg[~np.isnan(xg)], [0.1, 0.9]), 3)
            gg = np.unique(np.concatenate([q, [base_g]]))

        nx, ng = len(gx), len(gg)
        hit = np.flatnonzero(_is_base(gx, base_x))
        ix = int(hit[0]) if len(hit) else None
        hit = np.flatnonzero(_is_base(gg, base_g))
        ig = int(hit[0]) if len(hit) else None
        if ix is None or ig is None:
            stop("base cell not found in the grid")
        i_base = ig * nx + ix

        xvals = list(gx) * ng            # X varies fastest (= predict_factor_2d)
        gvals = [g for g in gg for _ in range(nx)]

        # Exposure and claims per cell. A continuous variable is binned onto
        # its own grid first (same midpoint rule as the main-effect
        # histogram), so the volume columns are filled for every combination
        # of types rather than only for categorical x categorical.
        # Interaction cells are exactly where a portfolio runs out of data, so
        # this is the place the thin-cell flag earns its keep.
        ci = cell_index(xv, gx, cx)
        cj = cell_index(gv, gg, cg, drop_outside=False)   # [RT-16]
        lin = np.where((ci >= 0) & (cj >= 0), cj * nx + ci, -1)
        expo, clm = volumes(lin, nx * ng)               # X varies fastest

        p2f = predict_factor_2d(model_freq, xv, gx, gv, gg, i_base)
        p2s = predict_factor_2d(model_sev, xv, gx, gv, gg, i_base)
        ff, fs = p2f.factor, p2s.factor

        # Pure interaction uplift: joint / (main_factor_x * main_factor_group)
        uf = ff / (np.tile(predict_factor(model_freq, xv, gx).factor, ng)
                   * np.repeat(predict_factor(model_freq, gv, gg).factor, nx))
        us = fs / (np.tile(predict_factor(model_sev, xv, gx).factor, ng)
                   * np.repeat(predict_factor(model_sev, gv, gg).factor, nx))

        n = nx * ng
        return pd.DataFrame({
            "Variable": [b1 + ":" + b2] * n,
            "Type": ["categorical" if cx else "continuous"] * n,
            "Level": _lvl(xvals),
            "LevelNum": _as_num_vec(xvals),
            "Group": pd.Series(_lvl(gvals), dtype=object),
            "IsBase": np.arange(n) == i_base,
            "Exposure": expo,
            "ClaimCount": clm,
            "Factor_Frequency": ff,
            "Factor_Severity": fs,
            "Factor_Premium": ff * fs,
            "SE_Frequency": p2f.se,
            "SE_Severity": p2s.se,
            "Uplift_Frequency": uf,
            "Uplift_Severity": us,
            "Uplift_Premium": uf * us,
            "XVar": [xv] * n,
            "GroupVar": pd.Series([gv] * n, dtype=object),
        })

    int_rows = []
    for bvs in gather_pairs():
        try:
            int_rows.append(interaction_block(bvs))
        except Exception as e:
            warn("Interaction ", ":".join(bvs), " skipped: ", _msg(e))
    if int_rows:
        out = pd.concat([out] + int_rows, ignore_index=True)
    else:
        # [RT-06]
        # An uplift only exists relative to the two main effects it sits
        # between, so on a tariff without a single interaction term these
        # three columns are NA in every row. Drop them rather than ship a
        # block of emptiness that has to be explained. `Group` and `GroupVar`
        # stay: `Group.isna()` is how the rest of the package tells a main
        # effect from an interaction, so those two are structure, not payload.
        out = out.drop(columns=["Uplift_Frequency", "Uplift_Severity",
                                "Uplift_Premium"])

    # [RT-07]
    # Thin-cell flag -------------------------------------------------------------
    # Levels backed by very few claims: the GLM applies no shrinkage to a
    # categorical level, so its factor comes essentially from those few
    # claims and is correspondingly unstable. Flagged here, dimmed by
    # make_rating_plot() and greyed out by export_rating_table().
    cc = out["ClaimCount"].to_numpy(dtype=float)
    if np.isnan(cc).any():
        out["IsThin"] = pd.Series([None if np.isnan(c) else bool(c < min_claims)
                                   for c in cc], dtype=object)
    else:
        out["IsThin"] = cc < min_claims

    # Warn for thin categorical main-effect levels (thin tail bins are
    # normal for continuous grids and would make the warning noisy). A level
    # with exposure and no claims at all is worth its own words: its
    # frequency factor is the fit running towards zero, not an estimate.
    main_cat = (out["Group"].isna() & (out["Type"] == "categorical")).to_numpy()
    thin = np.array([bool(t) if t is not None else False for t in out["IsThin"]])
    n_thin = int(np.sum(thin & main_cat))
    zero = main_cat & (cc == 0) & (out["Exposure"].to_numpy(dtype=float) > 0)
    if n_thin > 0:
        extra = ""
        if zero.any():
            nz = int(zero.sum())
            cells = (out["Variable"][zero] + " = " + out["Level"][zero]).tolist()
            extra = (" %d%s no claims at all (%s): their frequency factor is "
                     "the fit running towards zero, not an estimate; group "
                     "them with a level that has claims."
                     % (nz, " of them has" if nz == 1 else " of them have",
                        ", ".join(cells[:5])))
        warn(_FN, ": ", n_thin, " categorical level(s) have fewer than ",
             _num(min_claims), " claims; the corresponding factors rest on "
             "very little experience (see the IsThin and SE_ columns).", extra)

    # [RT-10]
    # Cells a model has no factor for: a level the model was not fitted on.
    # Said once, naming the cells, because an NA in a tariff table is a value
    # somebody has to supply by hand.
    for m, col, nm in ((model_freq, "Factor_Frequency", "frequency"),
                       (model_sev, "Factor_Severity", "severity")):
        if m is None:
            continue
        i = np.flatnonzero(out[col].isna().to_numpy())
        if not len(i):
            continue
        cells = []
        for k in i:
            g = out["Group"].iloc[k]
            c = "%s = %s%s" % (out["Variable"].iloc[k], out["Level"].iloc[k],
                               "" if _isna(g) else " x %s" % g)
            if c not in cells:
                cells.append(c)
        warn(_FN, ": no ", nm, " factor for ", ", ".join(cells[:5]),
             " and %d more" % (len(cells) - 5) if len(cells) > 5 else "",
             ": the ", nm, " model was not fitted on that level",
             " (a level without claims leaves no severity data)"
             if nm == "severity" else "",
             ", so ", col, " and Factor_Premium are NA there. Group the level "
             "with one the model knows, or supply the factor yourself.")

    # [OFF-06] [RT-01] [RT-03]
    # Intercepts (base values on the response scale, exposure = 1) ---------------
    # The factors are ratios of two predictions that share the same base row,
    # so whatever the offset contributes cancels there. The intercept is the
    # one number that carries it, so it is the one that has to have the
    # offset taken back out on the link scale rather than trusted to vanish
    # because a column was set to 1 - which only works for offset(log(x)).
    def intercept_val(model):
        if model is None:
            return float("nan")
        return float(np.asarray(U.predict_no_offset(model, base_df_for(model)),
                                dtype=float)[0])

    i_freq = intercept_val(model_freq)
    i_sev = intercept_val(model_sev)

    out = out.reset_index(drop=True)
    out.attrs["intercept_frequency"] = i_freq
    out.attrs["intercept_severity"] = i_sev
    out.attrs["intercept_premium"] = (i_freq * i_sev
                                      if not (math.isnan(i_freq) or math.isnan(i_sev))
                                      else float("nan"))
    out.attrs["base_values"] = {bv: base_value_raw(bv) for bv in uniq_bases}
    return out


def _msg(e):
    """An exception's message, as R's conditionMessage()."""
    return str(e.args[0]) if getattr(e, "args", None) else str(e)
