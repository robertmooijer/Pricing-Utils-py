"""Small helpers for the one-way, A/E, PDP and rating plots.

They carry over three pieces of R behaviour the plotting code relies on
without stating it:

* ``match.arg()``, with its partial matching;
* the order R puts things in: ``dplyr::arrange()`` and ``data.table``'s
  ``order()`` sort text by code point (the C locale), ``factor()`` and
  ``sort()`` sort it in the session's locale (``a A b C e é``);
* what ``plotly::plotly_build()`` adds to a figure in R: an axis that holds
  a factor, character or logical vector becomes a category axis in the
  order of the factor levels (or sorted), whatever the order of the rows.
"""
from __future__ import annotations

import unicodedata

import numpy as np
import pandas as pd

from . import _utils as U
from ._core import stop


def match_arg(arg, choices, fn=None, name="arg"):
    """R's match.arg(): the first choice when `arg` is None, otherwise the
    one choice `arg` is a (partial) match for."""
    if arg is None:
        return choices[0]
    if isinstance(arg, (list, tuple)) and list(arg) == list(choices):
        return choices[0]
    if isinstance(arg, str):
        if arg in choices:
            return arg
        hit = [c for c in choices if c.startswith(arg)] if arg else []
        if len(hit) == 1:
            return hit[0]
    stop((fn + ": " if fn else ""), "'", name, "' should be one of ",
         ", ".join('"%s"' % c for c in choices))


# ---- order -------------------------------------------------------------------


def collate_key(s):
    """A sort key for text in the order of R's sort() and factor() under a
    Western locale: case and accents only break ties (a A b C e é)."""
    s = str(s)
    base = "".join(c for c in unicodedata.normalize("NFD", s)
                   if not unicodedata.combining(c)).casefold()
    return (base, s.casefold(), s.swapcase())


def is_categorical(x):
    return isinstance(getattr(x, "dtype", None), pd.CategoricalDtype)


def is_num(x):
    """R's is.numeric(): numbers, not logicals or factors."""
    dt = getattr(x, "dtype", None)
    if dt is None:
        x = pd.Series(x)
        dt = x.dtype
    if isinstance(dt, pd.CategoricalDtype):
        return False
    return pd.api.types.is_numeric_dtype(dt) and not pd.api.types.is_bool_dtype(dt)


def is_discrete(x):
    """plotly's is.discrete(): a factor, character or logical vector."""
    return not is_num(x)


def r_sort_unique(x):
    """sort(unique(x)) as R does it: NA dropped, text in locale order."""
    s = pd.Series(x).reset_index(drop=True)
    s = s[s.notna()]
    if is_categorical(s):
        s = s.astype(object)
    vals = list(pd.unique(s))
    if is_num(pd.Series(vals)) or pd.api.types.is_bool_dtype(pd.Series(vals)):
        return sorted(vals)
    return sorted(vals, key=collate_key)


def r_levels(x):
    """The levels R's factor() gives `x`, as text: a factor keeps its own
    levels that occur, numbers sort numerically, text in locale order."""
    s = pd.Series(x).reset_index(drop=True)
    if is_categorical(s):
        present = set(s.cat.codes[s.cat.codes >= 0])
        return [U._txt(c) for i, c in enumerate(s.cat.categories) if i in present]
    return [U._txt(v) for v in r_sort_unique(s)]


def as_r_factor(x, levels=None):
    """x as a pandas Categorical of text labels, as R's factor(x)."""
    s = pd.Series(x).reset_index(drop=True)
    lev = r_levels(s) if levels is None else list(levels)
    vals = [None if pd.isna(v) else U._txt(v) for v in s.astype(object)]
    return pd.Categorical(vals, categories=lev)


def _codes(s, na_first=False):
    """Integer sort codes of a column, NA last (or first)."""
    s = pd.Series(s).reset_index(drop=True)
    codes, _ = pd.factorize(s, sort=True)
    codes = np.asarray(codes, dtype=float)
    codes[codes < 0] = -np.inf if na_first else np.inf
    return codes


def arrange(df, cols, na_first=False):
    """dplyr::arrange() (NA last) or data.table's order() (NA first): a
    stable sort on the columns, factors by level, text by code point."""
    if not len(df):
        return df.reset_index(drop=True)
    keys = [_codes(df[c], na_first) for c in reversed(cols)]
    o = np.lexsort(keys)
    return df.iloc[o].reset_index(drop=True)


# ---- numbers --------------------------------------------------------------------


def wmean(x, w, na_rm=False):
    """R's weighted.mean(): sum((x * w)[w != 0]) / sum(w)."""
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)
    if na_rm:
        keep = ~np.isnan(x)
        x, w = x[keep], w[keep]
    with np.errstate(invalid="ignore", divide="ignore"):
        xw = x * w
        return float(np.sum(xw[w != 0]) / np.sum(w))


def fmt_big(x, digits=0):
    """format()/formatC() with big.mark = ",": 1,234 or 1,234.5678."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "NA"
    return "{:,.{d}f}".format(float(x), d=digits)


# ---- plotly ---------------------------------------------------------------------


def plot_x(x):
    """Values as plotly receives them from R: a factor as its labels,
    missing values as null."""
    s = pd.Series(x).reset_index(drop=True)
    if is_categorical(s):
        return [None if pd.isna(v) else U._txt(v) for v in s.astype(object)]
    if pd.api.types.is_bool_dtype(s.dtype):
        return [None if pd.isna(v) else bool(v) for v in s.astype(object)]
    if is_num(s):
        return s.to_numpy(dtype=float, na_value=np.nan)
    return [None if pd.isna(v) else v for v in s.astype(object)]


def category_axis(*xs):
    """What plotly_build() adds in R to an axis whose traces hold discrete
    data: type 'category' with the factor levels (or the sorted values) as
    the category order. None for a numeric axis."""
    xs = [pd.Series(x).reset_index(drop=True) for x in xs if x is not None]
    if not xs:
        return None
    disc = [is_discrete(x) for x in xs]
    if not any(disc) or not all(disc):
        return None
    cats = []
    for x in xs:
        lev = ([U._txt(c) for c in x.cat.categories] if is_categorical(x)
               else r_sort_unique(x))
        for v in lev:
            if v not in cats:
                cats.append(v)
    if not all(is_categorical(x) for x in xs):
        if all(isinstance(v, (bool, np.bool_)) for v in cats):
            cats = sorted(cats)
        else:
            cats = sorted(cats, key=collate_key)
    return {"type": "category", "categoryorder": "array", "categoryarray": cats}


def fill_axis(cfg, extra):
    """Add plotly_build()'s category settings to an axis without
    overriding what the function set itself (R's %||%)."""
    if extra:
        for k, v in extra.items():
            if cfg.get(k) is None:
                cfg[k] = v
    return cfg


def n2mfrow(n):
    """grDevices::n2mfrow(), which facet_wrap() uses for its grid:
    (columns, rows)."""
    if n <= 3:
        return (n, 1)
    if n <= 6:
        return ((n + 1) // 2, 2)
    if n <= 12:
        return ((n + 2) // 3, 3)
    nrow = int(np.ceil(np.sqrt(n)))
    return (nrow, int(np.ceil(n / nrow)))
