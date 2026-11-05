"""add_groupings(): the grouped columns a model was fitted on, added to a
dataset you want to score; the port of R/groupings.R."""
from __future__ import annotations

import pandas as pd

from . import _utils as U
from ._core import stop, warn
from .model import GLMFit


def _as_list(x):
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    try:
        return [str(v) for v in list(x)]
    except TypeError:
        return [str(x)]


def add_groupings(data, model_freq=None, model_sev=None, merge_vars=None):
    """Add the grouped columns a model needs to a dataset you want to score.

    A grouping is made where it is needed: urbanisation 4 and 5 taken
    together next to the average claim, brands grouped one way for the
    frequency and another way for the severity, an age band for a severity
    model next to a spline in the frequency model. Those columns then live
    in the data each model was fitted on and nowhere else, while
    ``predict()`` needs every one of them on the data it scores. A fitted
    model (:class:`~pricingtools_rmo.model.GLMFit`) carries its own data, so
    the relation is read off the model that uses the column and the column
    is added here, from the source columns that a new portfolio does have.

    It takes the same ``merge_vars`` declaration as
    :func:`~pricingtools_rmo.rating_table.make_rating_table`, and the table
    uses this same code, so one declaration serves both and the two cannot
    disagree about which policy sits in which group.

    What is refused and what is not
    -------------------------------
    Anything that would be a guess is an error rather than a column with a
    value in it:

    * a source value the model's data never saw - a new brand, or a level
      with no claims, which therefore has no group in the severity data;
    * a grouping that is not a function of its source in the model's data,
      one value occurring with two groups;
    * one column name that the two models group differently, which cannot
      be scored for both at once; give the two groupings separate names and
      declare both, ``{"MERK": ["MERK_GR_F", "MERK_GR_S"]}``;
    * a grouped column that is already in ``data`` and disagrees with the
      model. It is checked, never overwritten.

    A grouped column that is already in ``data`` keeps its value for a
    source value the model's data never saw, since there is nothing to check
    it against (the R version refuses that case; see the porting notes).

    A continuous source outside the range the model was fitted on is not
    refused. The outer bands of a banding are open in practice - nobody over
    80 in the claims data does not make 85 a new class - so it takes the
    outer band, with a warning saying how many rows did. When the same
    source is also a continuous term of a model, such as ``ns(LEEFTIJD, 4)``,
    the warning says so too: that term extrapolates beyond its range in
    ``predict()``, which no grouping can prevent, and capping the values is
    the usual remedy.

    Whether a source is continuous or categorical is read off the models: a
    factor term is categorical, a numeric term continuous. A source no model
    uses directly is categorical unless it is numeric with more than 25
    distinct values, so a numeric code such as urbanisation 1 to 5 is looked
    up exactly and an unknown code is an error, not a neighbouring group.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset to score, holding the source columns.
    model_freq, model_sev : GLMFit, optional
        The fitted models that use the grouped columns. At least one is
        required.
    merge_vars : dict
        Keyed by the source column, the grouped column(s) derived from it,
        e.g. ``{"Stedelijkheid": "Stedelijkheid_groep",
        "MERK": ["MERK_GR_F", "MERK_GR_S"], "LEEFTIJD": "LFT_KL"}``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``data`` with the grouped columns added, each with the
        type and categories it has in the model's data. Columns that were
        already there are returned unchanged after the check. The input is
        not modified.

    See Also
    --------
    make_rating_table : its ``merge_vars`` takes the same declaration.

    Examples
    --------
    >>> new = pd.DataFrame({"Stedelijkheid": [1, 4, 5]})
    >>> new = add_groupings(new, model_sev=m_sev,
    ...                     merge_vars={"Stedelijkheid": "Stedelijkheid_groep"})
    >>> m_sev.predict(new)
    """
    fn = "add_groupings"
    if model_freq is None and model_sev is None:
        stop(fn, ": provide at least one model.")
    for m in (model_freq, model_sev):
        if m is not None and not isinstance(m, GLMFit):
            stop(fn, ": the models must be glm objects (GLMFit, from fit_glm()).")
    if merge_vars is None:
        stop(fn, ": 'merge_vars' is required, e.g. "
                 "{'Stedelijkheid': 'Stedelijkheid_groep'}.")
    if isinstance(merge_vars, pd.Series):      # R's named vector
        merge_vars = {k: v for k, v in merge_vars.items()} \
            if all(isinstance(k, str) for k in merge_vars.index) else list(merge_vars)
    if not isinstance(merge_vars, dict) or not merge_vars \
            or any(not isinstance(k, str) or not k for k in merge_vars):
        stop(fn, ": 'merge_vars' must be named by the source column, e.g. "
                 "{'Stedelijkheid': 'Stedelijkheid_groep'}.")
    data = U.as_df(data)
    U.check_cols(data, list(merge_vars), fn)

    # [MV-10]
    used = []
    for m in (model_freq, model_sev):
        if m is not None:
            for v in m.rhs_vars():
                if v not in used:
                    used.append(v)
    # the caller's frame is left alone, as R's copy-on-modify leaves it
    data = data.copy()
    seen = []

    for pv, als in merge_vars.items():
        for a in _as_list(als):
            if a == pv:
                stop(fn, ": 'merge_vars' maps '", pv, "' onto itself.")
            if a in seen:
                stop(fn, ": '", a, "' is declared more than once.")
            seen.append(a)
            if a not in used and a in data.columns:
                warn(fn, ": no model uses '", a, "', so it is left as it is.")
                continue
            vals = U.grouping_column(data, pv, a, model_freq, model_sev, fn)
            data[a] = pd.Series(vals, index=data.index) \
                if not isinstance(vals, pd.Series) else vals
    return data
