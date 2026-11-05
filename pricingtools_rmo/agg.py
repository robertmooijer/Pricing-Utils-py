"""agg_all(): raw policy rows to frequency and severity per level; the port
of R/agg.R."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import _utils as U
from ._core import stop, warn
from ._plot_common import as_r_factor


def agg_all(d, col, by_year,
            exposure_col="Exposure",
            claims_col="AantalClaims",
            loss_col="SCHADELAST",
            year_col="BOEKJAAR"):
    """Aggregate raw policy rows to frequency/severity per level.

    Goes from raw rows to an aggregate per level of `col` (and optionally
    per accounting year). This is the aggregation engine used internally by
    :func:`make_plot`; call it directly when you want the summary table
    itself.

    NA values in the input columns and groups with non-positive exposure
    are reported with a warning instead of being silently averaged away.

    Parameters
    ----------
    d : pandas.DataFrame
        Raw rows.
    col : str
        Name of the grouping column.
    by_year : bool
        True = also split by accounting year.
    exposure_col : str
        Exposure column (default ``"Exposure"``).
    claims_col : str
        Claim count column (default ``"AantalClaims"``).
    loss_col : str
        Loss amount column (default ``"SCHADELAST"``).
    year_col : str
        Accounting-year column (default ``"BOEKJAAR"``).

    Returns
    -------
    pandas.DataFrame
        Standardised column names: `col`, [``Year``,] ``Exposure``,
        ``ClaimCount``, ``Loss``, ``Frequency`` (``ClaimCount / Exposure``,
        NaN when exposure is not positive) and ``Severity``
        (``Loss / ClaimCount``, NaN when there are no claims). The groups
        are in the order they first occur in the data, as data.table's
        ``by =`` gives them; ``Year`` is a categorical of the year labels.

    Examples
    --------
    >>> d = pd.DataFrame({"REGIO": ["N", "N", "Z"], "Exposure": [1, 0.5, 1],
    ...                   "AantalClaims": [0, 1, 2],
    ...                   "SCHADELAST": [0, 500, 3000]})
    >>> agg_all(d, "REGIO", by_year=False)
    """
    d = U.as_df(d)
    U.check_cols(d, [col, exposure_col, claims_col, loss_col,
                     year_col if by_year else None], "agg_all")

    # [DATA-02]
    # Report data-quality issues explicitly instead of silently averaging away
    for cc in (exposure_col, claims_col, loss_col):
        n_na = int(d[cc].isna().sum())
        if n_na > 0:
            warn("agg_all: ", n_na, " NA value(s) in column '", cc,
                 "' are ignored (na.rm = TRUE).")

    # [DATA-02]
    # The output columns are fixed names, so grouping on one of them would
    # produce a data.frame with two columns of the same name rather than an
    # error, and every later rename or lookup would then pick the wrong one.
    created = ["Exposure", "ClaimCount", "Loss", "Frequency", "Severity"] + \
        (["Year"] if by_year else [])
    clash = [c for c in dict.fromkeys([col] + ([year_col] if by_year else []))
             if c in created]
    if clash:
        stop("agg_all: cannot group on '", ", ".join(clash), "', which is also "
             "the name of a column agg_all creates. Rename it first.")

    if by_year and col == year_col:
        stop("agg_all: 'col' and 'year_col' are both '", col, "'; pass "
             "by_year = FALSE to group on the year itself.")

    grp = [col, year_col] if by_year else [col]
    # data.table's by = keeps the groups in the order they first occur and
    # keeps NA as a group of its own
    codes = pd.DataFrame({"k%d" % i: pd.factorize(d[g], use_na_sentinel=False)[0]
                          for i, g in enumerate(grp)})
    gid = codes.groupby(list(codes.columns), sort=False).ngroup().to_numpy()
    first = np.unique(gid, return_index=True)[1]

    def gsum(c):
        return pd.Series(d[c].to_numpy()).groupby(gid, sort=True).sum().to_numpy()

    out = pd.DataFrame({col: d[col].iloc[first].reset_index(drop=True)})
    if by_year:
        out["Year"] = d[year_col].iloc[first].reset_index(drop=True)
    out["Exposure"] = gsum(exposure_col)
    out["ClaimCount"] = gsum(claims_col)
    out["Loss"] = gsum(loss_col)

    n_zero = int(np.sum(out["Exposure"].to_numpy(dtype=float) <= 0))
    if n_zero > 0:
        warn("agg_all: ", n_zero, " group(s) with Exposure <= 0; Frequency "
             "is NA there.")

    ex = out["Exposure"].to_numpy(dtype=float)
    cc = out["ClaimCount"].to_numpy(dtype=float)
    ls = out["Loss"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["Frequency"] = np.where(ex > 0, cc / ex, np.nan)
        out["Severity"] = np.where(cc > 0, ls / cc, np.nan)
    if by_year:
        out["Year"] = as_r_factor(out["Year"])
    return out
