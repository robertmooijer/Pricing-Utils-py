"""Excel export of a rating table (the port of R/export.R)."""
from __future__ import annotations

import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

from . import _utils as U
from ._core import stop
from .colors import ta_lightblue, ta_navy


# [EXP-02]
# The rating table keeps Level as text, because a categorical level is
# text. For a continuous variable that same column holds stringified
# numbers, and Excel then flags every cell as "number stored as text". Put
# the numeric LevelNum in its place there, and do the same for a Group
# column whose values are all numeric (a continuous interaction partner).
def _numeric_levels(d, tbl, rows):
    d = d.copy()
    if "Level" in d.columns and "LevelNum" in tbl.columns \
            and set(tbl.loc[rows, "Type"].unique()) == {"continuous"}:
        num = tbl.loc[rows, "LevelNum"].to_numpy(dtype=float)
        if not np.isnan(num).any():
            d["Level"] = num
    if "Group" in d.columns and len(d):
        g = pd.to_numeric(d["Group"], errors="coerce")
        if not g.isna().any():
            d["Group"] = g.to_numpy(dtype=float)
    return d


def _fmt6(a):
    """R's format(a, digits = 6, scientific = FALSE)."""
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return "-"
    return np.format_float_positional(float(a), precision=6, unique=False,
                                      fractional=False, trim="-")


def export_rating_table(rating_tbl, file="rating_table.xlsx", overwrite=True,
                        digits=4):
    """Export a rating table to a formatted Excel workbook.

    Writes the :func:`make_rating_table` output to a formatted ``.xlsx``
    file: an "Overview" sheet with the intercepts (per exposure unit), the
    base value per variable and a timestamp; a "Tariff" sheet with every
    factor under one lookup key; one sheet per main-effect variable (base
    row highlighted, thin rows greyed out); and one sheet per interaction
    with the long table plus a Level x Group matrix of the premium factor.
    Requires the 'openpyxl' package.

    Parameters
    ----------
    rating_tbl : pandas.DataFrame
        Output of :func:`make_rating_table`.
    file : str
        Output path of the ``.xlsx`` file.
    overwrite : bool
        Overwrite an existing file (default True).
    digits : int
        Number of decimals for the factor columns (default 4).

    Returns
    -------
    str
        The normalised path of the written file.
    """
    try:
        import openpyxl
        from openpyxl.styles import Font, PatternFill
        from openpyxl.utils import get_column_letter
    except ImportError:
        stop("export_rating_table: package 'openpyxl' is required "
             "(pip install openpyxl).")
    rating_tbl = U.as_df(rating_tbl)
    U.check_cols(rating_tbl, ["Variable", "Level", "Group", "IsBase"],
                 "export_rating_table")
    if os.path.exists(file) and not overwrite:
        stop("export_rating_table: file '", file, "' exists and overwrite = False.")
    attrs = rating_tbl.attrs
    tbl = rating_tbl.reset_index(drop=True)

    hdr_font = Font(color="FFFFFF", bold=True)
    hdr_fill = PatternFill("solid", fgColor=ta_navy.lstrip("#"))
    base_fill = PatternFill("solid", fgColor=ta_lightblue.lstrip("#"))
    thin_font = Font(color="999999", italic=True)
    fac_fmt = "0." + "0" * digits if digits > 0 else "0"
    int_fmt = "#,##0"

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    used = []

    # [EXP-02]
    # Excel treats sheet names case-insensitively, so REGIO and Regio would
    # otherwise both be accepted here and produce a corrupt workbook
    def sheet_name(nm):
        nm = re.sub(r"[^A-Za-z0-9 _.-]", "_", str(nm))[:31]
        base, i = nm, 1
        while nm.lower() in [u.lower() for u in used]:
            nm = base[:28] + "_" + str(i)
            i += 1
        used.append(nm)
        return nm

    def cell_value(v):
        if v is None:
            return None
        if isinstance(v, (float, np.floating)) and np.isnan(v):
            return None
        if isinstance(v, (np.bool_, bool)):
            return bool(v)
        if isinstance(v, np.integer):
            return int(v)
        if isinstance(v, np.floating):
            return float(v)
        if v is pd.NA:
            return None
        return v

    # Column widths from the actual content, with a floor, so a clipped
    # boolean does not render as #####
    def col_widths(d):
        out = []
        for c in d.columns:
            v = d[c]
            if pd.api.types.is_numeric_dtype(v) and not pd.api.types.is_bool_dtype(v):
                vals = v.to_numpy(dtype=float)
                vals = vals[~np.isnan(vals)]
                w = max([len(U.level_chr([round(x, digits)])[0]) for x in vals] or [1])
            else:
                w = max([len(str(x)) for x in v if x is not None and not pd.isna(x)] or [1])
            out.append(max(len(str(c)), w, 8) + 2)
        return out

    def write_df(ws, d, start_row=1, header=True):
        r = start_row
        if header:
            for j, c in enumerate(d.columns, start=1):
                cell = ws.cell(row=r, column=j, value=str(c))
                cell.font = hdr_font
                cell.fill = hdr_fill
            r += 1
        for row in d.itertuples(index=False):
            for j, v in enumerate(row, start=1):
                ws.cell(row=r, column=j, value=cell_value(v))
            r += 1

    def style_rows(ws, rows, ncol, fill=None, font=None):
        for r in rows:
            for j in range(1, ncol + 1):
                c = ws.cell(row=r, column=j)
                if fill is not None:
                    c.fill = fill
                if font is not None:
                    c.font = Font(color=font.color, italic=font.italic,
                                  bold=c.font.bold or font.bold)
                elif fill is base_fill:
                    c.font = Font(bold=True, color=c.font.color,
                                  italic=c.font.italic)

    def number_format(ws, rows, cols, fmt):
        for r in rows:
            for j in cols:
                ws.cell(row=r, column=j).number_format = fmt

    def set_widths(ws, widths):
        for j, w in enumerate(widths, start=1):
            ws.column_dimensions[get_column_letter(j)].width = w

    def write_sheet(nm, d):
        sn = sheet_name(nm)
        ws = wb.create_sheet(sn)
        write_df(ws, d)
        nc = d.shape[1]
        if "IsBase" in d.columns:
            base_rows = [i + 2 for i, b in enumerate(d["IsBase"]) if b is True or b == True]
            style_rows(ws, base_rows, nc, fill=base_fill)
        if "IsThin" in d.columns:
            thin_rows = [i + 2 for i, b in enumerate(d["IsThin"]) if b is True or b == True]
            style_rows(ws, thin_rows, nc, font=thin_font)
        fac_cols = [j + 1 for j, c in enumerate(d.columns)
                    if re.match(r"^(Factor|Uplift|SE_)", str(c))]
        number_format(ws, range(2, len(d) + 2), fac_cols, fac_fmt)
        int_cols = [j + 1 for j, c in enumerate(d.columns)
                    if c in ("Exposure", "ClaimCount")]
        number_format(ws, range(2, len(d) + 2), int_cols, int_fmt)
        set_widths(ws, col_widths(d))
        ws.freeze_panes = "A2"
        return ws

    # Overview sheet -------------------------------------------------------------
    bv = attrs.get("base_values")
    items = ["Generated", "Intercept frequency (per exposure unit)",
             "Intercept severity", "Intercept premium"]
    values = [datetime.now().strftime("%Y-%m-%d %H:%M"),
              _fmt6(attrs.get("intercept_frequency")),
              _fmt6(attrs.get("intercept_severity")),
              _fmt6(attrs.get("intercept_premium"))]
    if bv:
        for k, x in bv.items():
            items.append("Base value " + str(k))
            xs = x if isinstance(x, (list, tuple, np.ndarray)) else [x]
            values.append(", ".join(U.level_chr([v])[0] if not isinstance(v, str)
                                    else v for v in xs))
    ov = pd.DataFrame({"Item": items, "Value": values})
    ws = wb.create_sheet(sheet_name("Overview"))
    write_df(ws, ov)
    # [EXP-05]
    # The Value column mixes a timestamp, levels and numbers, so it is written
    # as text first and the numbers are then written over it as numbers. A
    # number stored as text is what a formula referring to the intercept
    # would pick up, and in a locale with a decimal comma "0.110053" is not a
    # number at all.
    ints = [attrs.get("intercept_frequency"), attrs.get("intercept_severity"),
            attrs.get("intercept_premium")]
    for k, a in enumerate(ints, start=1):
        if a is not None and np.isfinite(a):
            ws.cell(row=2 + k, column=2, value=float(a))
    if bv:
        for k, x in enumerate(bv.values(), start=1):
            if isinstance(x, (int, float, np.integer, np.floating)) \
                    and not isinstance(x, bool) and np.isfinite(x):
                ws.cell(row=5 + k, column=2, value=float(x))
    set_widths(ws, [max(len(str(i)) for i in items) + 2,
                    max(len(str(v)) for v in values) + 2])

    # [EXP-01] [EXP-03]
    # Tariff sheet: every factor of every variable under one another, with a
    # single lookup key, so a premium can be assembled with VLOOKUP instead
    # of hopping between sheets ------------------------------------------------
    tf_cols = [c for c in ("Factor_Frequency", "Factor_Severity", "Factor_Premium")
               if c in tbl.columns]
    is_int = tbl["Group"].notna().to_numpy()
    key = np.where(is_int,
                   tbl["Variable"].astype(str) + "|" + tbl["Level"].astype(str)
                   + "|" + tbl["Group"].astype(str),
                   tbl["Variable"].astype(str) + "|" + tbl["Level"].astype(str))
    tf = pd.DataFrame({"Key": key, "Variable": tbl["Variable"],
                       "Level": tbl["Level"], "Group": tbl["Group"],
                       "IsBase": tbl["IsBase"]})
    for c in tf_cols:
        tf[c] = tbl[c].to_numpy(dtype=float)
    # [EXP-04]
    # An interaction key carries the pure uplift, not the joint factor. The
    # joint factor already holds both main effects, so the instruction below -
    # every variable's factor, times each interaction key - counted them
    # twice: measured on a WA tariff, premiums 16% too high on average and up
    # to 3.4 times. With the uplift, main x main x uplift is the joint factor
    # and the instruction holds literally. The interaction sheets keep both.
    for c in tf_cols:
        up = c.replace("Factor_", "Uplift_")
        if is_int.any() and up in tbl.columns:
            tf.loc[is_int, c] = tbl.loc[is_int, up].to_numpy(dtype=float)
    ws = wb.create_sheet(sheet_name("Tariff"))
    ws.cell(row=1, column=1, value=(
        "Premium = intercept x the factor of every variable"
        + (" x, for each interaction, the factor of its key "
           "Variable|Level|Group. An interaction key holds the pure uplift on "
           "top of the two main effects" if is_int.any() else "")
        + ". Look up a factor with VLOOKUP on the Key column, e.g. "
        + '=VLOOKUP("' + str(tbl["Variable"].iloc[0]) + '|"&<cell>, Tariff!A:H, '
        + str(5 + len(tf_cols)) + ", FALSE). Intercepts are on the Overview "
        "sheet. An offset such as a bonus-malus discount is not a factor here "
        "and is applied on top."))
    write_df(ws, tf, start_row=2)
    tf_base = [i + 3 for i, b in enumerate(tf["IsBase"]) if b is True or b == True]
    style_rows(ws, tf_base, tf.shape[1], fill=base_fill)
    fc = [j + 1 for j, c in enumerate(tf.columns) if c in tf_cols]
    number_format(ws, range(3, len(tf) + 3), fc, fac_fmt)
    set_widths(ws, col_widths(tf))
    ws.freeze_panes = "A3"

    # Main-effect sheets ----------------------------------------------------------
    keep_main = [c for c in ("Level", "IsBase", "Exposure", "ClaimCount",
                             "IsThin", "Factor_Frequency", "Factor_Severity",
                             "Factor_Premium", "SE_Frequency", "SE_Severity")
                 if c in tbl.columns]
    main = tbl["Group"].isna().to_numpy()
    for v in pd.unique(tbl.loc[main, "Variable"]):
        rows = main & (tbl["Variable"] == v).to_numpy()
        d = tbl.loc[rows, keep_main].reset_index(drop=True)
        d = _numeric_levels(d, tbl, rows)
        write_sheet(v, d)

    # [EXP-01]
    # Interaction sheets: long table + premium-factor matrix ----------------------
    keep_int = [c for c in ("Level", "Group", "IsBase", "Exposure", "ClaimCount",
                            "IsThin", "Factor_Frequency", "Factor_Severity",
                            "Factor_Premium", "SE_Frequency", "SE_Severity",
                            "Uplift_Frequency", "Uplift_Severity",
                            "Uplift_Premium") if c in tbl.columns]
    for v in pd.unique(tbl.loc[~main, "Variable"]):
        rows = ~main & (tbl["Variable"] == v).to_numpy()
        d = tbl.loc[rows, keep_int].reset_index(drop=True)
        d = _numeric_levels(d, tbl, rows)
        ws = write_sheet(v, d)
        fac_col = next((c for c in ("Factor_Premium", "Factor_Frequency",
                                    "Factor_Severity")
                        if c in d.columns and d[c].notna().any()), None)
        if fac_col is not None:
            lv = list(pd.unique(d["Level"]))
            gv = list(pd.unique(d["Group"]))
            m = (d.assign(_L=pd.Categorical(d["Level"], categories=lv),
                          _G=pd.Categorical(d["Group"], categories=gv))
                 .pivot_table(index="_L", columns="_G", values=fac_col,
                              aggfunc="mean", observed=False))
            piv = pd.DataFrame({"Level": lv})
            for g in gv:
                piv[U.level_chr([g])[0] if not isinstance(g, str) else g] = \
                    m[g].reindex(lv).to_numpy()
            start = len(d) + 3
            ws.cell(row=start, column=1, value="Matrix " + fac_col +
                    " (rows = Level, columns = Group)")
            write_df(ws, piv, start_row=start + 1)
            number_format(ws, range(start + 2, start + 2 + len(piv)),
                          range(2, piv.shape[1] + 1), fac_fmt)

    wb.save(file)
    return os.path.abspath(file)
