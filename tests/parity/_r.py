"""Run R code against the R package pricingtoolsRmO and read the result back,
for the parity tests. Skipped when Rscript is not available."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pytest

RSCRIPT = os.environ.get(
    "RSCRIPT",
    shutil.which("Rscript")
    or r"C:\Users\rmooijer\AppData\Local\Programs\R\R-4.2.1\bin\Rscript.exe")


def have_r():
    return RSCRIPT is not None and os.path.exists(RSCRIPT)


requires_r = pytest.mark.skipif(not have_r(), reason="Rscript not available")


def run_r(code: str, frames: dict | None = None):
    """Write `frames` (name -> DataFrame) as CSV, run `code` in R with them
    loaded under the same names, and return the JSON the code assigns to
    `out` (a list) as a Python dict.

    Character columns are read back as character; columns listed in a
    frame's attrs["factors"] are turned into factors with the categories'
    order, so R and Python share the reference level."""
    with tempfile.TemporaryDirectory() as tmp:
        lines = ["suppressPackageStartupMessages({library(pricingtoolsRmO); "
                 "library(jsonlite)})"]
        for name, df in (frames or {}).items():
            path = os.path.join(tmp, name + ".csv").replace("\\", "/")
            df.to_csv(path, index=False)
            lines.append('%s <- read.csv("%s", stringsAsFactors = FALSE, '
                         'check.names = FALSE)' % (name, path))
            for col in df.columns:
                if isinstance(df[col].dtype, pd.CategoricalDtype):
                    lev = ", ".join('"%s"' % c for c in df[col].cat.categories)
                    lines.append('%s[["%s"]] <- factor(%s[["%s"]], levels = c(%s))'
                                 % (name, col, name, col, lev))
                elif pd.api.types.is_bool_dtype(df[col]):
                    lines.append('%s[["%s"]] <- as.logical(%s[["%s"]])'
                                 % (name, col, name, col))
        out_path = os.path.join(tmp, "out.json").replace("\\", "/")
        lines.append("out <- list()")
        lines.append(code)
        lines.append('write(toJSON(out, digits = NA, auto_unbox = TRUE, '
                     'null = "null", na = "null", dataframe = "columns"), "%s")'
                     % out_path)
        script = os.path.join(tmp, "run.R")
        with open(script, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        res = subprocess.run([RSCRIPT, script], capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError("R failed:\n" + res.stderr[-3000:])
        with open(out_path, encoding="utf-8") as fh:
            return json.load(fh)


def as_float(x):
    """R JSON values (null for NA) as a float array."""
    return np.array([np.nan if v is None else v for v in np.atleast_1d(x)],
                    dtype=float)
