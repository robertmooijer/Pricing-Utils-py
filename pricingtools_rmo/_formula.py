"""R-style model formulas and expressions, evaluated in Python.

A formula is written as in R:

    "AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR)
                    + offset(log(Exposure)) + offset(log(BM))"

The ``offset()`` terms are taken out and kept as expressions, so they can be
evaluated again on any data, which is what R's ``predict()`` does and what
the rest of the package relies on. The remaining right-hand side goes to
patsy, with the R functions a pricing formula uses (``log``, ``factor``,
``cut``, ``ns``, ``bs``, ``poly``, ``pmin``, ...) available under their R
names. Backticked names (`` `AUTO GEWICHT` ``) and ``^`` are translated.
"""
from __future__ import annotations

import ast
import math
import re

import numpy as np
import pandas as pd
import patsy
from patsy.builtins import C, I, Q

from .splines import bs, ns, poly


class FormulaError(ValueError):
    pass


# ---- R functions available in formulas and expressions ----------------------

def _pmin(*args):
    out = np.asarray(args[0], dtype=float)
    for a in args[1:]:
        out = np.minimum(out, np.asarray(a, dtype=float))
    return out


def _pmax(*args):
    out = np.asarray(args[0], dtype=float)
    for a in args[1:]:
        out = np.maximum(out, np.asarray(a, dtype=float))
    return out


def _ifelse(cond, yes, no):
    return np.where(np.asarray(cond), yes, no)


def _format_breaks(breaks, digits):
    out = []
    for b in breaks:
        if np.isinf(b):
            out.append("Inf" if b > 0 else "-Inf")
        else:
            s = "%.*g" % (digits, b)
            out.append(s)
    return out


def r_cut(x, breaks, labels=None, right=True, include_lowest=False,
          dig_lab=3):
    """R's cut(): right-closed bins labelled "(17,24]", as a pandas
    Categorical whose categories keep the order of the breaks."""
    x = np.asarray(x, dtype=float)
    breaks = np.atleast_1d(np.asarray(breaks, dtype=float))
    if breaks.size == 1:
        nb = int(breaks[0])
        lo, hi = np.nanmin(x), np.nanmax(x)
        dx = hi - lo if hi > lo else abs(lo)
        breaks = np.linspace(lo, hi, nb + 1)
        breaks[0] -= dx / 1000
        breaks[-1] += dx / 1000
    breaks = np.sort(breaks)
    nb = len(breaks)
    if labels is None:
        for dig in range(dig_lab, max(12, dig_lab) + 1):
            ch = _format_breaks(breaks, dig)
            if all(ch[i] != ch[i + 1] for i in range(nb - 1)):
                break
        if right:
            labels = ["(" + ch[i] + "," + ch[i + 1] + "]" for i in range(nb - 1)]
            if include_lowest:
                labels[0] = "[" + labels[0][1:]
        else:
            labels = ["[" + ch[i] + "," + ch[i + 1] + ")" for i in range(nb - 1)]
            if include_lowest:
                labels[-1] = labels[-1][:-1] + "]"
    codes = np.full(len(x), -1)
    for i in range(nb - 1):
        a, b = breaks[i], breaks[i + 1]
        if right:
            sel = (x > a) & (x <= b)
            if include_lowest and i == 0:
                sel |= x == a
        else:
            sel = (x >= a) & (x < b)
            if include_lowest and i == nb - 2:
                sel |= x == b
        codes[sel] = i
    return pd.Categorical.from_codes(codes, categories=list(labels))


R_FUNCTIONS = {
    "log": np.log, "log2": np.log2, "log10": np.log10, "log1p": np.log1p,
    "exp": np.exp, "sqrt": np.sqrt, "abs": np.abs,
    "floor": np.floor, "ceiling": np.ceil, "round": np.round,
    "pmin": _pmin, "pmax": _pmax, "ifelse": _ifelse,
    "cut": r_cut, "ns": ns, "bs": bs, "poly": poly,
    "I": I, "Q": Q, "C": C, "factor": C,
    "as_numeric": lambda x: np.asarray(x, dtype=float),
    "np": np, "pi": math.pi, "Inf": np.inf,
}


# ---- text-level translation --------------------------------------------------

_BACKTICK = re.compile(r"`([^`]*)`")


def translate(expr: str, colfun: str = "Q") -> str:
    """R expression text to Python: backticks, ^, TRUE/FALSE, named args."""
    s = _BACKTICK.sub(lambda m: '%s("%s")' % (colfun, m.group(1)), expr)
    s = s.replace("^", "**")
    s = re.sub(r"\bTRUE\b", "True", s)
    s = re.sub(r"\bFALSE\b", "False", s)
    # R argument names with dots, e.g. include.lowest = TRUE
    s = re.sub(r"\b(include|dig|Boundary)\.(lowest|lab|knots)\b",
               lambda m: {"include.lowest": "include_lowest",
                          "dig.lab": "dig_lab",
                          "Boundary.knots": "boundary_knots"}[m.group(0)], s)
    s = re.sub(r"(?<![\w.])c\(", "_c_(", s)
    return s


def _c(*args):
    out = []
    for a in args:
        out.extend(np.atleast_1d(a).tolist())
    return out


R_FUNCTIONS["_c_"] = _c


def split_top(text: str, seps: str):
    """Split at top-level separators (outside brackets and quotes), keeping
    the separator in front of each part."""
    parts, depth, cur, sign, quote = [], 0, [], "+", None
    for ch in text:
        if quote:
            cur.append(ch)
            if ch == quote:
                quote = None
            continue
        if ch in "\"'`":
            quote = ch
            cur.append(ch)
        elif ch in "([{":
            depth += 1
            cur.append(ch)
        elif ch in ")]}":
            depth -= 1
            cur.append(ch)
        elif ch in seps and depth == 0:
            parts.append((sign, "".join(cur).strip()))
            sign, cur = ch, []
        else:
            cur.append(ch)
    parts.append((sign, "".join(cur).strip()))
    return [(s, t) for s, t in parts if t]


class ParsedFormula:
    """An R formula split into response, patsy right-hand side and offsets."""

    def __init__(self, formula: str):
        self.formula = " ".join(formula.split())
        if "~" not in formula:
            raise FormulaError("The formula needs a '~': " + formula)
        lhs, rhs = formula.split("~", 1)
        self.response = lhs.strip()
        if not self.response:
            raise FormulaError("The formula has no response: " + formula)
        terms, offsets = [], []
        for sign, t in split_top(rhs, "+-"):
            if re.fullmatch(r"offset\s*\((.*)\)", t, flags=re.S):
                if sign == "-":
                    raise FormulaError("An offset cannot be subtracted.")
                inner = re.fullmatch(r"offset\s*\((.*)\)", t, flags=re.S).group(1)
                offsets.append(inner.strip())
            else:
                terms.append((sign, t))
        if any(t == "." for _, t in terms):
            raise FormulaError("'.' (all other columns) is not supported; "
                               "name the columns.")
        rhs_txt = ""
        for i, (sign, t) in enumerate(terms):
            rhs_txt += (t if i == 0 and sign == "+" else " %s %s" % (sign, t))
        self.rhs = translate(rhs_txt.strip() or "1")
        self.offsets = offsets

    def __repr__(self):
        return "ParsedFormula(%r)" % self.formula


# ---- evaluating expressions on data -------------------------------------------

def expr_vars(expr: str) -> list:
    """The data columns an R expression refers to, like R's all.vars()."""
    s = translate(expr, colfun="_col_")
    try:
        tree = ast.parse(s, mode="eval")
    except SyntaxError as e:
        raise FormulaError("Cannot parse expression '%s': %s" % (expr, e))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in ("_col_", "Q") and node.args \
                and isinstance(node.args[0], ast.Constant):
            out.append(node.args[0].value)
        elif isinstance(node, ast.Name) and node.id not in R_FUNCTIONS \
                and node.id not in ("_col_", "True", "False", "None"):
            # a function name used as a call is not a variable
            out.append(node.id)
    funcs = {n.func.id for n in ast.walk(tree)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    seen, res = set(), []
    for v in out:
        if v in funcs and v not in ("_col_",):
            continue
        if v not in seen:
            seen.add(v)
            res.append(v)
    return res


def eval_expr(expr: str, data: pd.DataFrame):
    """Evaluate an R expression on the columns of a data frame, as numpy."""
    s = translate(expr, colfun="_col_")
    ns_ = dict(R_FUNCTIONS)

    def _col_(name):
        if name not in data.columns:
            raise FormulaError("Column '%s' not found." % name)
        return _as_array(data[name])

    ns_["_col_"] = _col_
    ns_["Q"] = _col_
    for v in expr_vars(expr):
        if v in data.columns and v.isidentifier():
            ns_[v] = _as_array(data[v])
    try:
        val = eval(compile(s, "<expr>", "eval"), {"__builtins__": {}}, ns_)
    except NameError as e:
        raise FormulaError("Cannot evaluate '%s': %s" % (expr, e))
    if isinstance(val, pd.Series):
        val = val.to_numpy()
    return val


def _as_array(col):
    if isinstance(col.dtype, pd.CategoricalDtype):
        return col
    try:
        return col.to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError):
        return col.to_numpy()


def patsy_env():
    return patsy.EvalEnvironment([dict(R_FUNCTIONS)])


def factor_vars(code: str) -> list:
    """Data columns behind one patsy factor code, e.g. 'ns(LEEFTIJD, 3)'."""
    return expr_vars(code)


def r_code(code: str) -> str:
    """A patsy factor code written back the way R deparses the term:
    c() for _c_(), backticks for Q("..."), ^ for **, TRUE/FALSE."""
    s = re.sub(r"""\bQ\((['"])(.*?)\1\)""",
               lambda m: m.group(2) if m.group(2).isidentifier()
               else "`" + m.group(2) + "`", code)
    s = s.replace("_c_(", "c(").replace(" ** ", "^").replace("**", "^")
    s = re.sub(r"\bTrue\b", "TRUE", s)
    s = re.sub(r"\bFalse\b", "FALSE", s)
    return s
