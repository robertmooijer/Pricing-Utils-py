"""R-compatible spline and polynomial bases for model formulas.

``ns()``, ``bs()`` and ``poly()`` reproduce R's ``splines::ns``,
``splines::bs`` and ``stats::poly``: the same knots (type-7 quantiles of the
training data), the same boundary handling and extrapolation, and the same
basis. A model fitted with ``ns(LEEFTIJD, 3)`` therefore spans exactly the
space R's does, so fitted values and every rating factor agree with R to
machine precision.

They are patsy stateful transforms: the knots are learned once, on the data
the model is fitted on, and reused when the model scores other data, which
is what R's ``makepredictcall`` does.
"""
from __future__ import annotations

import numpy as np
from patsy import stateful_transform
from scipy.interpolate import BSpline


def _spline_design(knots, x, order, derivs=None):
    """B-spline basis of a given order at x, like R's splines::splineDesign.

    Columns are the ``len(knots) - order`` basis functions; ``derivs`` gives
    the derivative to take per value of x (0 = the basis itself).
    """
    knots = np.asarray(knots, dtype=float)
    x = np.asarray(x, dtype=float)
    k = order - 1
    nb = len(knots) - order
    derivs = np.zeros(len(x), dtype=int) if derivs is None else np.asarray(derivs)
    out = np.zeros((len(x), nb))
    for j in range(nb):
        c = np.zeros(nb)
        c[j] = 1.0
        spl = BSpline(knots, c, k, extrapolate=True)
        for d in np.unique(derivs):
            sel = derivs == d
            f = spl if d == 0 else spl.derivative(int(d))
            out[sel, j] = f(x[sel])
    return out


def _null_space_projection(const):
    """Q[:, 2:] of the Householder QR of t(const), as R's qr.qty() uses it."""
    q, _ = np.linalg.qr(const.T, mode="complete")
    return q[:, const.shape[0]:]


class _NS:
    """Natural cubic spline basis, R's splines::ns()."""

    def __init__(self):
        self._x = []
        self.knots = None
        self.boundary = None
        self.intercept = False

    def memorize_chunk(self, x, df=None, knots=None, intercept=False,
                       boundary_knots=None):
        self._x.append(np.asarray(x, dtype=float).ravel())

    def memorize_finish(self):
        pass

    def _setup(self, x, df, knots, intercept, boundary_knots):
        x = x[~np.isnan(x)]
        self.intercept = bool(intercept)
        if boundary_knots is None:
            boundary_knots = (np.min(x), np.max(x))
        self.boundary = np.asarray(boundary_knots, dtype=float)
        if knots is None:
            if df is None:
                knots = np.array([])
            else:
                n_ik = int(df) - 1 - int(self.intercept)
                n_ik = max(n_ik, 0)
                inside = x[(x >= self.boundary[0]) & (x <= self.boundary[1])]
                probs = np.linspace(0, 1, n_ik + 2)[1:-1]
                knots = (np.quantile(inside, probs) if n_ik > 0
                         else np.array([]))
        self.knots = np.asarray(knots, dtype=float)

    def transform(self, x, df=None, knots=None, intercept=False,
                  boundary_knots=None):
        if self.knots is None:
            self._setup(np.concatenate(self._x), df, knots, intercept,
                        boundary_knots)
        x = np.asarray(x, dtype=float).ravel()
        return ns_basis(x, self.knots, self.boundary, self.intercept)


def ns_basis(x, knots, boundary, intercept=False):
    """The ns() basis for fixed knots, with R's linear extrapolation."""
    x = np.asarray(x, dtype=float)
    a_knots = np.sort(np.concatenate([np.repeat(boundary, 4), knots]))
    nb = len(a_knots) - 4
    basis = np.full((len(x), nb), np.nan)
    ok = ~np.isnan(x)
    lo = ok & (x < boundary[0])
    hi = ok & (x > boundary[1])
    inside = ok & ~lo & ~hi
    if inside.any():
        basis[inside] = _spline_design(a_knots, x[inside], 4)
    for sel, piv in ((lo, boundary[0]), (hi, boundary[1])):
        if sel.any():
            tt = _spline_design(a_knots, [piv, piv], 4, derivs=[0, 1])
            xl = np.column_stack([np.ones(sel.sum()), x[sel] - piv])
            basis[sel] = xl @ tt
    const = _spline_design(a_knots, boundary, 4, derivs=[2, 2])
    if not intercept:
        const = const[:, 1:]
        basis = basis[:, 1:]
    q = _null_space_projection(const)
    return basis @ q


class _BS:
    """B-spline basis, R's splines::bs()."""

    def __init__(self):
        self._x = []
        self.knots = None
        self.boundary = None
        self.degree = 3
        self.intercept = False

    def memorize_chunk(self, x, df=None, knots=None, degree=3,
                       intercept=False, boundary_knots=None):
        self._x.append(np.asarray(x, dtype=float).ravel())

    def memorize_finish(self):
        pass

    def _setup(self, x, df, knots, degree, intercept, boundary_knots):
        x = x[~np.isnan(x)]
        self.degree = int(degree)
        self.intercept = bool(intercept)
        order = self.degree + 1
        if boundary_knots is None:
            boundary_knots = (np.min(x), np.max(x))
        self.boundary = np.asarray(boundary_knots, dtype=float)
        if knots is None:
            if df is None:
                knots = np.array([])
            else:
                n_ik = int(df) - order + (1 - int(self.intercept))
                n_ik = max(n_ik, 0)
                inside = x[(x >= self.boundary[0]) & (x <= self.boundary[1])]
                probs = np.linspace(0, 1, n_ik + 2)[1:-1]
                knots = (np.quantile(inside, probs) if n_ik > 0
                         else np.array([]))
        self.knots = np.asarray(knots, dtype=float)

    def transform(self, x, df=None, knots=None, degree=3, intercept=False,
                  boundary_knots=None):
        if self.knots is None:
            self._setup(np.concatenate(self._x), df, knots, degree, intercept,
                        boundary_knots)
        x = np.asarray(x, dtype=float).ravel()
        return bs_basis(x, self.knots, self.boundary, self.degree,
                        self.intercept)


def bs_basis(x, knots, boundary, degree=3, intercept=False):
    """The bs() basis for fixed knots, with R's Taylor extrapolation."""
    x = np.asarray(x, dtype=float)
    order = degree + 1
    a_knots = np.sort(np.concatenate([np.repeat(boundary, order), knots]))
    nb = len(a_knots) - order
    basis = np.full((len(x), nb), np.nan)
    ok = ~np.isnan(x)
    lo = ok & (x < boundary[0])
    hi = ok & (x > boundary[1])
    inside = ok & ~lo & ~hi
    if inside.any():
        basis[inside] = _spline_design(a_knots, x[inside], order)
    fact = np.array([np.prod(np.arange(1, d + 1)) for d in range(order)],
                    dtype=float)
    for sel, piv in ((lo, boundary[0]), (hi, boundary[1])):
        if sel.any():
            tt = _spline_design(a_knots, np.repeat(piv, order), order,
                                derivs=np.arange(order))
            xl = np.column_stack([(x[sel] - piv) ** d / fact[d]
                                  for d in range(order)])
            basis[sel] = xl @ tt
    if not intercept:
        basis = basis[:, 1:]
    return basis


class _Poly:
    """Orthogonal polynomials, R's stats::poly()."""

    def __init__(self):
        self._x = []
        self.alpha = None
        self.norm2 = None

    def memorize_chunk(self, x, degree=1):
        self._x.append(np.asarray(x, dtype=float).ravel())

    def memorize_finish(self):
        pass

    def transform(self, x, degree=1):
        degree = int(degree)
        if self.alpha is None:
            xt = np.concatenate(self._x)
            xt = xt[~np.isnan(xt)]
            xbar = xt.mean()
            xc = xt - xbar
            X = np.vander(xc, degree + 1, increasing=True)
            q, r = np.linalg.qr(X)
            z = q * np.diag(r)
            norm2 = (z ** 2).sum(axis=0)
            alpha = ((xc[:, None] * z ** 2).sum(axis=0) / norm2 + xbar)[:degree]
            self.alpha = alpha
            self.norm2 = np.concatenate([[1.0], norm2])
        x = np.asarray(x, dtype=float).ravel()
        n = len(x)
        Z = np.ones((n, degree + 1))
        Z[:, 1] = x - self.alpha[0]
        for i in range(1, degree):
            Z[:, i + 1] = ((x - self.alpha[i]) * Z[:, i]
                           - (self.norm2[i + 1] / self.norm2[i]) * Z[:, i - 1])
        Z = Z / np.sqrt(self.norm2[1:])
        return Z[:, 1:]


# [PY-03] knots learned on the fit data and reused on new data, as in R
ns = stateful_transform(_NS)
bs = stateful_transform(_BS)
poly = stateful_transform(_Poly)
