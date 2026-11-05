"""fit_glm(): a GLM that remembers what an R glm remembers.

The R package reads everything it needs off a fitted ``glm``: the formula,
the data it was fitted on, the offset expressions, the prior weights, the
factor levels. A statsmodels model keeps the numbers but not the
expressions - an offset is just an array - so this module fits the GLM with
statsmodels and keeps the rest alongside. The result, a :class:`GLMFit`,
predicts on new data the way R's ``predict()`` does: it rebuilds the design
with the training levels and spline knots, and re-evaluates every offset
expression on the new rows.

Estimation is statsmodels' IRLS, so coefficients, deviances and standard
errors are those of R's ``glm()`` to within the convergence tolerance.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import patsy
from scipy import special, stats
import statsmodels.api as sm
from statsmodels.genmod import families as F

from ._core import PricingError, warn
from ._formula import (ParsedFormula, eval_expr, expr_vars, factor_vars,
                       patsy_env, r_code)

_LINKS = {
    "log": F.links.Log, "identity": F.links.Identity,
    "inverse": F.links.InversePower, "logit": F.links.Logit,
    "sqrt": F.links.Sqrt, "probit": F.links.Probit,
}
_DEFAULT_LINK = {
    "poisson": "log", "quasipoisson": "log", "gamma": "log",
    "gaussian": "identity", "binomial": "logit", "tweedie": "log",
    "negbin": "log", "inverse.gaussian": "log",
}


def _sm_family(family, link, var_power, theta):
    lk = _LINKS[link]()
    if family in ("poisson", "quasipoisson"):
        return F.Poisson(lk)
    if family == "gamma":
        return F.Gamma(lk)
    if family == "gaussian":
        return F.Gaussian(lk)
    if family == "binomial":
        return F.Binomial(lk)
    if family == "tweedie":
        return F.Tweedie(link=lk, var_power=var_power)
    if family == "negbin":
        return F.NegativeBinomial(lk, alpha=1.0 / theta)
    if family == "inverse.gaussian":
        return F.InverseGaussian(lk)
    raise PricingError("Unknown family '%s'." % family)


class Term:
    """One model term: its label, order, factor codes and data columns."""

    def __init__(self, label, factors, cols, columns):
        self.label = label
        self.factors = factors          # patsy factor codes, e.g. "ns(LEEFTIJD, 3)"
        self.order = len(factors)
        self.cols = cols                # slice into the full design matrix
        self.columns = columns          # data columns behind the term

    def __repr__(self):
        return "Term(%s)" % self.label


class GLMFit:
    """A fitted GLM with its formula, data, offsets and levels.

    Attributes mirror R's glm object where the package uses them:
    ``family``, ``link``, ``data`` (the rows the fit used, original columns),
    ``y``, ``prior_weights``, ``offset``, ``fitted`` (mu), ``coef``,
    ``vcov``, ``deviance``, ``null_deviance``, ``df_residual``, ``aic``,
    ``xlevels``, ``terms``, ``theta`` (negative binomial).
    """

    # ---- link and variance ---------------------------------------------------

    def linkinv(self, eta):
        return self.sm_family.link.inverse(np.asarray(eta, dtype=float))

    def linkfun(self, mu):
        return self.sm_family.link(np.asarray(mu, dtype=float))

    def mu_eta(self, eta):
        return self.sm_family.link.inverse_deriv(np.asarray(eta, dtype=float))

    def variance(self, mu):
        return self.sm_family.variance(np.asarray(mu, dtype=float))

    # ---- what the model is built from ------------------------------------------

    @property
    def term_labels(self):
        return [t.label for t in self.terms]

    def rhs_vars(self):
        """Every data column on the right-hand side, offsets included, as
        R's all.vars(formula) minus the response."""
        resp = set(expr_vars(self.parsed.response))
        out = []
        for v in self._rhs_columns + self.offset_vars():
            if v not in resp and v not in out:
                out.append(v)
        return out

    def response_vars(self):
        return expr_vars(self.parsed.response)

    @property
    def offset_exprs(self):
        return list(self.parsed.offsets)

    def offset_vars(self):
        out = []
        for e in self.parsed.offsets:
            for v in expr_vars(e):
                if v not in out:
                    out.append(v)
        return out

    def weights_vars(self):
        if isinstance(self.weights_expr, str):
            return expr_vars(self.weights_expr)
        return []

    def offset_value(self, data, idx=None):
        """The offset's contribution to the linear predictor on `data`;
        `idx` limits it to some of the offset expressions."""
        ex = self.parsed.offsets
        if idx is not None:
            ex = [ex[i] for i in idx]
        n = len(data)
        if not ex:
            return np.zeros(n)
        tot = np.zeros(n)
        for e in ex:
            v = np.asarray(eval_expr(e, data), dtype=float)
            if v.ndim == 0:
                v = np.repeat(float(v), n)
            # [OFF-03] An offset that does not give one value per row of
            # this data cannot be neutralised; recycling it would give a
            # confidently wrong number, so it is refused.
            if len(v) != n:
                raise PricingError(
                    "pricingtools: the model's offset (%s) does not evaluate "
                    "to one value per row of this data." % e)
            tot = tot + v
        return tot

    # ---- design on new data -------------------------------------------------------

    def _align(self, nd):
        """Bring the columns of new data to the types they had at fit time,
        so '2024' matches a numeric year and a Categorical is read by value."""
        nd = nd.copy()
        for v in self._rhs_columns:
            if v not in nd.columns or v not in self.data.columns:
                continue
            col, tmpl = nd[v], self.data[v]
            if isinstance(col.dtype, pd.CategoricalDtype):
                col = pd.Series(np.asarray(col, dtype=object), index=nd.index)
            # a nullable boolean (what coerce_like() returns for a logical
            # column) is a dtype patsy cannot read
            if isinstance(col.dtype, pd.BooleanDtype):
                col = pd.Series(np.asarray(col, dtype=object), index=nd.index)
            if pd.api.types.is_bool_dtype(tmpl) and not pd.api.types.is_bool_dtype(col):
                m = {"TRUE": True, "True": True, "true": True, "1": True,
                     "FALSE": False, "False": False, "false": False, "0": False}
                col = col.map(lambda z: m.get(str(z), z))
            elif pd.api.types.is_numeric_dtype(tmpl) and not pd.api.types.is_bool_dtype(tmpl) \
                    and not pd.api.types.is_numeric_dtype(col):
                col = pd.to_numeric(col, errors="coerce")
            elif not pd.api.types.is_numeric_dtype(tmpl) \
                    and not pd.api.types.is_bool_dtype(tmpl) \
                    and pd.api.types.is_numeric_dtype(col):
                col = col.astype(object).map(
                    lambda z: z if pd.isna(z) else _num_str(z))
            nd[v] = col
        return nd

    def model_matrix(self, newdata=None):
        """Full design matrix (aliased columns included) for `newdata`, or the
        training matrix. Rows with a missing predictor are NaN."""
        if newdata is None:
            return self._X_full
        nd = self._align(_as_df(newdata))
        miss = [v for v in self._rhs_columns if v not in nd.columns]
        if miss:
            raise PricingError("Column(s) needed for prediction not found: "
                               + ", ".join(miss))
        na = np.zeros(len(nd), dtype=bool)
        for v in self._rhs_columns:
            na |= pd.isna(nd[v]).to_numpy()
        X = np.full((len(nd), len(self.design_info.column_names)), np.nan)
        if (~na).any():
            sub = nd.loc[~na].reset_index(drop=True)
            try:
                # a term that turns a value into NA (cut() outside its
                # breaks) gives an NA prediction for that row, as in R
                Xi = patsy.build_design_matrices(
                    [self.design_info], sub, NA_action="drop",
                    return_type="dataframe")[0]
            except patsy.PatsyError as e:
                raise PricingError(_patsy_message(e)) from None
            pos = np.flatnonzero(~na)[Xi.index.to_numpy()]
            X[pos] = Xi.to_numpy()
        return X

    def predict(self, newdata=None, type="response", drop_offsets=None):
        """Prediction as R's predict(): every offset included, unless the
        indices of offset expressions to subtract are given."""
        if newdata is None:
            eta = self.eta.copy()
            if drop_offsets is not None and len(drop_offsets):
                eta = eta - self.offset_value(self.data, drop_offsets)
        else:
            nd = _as_df(newdata)
            X = self.model_matrix(nd)[:, self.keep]
            eta = X @ self.beta
            keep = [i for i in range(len(self.parsed.offsets))
                    if drop_offsets is None or i not in set(drop_offsets)]
            if keep:
                eta = eta + self.offset_value(nd, keep)
        return self.linkinv(eta) if type == "response" else eta

    # ---- residuals and influence -------------------------------------------------

    def residuals(self, type="deviance"):
        y, mu, w = self.y, self.fitted, self.prior_weights
        if type == "response":
            return y - mu
        if type == "pearson":
            return (y - mu) * np.sqrt(w) / np.sqrt(self.variance(mu))
        if type == "deviance":
            return self.sm_family.resid_dev(y, mu, var_weights=w)
        if type == "working":
            return (y - mu) / self.mu_eta(self.eta)
        raise PricingError("Unknown residual type '%s'." % type)

    def _working_weights(self):
        return self.prior_weights * self.mu_eta(self.eta) ** 2 / self.variance(self.fitted)

    def hatvalues(self):
        Xw = self._X_full[:, self.keep] * np.sqrt(self._working_weights())[:, None]
        q, _ = np.linalg.qr(Xw)
        return (q ** 2).sum(axis=1)

    def cooks_distance(self):
        h = self.hatvalues()
        res = self.residuals("pearson")
        with np.errstate(divide="ignore", invalid="ignore"):
            d = (res / (1 - h)) ** 2 * h / (self.dispersion_summary * self.rank)
        d[np.isinf(d)] = np.nan
        return d

    def rstandard(self, type="deviance"):
        h = self.hatvalues()
        res = self.residuals(type)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = res / np.sqrt(self.dispersion_summary * (1 - h))
        r[np.isinf(r)] = np.nan
        return r

    # ---- refit ---------------------------------------------------------------------

    def refit(self, data):
        """The same model fitted on other data, as R's update(model, data =)."""
        return fit_glm(self.formula, data, family=self.family, link=self.link,
                       weights=self.weights_expr, var_power=self.var_power,
                       theta=self.theta if self.theta_fixed else None)

    def summary_table(self):
        se = np.sqrt(np.diag(self.vcov.to_numpy()))
        b = self.coef.dropna()
        return pd.DataFrame({"Estimate": b, "Std. Error": se,
                             "z value": b / se}, index=b.index)

    def __repr__(self):
        return ("GLMFit(%s, family=%s, link=%s, n=%d)"
                % (self.formula, self.family, self.link, self.nobs))


def _r_num(x):
    """A number as R's format() writes it, up to 7 significant digits."""
    return np.format_float_positional(float(x), precision=7, unique=False,
                                      fractional=False, trim="-")


def _num_str(z):
    f = float(z)
    return str(int(f)) if f.is_integer() else repr(f)


def _patsy_message(e):
    msg = str(e).split("\n")[0]
    return "The data cannot be run through the model: " + msg


def _as_df(x):
    if isinstance(x, pd.DataFrame):
        return x
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def _aliased(X, tol=1e-7):
    """Columns that are (nearly) a linear combination of the columns before
    them, as R's pivoting QR (dqrdc2) leaves them out."""
    keep = []
    Q = np.zeros((X.shape[0], 0))
    for j in range(X.shape[1]):
        x = X[:, j]
        nx = np.linalg.norm(x)
        if nx == 0:
            continue
        r = x - Q @ (Q.T @ x)
        r = r - Q @ (Q.T @ r)
        nr = np.linalg.norm(r)
        if nr > tol * nx:
            keep.append(j)
            Q = np.column_stack([Q, r / nr])
    mask = np.zeros(X.shape[1], dtype=bool)
    mask[keep] = True
    return mask


def _theta_ml(y, mu, w, limit=25, eps=np.finfo(float).eps ** 0.25):
    """MASS::theta.ml: Newton steps on the score for theta."""
    n = w.sum()

    def score(th):
        return np.sum(w * (special.digamma(th + y) - special.digamma(th)
                           + np.log(th) + 1 - np.log(th + mu)
                           - (y + th) / (mu + th)))

    def info(th):
        return np.sum(w * (-special.polygamma(1, th + y) + special.polygamma(1, th)
                           - 1 / th + 2 / (mu + th) - (y + th) / (mu + th) ** 2))

    t0 = n / np.sum(w * (y / mu - 1) ** 2)
    it, delta = 0, 1.0
    while True:
        it += 1
        if it >= limit or abs(delta) <= eps:
            break
        t0 = abs(t0)
        delta = score(t0) / info(t0)
        t0 = t0 + delta
    if t0 < 0:
        t0 = 0.0
        warn("fit_glm: theta estimate truncated at zero")
    if it == limit:
        warn("fit_glm: iteration limit reached in theta.ml")
    return float(t0)


def _nb_loglik(th, mu, y, w):
    return np.sum(w * (special.gammaln(th + y) - special.gammaln(th)
                       - special.gammaln(y + 1) + th * np.log(th)
                       + y * np.log(mu + (y == 0)) - (th + y) * np.log(th + mu)))


def fit_glm(formula, data, family="poisson", link=None, weights=None,
            var_power=1.5, theta=None, maxiter=100, tol=1e-10):
    """Fit a GLM from an R-style formula.

    Parameters
    ----------
    formula : str
        R formula, e.g. ``"AantalClaims ~ ns(LEEFTIJD, 3) + REGIO +
        factor(BOEKJAAR) + offset(log(Exposure)) + offset(log(BM))"``.
        ``offset()`` terms may appear any number of times; ``factor()``,
        ``cut()``, ``ns()``, ``bs()``, ``poly()``, ``log()``, ``pmin()``,
        ``I()`` and backticked names work as in R.
    data : pandas.DataFrame
    family : str
        ``"poisson"``, ``"quasipoisson"``, ``"gamma"``, ``"gaussian"``,
        ``"binomial"``, ``"tweedie"`` (with ``var_power``) or ``"negbin"``
        (theta estimated as MASS::glm.nb does, or fixed with ``theta``).
    link : str, optional
        Defaults to ``"log"`` for every family except gaussian (identity) and
        binomial (logit). Note that R's Gamma default is the inverse link;
        pricing models use the log link, so that is the default here.
    weights : str, optional
        Prior weights: a column name or an expression, e.g.
        ``"AantalClaims"`` for a severity model on the average claim.

    Rows with a missing value in any variable the model uses are left out,
    as R's na.omit does.
    """
    data = _as_df(data)
    family = family.lower() if family != "Gamma" else "gamma"
    if family not in _DEFAULT_LINK:
        raise PricingError("fit_glm: unknown family '%s'." % family)
    link = link or _DEFAULT_LINK[family]
    if link not in _LINKS:
        raise PricingError("fit_glm: unknown link '%s'." % link)

    pf = ParsedFormula(formula)
    desc = patsy.ModelDesc.from_formula(pf.rhs)
    rhs_cols = []
    for t in desc.rhs_termlist:
        for f in t.factors:
            for v in factor_vars(f.code):
                if v not in rhs_cols:
                    rhs_cols.append(v)
    resp_cols = expr_vars(pf.response)
    off_cols = [v for e in pf.offsets for v in expr_vars(e)]
    w_cols = expr_vars(weights) if isinstance(weights, str) else []
    used = list(dict.fromkeys(resp_cols + rhs_cols + off_cols + w_cols))
    miss = [v for v in used if v not in data.columns]
    if miss:
        raise PricingError("fit_glm: column(s) not found in the data: "
                           + ", ".join(miss))

    na = np.zeros(len(data), dtype=bool)
    for v in used:
        na |= pd.isna(data[v]).to_numpy()
    d = data.loc[~na].copy()
    # R's model.frame drops unused factor levels
    for v in rhs_cols:
        if isinstance(d[v].dtype, pd.CategoricalDtype):
            d[v] = d[v].cat.remove_unused_categories()

    X = patsy.dmatrix(pf.rhs, d, eval_env=patsy_env(), NA_action="raise")
    di = X.design_info
    X = np.asarray(X, dtype=float)
    y = np.asarray(eval_expr(pf.response, d), dtype=float)
    if weights is None:
        w = np.ones(len(d))
    elif isinstance(weights, str):
        w = np.asarray(eval_expr(weights, d), dtype=float)
    else:
        w = np.asarray(weights, dtype=float)[~na]
    off = None
    if pf.offsets:
        off = np.zeros(len(d))
        for e in pf.offsets:
            off = off + np.asarray(eval_expr(e, d), dtype=float)

    keep = _aliased(X)
    Xk = X[:, keep]
    scale = "X2" if family in ("quasipoisson", "gamma", "gaussian",
                               "tweedie", "inverse.gaussian") else 1.0

    def _fit(fam, start=None):
        mod = sm.GLM(y, Xk, family=fam, offset=off, var_weights=w)
        return mod.fit(scale=scale, tol=tol, maxiter=maxiter,
                       start_params=start)

    theta_fixed = theta is not None
    if family == "negbin" and theta is None:
        # MASS::glm.nb: Poisson start, then alternate the GLM and theta.ml
        res = _fit(_sm_family("poisson", link, var_power, None))
        mu = res.fittedvalues
        th = _theta_ml(y, mu, w)
        d1 = np.sqrt(2 * max(1, res.df_resid))
        Lm = _nb_loglik(th, mu, y, w)
        Lm0 = Lm + 2 * d1
        d2 = delta = 1.0
        it = 0
        while it < 25 and (abs(Lm0 - Lm) / d1 + abs(delta) / d2) > 1e-8:
            it += 1
            res = _fit(_sm_family("negbin", link, var_power, th), res.params)
            t0 = th
            th = _theta_ml(y, res.fittedvalues, w)
            mu = res.fittedvalues
            delta = t0 - th
            Lm0, Lm = Lm, _nb_loglik(th, mu, y, w)
        res = _fit(_sm_family("negbin", link, var_power, th), res.params)
        theta = th
    else:
        res = _fit(_sm_family(family, link, var_power, theta))

    m = GLMFit()
    m.formula = pf.formula
    m.parsed = pf
    m.family = family
    m.link = link
    m.var_power = var_power
    m.theta = theta
    m.theta_fixed = theta_fixed
    m.sm_family = res.model.family
    # the family as R names it, for tables and titles
    m.family_label = {"gamma": "Gamma", "tweedie": "Tweedie",
                      "inverse.gaussian": "inverse.gaussian"}.get(family, family)
    if family == "negbin":
        m.family_label = "Negative Binomial(%s)" % _r_num(round(theta, 4))
    m.weights_expr = weights if isinstance(weights, str) else None
    m.data = d
    m.design_info = di
    m._X_full = X
    m._rhs_columns = rhs_cols
    m.keep = keep
    m.beta = np.asarray(res.params, dtype=float)
    names = list(di.column_names)
    coef = np.full(len(names), np.nan)
    coef[keep] = m.beta
    m.coef = pd.Series(coef, index=names)
    kn = [n for n, k in zip(names, keep) if k]
    m.vcov = pd.DataFrame(np.asarray(res.cov_params()), index=kn, columns=kn)
    m.y = y
    m.prior_weights = w
    m.offset = off
    m.eta = Xk @ m.beta + (off if off is not None else 0.0)
    m.fitted = m.linkinv(m.eta)
    m.nobs = len(y)
    m.rank = int(keep.sum())
    m.df_residual = m.nobs - m.rank
    m.deviance = float(np.sum(m.sm_family.resid_dev(y, m.fitted, var_weights=w) ** 2))
    m.null_deviance = float(res.null_deviance)
    pear = (y - m.fitted) * np.sqrt(w) / np.sqrt(m.variance(m.fitted))
    m.dispersion = float(np.sum(pear ** 2) / m.df_residual) if m.df_residual > 0 else np.nan
    m.dispersion_summary = (1.0 if family in ("poisson", "binomial", "negbin")
                            else m.dispersion)
    m.aic = _aic(m)
    m.converged = bool(getattr(res, "converged", True))
    m.result = res

    # Terms, their columns and the levels of the categorical factors
    m.terms = []
    for t in di.terms:
        if not t.factors:
            continue
        cols = []
        for f in t.factors:
            for v in factor_vars(f.code):
                if v not in cols:
                    cols.append(v)
        m.terms.append(Term(t.name(), [f.code for f in t.factors],
                            di.term_slices[t], cols))
    # R's terms(): by degree, in formula order within a degree, and the
    # factors of an interaction in their order of first appearance in the
    # formula, so labels read "REGIO:LEEFTIJD" where R writes that. patsy
    # orders its design columns its own way; Term.cols still points there.
    first, pos = {}, {}
    for i, t in enumerate(desc.rhs_termlist):
        for f in t.factors:
            first.setdefault(f.code, len(first))
        pos.setdefault(frozenset(f.code for f in t.factors), i)
    for t in m.terms:
        t.factors = sorted(t.factors, key=lambda c: first.get(c, len(first)))
        t.label = ":".join(r_code(c) for c in t.factors)
    m.terms.sort(key=lambda t: (t.order, pos.get(frozenset(t.factors), len(pos))))
    m.has_intercept = any(not t.factors for t in di.terms)
    m.xlevels = {}
    for f, fi in di.factor_infos.items():
        if fi.type == "categorical":
            m.xlevels[f.code] = list(fi.categories)
    return m


def _aic(m):
    y, mu, w, fam = m.y, m.fitted, m.prior_weights, m.family
    p = m.rank
    with np.errstate(divide="ignore", invalid="ignore"):
        if fam == "poisson":
            return float(-2 * np.sum(w * stats.poisson.logpmf(y, mu)) + 2 * p)
        if fam == "gamma":
            n = w.sum()
            disp = m.deviance / n
            return float(-2 * np.sum(w * stats.gamma.logpdf(y, a=1 / disp, scale=mu * disp))
                         + 2 + 2 * p)
        if fam == "gaussian":
            nobs = len(y)
            return float(nobs * (np.log(2 * np.pi * m.deviance / nobs) + 1) + 2
                         - np.sum(np.log(w[w > 0])) + 2 * p)
        if fam == "binomial":
            mm = w if np.all(np.abs(w - np.round(w)) < 1e-8) else np.ones_like(w)
            return float(-2 * np.sum(np.where(mm > 0, w / mm, 0)
                                     * stats.binom.logpmf(np.round(mm * y), np.round(mm), mu))
                         + 2 * p)
        if fam == "negbin":
            return float(-2 * _nb_loglik(m.theta, mu, y, w) + 2 * p + 2)
    return float("nan")
