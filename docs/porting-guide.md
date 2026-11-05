# Porting guide: R pricingtoolsRmO -> Python pricingtools_rmo

How the R package is translated, so every module reads the same way and the
two packages stay comparable. The R source is the reference:
`C:\Users\rmooijer\OneDrive - House Of HR NV\Bureaublad\GLM Utils\R\`, its
tests in `...\GLM Utils\tests\testthat\`, its design record in
`...\GLM Utils\TECHNICAL.md` and the user documentation in `...\README.md`.

## Environment

- Python 3.11, virtual environment in `.venv` of this repo:
  `.venv\Scripts\python.exe -m pytest tests\test_<module>.py -q`
- Code must run on Python 3.11 (no 3.12+ syntax) and on pandas 2.x and 3.x.
- R for parity checks: `C:\Users\rmooijer\AppData\Local\Programs\R\R-4.2.1\bin\Rscript.exe`,
  with the R package installed (`library(pricingtoolsRmO)`).

## The core (already built - use it, do not re-implement it)

`pricingtools_rmo/model.py` - `fit_glm(formula, data, family, link=None,
weights=None, var_power=1.5, theta=None)` returns a `GLMFit`, the
counterpart of an R `glm` object:

| R | Python (`m` is a GLMFit) |
|---|---|
| `family(m)$family` | `m.family` (`"poisson"`, `"quasipoisson"`, `"gamma"`, `"gaussian"`, `"binomial"`, `"tweedie"`, `"negbin"`); note R's `"Gamma"` is `"gamma"` here |
| `family(m)$link` | `m.link` (`"log"`, `"identity"`, ...) |
| `family(m)$linkinv/linkfun/mu.eta/variance` | `m.linkinv()`, `m.linkfun()`, `m.mu_eta()`, `m.variance()` |
| `model$data` aligned to the fit, `.glm_training_data()` | `m.data` (rows used, original columns, original index); `U.glm_training_data(m)` |
| `model.frame(m)[[1]]` (response) | `m.y` |
| `m$prior.weights`, `m$offset` | `m.prior_weights`, `m.offset` (None without offset) |
| `fitted(m)`, `predict(m, type = "link")` | `m.fitted`, `m.eta` |
| `predict(m, newdata, type = "response")` | `m.predict(newdata)` (all offsets in) ; `m.predict(nd, drop_offsets=[i])` takes offset expression i out |
| `coef(m)`, `vcov(m)` | `m.coef` (Series, NaN for aliased), `m.vcov` (DataFrame, non-aliased only) |
| `deviance`, `null.deviance`, `df.residual`, `AIC`, `rank`, `nobs` | same names with underscores: `m.deviance`, `m.null_deviance`, `m.df_residual`, `m.aic`, `m.rank`, `m.nobs` |
| Pearson dispersion `sum(pearson^2)/df` | `m.dispersion`; summary dispersion (1 for poisson/binomial/negbin) `m.dispersion_summary` |
| `residuals(m, type)` | `m.residuals("pearson"/"deviance"/"response"/"working")` |
| `hatvalues`, `cooks.distance`, `rstandard(type = "deviance")` | `m.hatvalues()`, `m.cooks_distance()`, `m.rstandard()` |
| `model.matrix(m)` / on new data | `m.model_matrix(newdata=None)` (full, aliased columns included; `m.keep` marks the estimated ones) |
| `attr(terms(m), "term.labels")` | `m.term_labels`; `m.terms` is a list of `Term(label, factors, order, cols, columns)` - `factors` are the patsy factor codes, `cols` the slice into the design matrix (R's `assign`), `columns` the data columns |
| `m$xlevels` | `m.xlevels` (dict factor code -> list of levels, original values) |
| `.rhs_vars(m)` (all.vars of the formula minus the response, offsets included) | `m.rhs_vars()` |
| `.offset_exprs(m)`, `.offset_vars(m)`, `.offset_value(m, d, idx)` | `m.offset_exprs`, `m.offset_vars()`, `m.offset_value(d, idx)` |
| `m$call$weights` variables | `m.weights_vars()`, `m.weights_expr` |
| `m$theta` (glm.nb) | `m.theta` |
| `update(m, data = d)` | `m.refit(d)` |

Factor codes are patsy's, with the R text translated: `factor(BOEKJAAR)`,
`cut(LEEFTIJD, _c_(17, 24, 40))`, `Q("AUTO GEWICHT")`. Use
`U.term_base_var(code, data)` to get the column behind a code, as in R.

`pricingtools_rmo/_utils.py` (import as `from . import _utils as U`) is the
port of `R/utils-internal.R`, with the same names minus the leading dot:
`check_cols`, `as_df`, `check_range`, `coerce_like`, `glm_training_data`,
`glm_newdata_parts`, `unseen_levels`, `unseen_message`, `model_label`,
`range_warning`, `apply_at`, `glm_ae_parts`, `glm_var_values`,
`group_values` (returns a pandas Categorical), `offset_exprs`,
`offset_value`, `predict_no_offset`, `offset_split`, `kept_offsets`,
`kept_label`, `offset_is_log` (None/True/False for R's NA/TRUE/FALSE),
`offset_vars`, `rhs_vars`, `default_nthread`, `model_rate`, `old_new_rates`,
`check_ae_scale`, `exposure_bins` (1-based bins), `gini_exposure`,
`gini_ordered`, `gini_boot(score, actual, weight, n_boot, conf, rng)`,
`top_share`, `large_loss_note`, `check_boot`, `drop_reasons(dict)`,
`nice_step`, `list_values`, `step_grid`, `level_chr` (list of str),
`snap_to_step`, `term_base_var`, `model_base_vars`, `predictable_rows`,
`source_kind`, `grouping_column`, `nb_theta`, `dev_resid_moments`,
`nil_claims_warning`, `year_warning`, `as_na_omit` (identity: Python models
are always na.omit). `_txt(v)` prints a value the way R prints a level
(`2024`, not `2024.0`; `TRUE`).

`pricingtools_rmo/_core.py`:

| R | Python |
|---|---|
| `stop(fn, ": ...")` | `stop(fn, ": ...")` - raises `PricingError` |
| `warning(...)` | `warn(...)` - `PricingWarning` |
| `message(...)` | `message(...)` - `PricingMessage` (a warning category) |
| a returned `list(...)` | `Result(...)` - a dict with attribute access |
| `attr(p, "cells") <- x` | `attach(fig, "cells", x)`; read as `fig.cells` |
| `attr(tbl, "intercept_frequency")` on a data frame | `tbl.attrs["intercept_frequency"]` |
| `plotly::config(...)` | `set_config(fig, "<filename>")` |

`pricingtools_rmo/_formula.py`: `r_cut()` is R's `cut()` with R labels;
`eval_expr(expr, data)` evaluates an R expression on a data frame;
`expr_vars(expr)` is R's `all.vars()`.

`pricingtools_rmo/colors.py`: `ta_navy`, `ta_blue`, `ta_lightblue`,
`ta_gold`, `ta_muted`, `ta_years_base`, `ta_year_palette(n)`.

## Conventions

- **Same names, same defaults.** Function and argument names are the R
  ones. `NULL` is `None`; `c(lo, hi)` is a 2-tuple or list; a named R list
  argument (`at`, `base`, `merge_vars`, `grid_step`) is a dict; a character
  vector is a list of str. Dutch column defaults stay (`"Exposure"`,
  `"AantalClaims"`, `"SCHADELAST"`, `"BOEKJAAR"`).
- **Input** is a pandas DataFrame; call `U.as_df()` at the entry point.
- **Output**: R data.frames are pandas DataFrames with the same column
  names (index reset to 0..n-1); R lists are `Result`; R vectors are numpy
  arrays or lists.
- **Messages keep R's English text**, including the `fn: ` prefix, so a
  search finds the same message in both packages. Use `stop`, `warn`,
  `message` from `_core`.
- **Decision tags**: keep every `# [ID]` comment of the R code at the same
  logical place, and keep R's explanatory comments where they still apply.
- **Docstrings**: numpy style, translated from the roxygen block, content
  kept - it is the user documentation.
- **Randomness**: an R `seed` argument becomes `seed=None`; create
  `rng = np.random.default_rng(seed)` and pass it on. Never use numpy's
  global random state. Results then differ from R's random streams, which
  is expected; everything deterministic must match R.
- **Plots** are `plotly.graph_objects` figures with the same traces, trace
  names, hover templates, titles, axes, colours and layout as the R
  version; finish with `set_config(fig, "<same filename as R>")`. Write
  non-ASCII characters in strings as `\u2013` escapes, as the R source
  does.
- **Grouping**: R's dplyr/data.table group-bys become pandas `groupby`;
  pass `observed=True` for Categorical keys where R drops empty groups,
  and keep R's ordering of the result (factor level order, or sorted).
- **Numbers**: R's `quantile()` default is numpy's default (type 7); R's
  `round()` and numpy's both round half to even; `weighted.mean` is
  `np.average(x, weights=w)`.

## Tests

- Port the R test file(s) for your module to `tests/test_<module>.py`, one
  Python test per R `test_that()`, named after it in snake_case. Use the
  fixtures in `tests/conftest.py` (`dat`, `m_freq`, `d_sev`, `m_sev`,
  `m_freq2`, `tbl`, `tbl2`; helpers `trace`, `trace_y`).
- Every R expectation gets a Python equivalent. `expect_error(..., "x")` ->
  `pytest.raises(PricingError, match="x")`; `expect_warning` ->
  `pytest.warns(PricingWarning, match=...)`; `expect_message` ->
  `pytest.warns(PricingMessage, match=...)`.
- The data differ from R's (numpy generator), so a statistical threshold
  may need re-checking; keep the property the test states, and if a seed or
  size has to change say so in a comment. Never weaken a test to make it
  pass.
- Port the tests of `test-review-findings.R` that concern your module.
- **Parity with R**: for deterministic outputs add
  `tests/parity/test_parity_<module>.py`, marked `@pytest.mark.parity` and
  `@requires_r` (from `tests/parity/_r.py`). Build data in Python, fit the
  same formula in both, run the R function through `run_r(code, frames)`,
  and compare (rtol 1e-6 for fitted quantities, exact for structure,
  labels and counts). Keep each parity test under ~20 seconds.
- Run only your own test files; other modules are being written at the
  same time.

## Boundaries

- Do not edit the core files (`model.py`, `_formula.py`, `_utils.py`,
  `_core.py`, `splines.py`, `colors.py`, `tests/conftest.py`,
  `tests/parity/_r.py`) unless a bug blocks you; then keep the fix minimal
  and list it in your report with the reason.
- Do not edit `pricingtools_rmo/__init__.py`, `README.md`, `TECHNICAL.md`,
  and do not commit; that is done centrally.
