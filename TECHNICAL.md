# pricingtools_rmo: technical documentation

This document records **how the Python package is built and why**: every
assumption it makes about the models and the data, every choice between two
defensible options, what it refuses and what it tolerates, and where in the
code each of those lives. It is the Python counterpart of `TECHNICAL.md` in
the R package pricingtoolsRmO, of which this package is a port, and follows
it entry for entry. It complements the README and the docstrings, which
explain how to use the functions and how to read their output.

It is written to be searched, by a person or by an AI assistant. Every
decision has a stable identifier such as `OFF-05`, the same identifier as in
the R package. The same identifier appears as a comment tag, `# [OFF-05]`, at
the place in the code that implements it (in a few modules as `[ID]` in the
module docstring), so one search finds both the reasoning and the code:

```sh
grep -rn "OFF-05" pricingtools_rmo/ tests/ TECHNICAL.md
```

The same search in the R repository finds the R code, so the two
implementations of one decision can be read side by side. The methods are
identical; where Python differs in how it does something, the entry says so.
Decisions about the port itself are in their own group, `PY`
([PY: the port itself](#py-the-port-itself)), and section 4 lists every
difference from the R package with the measured parity.

Identifiers without a tag in the code: `CONV-01`, `CONV-03`, `CONV-05`,
`CONV-06`, `CONV-07`, `CONV-09`, `AE-04`, `PY-08`.

The conventions in section 2 apply everywhere and are tagged only where they
have one concrete place (`CONV-02`, `CONV-04`, `CONV-08`). `AE-04` documents
behaviour of the statistics rather than of the code. `PY-08` lives in
`tests/parity/`, not in the package.

Each register entry has the same fields:

- **Decision**: what the code does.
- **Why**: the reasoning, including the alternative that was rejected.
- **Where**: module and function. Line numbers are deliberately not given,
  because they drift; the tag is the anchor.
- **Evidence**: measurements behind the choice, where there are any. Unless
  marked *Parity*, these were measured on the R package; the method is the
  same here. *Parity* figures compare the two packages (section 4.1).
- **Pinned by**: the Python tests that fail if the behaviour changes, as
  `file: test name`. Where an R test has no Python counterpart, the entry
  says so and section 6.3 lists it.
- **History (R package)**: only where a later change altered results, with
  the commit. These commits are in the R repository, not in this one. The
  port follows the R package at commit `b3faab2`, so every change listed is
  already in the Python code.

Contents

1. [Python architecture](#1-python-architecture)
2. [Cross-cutting conventions](#2-cross-cutting-conventions)
3. [Decision and assumption register](#3-decision-and-assumption-register)
   - [OFF: offsets and the link](#off-offsets-and-the-link)
   - [RT: rating table](#rt-rating-table)
   - [MV: merged and grouped columns](#mv-merged-and-grouped-columns)
   - [SCORE: data a model was not fitted on](#score-data-a-model-was-not-fitted-on)
   - [RATE: premium impact and lift](#rate-premium-impact-and-lift)
   - [AE: actual versus expected](#ae-actual-versus-expected)
   - [INT: interactions](#int-interactions)
   - [PDP: partial dependence](#pdp-partial-dependence)
   - [DIAG: diagnostics, Q-Q and influence](#diag-diagnostics-q-q-and-influence)
   - [SCR: feature screening](#scr-feature-screening)
   - [EXP: Excel export](#exp-excel-export)
   - [REP: HTML report](#rep-html-report)
   - [DATA: data handling and one-way aggregation](#data-data-handling-and-one-way-aggregation)
   - [PY: the port itself](#py-the-port-itself)
4. [Differences from the R package](#4-differences-from-the-r-package)
5. [Known limitations and open points](#5-known-limitations-and-open-points)
6. [Test map](#6-test-map)
7. [Glossary](#7-glossary)

---

## 1. Python architecture

### 1.1 What the package is

A Python package for GLM-based non-life pricing built around a frequency and
severity model pair: Poisson (or quasi-Poisson) counts with an exposure
offset, and a Gamma severity on the average claim weighted by the claim
count, both with a log link. It is the port of the R package pricingtoolsRmO:
the same functions under the same names, with the same arguments, defaults,
messages, tables and plots.

One thing is different by construction. The R package works on any fitted
`glm` and never fits a pricing model itself. In Python there is no fitted
object that remembers what an R `glm` remembers, so the package fits the
model too: `fit_glm()` takes an R formula and returns a `GLMFit` (PY-01).
Everything else explains, checks, tabulates and compares models fitted that
way.

All plots are `plotly.graph_objects` figures. There is no state between
calls: every function takes models and data and returns a value.

### 1.2 The model: fit_glm() and GLMFit

`model.py`, `fit_glm(formula, data, family, link=None, weights=None,
var_power=1.5, theta=None)` parses the formula (`_formula.py`, 1.3), builds
the design matrix with patsy, fits with statsmodels' IRLS and returns a
`GLMFit`. The `GLMFit` is the counterpart of an R `glm` object: it holds the
numbers statsmodels returns and, alongside them, everything the package reads
off an R fit.

| R | Python (`m` is a GLMFit) |
|---|---|
| `family(m)$family`, `family(m)$link` | `m.family` (`"poisson"`, `"quasipoisson"`, `"gamma"`, `"gaussian"`, `"binomial"`, `"tweedie"`, `"negbin"`), `m.link`; R's name for tables in `m.family_label` |
| `family(m)$linkinv`, `linkfun`, `mu.eta`, `variance` | `m.linkinv()`, `m.linkfun()`, `m.mu_eta()`, `m.variance()` |
| `model$data`, `.glm_training_data()` | `m.data`: the rows the fit used, original columns, original index |
| response, `m$prior.weights`, `m$offset` | `m.y`, `m.prior_weights`, `m.offset` (None without offset) |
| `fitted(m)`, `predict(m, type = "link")` | `m.fitted`, `m.eta` |
| `predict(m, newdata, type = "response")` | `m.predict(newdata)`, every offset in; `drop_offsets=[i]` leaves offset expression `i` out |
| `coef(m)`, `vcov(m)` | `m.coef` (NaN for an aliased column), `m.vcov` (estimated columns only) |
| `deviance`, `df.residual`, `AIC`, `rank`, `nobs` | `m.deviance`, `m.null_deviance`, `m.df_residual`, `m.aic`, `m.rank`, `m.nobs` |
| Pearson dispersion | `m.dispersion`; the summary dispersion (1 for Poisson, binomial, negative binomial) `m.dispersion_summary` |
| `residuals()`, `hatvalues()`, `cooks.distance()`, `rstandard()` | `m.residuals(type)`, `m.hatvalues()`, `m.cooks_distance()`, `m.rstandard()` |
| `model.matrix(m)` | `m.model_matrix(newdata=None)`, aliased columns included; `m.keep` marks the estimated ones |
| `attr(terms(m), "term.labels")` | `m.term_labels`; `m.terms` is a list of `Term(label, factors, order, cols, columns)` in R's order (PY-04) |
| `m$xlevels` | `m.xlevels`: patsy factor code to its levels |
| `all.vars()` of the right-hand side | `m.rhs_vars()` (offset variables included), `m.response_vars()` |
| offset expressions, their variables, their value | `m.offset_exprs`, `m.offset_vars()`, `m.offset_value(data, idx)` |
| `m$call$weights` | `m.weights_expr`, `m.weights_vars()` |
| `m$theta` of `MASS::glm.nb()` | `m.theta` |
| `update(m, data = d)` | `m.refit(d)` |

Prediction on new data follows R's `predict()`: `GLMFit._align()` brings the
columns to the types they had at fit time (a year typed as text finds the
numeric year, a nullable boolean becomes a plain one), `model_matrix()`
rebuilds the design with the stored patsy design info, which carries the
training levels and spline knots, and `predict()` re-evaluates every offset
expression on the new rows. A row with a missing predictor predicts as NaN.

Aliased columns are found the way R's pivoting QR leaves them out
(`_aliased()`), and keep a NaN coefficient. A negative binomial is fitted as
`MASS::glm.nb()` fits it: a Poisson start, then the GLM and the maximum
likelihood theta (`_theta_ml()`) in turn, unless `theta` is given.

### 1.3 Formulas and offsets: _formula.py

A formula is written as in R, for example
`"AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR) + offset(log(Exposure)) + offset(log(BM))"`.

- `ParsedFormula` splits it into the response, the right-hand side and the
  offset expressions. Every `offset()` term is taken out and kept as text, so
  it can be evaluated again on any data (OFF-01, OFF-02).
- `translate()` turns the R text into text patsy and Python read: backticked
  names become `Q("AUTO GEWICHT")`, `^` becomes `**`, `TRUE` and `FALSE`
  become Python booleans, `c()` and the dotted argument names `include.lowest`,
  `dig.lab` and `Boundary.knots` are renamed (PY-02).
- `R_FUNCTIONS` makes the R functions a pricing formula uses available under
  their R names: `log`, `exp`, `sqrt`, `pmin`, `pmax`, `ifelse`, `cut`
  (`r_cut()`, with R's labels such as `"(17,24]"`), `factor` (patsy's `C`),
  `ns`, `bs`, `poly`, `I`.
- `eval_expr()` evaluates an R expression on the columns of a data frame and
  nothing else; `expr_vars()` is R's `all.vars()`; `r_code()` writes a patsy
  factor code back the way R deparses the term (PY-04).

### 1.4 Spline bases: splines.py

`ns()`, `bs()` and `poly()` reproduce `splines::ns`, `splines::bs` and
`stats::poly`: the same knots, boundary handling, basis and extrapolation
(PY-03). They are patsy stateful transforms, so the knots are learned once on
the fit data and reused when the model scores other data, which is what R's
`makepredictcall` does. They are exported at package level, as the R package
re-exports `ns` and `bs`.

### 1.5 Shared helpers: _utils.py, _core.py, _plot_common.py

- `_utils.py` is the port of `R/utils-internal.R`, with the same helper names
  minus the leading dot (index in 1.9). Two of them are no-ops kept so the
  code reads like the R code: `as_na_omit()` returns the model unchanged
  (DATA-06), and `glm_training_data()` reads what `GLMFit` already holds
  (DATA-01, DATA-05). `_txt()` prints a value the way R prints a level
  (`2024`, not `2024.0`; `TRUE`).
- `_core.py` maps R's conditions and return values onto Python:
  `stop()` raises `PricingError`, `warn()` and `message()` issue
  `PricingWarning` and `PricingMessage` (PY-09); an R list is a `Result`,
  a table attached to a plot is set with `attach()`, plotly's config is kept
  by `set_config()` (PY-06).
- `_plot_common.py` carries R behaviour the plotting code relied on without
  stating it: `match.arg()` with partial matching (`match_arg()`), the order
  R puts text in (`arrange()`, `r_sort_unique()`, `r_levels()`), and the
  category axis in factor-level order that R's plotly builds by itself
  (`category_axis()`).

### 1.6 File map

| Module | R counterpart | Exported | Internal | Responsibility |
|---|---|---|---|---|
| `model.py` | R's `glm()`, `MASS::glm.nb()` | `fit_glm()`, `GLMFit` | `Term`, `_aliased()`, `_theta_ml()`, `_aic()` | fits a GLM from an R formula and keeps what an R `glm` keeps (PY-01) |
| `_formula.py` | R's formula handling | none | `ParsedFormula`, `translate()`, `R_FUNCTIONS`, `r_cut()`, `eval_expr()`, `expr_vars()`, `r_code()` | R formula text, offsets and expressions in Python (PY-02) |
| `splines.py` | `R/reexports.R` (`splines::ns`, `splines::bs`), `stats::poly` | `ns`, `bs`, `poly` | `ns_basis()`, `bs_basis()`, `_spline_design()` | R-compatible bases (PY-03) |
| `_core.py` | `stop()`, `warning()`, `message()`, lists, attributes | `PricingError`, `PricingWarning`, `PricingMessage`, `Result` | `stop()`, `warn()`, `message()`, `attach()`, `set_config()` | conditions and result containers (PY-06, PY-09) |
| `_utils.py` | `R/utils-internal.R` | none | all shared helpers (1.9) | offset handling, rates, training data, grids, grouped columns, binning, Gini |
| `_plot_common.py` | what dplyr and plotly do implicitly in R | none | `match_arg()`, `arrange()`, `category_axis()`, `wmean()` | R's argument matching, sort order and category axes |
| `rating_table.py` | `R/rating-table.R` | `make_rating_table()` | closures inside the function; `_r_terms()`, `_r_factor_codes()` | multiplicative factors, intercepts, interactions, thin cells, `base`, `merge_vars` |
| `groupings.py` | `R/groupings.R` | `add_groupings()` | none | adds grouped columns to data to be scored, sharing code with `merge_vars` |
| `rating_plot.py` | `R/rating-plot.R` | `make_rating_plot()` | none | plots one variable or interaction of a rating table |
| `export.py` | `R/export.R` | `export_rating_table()` | `_numeric_levels()` | formatted `.xlsx` with an Overview, a Tariff lookup sheet, one sheet per variable |
| `impact.py` | `R/impact.R` | `premium_impact()` | `_as_mask()`, `_r_logical()` (the spotlight expression) | per-policy dislocation, rebase, contribution, spotlight |
| `lift.py` | `R/lift.R` | `model_lift()`, `double_lift()` | none | lift chart, Gini with a bootstrap interval, Lorenz curve, double lift with an interval behind its verdict |
| `ae.py` | `R/ae.R` | `plot_glm_predictor()` | `_model_frame_column()` | actual versus expected per level or bin, in-sample or on new data, levels or ratio |
| `pdp.py` | `R/pdp.R` | `make_pdp()` | none | partial dependence on the response scale with an observed overlay |
| `agg.py` | `R/agg.R` | `agg_all()` | none | raw rows to frequency and severity per level |
| `plot_oneway.py` | `R/plot-oneway.R` | `make_plot()` | `_facet_plot()` | observed one-way plot with volume bars |
| `diagnostics.py` | `R/diagnostics.R` | `glm_diagnostics()`, `glm_collinearity()`, `plot_glm_residuals()` | `_r_format()` | fit table, generalised VIF, binned residuals |
| `qq_influence.py` | `R/qq-influence.R` | `plot_glm_qq()`, `plot_glm_influence()` | `_qr_dispersion()`, `_quantile_residuals()` | randomised quantile residuals, leverage and Cook's distance |
| `interactions.py` | `R/interactions.R` | `plot_residual_heatmap()`, `detect_interactions()` | `_ta_diverging`, `_rake()`, `_dev_of()` | cell-level A/E, interaction ranking by raking and simulation |
| `screen.py` | `R/screen.R` | `screen_features()` | `_fam_deviance()`, `_screen_levels()`, `_screen_matrix()` | boosted challenger on the GLM margin |
| `report.py` | `R/report.R` | `pricing_report()` | `safe()`, `_fmt_num()`, `_html_table()` | one HTML file with diagnostics, one-ways, A/E, PDP, rating plots, interaction scan |
| `colors.py` | `R/colors.R` | `ta_navy`, `ta_blue`, `ta_lightblue`, `ta_gold`, `ta_muted`, `ta_years_base`, `ta_year_palette()` | none | house colours |
| `__init__.py` | `NAMESPACE`, `R/pricingtoolsRmO-package.R` | every function above, `__version__` | none | the public interface |

### 1.7 How the main calls flow

**Fitting and predicting.** `fit_glm()` parses the formula
(`ParsedFormula`), builds the design matrix with patsy in an environment that
holds `R_FUNCTIONS`, drops rows with a missing value in any variable the
model uses (as R's `na.omit`), fits with statsmodels and records terms,
levels and offsets on the `GLMFit`. `GLMFit.predict(newdata)` aligns the
column types (`_align()`), rebuilds the design (`model_matrix()`), adds the
offset expressions evaluated on the new rows (`offset_value()`) and applies
the inverse link.

**Rating table.** `make_rating_table()` validates `merge_vars` and derives or
checks grouped columns (`grouping_column()`), resolves every model term to
its base column (`term_base_var()`), picks the base value per variable,
builds a grid per continuous variable (`nice_step()`, `list_values()`,
`step_grid()`), predicts each grid row against a base row with every other
variable at its base, divides, and adds two-way interaction blocks. The
intercepts are predictions of the base row with the whole offset taken out
(`predict_no_offset()`).

**Rates for comparison.** `premium_impact()` and `double_lift()` call
`old_new_rates()`, which applies `at` (`apply_at()`) and calls
`model_rate()` per model; `model_lift()` calls `apply_at()` and
`model_rate()` directly. `model_rate()` first names any value the model was
not fitted on (`unseen_levels()`, an error) and any numeric value outside
the fitted range (`range_warning()`), then asks `offset_split()` which
offset expressions to subtract and calls `predict_no_offset()` with only
those.

**Models read as fitted with na.omit.** Every function that mixes residuals,
fitted values or predictions with the training rows calls `as_na_omit()`
first, as in R (DATA-06). In Python that call returns the model unchanged: a
`GLMFit` is always fitted as with `na.omit`.

**Actual versus expected.** `plot_glm_predictor()` takes its rows from
`glm_training_data()` (in-sample) or `glm_newdata_parts()` (new data).
`plot_residual_heatmap()` and `detect_interactions()` take actual, expected
and volume per row from `glm_ae_parts()` and group them with
`group_values()`.

**Grouped columns.** `add_groupings()` and the `merge_vars` part of
`make_rating_table()` both call `grouping_column()`, which reads the
source-to-group relation from the data each model carries and decides
between exact lookup and banding with `source_kind()`.

### 1.8 Randomness, results and conditions

Functions with a `seed` argument create a numpy Generator and pass it down;
the global random state is never used (CONV-08, PY-05). R lists are returned
as `Result`, data frame attributes are `DataFrame.attrs`, tables attached to
a plot are figure attributes (PY-06). R's `stop()`, `warning()` and
`message()` are `PricingError`, `PricingWarning` and `PricingMessage`
(PY-09).

### 1.9 Internal helper index

All in `_utils.py` unless stated otherwise. The R name is the same with a
leading dot.

| Helper | Purpose | Called from |
|---|---|---|
| `check_cols(d, cols, fn)` | stop with a named message when columns are missing | most exported functions |
| `as_df(x)` | normalise any table to a pandas DataFrame | most exported functions |
| `_txt(v)` | a value as R prints it in a level; Python only | nearly every module |
| `check_range(rng, fn, arg)` | validate `(lo, hi)` axis ranges | every plot with `y_range`, `x_range` or `z_range` |
| `coerce_like(vals, template)` | coerce grid values to a column's type and categories; a logical stays logical | `make_rating_table()`, `make_pdp()`, `apply_at()` |
| `as_na_omit(model)` | R reads an `na.exclude` fit as `na.omit`; here it returns the model unchanged | `ae.py`, `diagnostics.py`, `interactions.py`, `pdp.py`, `qq_influence.py` |
| `glm_training_data(model, fn)` | training rows, response, prior weights and offset, aligned | `ae.py`, `diagnostics.py`, `lift.py`, `pdp.py`, `qq_influence.py`, `screen.py`, `glm_ae_parts()` |
| `glm_newdata_parts(model, newdata, fn)` | the same pieces evaluated on other data | `plot_glm_predictor()` |
| `unseen_levels(model, data)` | values of a factor term the model was not fitted on, per column, with a row mask | `model_rate()`, `glm_newdata_parts()`, `predictable_rows()`, `make_rating_table()` |
| `_eval_factor_values(code, data)` | the values of a patsy factor code on data, without patsy's categorical box; Python only | `unseen_levels()`, `ae.py`, `make_rating_table()` |
| `unseen_message(un, label, fn, at_hint)` | the error text for those values | `model_rate()`, `glm_newdata_parts()` |
| `model_label(label)` | "model" or "frequency model" in a message | `unseen_message()`, `range_warning()` |
| `range_warning(model, data, label, fn)` | warn when numeric predictors leave the fitted range | `model_rate()` |
| `apply_at(data, at, fn)` | set columns to one value for every row (`at={"BOEKJAAR": 2024}`) | `old_new_rates()`, `model_lift()` |
| `rhs_vars(model)` | variables on the right-hand side, response excluded, offsets included | `pdp.py`, `screen.py` |
| `glm_ae_parts(model, fn)` | actual, expected, volume and claim weight per fitted row | `plot_residual_heatmap()`, `detect_interactions()` |
| `glm_var_values(parts, var, fn)` | a variable aligned with the fitted rows | `interactions.py` |
| `group_values(x, n_bins)` | categorical as is, few numeric values exact, else quantile bins; returns a pandas Categorical | `interactions.py` |
| `offset_exprs(model)` | every offset expression; reads `GLMFit.offset_exprs` | the documented entry point; callers read the model directly |
| `offset_value(model, data, idx)` | evaluated offset, optionally only some expressions; reads `GLMFit.offset_value()` | as above |
| `predict_no_offset(model, newdata, idx)` | `linkinv(eta - offset)` | `model_rate()`, `make_pdp()`, `make_rating_table()` |
| `offset_split(model, exposure_col, keep_offsets)` | which offset expressions to subtract and which stay | `model_rate()`, `kept_offsets()`, `make_pdp()` |
| `kept_offsets(models, exposure_col, keep_offsets)` | names of the offsets that stay in a rate | `old_new_rates()`, `model_lift()`, `make_pdp()` |
| `kept_label(kept)` | text for the summary line | `premium_impact()` |
| `offset_is_log(model)` | is every offset a `log()`: None, True or False for R's NA, TRUE, FALSE | `ae.py`, `pdp.py`, `glm_ae_parts()` |
| `offset_vars(model)` | variables appearing in any offset | many |
| `default_nthread()` | thread count respecting `OMP_THREAD_LIMIT` | `screen_features()` |
| `model_rate(model, data, exposure_col, fn, keep_offsets, label)` | predicted rate per unit of exposure, after the unseen-level and range checks | `model_lift()`, `old_new_rates()` |
| `old_new_rates(...)` | old and new rate per policy from models or a premium column | `premium_impact()`, `double_lift()` |
| `check_ae_scale(actual, predicted, fn, actual_col)` | warn when the overall A/E suggests mismatched units | `model_lift()`, `double_lift()` |
| `exposure_bins(score, weight, n_bins, fn)` | equal-weight bins ordered by a score, numbered from 1 | `model_lift()`, `double_lift()` |
| `gini_exposure(score, actual, weight)` | exposure-weighted Gini | `model_lift()` |
| `gini_ordered(w, a)` | the same on vectors already in score order | `gini_exposure()`, `gini_boot()` |
| `gini_boot(score, actual, weight, n_boot, conf, rng)` | Poisson-weight bootstrap interval for the Gini, drawn from the Generator `rng` | `model_lift()` |
| `top_share(actual, k)` | share of the total held by the `k` largest records | `large_loss_note()` |
| `large_loss_note(actual, actual_col)` | the sentence naming large losses, when the ten largest carry 10% or more | `model_lift()`, `double_lift()` |
| `check_boot(n_boot, conf_level, fn)` | validate the bootstrap arguments | `model_lift()`, `double_lift()` |
| `drop_reasons(flags)` | count dropped rows per reason; `flags` is a dict | `premium_impact()`, `model_lift()`, `double_lift()` |
| `nil_claims_warning(...)` | warn when claims without payment make frequency x severity too high | `make_rating_table()`, `model_lift()` |
| `year_warning(new_models, old_models, year_col, at, fn)` | warn when a model prices each policy at its own accounting year | `premium_impact()` |
| `predictable_rows(model, newd)` | rows of a prediction frame without an unseen level | `make_rating_table()` |
| `dev_resid_moments(model)` | mean and variance of each row's deviance residual under the fitted family | `plot_glm_residuals()` |
| `nb_theta(model)` | theta of a negative binomial fit, or None | `dev_resid_moments()`, `_quantile_residuals()` |
| `nice_step(rng, n, integer_data)` | a 1, 2 or 5 times 10^k step | `make_rating_table()` |
| `list_values(uv, n, integer_data)` | list a whole-number column value by value | `make_rating_table()` |
| `step_grid(lo, hi, step, base)` | grid on whole multiples of the step, base included | `make_rating_table()` |
| `level_chr(x)` | grid values as text without scientific notation, as a list of str | `make_rating_table()`, `export.py`, `_txt()` |
| `snap_to_step(x, step, lo, hi)` | round a value onto the grid, clamped | `make_rating_table()` |
| `term_base_var(term, data)` | the column behind a term label or patsy factor code | `make_rating_table()`, `model_base_vars()`, `source_kind()`, `plot_glm_predictor()` |
| `source_kind(pv, models, src)` | is a grouping source categorical or continuous | `grouping_column()` |
| `grouping_column(data, pv, a, model_freq, model_sev, fn)` | derive or verify one grouped column | `make_rating_table()`, `add_groupings()` |
| `model_base_vars(model, data)` | base columns of a model's terms | `pricing_report()`, `detect_interactions()`, `plot_glm_influence()` |

R's `.model_data()` has no counterpart: a `GLMFit` always holds its
training rows (DATA-05).

### 1.10 Dependencies

| Package | Kind | Used for |
|---|---|---|
| `numpy`, `pandas`, `scipy` | required | arrays, tables, distributions, integration |
| `statsmodels` | required | the IRLS fit inside `fit_glm()` |
| `patsy` | required | design matrices, stateful transforms for the spline bases |
| `plotly` | required | every plot, and plotly.js for `pricing_report()` |
| `openpyxl` | extra `excel` | `export_rating_table()` only |
| `xgboost` | extra `screen` | `screen_features()` only |
| `pytest` | extra `dev` | the tests |
| R with pricingtoolsRmO and `jsonlite` | outside Python | the parity tests only (PY-08) |

Optional packages are imported inside the function that needs them, which
stops with a `pip install` hint when the import fails. The package runs on
Python 3.11 and later and on pandas 2.x and 3.x; CI tests both.

---

## 2. Cross-cutting conventions

#### CONV-01 Dutch column defaults

The column arguments default to `Exposure`, `AantalClaims`, `SCHADELAST`
and `BOEKJAAR`. That is the owner's data, and the defaults are deliberately
kept rather than translated, in R and in Python. Every function that reads
such a column takes it as an argument.

#### CONV-02 Normalise once at the entry point

In R, a data.table reads `d[, cols]` as an expression and a tibble returns a
tibble where a vector is expected, so input is converted with `.as_df()`
once, at the start, instead of guarded at every call site. Python does the
same with `as_df()`: a pandas DataFrame passes through, a table with
`to_pandas()` (polars, pyarrow) is converted, anything else goes through
`pd.DataFrame()`, so a dict of columns works too. Everything below the entry
point then uses pandas only. Where R returned a data.table for a data.table
(`add_groupings()`), Python returns a new pandas DataFrame and leaves the
caller's frame alone.

- **Where.** `_utils.py`, `as_df()`.
- **Pinned by.** `test_agg_plots.py`: `test_agg_all_and_make_plot_accept_other_table_input`;
  `test_pdp.py`: `test_make_pdp_accepts_other_table_input`;
  `test_rating_table.py`: `test_make_rating_table_accepts_non_pandas_input`;
  `test_screen.py`: `test_model_data_from_non_pandas_input_is_handled`;
  `test_groupings.py`: `test_the_input_is_left_alone`.

#### CONV-03 Refuse rather than guess; warn when the result is still right

The line between an error and a warning is whether the output would still
be correct. An unknown factor level, an offset that cannot be evaluated per
row, a grouping that means two things, or a column that is missing are
errors, because any number produced would be a guess. A non-log link, a
thin cell, a misspelled `grid_step` name, an undeclared regrouping, or a
value outside the fitted range of a banding are warnings, because the result
is still correct and only needs reading with care. Errors are `PricingError`
and warnings `PricingWarning` (PY-09). Both are prefixed with the function
name, so the message says where it comes from; R's `call. = FALSE` has no
counterpart, and `warn()` points the warning at the caller's line.

#### CONV-04 Partial output over no output

Where a function builds many independent pieces, one failing piece becomes
a warning and is skipped: a variable in `make_rating_table()`, an
interaction pair in `make_rating_table()` and `detect_interactions()`, a
plot in `pricing_report()`. The function fails only when nothing at all
could be produced. The cost is that a skipped piece is easy to miss in a
long run, so the warning always names it.

- **Where.** `rating_table.py`, `make_rating_table()` (the loops over
  `main_block()` and `interaction_block()`); `report.py`, `safe()`.

#### CONV-05 The model is read, not assumed

Link, inverse link, offset expressions, prior weights, factor levels and the
training data are all taken from the fitted object: in R from `family()`,
`terms()`, `model$call`, `model$prior.weights`, `model$xlevels` and
`model$data`; here from `GLMFit` (`m.link`, `m.linkinv()`,
`m.offset_exprs`, `m.prior_weights`, `m.xlevels`, `m.data`). In R several
functions therefore need a model fitted with a `data =` argument, and say
so. A `GLMFit` always carries its data (PY-01), so in Python that condition
always holds.

#### CONV-06 ASCII-only source

All R files are ASCII, with non-ASCII characters in messages and plot titles
written as `\uXXXX` escapes; that is required for a clean `R CMD check`.
Python source is UTF-8, so nothing requires it here. The porting guide
still asks for escapes so that strings read the same as in R, and most
modules follow it; a few plot titles and trace names hold the characters
literally (an en dash, `±`, `·`, `×`). The rendered text is identical.

#### CONV-07 Plots

Every plot is a plotly figure in the house colours (`colors.py`), with a
volume bar on a secondary axis where that applies, and a `y_range` (or
`x_range`, `z_range`) argument so that plots of different variables can
share a scale. Categorical axes show markers only, never a line through
categories. Python sets the order of a category axis explicitly, where R's
plotly takes it from the factor. The modebar configuration R sets with
`plotly::config()` is kept as `fig.pricing_config` (PY-06).

- **Pinned by.** `test_y_range.py`: `test_all_plot_functions_auto_scale_the_y_axis_by_default`,
  `test_all_plot_functions_honour_a_fixed_y_range`,
  `test_make_plot_facet_mode_pins_all_facets_to_the_fixed_range`,
  `test_an_invalid_range_is_rejected_everywhere`.

#### CONV-08 Randomness

Functions with a `seed` argument (`plot_glm_qq()`, `plot_glm_influence()`,
`detect_interactions()`, `screen_features()`, `model_lift()`,
`double_lift()`) create a numpy Generator, `np.random.default_rng(seed)`,
and pass it to everything random they call. Unlike R, where these functions
call `set.seed()` and so change the global random state, they leave numpy's
global state alone. Without `seed` each call draws fresh entropy. The same
seed gives other draws than in R (PY-05).

- **Where.** `interactions.py`, `detect_interactions()`; `lift.py`,
  `model_lift()`, `double_lift()`; `qq_influence.py`, `plot_glm_qq()`,
  `plot_glm_influence()`; `screen.py`, `screen_features()`.
- **Pinned by.** `test_diagnostics.py`: `test_na_rows_give_the_same_result_as_complete_rows`
  (one seed, the same randomised quantile residuals twice).

#### CONV-09 Documentation style

The README and this document are in English and use no em dashes. The
docstrings are numpy style, translated from the roxygen blocks of the R
package with their content kept: they are the user documentation. Each
register entry here is written so it can be read on its own.

---

## 3. Decision and assumption register

### OFF: offsets and the link

#### OFF-01 Offsets are read from both places predict() reads them

- **Decision.** In R an offset is collected from the formula terms
  (`attr(terms, "offset")`) and from the `offset =` argument
  (`model$call$offset`); a model can carry both, and they add. In Python a
  model has one place for offsets: every `offset()` term of the formula, any
  number of them, is kept as an expression by `ParsedFormula` and read back
  as `GLMFit.offset_exprs`. They add. `fit_glm()` has no `offset =`
  argument (PY-02).
- **Why.** `glm(y ~ x, offset = log(E))` never reaches `terms()`. Reading
  only the terms left that offset in the prediction, silently and 70% out.
  `predict.lm` reads exactly these two places, so following it keeps the
  package in step with what `predict()` itself puts in the linear predictor.
  In Python, `GLMFit.predict()` adds exactly the expressions in
  `offset_exprs`, so the two cannot disagree; the second place was left out
  rather than ported, because it is what made this rule necessary in R.
- **Where.** `_utils.py`, `offset_exprs()`; `_formula.py`, `ParsedFormula`;
  `model.py`, `GLMFit.predict()`.
- **Pinned by.** `test_core.py`: `test_formula_parts_and_offsets`,
  `test_rhs_vars_include_offset_vars`; `test_impact.py`:
  `test_offsets_from_the_formula_add_up`. R's "an offset passed as an
  argument is neutralised too" has nothing to test here.

#### OFF-02 An offset is removed by subtraction on the link scale

- **Decision.** `rate = linkinv(eta - offset)`, with the offset evaluated
  from the model's own expressions. The column is never simply set to 1. In
  Python, `predict_no_offset()` calls `GLMFit.predict(newdata,
  drop_offsets=idx)`, which leaves the chosen expressions out of the linear
  predictor.
- **Why.** Setting the exposure column to 1 only neutralises
  `offset(log(Exposure))`. With `offset(Exposure)` it leaves 1 instead of
  0, a factor e too high; with `offset(log(Maanden / 12))` it leaves
  `log(1/12)`, a factor 12 out. Subtraction is exact for any link and any
  expression.
- **Where.** `_utils.py`, `offset_value()`, `predict_no_offset()`;
  `model.py`, `GLMFit.offset_value()`, `GLMFit.predict()`.
- **Evidence.** Rate per unit of exposure on one model and data:
  `offset(log(Exposure))` 0.13063 both ways; `offset(Exposure)` 0.11764
  with the shortcut against 0.04328 correct; `offset(log(Maanden/12))`
  0.01089 against 0.13063.
- **Pinned by.** `test_core.py`: `test_prediction_reevaluates_offsets`,
  `test_model_rate_keeps_bm_and_drops_exposure`; `test_pdp.py`:
  `test_make_pdp_neutralises_a_non_log_offset_and_keeps_the_collapse`
  (`offset(Exposure)`); `test_rating_table.py`:
  `test_the_intercept_is_per_unit_of_exposure_whatever_the_offset_is_called`.
  R's three cases "an offset without a log is neutralised correctly", "an
  offset on an expression is neutralised correctly" and "the usual log link
  with a log offset is unchanged" are ported as `test_impact.py`:
  `test_an_offset_without_a_log_is_neutralised_correctly`,
  `test_an_offset_on_an_expression_is_neutralised_correctly`,
  `test_the_usual_log_link_with_a_log_offset_is_unchanged`, with
  `test_offset_value_evaluates_the_models_own_offset` and
  `test_a_model_without_an_offset_is_untouched`.

#### OFF-03 A pre-computed offset vector is refused

- **Decision.** If an offset expression does not evaluate to one value per
  row of the data being scored, the function stops. In Python the usual way
  in, `offset(off_vec)` with a vector from the session, is already refused by
  `fit_glm()`: formula expressions are evaluated on the columns of `data`
  only (`eval_expr()`), so `off_vec` is a column that is not found.
  `GLMFit.offset_value()` keeps the per-row check for any expression that
  still gives the wrong length (`np.repeat(0.1, n)`), and `model_rate()`
  turns that, or a column the scoring data lacks, into R's message ("the
  model has an offset that could not be evaluated on this data").
- **Python difference.** Only the offsets that stay in a rate are evaluated
  on the scoring data; the ones taken out are left out of the linear
  predictor, not subtracted from it. A scoring data set without the column
  of an offset that comes out anyway therefore still gives the exact rate,
  where R stops because it has to evaluate the offset to subtract it.
- **Why.** `offset = off_vec` stores only the name. Evaluated on other data
  it returns the fitted vector, of the fitted length, and `predict()`
  recycles it without a word. A wrong number delivered confidently is worse
  than an error that says to put the offset in the formula.
- **Where.** `model.py`, `GLMFit.offset_value()`, `fit_glm()`; `_utils.py`,
  `model_rate()`.
- **Pinned by.** `test_impact.py`:
  `test_an_offset_that_cannot_be_evaluated_per_row_is_refused` (adapted: a
  formula offset with a fixed length and a missing column, since there is
  no `offset =` argument).
- **History (Python port).** Before the port's offset tests were completed,
  a missing column of a kept offset escaped as a bare `FormulaError` ("name
  'BM' is not defined") instead of R's message; `model_rate()` now catches
  it.

#### OFF-04 exp(offset) is an exposure only for a log offset under a log link

- **Decision.** Code that recovers a volume as `exp(offset)` does so only
  when every offset term is a `log()` and the link is log. Otherwise it
  falls back to the prior weights, with a warning. `offset_is_log()` returns
  None without an offset, where R returns NA.
- **Why.** With `offset(Exposure)`, `exp(offset)` is `exp(Exposure)`, which
  is not a volume, and a counts-per-exposure computation weighted by it is
  nonsense.
- **Where.** `_utils.py`, `offset_is_log()`, `glm_ae_parts()`; `ae.py`,
  `plot_glm_predictor()`; `pdp.py`, `make_pdp()`.
- **Pinned by.** `test_agg_plots.py`:
  `test_exp_offset_is_not_read_as_an_exposure_when_it_is_not_one`;
  `test_pdp.py`: `test_make_pdp_neutralises_a_non_log_offset_and_keeps_the_collapse`;
  `test_impact.py`: `test_offset_is_log_distinguishes_a_log_offset_from_any_other`.

#### OFF-05 Rates for comparison keep every offset except the exposure

- **Decision.** In `premium_impact()`, `model_lift()`, `double_lift()` and,
  since PDP-05, `make_pdp()`, only the offset expressions that mention
  `exposure_col` are subtracted.
  Any other offset, typically a bonus-malus scale, stays in the rate.
  `keep_offsets=False` subtracts everything. Two cases fall back to
  subtracting everything: a model whose offset does not mention
  `exposure_col` at all (it offsets on `Maanden` or `Ex`, which is an
  exposure under another name; see OFF-09), and one expression holding the
  exposure and something else, `offset(log(Exposure * BM))`, which cannot be
  split and raises a warning.
- **Why.** The exposure says how long a policy was on risk, the BM how much
  it pays. Every comparison these functions make is against something that
  includes the BM: the premium a customer pays, the claims that happened.
  Removing it puts every policy at the class without discount.
- **Where.** `_utils.py`, `offset_split()`, `model_rate()`,
  `kept_offsets()`, `old_new_rates()`; `impact.py`, `premium_impact()`;
  `lift.py`, `model_lift()`, `double_lift()`.
- **Evidence.** On a book with an average discount of 20%, a correct model
  with an in-sample A/E of exactly 1.000 read 0.800 on `model_lift()` with
  the BM removed, in every bin. `premium_impact()` against a current premium
  including discount made an unchanged tariff look like a shift: -20.0% for
  no discount, -11.1%, 0.0%, +14.3%, +33.3% for a 40% discount. With the BM
  kept, 0.0% everywhere. *Parity:* `premium_impact` rates, by_level and rate
  change 1e-15 to 2e-9, on models with an exposure and a BM offset.
- **Consequence if changed.** An old premium column must be on the same
  footing as the kept offsets. With the default, that is the premium
  including discount. Dividing such a premium by the BM yourself now takes
  the discount out twice (+25% and -25% at the extremes, measured).
- **Pinned by.** `test_impact.py`:
  `test_a_bonus_malus_offset_stays_in_the_rate_the_exposure_goes`,
  `test_premium_impact_compares_a_premium_incl_discount_like_for_like`,
  `test_offsets_from_the_formula_add_up`; `test_lift.py`:
  `test_model_lift_reads_a_bm_model_against_its_own_claims`;
  `test_core.py`: `test_model_rate_keeps_bm_and_drops_exposure`;
  `parity/test_parity_impact_lift.py`: `test_premium_impact_parity`,
  `test_model_lift_parity`, `test_double_lift_parity`.
- **History (R package).** Before commit `626bb2a` every offset was removed
  in these three functions. Results change only for models with a second
  offset.

#### OFF-06 The rating table removes the whole offset

- **Decision.** The rating table's base row holds every offset variable at
  1 and its intercepts are computed with the whole offset subtracted, so
  the table is at BM = 1.
- **Why.** A tariff table describes a base policy, and the BM discount is
  applied per policy on top, the same way the exposure is. A BM factor is
  not estimated, so it has no block in the table.
- **Where.** `rating_table.py`, `base_value_raw()` (offset variables
  return 1) and `intercept_val()`; `pdp.py`, `make_pdp()` (the tag marks
  where the PDP now departs from this, PDP-05).
- **Consequence.** A premium built from the table is
  `intercept_premium x factors x BM`. Verified per policy against
  `predict()` with Exposure = 1 and the policy's BM.
- **Pinned by.** `test_rating_table.py`:
  `test_the_intercept_is_per_unit_of_exposure_whatever_the_offset_is_called`,
  `test_intercept_x_factors_reconstructs_the_model_prediction_exactly`;
  `test_groupings.py`: `test_the_grouped_columns_are_derived_from_the_source_columns`
  (premium chain with the BM on top, per policy); `test_impact.py`:
  `test_the_rating_table_intercept_follows_the_models_own_offset`.
- **History (R package).** Until the review fixes of October 2026 the PDP
  removed the whole offset too, on the argument that a partial effect is
  cleaner without a policy-specific discount. It was then drawn against an
  observed line at each policy's own BM and sat 35% above it on a correct
  model. It now keeps the BM (PDP-05).

#### OFF-07 A non-log link is tolerated, with a warning

- **Decision.** With a link other than log, offsets are still removed, and
  the rating table is still built, but a warning says the result is not
  multiplicative and should not be multiplied by a second model.
- **Why.** Refusing would block an inspection that is still informative;
  silence would let a sum be read as a product.
- **Where.** `_utils.py`, `model_rate()`; `rating_table.py`,
  `check_link()`.
- **Pinned by.** `test_rating_table.py`: `test_non_log_link_raises_a_warning`;
  `test_impact.py`: `test_a_non_log_link_removes_the_offset_and_says_what_that_means`
  (the warning in `model_rate()` and the rate with the offset removed).

#### OFF-08 Volumes are divided by the exposure column, not by exp(offset)

- **Decision.** Where a plot shows a frequency or a volume bar, it divides
  by `exposure_col` when that column exists, and only falls back to
  `exp(offset)` without it.
- **Why.** With a second offset, `exp(offset)` is `Exposure x BM`. The A/E
  is the same either way, but the axis would read "claims per BM-adjusted
  policy-year", which nobody reads off a chart.
- **Where.** `ae.py`, `plot_glm_predictor()`; `pdp.py`, `make_pdp()`
  (the PDP average is weighted by the same column as its observed line).
- **Known gap.** `glm_ae_parts()` in `_utils.py`, behind
  `plot_residual_heatmap()` and `detect_interactions()`, still takes
  `exp(offset)` as the volume. With a BM offset, the tooltip exposure and
  `MaxAE_ExposureShare` are BM-weighted; the A/E itself is unaffected. See
  section 5.
- **Pinned by.** `test_agg_plots.py`:
  `test_a_second_offset_does_not_end_up_on_the_exposure_axis_ae`,
  `test_the_exposure_column_falls_back_and_can_be_named`; `test_pdp.py`:
  `test_a_second_offset_does_not_end_up_on_the_exposure_axis_pdp`.

#### OFF-09 An offset on another column is followed, and said out loud

- **Decision.** When the model offsets on a column other than
  `exposure_col` (`offset(log(Ex))`, `offset(log(Maanden / 12))`), that
  offset is the one removed, and a warning says the rates are per unit of
  that variable while the weighting uses `exposure_col`.
- **Why.** Setting `exposure_col` to 1 would leave the real offset in place
  and return a prediction per record instead of a rate, silently.
- **Where.** `_utils.py`, `model_rate()`; `rating_table.py`,
  `make_rating_table()` (the `offset_vars` warning); `pdp.py`,
  `make_pdp()`.
- **Pinned by.** `test_impact.py`:
  `test_an_offset_on_another_column_is_neutralised_and_said_out_loud`;
  `test_rating_table.py`:
  `test_the_intercept_is_per_unit_of_exposure_whatever_the_offset_is_called`.

### RT: rating table

#### RT-01 A factor is a ratio of two predictions sharing one base row

- **Decision.** `Factor(level) = predict(level, everything else at base) /
  predict(base row)`. The intercept is the base row's prediction with the
  offset removed. `Factor_Premium = Factor_Frequency x Factor_Severity`.
- **Why.** Predicting rather than exponentiating coefficients handles
  splines, inline transforms, interactions and any parametrisation the same
  way, and guarantees the reconstruction identity:
  `intercept x product of factors = predict()` at exposure 1, for a log
  link. With an interaction, the interaction block's joint factor replaces
  the two main-effect factors of that pair. In Python it also keeps the
  table independent of patsy's coding of the design matrix (PY-04).
- **Where.** `rating_table.py`, `predict_factor()`, `predict_factor_2d()`,
  `intercept_val()`.
- **Evidence.** *Parity:* `Factor_*` 1e-10 from R on three tables;
  structure, labels and base values exact.
- **Pinned by.** `test_rating_table.py`:
  `test_intercept_x_factors_reconstructs_the_model_prediction_exactly`,
  `test_intercept_is_per_unit_of_exposure`,
  `test_every_variable_has_exactly_one_base_row_with_factor_1`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`.

#### RT-02 The base point

- **Decision.** Categorical: the first level of the model's `xlevels`, or
  with `base_level="exposure"` the level with the most exposure; the
  accounting year (`year_col`) its latest level (RT-15).
  Continuous: the median, snapped onto the grid. Offset variables: 1. A
  `base` argument overrides any of these per variable; a categorical value
  must be an existing level, a continuous value is used exactly as given
  (with a warning outside the data). A merged column follows its source
  (MV-09).
- **Why.** The base is a reference point, not a recommendation: moving it
  moves the factors and the intercept against each other and leaves every
  premium unchanged. The defaults give a row with factor 1 that is a value
  someone would quote.
- **Where.** `rating_table.py`, `base_value_raw()`, `grid_of()`, the
  `base` validation block in `make_rating_table()`.
- **Pinned by.** `test_rating_table.py`:
  `test_base_level_exposure_picks_the_largest_level_as_reference`,
  `test_the_base_can_be_chosen_per_variable_default_unchanged`,
  `test_a_wrong_base_is_refused_or_reported`;
  `parity/test_parity_rating_table.py`: `test_rating_table_parity_covers_what_it_should`.

#### RT-03 A variable missing from one model gets a factor of 1; a missing model gets NA

- **Decision.** A variable in the frequency model but not in the severity
  model gets `Factor_Severity = 1`, and vice versa. A model that is not
  supplied at all gives NaN in its factor column, so with one model only,
  `Factor_Premium` and `intercept_premium` are NaN and the single model's
  own columns are the ones to use.
- **Why.** A neutral 1 keeps `Factor_Premium` correct when both models are
  there. When one is absent there is no premium to report, and a missing
  value says so rather than pretending the missing side is 1.
- **Where.** `rating_table.py`, `predict_factor()`; the
  `intercept_premium` entry of `tbl.attrs`, set at the end of
  `make_rating_table()`.

#### RT-04 Readable grids for continuous variables

- **Decision.** The step is about `grid_res` points across the range,
  rounded to 1, 2 or 5 times a power of ten, never fractional for a column
  of whole numbers. A whole-number column with at most `2 x grid_res`
  distinct values is listed on its own values. The base is inserted into
  the grid if the clamping left it off. `grid_step` overrides globally or
  per variable (a dict); a name that matches no continuous variable is a
  warning.
- **Why.** A tariff lists ages 18, 19, 20 and weights 800, 850, 900, not
  19.2653. Listing "values" rather than "a step of 1" keeps a coded column
  (five sums insured spread over a million) to five rows instead of a
  million.
- **Where.** `_utils.py`, `nice_step()`, `list_values()`, `step_grid()`,
  `snap_to_step()`; `rating_table.py`, `grid_of()`, the `grid_step` check in
  `make_rating_table()`.
- **Pinned by.** `test_rating_table.py`:
  `test_continuous_grids_run_in_readable_steps`,
  `test_grid_step_can_be_set_globally_or_per_variable`,
  `test_a_wide_integer_column_with_few_values_is_listed_not_stepped`;
  `test_core.py`: `test_grid_helpers`.

#### RT-05 Volumes per grid point, and trimming

- **Decision.** Categorical: exposure and claims per level. Continuous:
  each policy goes to its nearest grid point, splitting on the midpoints.
  With `trim` other than `(0, 1)`, policies outside the trimmed range are
  left out of the volumes rather than swept into the edge points.
- **Why.** Sweeping them in would make the outer bars overstate their
  volume for values the grid does not cover. The trimmed exposure is
  therefore below the portfolio total by design.
- **Where.** `rating_table.py`, `cell_index()`, `volumes()`.
- **Evidence.** *Parity:* `Exposure` 2e-15 from R, including a trimmed
  grid.
- **Pinned by.** `test_rating_table.py`:
  `test_a_trimmed_grid_excludes_the_policies_it_does_not_cover`.

#### RT-06 Two-way interactions

- **Decision.** Only two-way terms get a block. The continuous variable is
  the x-axis when exactly one is continuous; with two categoricals the one
  with more levels is. A continuous group variable is shown in three
  slices: its 10% and 90% quantiles (rounded to three decimals) and its
  base. The factor is the joint relativity against the base cell; the
  uplift is `joint / (main_x x main_group)`. Without any interaction term
  the three `Uplift_*` columns are dropped.
- **Why.** Higher-order blocks cannot be read in a table. Uplift columns
  that are missing in every row would only need explaining. `Group` and
  `GroupVar` always stay: `Group.isna()` is how a main effect is told from
  an interaction.
- **Where.** `rating_table.py`, `gather_pairs()`, `interaction_block()`,
  `predict_factor_2d()`.
- **Evidence.** *Parity:* `Uplift_*` 1e-13 from R.
- **Pinned by.** `test_rating_table.py`:
  `test_interaction_rows_have_a_unit_base_cell_and_unit_uplift_at_reference`,
  `test_interaction_cells_carry_exposure_and_claims_for_any_variable_type`,
  `test_the_uplift_columns_are_absent_without_an_interaction`.

#### RT-07 Thin cells

- **Decision.** `IsThin = ClaimCount < min_claims` (default 30). Only thin
  categorical main-effect levels raise a warning, which names the levels
  without any claim separately. The flag stays a volume flag; the
  precision of a factor is in its standard error (RT-12).
- **Why.** A GLM applies no shrinkage to a categorical level, so its factor
  rests on those few claims. Thin tail points of a continuous grid are
  normal and would make the warning noise. A level without claims gets a
  frequency factor that is the fit running towards zero, not an
  estimate.
- **Where.** `rating_table.py`, the thin-cell block of
  `make_rating_table()`.
- **Pinned by.** `test_rating_table.py`: `test_the_thin_cell_column_is_correct`.

#### RT-08 Terms are resolved to the column behind them

- **Decision.** `ns(LEEFTIJD, 4)`, `factor(BOEKJAAR)`, `I(KM^2)`,
  `log(KM)` and backticked names resolve to their data column. Inline
  `factor()` (patsy's `C()` after translation) makes a numeric column
  categorical through `xlevels`.
- **Why.** A tariff is per column, not per term.
- **Where.** `_utils.py`, `term_base_var()`; `rating_table.py`, the
  xlevels-by-column block of `make_rating_table()` (R's closure
  `xlev_map`), `var_is_cat()`.
- **Pinned by.** `test_rating_table.py`:
  `test_rating_table_builds_incl_inline_factor_boekjaar`,
  `test_i_terms_resolve_to_their_underlying_column`,
  `test_a_column_name_with_a_space_survives_the_whole_rating_table`.

#### RT-09 Small robustness choices

- **Decision.** Levels are written as R's `format(scientific = FALSE)`
  writes them; base rows keep column names as they are.
- **Why.** `as.character(1e5)` is `"1e+05"`, unusable as a tariff level and
  as a lookup key. In R, without `check.names = FALSE` a column called
  `AUTO GEWICHT` becomes `AUTO.GEWICHT` and every prediction fails. A pandas
  DataFrame never renames a column, so in Python that half holds by
  construction, and `Q("AUTO GEWICHT")` carries the name through patsy
  (PY-02).
- **Where.** `_utils.py`, `level_chr()`; `rating_table.py`,
  `base_df_for()`.
- **Pinned by.** `test_rating_table.py`:
  `test_round_large_levels_are_not_written_in_scientific_notation`,
  `test_a_column_name_with_a_space_survives_the_whole_rating_table`;
  `test_core.py`: `test_grid_helpers`.

#### RT-10 A level a model was not fitted on is an empty cell

- **Decision.** A level that is in `data` but not in a model's `xlevels`
  keeps its row: that model's factor is NaN there, and so is
  `Factor_Premium`, with one warning per model naming the cells. A base
  row the model does not know is an error, raised once before any factor
  is computed, since every factor is a ratio against it.
- **Why.** A level with exposure and no claims is the usual case: the
  frequency model knows it, the severity data has no row for it. Predicting
  the whole variable failed on that one level, and the variable then left
  the table with a warning, so the frequency factors of every other level
  went with it.
- **Where.** `rating_table.py`, `predict_rows()`, the base-row check and
  the NaN-cell warning in `make_rating_table()`; `_utils.py`,
  `predictable_rows()`.
- **Evidence.** The review's WA book: Waterstof with 121 exposure and no
  claims dropped `BRANDSTOF` from the table entirely.
- **Pinned by.** `test_rating_table.py`:
  `test_a_level_without_claims_does_not_drop_the_variable_from_the_table`.

#### RT-11 A term that bands a column is read on its bands

- **Decision.** When a model's factor levels are not values of the column
  behind them, as with `cut(LEEFTIJD, ...)` or an `ifelse()`, the variable
  is reported on those levels. Each level is put into the prediction frame
  as the most frequent value of the column within it, which falls in it by
  definition, and the volumes are counted on the evaluated term.
  `factor(BOEKJAAR)` needs none of this, since `"2021"` is a year.
- **Why.** Predicting on the label put `"(24,40]"` into `LEEFTIJD`, which
  gave a missing value in every row, the intercept included, because every
  prediction shares the base row.
- **Where.** `rating_table.py`, the decoder block of `make_rating_table()`,
  `to_model()`, `row_level()`, `main_block()`.
- **Evidence.** Intercept and every factor NA before; afterwards the
  reconstruction of `predict()` holds to 2e-16. *Parity:* table C of the
  parity test has a `cut()` term and matches R.
- **Pinned by.** `test_rating_table.py`:
  `test_cut_in_the_formula_does_not_give_a_table_full_of_na`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`.

#### RT-12 Standard errors of the factors

- **Decision.** `SE_Frequency` and `SE_Severity` hold the standard error of
  `log(Factor)` in each row, `sqrt(d' V d)` with `d` the difference of the
  row's and the base's model-matrix rows (`GLMFit.model_matrix()`) and `V`
  the model's covariance matrix (`GLMFit.vcov`). 0 at the base and for a
  variable the model does not use, NaN where there is no factor or the link
  is not log. On an interaction row it belongs to the joint relativity
  against the base cell.
- **Why.** The model matrix is built exactly as `predict()` builds it, so
  this holds for splines, inline factors and interactions alike. A volume
  flag cannot say how precise a factor is: a level with 31 claims passes
  `min_claims = 30` with a standard error of 0.18, a 95% interval of x0.70
  to x1.42.
- **Where.** `rating_table.py`, `se_of()`.
- **Evidence.** For a categorical level against the reference the column
  equals the coefficient's own standard error (0.07675 for one region);
  Waterstof on the review's book: 31 claims, estimate 1.167 against a true
  1.000, `IsThin = FALSE`. *Parity:* `SE_*` 1e-9 from R.
- **Limitation.** The covariance of a Poisson model assumes a dispersion of
  1; fit a quasipoisson for standard errors that carry the overdispersion.
- **Pinned by.** `test_rating_table.py`:
  `test_the_rating_table_carries_standard_errors_of_its_factors`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`.

#### RT-13 Claims without payment are pointed out

- **Decision.** When the frequency response counts `claims_col` and at
  least 1% of those claims sit in records with `loss_col = 0`, a warning
  says that frequency x severity overstates the risk premium by about that
  share. Raised by `make_rating_table()` and `model_lift()`.
- **Why.** A Gamma severity cannot take a zero, so the zero amounts leave
  the severity data while the frequency model still counts them. Only
  records whose every claim is nil are visible, so the share is a lower
  bound.
- **Where.** `_utils.py`, `nil_claims_warning()`.
- **Evidence.** With 15% nil claims: predicted over actual losses 1.153.
- **Pinned by.** `test_rating_table.py`: `test_claims_without_payment_are_pointed_out`;
  `test_lift.py`: `test_claims_without_payment_are_pointed_out`.

#### RT-14 A logical predictor is a two-level categorical variable

- **Decision.** A logical column is reported as a categorical variable
  with the levels `FALSE` and `TRUE`, `FALSE` first, so it is the base by
  default (`base_level="exposure"` and `base` choose as for any factor).
  Its grid values go into `predict()` as `True` / `False`: `coerce_like()`
  returns a nullable boolean and `GLMFit._align()` turns it into plain
  booleans patsy can read.
- **Why.** The model codes a logical as `GARAGE[T.True]` against `False`
  (R: `GARAGETRUE`), so `FALSE` is the model's reference. R's `xlevels`
  leave a logical out (and so does the xlevels-by-column block here, to
  read a term as R does), and reading the column as continuous made the
  base the most frequent value, wrote the levels to Excel as numbers and
  drew the rating plot as a line between FALSE and TRUE.
- **Where.** `rating_table.py`, `var_is_cat()`, the xlevels-by-column block
  of `make_rating_table()`, `grid_of()`; `_utils.py`, `coerce_like()`;
  `model.py`, `GLMFit._align()`.
- **Pinned by.** `test_rating_table.py`: `test_a_logical_predictor_does_not_break_the_table`,
  `test_a_logical_predictor_is_a_two_level_rating_factor_with_false_as_base`;
  `parity/test_parity_rating_table.py`: `test_rating_table_parity_covers_what_it_should`
  (table C holds a logical predictor).
- **History.** Until November 2026 both packages read a logical as
  continuous; found while porting, fixed in both.

#### RT-15 The accounting year sits at its latest level

- **Decision.** When a model uses `year_col` (default `"BOEKJAAR"`), its
  default base is the latest year, for a `factor(BOEKJAAR)` as for a
  numeric trend term. `base` still overrides it; `year_col=None` returns
  to the general rule.
- **Why.** The intercept is then the current price level instead of the
  oldest one, and the year factors say how much cheaper the past was. The
  other factors do not change.
- **Where.** `rating_table.py`, `base_value_raw()`, `grid_of()`,
  `latest_of()`.
- **Evidence.** On the review's WA book the premium intercept was 207.2 at
  the first year and 508.9 at the latest (with the region base moving as
  well).
- **Pinned by.** `test_rating_table.py`:
  `test_the_accounting_year_sits_at_its_latest_level_by_default`,
  `test_intercept_is_per_unit_of_exposure`;
  `parity/test_parity_rating_table.py`: `test_rating_table_parity_covers_what_it_should`.

#### RT-16 Trimming leaves the interaction slices alone

- **Decision.** `trim` keeps policies outside the trimmed range out of the
  volumes of a grid that spans that range (RT-05), but not out of the three
  slices of a continuous group variable in an interaction.
- **Why.** The slices stand for the whole distribution. Cutting at the
  outer slices threw away everything below the 10% and above the 90%
  quantile.
- **Where.** `rating_table.py`, `cell_index()` (`drop_outside=False`) in
  `interaction_block()`.
- **Evidence.** The review's book with `trim = c(0.005, 0.995)`: 24,815
  exposure in the main effect against 19,992 in the interaction cells.
- **Pinned by.** `test_rating_table.py`:
  `test_trim_does_not_take_exposure_out_of_the_slices_of_a_group_variable`.

### MV: merged and grouped columns

#### MV-01 One tariff field under two column names becomes one block

- **Decision.** `merge_vars={"Stedelijkheid": "Stedelijkheid_groep"}`
  reports the grouped column inside the block of its source. Every
  prediction frame moves the source and all its groupings together.
- **Why.** Without it the table is correct but holds two blocks, each
  padded with neutral factors, and anyone building the tariff from one
  block silently drops the other.
- **Where.** `rating_table.py`, `set_var()`, `canon()`, `spellings()`,
  `alias_values()`.
- **Evidence.** One policy in class 5: 506.20 with both blocks or with the
  merged block, 401.76 with the source block alone (20.6% too low).
- **Pinned by.** `test_rating_table.py`:
  `test_two_columns_for_one_rating_factor_can_be_merged_into_one_block`;
  `parity/test_parity_rating_table.py`: `test_rating_table_parity_covers_what_it_should`.

#### MV-02 A grouping must be a function of its source

- **Decision.** Every source value maps to exactly one group, checked in
  `data` and in each model's own data.
- **Why.** If a value occurs with two groups there is no single factor for
  it, and taking the first would be wrong for part of the book.
- **Where.** `rating_table.py`, the `merge_vars` loop of
  `make_rating_table()`; `_utils.py`, `grouping_column()`.
- **Pinned by.** `test_rating_table.py`: `test_a_merge_declaration_that_cannot_hold_is_refused`.

#### MV-03 A grouped column missing from the data is derived from the model

- **Decision.** When `data` lacks the grouped column, it is derived from
  the data of the model that uses it, which every `GLMFit` carries
  (`m.data`). The caller's data frame is not changed.
- **Why.** The grouping is made where it is needed, next to the average
  claim, so it lives in the severity data and nowhere else. Asking the user
  to rebuild it for every dataset invites exactly the inconsistencies MV-04
  guards against.
- **Where.** `_utils.py`, `grouping_column()`; `rating_table.py`, the
  `merge_vars` loop of `make_rating_table()`.
- **Pinned by.** `test_rating_table.py`:
  `test_a_regrouped_column_missing_from_data_is_taken_from_the_model`;
  `test_groupings.py`: `test_the_grouped_columns_are_derived_from_the_source_columns`;
  `test_core.py`: `test_grouping_column_derived_and_checked`.
- **History (R package).** Added in commit `2779b8b`; shared with
  `add_groupings()` since `094d345`.

#### MV-04 One column name may not hold two groupings

- **Decision.** When both models use a column of the same name, the
  mapping each one's data defines is compared, and a difference is an error
  naming the value it turns on.
- **Why.** A grouping made twice need not agree, and a scored dataset can
  hold only one of the two. With different labels the result is a
  prediction failure; with the same labels nothing fails and one model is
  read on the other's grouping.
- **Where.** `_utils.py`, `grouping_column()`.
- **Evidence.** On a bench of ten scenarios, before this check: one brand's
  severity factor 36% out (same labels G1/G2, different membership) and the
  ages between two cut points 16% out (same labels LAAG/HOOG, cuts at 40
  and 50). Both silent. Six sound scenarios, including a spline against
  bands and an interaction, reconstruct the models to machine precision
  before and after.
- **Pinned by.** `test_rating_table.py`:
  `test_one_column_name_with_two_groupings_is_refused_not_averaged_over`,
  `test_two_bandings_of_one_continuous_variable_are_kept_apart`;
  `test_groupings.py`: `test_one_name_grouped_two_ways_by_the_two_models_is_refused`.
- **History (R package).** Added in commit `9245289`. Since `094d345` the
  message names both models rather than `'data'` and one model.

#### MV-05 A grouped column already present is checked, never overwritten

- **Decision.** If the data has the grouped column, every row is compared
  with what the model's data says, and a disagreement is an error with the
  row and the values. A source value the model's data never saw cannot be
  compared, and keeps the caller's value (PY-07).
- **Why.** A wrong mapping in a source system would otherwise be scored.
- **Where.** `_utils.py`, `grouping_column()`.
- **Pinned by.** `test_groupings.py`: `test_anything_that_would_be_a_guess_is_refused`,
  `test_a_grouped_column_already_in_data_is_kept_for_a_value_the_model_never_saw`
  (the check on the values the model's data does know);
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r` (table D).

#### MV-06 Categorical or continuous source

- **Decision.** Read off the models first: a factor term on the source
  means categorical, a numeric term means continuous. A source no model
  uses directly falls back on its own column: numeric with more than 25
  distinct values is continuous, anything else categorical.
- **Why.** A numeric code (urbanisation 1 to 5) must be looked up exactly,
  so an unknown code is an error and not a neighbour's group. The
  threshold of 25 is the same `discrete_cutoff` the one-way plots use.
- **Where.** `_utils.py`, `source_kind()`.
- **Pinned by.** `test_groupings.py`: `test_anything_that_would_be_a_guess_is_refused`
  (numeric code 6).
- **Limitation.** A genuine measurement with at most 25 distinct values,
  used only through a band, is treated as a code. See section 5.

#### MV-07 What happens to a value the model's data never saw

- **Decision.** Categorical source: error, with the values. Continuous
  source inside the fitted range: the band of the nearest observed value.
  Continuous source outside the range: the outer band, with a warning; if
  the same source is also a continuous term of a model, the warning names
  that model's extrapolation and the capping remedy.
- **Why.** The outer bands of a banding are open in practice; nobody over
  80 in the claims data does not make 85 a new class. The real cost of
  going outside the range is in a spline, which `predict()` extrapolates
  linearly on the log scale without a word, and no grouping can prevent
  that.
- **Where.** `_utils.py`, `grouping_column()`; `rating_table.py`,
  `alias_values()` for grid points.
- **Pinned by.** `test_groupings.py`:
  `test_a_continuous_source_outside_the_fitted_range_takes_the_outer_band`;
  `test_rating_table.py`: `test_a_band_of_a_continuous_variable_can_be_merged_into_it`,
  `test_a_band_missing_from_data_is_derived_per_nearest_observed_value`.

#### MV-08 An undeclared regrouping across the two models is pointed out

- **Decision.** When one column is a function of another and each is used
  by one model only, a warning suggests the `merge_vars` line. Skipped when
  both columns are numeric, when either has fewer than two or more than
  1000 distinct values, or when both have the same number of values.
- **Why.** The two-block table is right but misleading. Only across models:
  two nested factors inside one model (a postcode and its region) are
  genuinely two rating factors.
- **Where.** `rating_table.py`, the regrouping-hint block of
  `make_rating_table()`.
- **Pinned by.** `test_rating_table.py`:
  `test_two_columns_for_one_rating_factor_can_be_merged_into_one_block`
  (the warning when undeclared).

#### MV-09 What a merged column may not do

- **Decision.** A merged column cannot be named in `base` (set it on the
  source; both move together and the base row stays a real policy). An
  interaction on it is reported under the source's name; an interaction
  between a column and its own grouping disappears. Chains (a grouping of
  a grouping) are refused. A declaration that neither model uses is a
  warning.
- **Where.** `rating_table.py`, `base_value_raw()`, the `base` and
  `merge_vars` validation in `make_rating_table()`, `gather_pairs()`.
- **Pinned by.** `test_rating_table.py`:
  `test_an_interaction_on_a_merged_column_is_reported_under_one_name`,
  `test_a_merge_declaration_that_cannot_hold_is_refused`.

#### MV-10 add_groupings() is the same derivation for data to be scored

- **Decision.** `add_groupings(data, model_freq, model_sev, merge_vars)`
  runs MV-02 to MV-07 on any dataset and returns it with the grouped
  columns added, typed and levelled as in the model's data (a pandas
  Categorical where that data holds one). The input is copied, not
  modified, and comes back as a pandas DataFrame whatever table type went
  in (CONV-02).
- **Why.** `predict()` needs every grouped column; one declaration then
  serves the rating table and every scored dataset, and the two cannot
  disagree about which policy sits in which group.
- **Where.** `groupings.py`, `add_groupings()`; `_utils.py`,
  `grouping_column()`.
- **Pinned by.** `test_groupings.py`, all tests; the first one,
  `test_groupings.py`: `test_the_grouped_columns_are_derived_from_the_source_columns`,
  checks that `predict()` after `add_groupings()` equals the rating table
  rebuilt with the same declaration, for every policy.

### SCORE: data a model was not fitted on

#### SCORE-01 A value the model never saw is named before predict()

- **Decision.** Before scoring, every factor term of the model is evaluated
  on the data and compared with its `xlevels` (`unseen_levels()`). A value
  the model does not know stops the function with the column, the values
  and their row count, and the advice to map them or leave the rows out;
  where the function has `at`, also to price a later year at the latest
  known one. Used by `model_rate()` (so `premium_impact()`, `model_lift()`,
  `double_lift()`) and by `plot_glm_predictor(data=...)`. The rating table
  turns the same check into empty cells (RT-10).
- **Why.** `predict()` raised "factor has new levels" from inside the
  offset evaluation, and the message then blamed the offset. Python would
  misdirect the same way without the check: `model_rate()` reports any
  failure of the prediction as an offset that could not be evaluated.
- **Where.** `_utils.py`, `unseen_levels()`, `unseen_message()`,
  `model_rate()`, `glm_newdata_parts()`.
- **Evidence.** The review: scoring 2025 gave "the model has an offset that
  could not be evaluated on this data ... factor factor(BOEKJAAR) has new
  level 2025".
- **Pinned by.** `test_impact.py`:
  `test_an_unknown_level_at_scoring_is_not_reported_as_an_offset_problem`,
  `test_at_prices_every_policy_at_one_accounting_year`;
  `test_agg_plots.py`: `test_unusable_monitoring_data_is_refused_rather_than_guessed_at`;
  `test_core.py`: `test_unseen_levels_named`.

#### SCORE-02 Scoring outside the fitted range is said once

- **Decision.** A numeric predictor with values outside the range of the
  model's training data raises one warning per call, with the row count
  and the range. A column the model turns into a factor is left to
  SCORE-01.
- **Why.** `predict()` extrapolates without a word: a spline linearly on
  the link scale beyond its last knot, a log term without limit. The
  Python bases extrapolate exactly as R's do (PY-03).
- **Where.** `_utils.py`, `range_warning()`, `model_rate()`.
- **Pinned by.** `test_impact.py`: `test_scoring_outside_the_fitted_range_is_pointed_out`.

### RATE: premium impact and lift

#### RATE-01 A rate is the product of the models, per unit of exposure

- **Decision.** Rate = frequency rate x severity rate, offsets handled per
  OFF-05; a model that is not supplied contributes 1.
- **Where.** `_utils.py`, `model_rate()`, `old_new_rates()`.

#### RATE-02 An old premium column

- **Decision.** `old_premium_basis="amount"` divides by `exposure_col`
  (an earned premium for the record); `"rate"` takes it as a premium per
  unit of exposure. It must be on the footing of the kept offsets (OFF-05).
  `double_lift()` divides by its weight column instead, which is the
  exposure unless both sides are severity models (RATE-12), so that rate x
  weight gives the amount back on both sides. An old premium column is
  read as it is: `at` (RATE-10) only changes what the models score.
- **Where.** `_utils.py`, `old_new_rates()` (`divisor_col`).
- **Pinned by.** `test_impact.py`: `test_rebase_separates_rate_level_change_from_dislocation`.

#### RATE-03 Rebase and rate-level change

- **Decision.** With `rebase=True` the new rates are scaled so the
  exposure-weighted totals match. The rate-level change is always measured
  before scaling.
- **Why.** That separates the two questions a dislocation analysis answers:
  the overall level and the redistribution. Deriving the level change from
  the scale factor would report 0 exactly when `rebase=False`.
- **Where.** `impact.py`, `premium_impact()`; `lift.py`, `double_lift()`.
- **Pinned by.** `test_impact.py`: `test_rebase_separates_rate_level_change_from_dislocation`;
  `test_lift.py`: `test_the_rate_level_change_is_reported_whether_or_not_we_rebase`.

#### RATE-04 Contribution, not mix

- **Decision.** Per level of a `by` column:
  `Contribution = 100 x (NewPremium - OldPremium) / old total premium`,
  which is `PremiumShare x TotalChangePct`. The contributions of one
  variable sum to the change in the total premium: exactly 0 after a
  rebase, the rate-level change without one. `MeanChangePct` (what a
  policy in the level sees on average) is reported next to it, and so are
  `OldPremium`, `NewPremium`, `PremiumShare` and `TotalChangePct`. The
  spotlight's contribution is computed the same way.
- **Why.** Old and new are compared on the same policies, so there is no
  mix effect to isolate; contribution answers which segment drives the
  change. A segment's part of the total is its premium change, not its
  exposure times its mean percentage, because a percentage on a cheap
  policy moves less money than the same percentage on an expensive one.
- **Where.** `impact.py`, `premium_impact()` (the `by_level` and spotlight
  blocks).
- **Evidence.** On the review's WA book after a rebase (total premium
  change +0.000%), the old definition summed to -4.008% for every `by`
  variable, the exposure-weighted mean change. Per region the two
  definitions disagreed by up to 2.1 points (Randstad +2.715 against
  +4.777).
- **Pinned by.** `test_impact.py`: `test_contributions_sum_to_the_portfolio_change`,
  `test_premium_impact_contributions_add_up_to_the_change_in_total_premium`;
  `parity/test_parity_impact_lift.py`: `test_premium_impact_parity`.
- **History (R package).** Until the review fixes of October 2026,
  `Contribution = ExposureShare x MeanChangePct`. That sums to the
  exposure-weighted mean change, which equals the change in total premium
  only when the old premium is flat per unit of exposure.

#### RATE-05 Spotlight, dropped rows, histogram

- **Decision.** A spotlight is evaluated against `data` or taken as a
  boolean array or Series, and reported after the book-level rebase. In
  Python the expression form is a string with an R-style condition on the
  data's columns, which cannot refer to the caller's own variables (section
  4). Rows without a usable new or old rate, a positive weight or an actual
  value are dropped with a warning that counts them per reason
  (`drop_reasons()`), in `premium_impact()`, `model_lift()` and
  `double_lift()`. A row can count under more than one reason. The
  histogram is pre-binned into 60 bars.
- **Why.** The spotlight shows how a segment moves relative to the book.
  One count for every reason reported a missing loss amount as an exposure
  problem. Pre-binning keeps the figure small: a plotly histogram would
  embed every policy.
- **Where.** `impact.py`, `premium_impact()`; `lift.py`, `model_lift()`,
  `double_lift()`; `_utils.py`, `drop_reasons()`.
- **Pinned by.** `test_impact.py`:
  `test_the_spotlight_reports_a_subset_without_filtering_the_book`,
  `test_spotlight_input_is_validated`,
  `test_histogram_is_pre_binned_into_60_bars`,
  `test_dropped_rows_are_counted_per_reason`; `test_lift.py`:
  `test_dropped_rows_are_reported_per_reason`.

#### RATE-06 Lift bins hold equal exposure

- **Decision.** Policies are ordered by predicted rate and cut at equal
  cumulative weight. Fewer distinct predictions than bins raises a warning;
  ties are counted at 12 significant digits. Bins are numbered from 1, as
  in R.
- **Why.** Equal policy counts would give bins unequal weight. Ties are
  counted at a tolerance because removing an offset leaves rounding noise
  of order 1e-16, which would otherwise make equal predictions look
  distinct.
- **Where.** `_utils.py`, `exposure_bins()`.
- **Evidence.** *Parity:* the lift tables match R to 1e-10 with continuous
  predictors; with many exactly tied predictions a bin boundary can fall on
  another policy than in R (section 4).
- **Pinned by.** `test_lift.py`: `test_bins_hold_roughly_equal_exposure`;
  `test_core.py`: `test_exposure_bins_and_gini`.

#### RATE-07 The lift weight follows the model

- **Decision.** `weight_col=None` resolves to `claims_col` for a
  severity-only lift and `exposure_col` otherwise.
- **Why.** A severity predicts a mean per claim; weighting it by exposure
  multiplies a per-claim amount by a per-year one and the totals do not
  reconcile.
- **Where.** `lift.py`, `model_lift()`.
- **Pinned by.** `test_lift.py`: `test_a_severity_only_lift_weights_by_claims_not_exposure`.

#### RATE-08 Gini and the units check

- **Decision.** The Gini is computed on the exposure-weighted Lorenz curve
  of actual losses ordered by prediction, and is missing (NaN) without
  losses. An overall A/E outside 0.5 to 2 raises a warning about mismatched
  units.
- **Why.** The Gini measures ordering only, not calibration. A wildly off
  A/E nearly always means counts were compared with amounts.
- **Where.** `_utils.py`, `gini_exposure()`, `gini_ordered()`,
  `check_ae_scale()`.
- **Evidence.** *Parity:* Gini 4e-14 from R.
- **Pinned by.** `test_lift.py`:
  `test_a_portfolio_without_losses_gives_na_rather_than_nan_for_the_gini`
  (R's name; Python has one missing value, NaN),
  `test_mismatched_units_are_caught`; `test_core.py`:
  `test_exposure_bins_and_gini`.

#### RATE-09 Double lift

- **Decision.** Bins are ordered by the new/old rate ratio. Each model is
  scored by the mean absolute distance of its A/E from 1 across the bins
  (`mad_new`, `mad_old`). `winner` is `"tie"` when the two are equal;
  otherwise it names a model only when the bootstrap interval of
  `mad_old - mad_new` (RATE-11) excludes 0, and is `"inconclusive"` when
  it does not. With `n_boot=0` the point values decide, as they did
  before the interval existed.
- **Why.** The point comparison is decided by a few large claims: on the
  review's 16,000-policy holdout it said "old" on the losses as recorded
  (0.543 against 0.405) and "new" on the same losses capped at 50,000
  (0.222 against 0.271). A verdict that flips with the treatment of ten
  claims is not a verdict.
- **Where.** `lift.py`, `double_lift()`.
- **Evidence.** *Parity:* `mad` 3e-10 from R with `n_boot=0`.
- **Pinned by.** `test_lift.py`:
  `test_double_lift_prefers_the_model_that_is_right_where_they_differ`,
  `test_double_lift_against_itself_shows_no_winner_and_a_flat_ae`,
  `test_the_double_lift_names_a_winner_only_when_the_interval_excludes_0`;
  `parity/test_parity_impact_lift.py`: `test_double_lift_parity`.
- **History (R package).** Until the review fixes of October 2026 the
  winner was the smaller point value, always.

#### RATE-10 Pricing at one accounting year

- **Decision.** `premium_impact()`, `model_lift()` and `double_lift()` take
  `at`, a dict of values every policy is priced at, for example
  `at={"BOEKJAAR": 2024}`. It is applied to the data the models score
  and never to an old premium column. Each value must be valid for its
  column (`coerce_like()`); a level the model does not know is refused by
  SCORE-01. `premium_impact()` warns when the new models use `year_col`
  against an old premium column, or when only one of two model sets uses
  it, and `at` does not set it. The lifts do not warn.
- **Why.** With the accounting year as a rating factor, every policy is
  priced at the level of its own historic year, and against a current
  premium that mixes inflation into the dislocation. Two model sets that
  both use the year price each policy at the same year on both sides,
  which is like for like. A lift compares with the claims of each
  policy's own year, which is exactly what pricing at its own year does.
- **Where.** `_utils.py`, `apply_at()`, `year_warning()`,
  `old_new_rates()`; `impact.py`, `premium_impact()`; `lift.py`,
  `model_lift()`.
- **Evidence.** On the review's WA book, with `BOEKJAAR` in the frequency
  model: a rate-level change of +11.1% with every policy at its own year,
  +34.7% with every policy at 2024.
- **Pinned by.** `test_impact.py`:
  `test_at_prices_every_policy_at_one_accounting_year`,
  `test_rebase_separates_rate_level_change_from_dislocation` (the warning);
  `test_lift.py`: `test_at_prices_every_policy_for_the_lift_too`;
  `test_core.py`: `test_apply_at_validates`.

#### RATE-11 Bootstrap intervals for the Gini and the double-lift verdict

- **Decision.** `model_lift()` reports `gini_ci`, a bootstrap interval for
  the Gini (`n_boot=200`, `conf_level=0.90`), and `double_lift()` one
  for `mad_old - mad_new`. Both resample policies with Poisson(1) counts,
  the large-sample stand-in for resampling with replacement, so the book is
  sorted once (`gini_boot()`) or the bins are kept as cut on the full data
  and every replicate reweights the policies within them (rebased again
  when `rebase=True`). The counts are drawn from the call's numpy
  Generator (PY-05). The interval is in the title and the return value.
  A warning is raised only when large losses are the reason: the ten
  largest records carry 10% or more of `actual_col` (`large_loss_note()`)
  and, for the Gini, the interval is wider than half the Gini and than
  0.05; for the double lift, the verdict is inconclusive.
- **Why.** A Gini on raw losses moves with a handful of large claims, and
  so does the double-lift verdict (RATE-09). Capping the losses is the
  remedy for that, and the warning says so. A wide interval on a small book
  has no such remedy; warning about it fired on an ordinary frequency lift
  of 20,000 policies and was noise.
- **Where.** `_utils.py`, `gini_boot()`, `gini_ordered()`, `top_share()`,
  `large_loss_note()`, `check_boot()`; `lift.py`, `model_lift()`,
  `double_lift()`.
- **Evidence.** The review's 16,000-policy holdout: Gini 0.230, 90%
  interval 0.143 to 0.326 (0.133 to 0.302 for the true risk premium), so
  the model and the truth could not be told apart on it. The Python draws
  differ from R's for the same seed, so its interval differs by the
  bootstrap's own noise.
- **Limitation.** The interval resamples policies only. The models are
  held fixed, so an in-sample interval ignores that they were fitted on
  the same rows.
- **Pinned by.** `test_lift.py`:
  `test_model_lift_reports_a_bootstrap_interval_around_the_gini`,
  `test_large_losses_that_decide_the_gini_or_the_verdict_are_named`,
  `test_the_double_lift_names_a_winner_only_when_the_interval_excludes_0`.

#### RATE-12 The double-lift weight follows the models

- **Decision.** `double_lift()` resolves `weight_col=None` the way
  `model_lift()` does (RATE-07): `claims_col` when both sides are severity
  models, otherwise `exposure_col`. The bins hold equal weight, the
  `Exposure` column of the table holds the weight, and an old premium
  amount is divided by it (RATE-02). A severity model on one side and a
  frequency or risk premium on the other is refused.
- **Why.** A severity predicts a mean per claim. Weighted by exposure, the
  predicted and actual totals do not reconcile, and the A/E per bin says
  more about claim frequency than about severity. The ratio of a rate per
  claim to a rate per exposure unit orders the book by neither tariff.
- **Where.** `lift.py`, `double_lift()`.
- **Evidence.** Two severity models on the review's claim data, weighted
  by exposure: the old model's A/E per bin ran from 0.76 to 1.50.
- **Pinned by.** `test_lift.py`:
  `test_a_double_lift_between_two_severity_models_is_weighted_by_claims`,
  `test_a_severity_model_against_a_frequency_model_is_refused`.

### AE: actual versus expected

#### AE-01 Counts mode and weighted-mean mode

- **Decision.** For a log-link model with a log offset, response and
  prediction are counts, divided by the exposure per bin. Otherwise both are
  prior-weight-weighted means.
- **Where.** `ae.py`, `plot_glm_predictor()`.

#### AE-02 Binning

- **Decision.** A numeric predictor with at most `n_bins` distinct values
  is not binned; above that, quantile bins (or equal width), each point at
  the weight-weighted mean of its bin.
- **Why.** Binning a few values merges neighbours, and a single outlier
  would move every equal-width bin.
- **Where.** `ae.py`, `plot_glm_predictor()`.
- **Pinned by.** `test_agg_plots.py`: `test_fewer_distinct_values_than_n_bins_means_no_binning_at_all`.

#### AE-03 Error bars on the observed point only

- **Decision.** The interval is on the observed point, from the family's
  variance function times the Pearson dispersion; Poisson and binomial hold
  the dispersion at 1.
- **Why.** The predicted line is the model; its parameter uncertainty is a
  different and smaller quantity. Holding the dispersion at 1 for Poisson
  is the assumption being examined.
- **Evidence.** Measured coverage on a correctly specified Poisson: 0.96
  for a binned continuous predictor, 0.98 for a categorical one outside the
  model. *Parity:* `plot_glm_predictor` 1e-9 from R.
- **Where.** `ae.py`, `plot_glm_predictor()`.
- **Pinned by.** `test_agg_plots.py`:
  `test_observed_points_carry_error_bars_that_match_the_theory`,
  `test_the_ci_switches_behave`; `parity/test_parity_plots.py`:
  `test_plot_glm_predictor_parity`.

#### AE-04 A categorical term in the model is pinned in-sample

- **Decision.** Documented, not corrected: in-sample, under a canonical
  link, observed equals predicted at every level of a categorical term in
  the model. The bar length still says how much evidence the level carries.
- **Where.** `ae.py`, the docstring of `plot_glm_predictor()`; see INT-01
  for the derivation.

#### AE-05 Monitoring on new data

- **Decision.** With `data`, the comparison runs on those rows. The
  response, weights and offsets are re-evaluated on them; rows with a
  missing value in a model variable are dropped as `fit_glm()` drops them;
  a level the model never saw is an error that names it (SCORE-01); the
  dispersion behind the bars is the fitted model's.
- **Why.** Recycling the fitted weights would pair one period's claim
  counts with another's rows.
- **Where.** `_utils.py`, `glm_newdata_parts()`; `ae.py`,
  `plot_glm_predictor()`.
- **Pinned by.** `test_agg_plots.py`:
  `test_plot_glm_predictor_evaluates_another_period_when_data_is_given`,
  `test_plot_glm_predictor_on_new_data_reevaluates_weights_and_offsets`,
  `test_unusable_monitoring_data_is_refused_rather_than_guessed_at`.

#### AE-06 The A/E ratio itself

- **Decision.** `ratio=True` draws one A/E line with a reference at 1;
  the bars are the observed bars divided by the prediction.
- **Where.** `ae.py`, `plot_glm_predictor()`.
- **Pinned by.** `test_agg_plots.py`: `test_ratio_true_plots_ae_against_a_reference_line_at_1`.

#### AE-07 A multi-column term is not a predictor

- **Decision.** Asking for `"ns(LEEFTIJD, 4)"` is an error that names the
  column to ask for instead.
- **Where.** `ae.py`, `plot_glm_predictor()`.
- **Pinned by.** `test_agg_plots.py`: `test_a_multi_column_model_term_is_refused_with_a_usable_message`.

### INT: interactions

#### INT-01 Why a one-way check cannot see an interaction

- **Decision.** The interaction tools look at cells, never at margins.
- **Why.** Under a canonical link the score equations force observed to
  equal predicted at every level of a categorical term in the model, so a
  Poisson frequency model is blind by identity. A Gamma log-link severity
  is not canonical, but its score pins the weighted mean of `y / mu` to 1,
  which is nearly the same: an omitted interaction of 35% moves the one-way
  A/E by 0.26%.
- **Where.** `interactions.py`, module docstring.
- **Pinned by.** `test_interactions.py`:
  `test_one_way_ae_is_blind_to_the_missing_interaction`,
  `test_each_link_pins_a_different_quantity_to_exactly_1`,
  `test_the_severity_residue_is_far_too_small_to_price_on`.

#### INT-02 The heatmap

- **Decision.** A/E per cell is `sum(actual) / sum(expected)`. Cells with
  fewer than `min_claims` claims are left uncoloured and marked. The scale
  diverges around 1 with a neutral midpoint. When every cell is thin, the
  colour layer is left out, with a warning, and the crosses with their
  tooltips are the plot.
- **Why.** A thin cell's A/E is noise, and colouring it would make it the
  loudest thing on the plot. R's plotly cannot build a heatmap whose values
  are all `NA`, whatever its colour range: it failed with "Wasn't able to
  determine range of domain", and on a 400-policy book that took the whole
  report down. The Python port keeps the same fallback, so the two packages
  draw the same plot.
- **Where.** `interactions.py`, `plot_residual_heatmap()`,
  `_ta_diverging`.
- **Evidence.** *Parity:* heatmap cells 1e-14 from R.
- **Pinned by.** `test_interactions.py`:
  `test_plot_residual_heatmap_returns_a_plot_plus_its_cell_table`,
  `test_thin_cells_are_not_coloured_and_z_range_is_validated`,
  `test_a_heatmap_with_every_cell_thin_still_builds`;
  `parity/test_parity_diagnostics.py`: `test_residual_heatmap_cells`.

#### INT-03 The interaction statistic

- **Decision.** Per pair, expected is raked to actual's row and column
  margins (iterative proportional fitting, 50 iterations, tolerance 1e-9),
  and the likelihood-ratio deviance against the saturated table is the
  statistic. For Poisson counts the null comes from `n_sim` simulations
  under the raked expected, scaled for overdispersion (INT-05); otherwise
  from a chi-square scaled by the Pearson dispersion.
- **Why.** Raking removes main-effect misfit first, so what remains is
  interaction. The deviance is unreliable in sparse tables, hence the
  simulation where the response allows it.
- **Where.** `interactions.py`, `detect_interactions()`, `_rake()`,
  `_dev_of()`; `_utils.py`, `glm_ae_parts()` decides `poisson_counts`.
- **Evidence.** *Parity:* the deterministic columns of
  `detect_interactions` 8e-10 from R; the simulated `Z` and `P` differ by
  design (PY-05).
- **Pinned by.** `test_interactions.py`:
  `test_detect_interactions_ranks_the_true_pair_first`,
  `test_no_interaction_present_means_no_signal`,
  `test_a_poisson_model_without_an_offset_still_gets_the_simulated_null`;
  `parity/test_parity_diagnostics.py`: `test_detect_interactions_deterministic_columns`.

#### INT-04 Signal and materiality are reported separately

- **Decision.** Pairs are ranked by `Z`; `MaxAE` (worst non-thin cell),
  its claims and its exposure share are reported alongside.
- **Why.** Significant and material are different questions.
- **Where.** `interactions.py`, `detect_interactions()`.

#### INT-05 The simulated null carries the overdispersion

- **Decision.** For count models the deviance is divided by the dispersion
  of a cell total, `phi = sum((y - mu)^2) / sum(mu) x n / df`, and the
  simulation draws Poisson counts with means `E / phi`. A dispersion below 1
  is left at 1. The value is in the `Dispersion` column, and a message
  names it from 1.2 upwards.
- **Why.** A Poisson null has no room for overdispersion, so the extra
  variance read as interaction. A cell total of variance `phi x mean`
  behaves like a Poisson count after dividing by `phi`, and the deviance is
  homogeneous of degree one, so `D / phi` against a null on `E / phi` is
  the same test on rescaled counts. The `phi` is the mean-weighted
  Pearson, not the row-level one: a total adds rows by their mean, and
  under a gamma frailty the rows with a larger mean carry more of the extra
  variance.
- **Where.** `interactions.py`, `detect_interactions()`.
- **Evidence.** Before: ten books with a dispersion of 1.9 and no
  interaction, 33 of 60 pairs at `P < 0.05`. With the row-level Pearson
  (1.72 against 1.91 for the cell totals of random partitions of the same
  rows) still 8 of 60. With the cell-total dispersion, 100 books per case:
  10 of 300 pairs (3.3%) at a dispersion of 1.94, 20 of 300 (6.7%) on pure
  Poisson; an interaction of 0.15 on the log scale found in 83% of the
  books on pure Poisson and 48% at a dispersion of 2.0.
- **Limitation.** One dispersion for the whole book. Under a gamma frailty
  the cells with larger expected counts carry more overdispersion than the
  average, so the test is slightly liberal there and slightly conservative
  elsewhere.
- **Pinned by.** `test_interactions.py`: `test_the_interaction_scan_is_not_fooled_by_overdispersion`.

### PDP: partial dependence

#### PDP-01 An arithmetic mean on the response scale

- **Decision.** For each grid value, the whole training data is set to it,
  predicted on the response scale with the exposure offset removed
  (PDP-05), and averaged with weights.
- **Why.** `exp(mean(link))` is a geometric mean and biased low.
- **Where.** `pdp.py`, `make_pdp()`.
- **Evidence.** *Parity:* `make_pdp` 1e-9 from R.
- **Pinned by.** `test_pdp.py`: `test_make_pdp_computes_the_exposure_weighted_mean_at_exposure_1`;
  `parity/test_parity_plots.py`: `test_make_pdp_parity`.

#### PDP-02 The weights

- **Decision.** Frequency: `exposure_col` (falling back to `exp(offset)`),
  or the prior weights for a rate model, or unweighted with a warning.
  Severity: the prior weights, normally the claim count.
- **Where.** `pdp.py`, `make_pdp()`.

#### PDP-03 Collapse to profiles

- **Decision.** The data is collapsed to unique profiles of the other
  predictors with summed weights before predicting.
- **Why.** `grid_res x profiles` predictions instead of `grid_res x rows`,
  with an identical result.
- **Where.** `pdp.py`, `make_pdp()`.
- **Pinned by.** `test_pdp.py`: `test_make_pdp_grid_collapse_and_layout`
  (the collapse changes nothing).

#### PDP-04 Its own grid

- **Decision.** Factor levels, the unique values when there are at most
  `grid_res`, or an evenly spaced sequence over the range. Not the rounded
  grid of the rating table.
- **Where.** `pdp.py`, `make_pdp()`. See section 5.
- **Pinned by.** `test_pdp.py`: `test_make_pdp_grid_collapse_and_layout`.

#### PDP-05 The PDP keeps every offset except the exposure

- **Decision.** `keep_offsets=True` (default): only the offset
  expressions on `exposure_col` are subtracted, as in OFF-05, and any other
  offset stays in at each policy's own value. The variables of the
  subtracted offsets are pinned to 1 for the collapse to profiles; the kept
  ones become part of the profile. `False` subtracts everything. The
  default y-axis title names the offsets kept.
- **Why.** The observed line is at each policy's own BM. A PDP at BM = 1
  against it showed a gap on a model that was exactly right.
- **Where.** `pdp.py`, `make_pdp()`.
- **Evidence.** The review's book with an average BM of 0.74: the PDP ran
  1.347 to 1.359 times the observed line at every level before; on the
  regression test it now sits within 5% of it.
- **Pinned by.** `test_pdp.py`:
  `test_the_pdp_sits_on_the_observed_line_with_a_bonus_malus_offset`,
  `test_a_second_offset_does_not_end_up_on_the_exposure_axis_pdp`.
- **History (R package).** Until the review fixes of October 2026 the PDP
  removed the whole offset (OFF-06).

### DIAG: diagnostics, Q-Q and influence

#### DIAG-01 Dispersion

- **Decision.** Pearson chi-squared over residual degrees of freedom; above
  1.2 for Poisson or binomial raises an overdispersion warning.
- **Where.** `diagnostics.py`, `glm_diagnostics()`.
- **Evidence.** *Parity:* `glm_diagnostics` 8e-11 from R.
- **Pinned by.** `test_diagnostics.py`: `test_overdispersion_raises_a_warning`;
  `parity/test_parity_diagnostics.py`: `test_glm_diagnostics_table`.

#### DIAG-02 Collinearity per term

- **Decision.** Generalised VIF per term (Fox and Monette), on
  log-determinants of the coefficient correlation matrix, scaled to
  `GVIF^(1/(2 DF))`, flagged at 3. Aliased coefficients are dropped with a
  warning. In Python `Flag` is a nullable boolean column.
- **Why.** A plain VIF is meaningless for a spline or a factor spanning
  several columns. Log-determinants because a spline's correlation matrix
  underflows to zero long before the model is singular.
- **Where.** `diagnostics.py`, `glm_collinearity()`.
- **Evidence.** *Parity:* `glm_collinearity` 2e-9 from R.
- **Pinned by.** `test_diagnostics.py`:
  `test_collinear_terms_are_flagged_and_independent_ones_are_not`,
  `test_a_clean_model_flags_nothing`,
  `test_multi_column_terms_use_the_generalised_form`,
  `test_a_model_with_fewer_than_two_terms_returns_nothing`;
  `parity/test_parity_diagnostics.py`: `test_glm_collinearity_table`.

#### DIAG-03 Binned residuals

- **Decision.** Mean residual per quantile bin of the fitted value or a
  predictor, with a band of `2 sqrt(sum of the row variances) / n`, which
  is `2 sqrt(phi / n)` for Pearson residuals; deviance residuals are
  centred first (DIAG-06). Points, never a line.
- **Why.** Bin means are independent draws; a line invites a trend.
- **Where.** `diagnostics.py`, `plot_glm_residuals()`.
- **Evidence.** *Parity:* `plot_glm_residuals` (Poisson) 1e-14 from R.
- **Pinned by.** `test_diagnostics.py`:
  `test_binned_residuals_are_drawn_as_points_not_a_line`,
  `test_binned_residual_plots_render_and_stay_within_the_band`;
  `test_agg_plots.py`: `test_binned_residuals_are_drawn_as_points_not_a_line`;
  `parity/test_parity_diagnostics.py`: `test_plot_glm_residuals_bins_and_bands`.

#### DIAG-04 Randomised quantile residuals for the Q-Q plot

- **Decision.** Dunn and Smyth (1996): each observation through its own
  fitted CDF, uniformly within the jump for a discrete response, then
  through the standard normal quantile function. Dispersion held at 1 for
  Poisson and binomial; estimated for Gamma and gaussian. Tails guarded at
  machine epsilon. A pointwise Beta band; at most `n_max` points drawn,
  extremes kept. Families without a closed-form CDF fall back to deviance
  residuals with a warning; a quasipoisson gets a caveat. The uniform draws
  come from the call's Generator (PY-05).
- **Why.** Classical residuals are not normal for counts.
- **Evidence.** On a correct Poisson with 20,000 rows: std. Pearson 8.34%
  outside a 5% band, std. deviance 1.31%, quantile residual 5.10% (KS
  p = 0.86). Identical to `statmod::qresid()` for Poisson, to 6e-14 for
  Gamma. *Parity:* Gamma quantile residuals 3e-9 from R (the continuous
  case, where nothing is random).
- **Where.** `qq_influence.py`, `_qr_dispersion()`,
  `_quantile_residuals()`, `plot_glm_qq()`.
- **Pinned by.** `test_qq_influence.py`:
  `test_quantile_residuals_are_standard_normal_on_a_correct_model`,
  `test_quantile_residuals_are_the_cdf_transform`,
  `test_the_qq_plot_flags_a_distribution_that_is_genuinely_wrong`,
  `test_a_family_without_a_closed_form_cdf_falls_back_and_says_so`,
  `test_a_quasipoisson_is_drawn_but_the_caveat_is_raised`;
  `parity/test_parity_diagnostics.py`: `test_quantile_residuals_and_qq_plot_of_a_gamma_model`.

#### DIAG-05 Influence is ranked, not thresholded

- **Decision.** Rows are ranked by Cook's distance and the top `n_label`
  labelled; `2p/n` is drawn as a leverage reference. Above `max_rows` (2e5)
  the model is refitted on a sample, with a message. The `Row` of the
  returned table is the data's index label.
- **Why.** On a large book `D > 1` never fires and `D > 4/n` flags
  thousands. Cook's distance cannot see a low-exposure keying error (an
  exposure of 0.001 with two claims: worst residual in the book, about 30th
  on Cook's D), so the plot has two axes.
- **Where.** `qq_influence.py`, `plot_glm_influence()`.
- **Evidence.** *Parity:* `plot_glm_influence` 3e-14 from R.
- **Pinned by.** `test_qq_influence.py`:
  `test_a_low_exposure_keying_error_is_invisible_to_cooks_distance`,
  `test_an_extreme_predictor_value_tops_the_ranking_and_masks_itself`,
  `test_plot_glm_influence_caps_the_row_count_with_a_message`;
  `parity/test_parity_diagnostics.py`: `test_plot_glm_influence_top_rows`.

#### DIAG-06 Deviance residuals are centred on their expectation

- **Decision.** With `residual_type="deviance"`, each row's residual is
  shown minus its expectation under the fitted model, and the band per bin
  uses the rows' own variances (`dev_resid_moments()`). The moments are
  exact sums over the response distribution, computed on a grid of the one
  parameter they depend on and interpolated: Poisson and quasipoisson on
  the mean count, negative binomial with unit weights, binomial for up to
  20 distinct trial counts, Gamma per prior weight (closed-form second
  moment, integrated first, with scipy's `quad` where R uses
  `integrate()`), Gaussian. Count families scale the variance by the
  Pearson dispersion. Other families are shown uncentred with a warning.
  The axis title says "centred".
- **Why.** A deviance residual is not centred: a claim-free row has
  `-sqrt(2 mu)`, so for small means the residuals are negative on average
  and a correct model sat below the band in every bin.
- **Where.** `_utils.py`, `dev_resid_moments()`; `diagnostics.py`,
  `plot_glm_residuals()`.
- **Evidence.** A correct Poisson book of 60,000 rows: mean binned deviance
  residual -0.220 against a band of 0.058, 50 of 50 bins outside it.
  *Parity:* Gamma deviance centring 6e-8 absolute from R, which is R's
  `integrate()` tolerance.
- **Pinned by.** `test_diagnostics.py`:
  `test_binned_deviance_residuals_of_a_correct_poisson_model_stay_in_the_band`;
  `parity/test_parity_diagnostics.py`: `test_plot_glm_residuals_bins_and_bands`.

#### DIAG-07 A term in an interaction is reported, not flagged

- **Decision.** `glm_collinearity()` marks every interaction term and every
  term whose variables appear in one (`InInteraction`), reports their GVIF
  and leaves `Flag` missing for them, with a warning naming them.
- **Why.** Their GVIF depends on how the variables are coded rather than on
  the data. Centring the numeric side of an interaction moves the
  coefficients of the other variable's main effect to another reference
  point, and with it their correlation with the interaction's. A term
  outside the interaction is unaffected, because its GVIF is invariant to
  any reparametrisation of the remaining terms.
- **Where.** `diagnostics.py`, `glm_collinearity()`.
- **Evidence.** The review: with `pmin(LEEFTIJD, 30):STEDELIJKHEID`,
  writing the age part as `pmin(LEEFTIJD, 30) - 30` moved the scaled GVIF
  of `STEDELIJKHEID` from 14.12 to 1.11, with the fitted values equal to
  2e-14.
- **Pinned by.** `test_diagnostics.py`:
  `test_a_gvif_that_depends_on_the_coding_is_not_flagged`,
  `test_splines_and_rank_deficiency_do_not_empty_the_table`.

#### DIAG-08 Negative binomial models

- **Decision.** A negative binomial fit gets randomised quantile residuals
  from the negative binomial CDF and centred deviance residuals, when its
  prior weights are 1. In R that is a fit from `MASS::glm.nb()` or a
  `glm()` with `MASS::negative.binomial(theta)`, with theta read from
  `model$theta` or the family's environment. In Python it is
  `fit_glm(family="negbin")`, with theta estimated as `glm.nb()` does or
  fixed with `theta=`, and read from `m.theta`.
- **Why.** In R its family name carries theta, "Negative Binomial(1.4)", so
  it fell through to the deviance fallback, which is not normal for counts.
  In Python the family is `"negbin"` and only the displayed `family_label`
  carries theta (PY-04), so the check is on the family itself.
- **Where.** `_utils.py`, `nb_theta()`; `qq_influence.py`,
  `_quantile_residuals()`.
- **Pinned by.** `test_qq_influence.py`:
  `test_a_glm_nb_model_gets_quantile_residuals_in_the_qq_plot`;
  `test_pdp.py`: `test_a_glm_nb_model_works_in_make_pdp`;
  `test_agg_plots.py`: `test_a_glm_nb_model_works_in_the_ae_plot`.

### SCR: feature screening

#### SCR-01 The booster is a diagnostic, never a pricing model

- **Decision.** xgboost is fitted with the GLM's log prediction as
  `base_margin`, so it can only model what the GLM leaves behind. No
  scorable object is returned.
- **Where.** `screen.py`, module docstring, `screen_features()`
  (`mk_from()`).

#### SCR-02 Supported models

- **Decision.** Poisson or quasi-Poisson counts with a log link and an
  offset (`count:poisson`, row weights 1, the offset in the margin), and
  Gamma with a log link (`reg:gamma`, prior weights). Anything else is
  refused.
- **Where.** `screen.py`, `screen_features()`.
- **Pinned by.** `test_screen.py`: `test_unsupported_families_and_bad_splits_are_refused`,
  `test_a_gamma_severity_model_screens_with_prior_weights`.

#### SCR-03 Out of sample on both sides

- **Decision.** The baseline is refitted on the training split
  (`GLMFit.refit()`); if that fails (usually a factor level missing from
  the split), the supplied model is used with a warning that its deviance
  is then in-sample. In Python the attempt includes the refitted model's
  predictions on the validation and test splits: patsy accepts a factor
  left with one level, so the failure R meets in the refit shows only when
  the refit meets the other levels.
- **Where.** `screen.py`, `screen_features()`; `model.py`,
  `GLMFit.refit()`.

#### SCR-04 A staged verdict

- **Decision.** Baseline, depth-1 booster, depth-`max_depth` booster. A
  depth-1 improvement of more than 0.1% means main effects are missing and
  the interaction ranking is not to be trusted yet.
- **Where.** `screen.py`, `screen_features()`.
- **Pinned by.** `test_screen.py`: `test_the_verdict_points_at_the_missing_main_effect`.

#### SCR-05 PermDeviance is a ranking, not a test

- **Decision.** Rank on the increase in test deviance when a feature is
  shuffled with the baseline held fixed. `Gain` is shown but not ranked on;
  in Python it is xgboost's `total_gain` normalised to sum to 1, which is
  R's `xgb.importance` Gain.
- **Evidence.** On 150,000 rows over three seeds, noise columns ranged
  from -3.8 to +2.3, a weak real predictor from -1.9 to +2.0, strong ones
  30 to 48.
- **Where.** `screen.py`, `screen_features()`.
- **Pinned by.** `test_screen.py`:
  `test_a_genuinely_predictive_feature_outranks_pure_noise`,
  `test_features_already_in_the_baseline_show_no_incremental_value`.

#### SCR-06 Performance levers

- **Decision.** `tree_method="hist"` by default; `nthread` one fewer than
  the cores, capped by `OMP_THREAD_LIMIT` (and 2 when R CMD check's
  `_R_CHECK_LIMIT_CORES_` is set, carried over from R); `max_rows=math.inf`
  by default (`None` means the same), with a message above 500,000 rows;
  the SHAP interaction array capped near 1 GB.
- **Evidence.** 2,000,000 rows, 6 candidates: 291.6 s under xgboost's old
  default against 108.4 s with `hist`, identical ranking.
- **Where.** `screen.py`, `screen_features()`; `_utils.py`,
  `default_nthread()`.
- **Pinned by.** `test_screen.py`:
  `test_max_rows_defaults_to_every_row_and_accepts_inf_or_none`,
  `test_tree_method_is_hist_by_default_and_passed_through`.

#### SCR-07 Candidates and near-duplicates

- **Decision.** Constant columns, factors with more than `max_levels` (50)
  levels and unsupported types are skipped with a warning naming them.
  Numeric pairs with `|cor| >= 0.95` are reported as near-duplicates.
  Factors are one-hot encoded by hand, so a single-level factor does not
  break the matrix.
- **Where.** `screen.py`, `screen_features()`, `_screen_levels()`,
  `_screen_matrix()`.
- **Evidence.** *Parity:* candidates, skipped columns and messages exact;
  boosting itself not comparable.
- **Pinned by.** `test_screen.py`:
  `test_awkward_columns_are_dropped_instead_of_crashing`,
  `test_a_single_level_factor_no_longer_breaks_the_design_matrix`,
  `test_near_duplicate_candidates_are_reported`,
  `test_an_uncorrelatable_pair_does_not_corrupt_the_near_duplicate_table`;
  `parity/test_parity_screen.py`: `test_screen_features_parity`.

#### SCR-08 Outcome columns are not candidates

- **Decision.** With `features=None`, the candidates exclude the
  response, the offset variables, the prior-weight variables and
  `claims_col`, `loss_col` and `exposure_col`, with a message naming the
  columns left out. Named explicitly in `features`, such a column stays,
  with a warning.
- **Why.** The booster takes whatever predicts the response, and the loss
  amount predicts the claim count perfectly: the ranking and the verdict
  were then about leakage, not risk.
- **Where.** `screen.py`, `screen_features()`.
- **Evidence.** The review's WA book: `SCHADELAST` ranked first with a
  `PermDeviance` of 7,188 and 97% of the gain, and the verdict said the
  baseline missed main effects worth 95.9% of its deviance. On a severity
  model the claim count, its weight, ranked second.
- **Pinned by.** `test_screen.py`: `test_screen_features_does_not_take_the_loss_amount_as_a_candidate`;
  `parity/test_parity_screen.py`: `test_screen_features_parity`.

### EXP: Excel export

#### EXP-01 Workbook layout

- **Decision.** An Overview sheet (timestamp, the three intercepts, the
  base value per variable), a Tariff sheet with every factor under a lookup
  key `Variable|Level` or `Variable|Level|Group` (an interaction key holds
  the uplift, EXP-04), one sheet per main effect
  (base row highlighted, thin rows greyed), and one per interaction with a
  Level x Group matrix of `Factor_Premium` (falling back to frequency or
  severity when that is empty). Written with openpyxl.
- **Where.** `export.py`, `export_rating_table()`.
- **Pinned by.** `test_export_report.py`:
  `test_export_rating_table_writes_a_structured_workbook`,
  `test_the_tariff_sheet_rebuilds_a_premium_by_lookup`.

#### EXP-02 Numbers stay numbers; names stay valid

- **Decision.** Continuous levels are written as numbers, also in the
  interaction matrix. Sheet names are sanitised, cut to 31 characters and
  deduplicated case-insensitively.
- **Why.** Excel flags numbers stored as text, and treats sheet names
  case-insensitively, so `REGIO` and `Regio` would corrupt the workbook.
- **Where.** `export.py`, `_numeric_levels()`, `sheet_name()`.
- **Pinned by.** `test_export_report.py`:
  `test_continuous_levels_are_written_as_numbers_not_text`,
  `test_the_matrix_under_an_interaction_keeps_numeric_levels_too`,
  `test_export_sheet_names_are_deduplicated_case_insensitively`.

#### EXP-03 Offsets are not in the export

- **Decision.** The BM scale and the exposure are not factors and do not
  appear; the Overview lists the base value of each variable in the table.
- **Where.** `export.py`, `export_rating_table()`.
- **Consequence.** A premium from the workbook is
  `Intercept premium x VLOOKUP factors x BM_KORTING`.

#### EXP-04 An interaction key on the Tariff sheet holds the uplift

- **Decision.** The factor columns of an interaction row on the Tariff
  sheet hold `Uplift_*`, the joint factor over the two main factors, and
  the instruction on the sheet says so. The interaction sheets keep the
  joint factor.
- **Why.** The instruction is "every variable's factor, times each
  interaction key". With the joint factor under the key, the two main
  effects were counted twice. With the uplift, main x main x uplift is the
  joint factor and the instruction holds literally.
- **Where.** `export.py`, the Tariff block of `export_rating_table()`.
- **Evidence.** The review rebuilt premiums by the instruction on a sample
  of policies: 16% too high on average, 5th to 95th percentile -21% to
  +79%, at most 3.4 times.
- **Pinned by.** `test_export_report.py`:
  `test_the_tariff_sheet_instruction_rebuilds_the_premium_with_an_interaction`.

#### EXP-05 The intercepts on the Overview sheet are numbers

- **Decision.** The Overview's value column is written as text and the
  intercepts and numeric base values are then written over it as numbers.
- **Why.** The column mixes a timestamp, levels and numbers. A number
  stored as text is what a formula referring to the intercept picks up, and
  in a locale with a decimal comma "0.110053" is not a number at all.
- **Where.** `export.py`, the Overview block of `export_rating_table()`.
- **Pinned by.** `test_export_report.py`:
  `test_the_intercepts_on_the_overview_sheet_are_numbers_not_text`.

### REP: HTML report

#### REP-01 One rating table, with the table's own arguments

- **Decision.** The report builds one rating table and passes `grid_res`,
  `base_level`, `trim`, `merge_vars`, `base`, `grid_step`, `year_col`,
  `loss_col`, `exposure_col` and `claims_col` on to it.
- **Why.** The report should show the tariff a table built with the same
  arguments shows.
- **Where.** `report.py`, `pricing_report()`.
- **Pinned by.** `test_export_report.py`:
  `test_the_report_passes_merge_vars_and_base_to_its_rating_table`.
- **History (R package).** Until the review fixes of October 2026 the
  report passed no `merge_vars`, `base` or `grid_step`, so a merged field
  showed as two blocks.

#### REP-02 What the report contains and how it fails

- **Decision.** Diagnostics (fit table, collinearity, binned residuals);
  per variable the one-way (severity only alongside a severity model), A/E,
  PDP and rating plot; the rating table's interaction plots and an
  interaction scan on the frequency model (`n_bins=10`, `n_sim=100`,
  top 3 heatmaps). A failing plot is replaced by a note (REP-03). plotly.js
  is written once, next to the HTML file.
- **Where.** `report.py`, `pricing_report()`.
- **Pinned by.** `test_export_report.py`:
  `test_pricing_report_writes_a_complete_html_report`,
  `test_the_one_way_block_covers_severity_not_just_frequency`.

#### REP-03 Every plot is built inside the guard

- **Decision.** Each plot is rendered inside the error guard, so a plot
  that cannot be built becomes a note: R passes it through
  `plotly::plotly_build()` there, Python renders it to HTML there. A number
  of 1e9 or more in a table is printed in scientific notation, and a
  missing value in a table cell prints as empty.
- **Why.** R's plotly builds a widget only when the page is written, so an
  error in one plot escaped the guard and took the whole report down at
  `save_html()`. `formatC(format = "d")` goes through an integer, so a
  large GVIF printed as "NA". The Python port keeps both rules, so a report
  reads the same in both packages.
- **Where.** `report.py`, `safe()`, `_fmt_num()`, `_html_table()`.
- **Evidence.** The review: "Wasn't able to determine range of domain" on a
  400-policy book.
- **Pinned by.** `test_export_report.py`:
  `test_pricing_report_writes_a_report_for_a_small_book_too`.

### DATA: data handling and one-way aggregation

#### DATA-01 Training data is aligned by row name

- **Decision.** In R the training data (`.model_data()`, DATA-05) is
  matched to the model frame by row name, not by position. In Python
  `GLMFit.data` holds exactly the rows the fit used, with their original
  index, in the order of `m.y`, `m.prior_weights` and `m.offset`, so the
  alignment holds by construction and `glm_training_data()` only reads it.
- **Why.** Rows dropped by `na.action` and non-numeric row names would
  otherwise misalign values and fitted values.
- **Where.** `_utils.py`, `glm_training_data()`.
- **Pinned by.** `test_rating_table.py`:
  `test_the_rating_table_does_not_depend_on_the_index_of_data`;
  `test_agg_plots.py`: `test_missing_values_in_the_data_do_not_break_the_ae_plot`;
  `test_screen.py`: `test_model_data_from_non_pandas_input_is_handled`.

#### DATA-02 Aggregation rules

- **Decision.** `Frequency = sum(claims) / sum(exposure)` (missing at zero
  exposure), `Severity = sum(loss) / sum(claims)` (missing without claims).
  Missing values are counted in a warning and ignored. Grouping on a column
  name that `agg_all()` creates is refused.
- **Why.** The severity of a group is total loss over total claims, which
  weights each policy by its claim count, the same weight as the severity
  model. A clash of output names would silently produce two columns of one
  name.
- **Where.** `agg.py`, `agg_all()`.
- **Evidence.** *Parity:* `agg_all`, `make_plot` 5e-15 from R.
- **Pinned by.** `test_agg_plots.py`:
  `test_agg_all_works_with_custom_column_names_and_validates_input`,
  `test_grouping_on_a_column_agg_all_creates_is_refused`;
  `parity/test_parity_plots.py`: `test_agg_all_parity`.

#### DATA-03 The volume bar is the metric's own denominator

- **Decision.** Under a frequency the bar is exposure; under a severity it
  is the claim count.
- **Why.** The level with the most exposure is often the one with the
  fewest claims, so exposure bars under a severity line point the wrong
  way.
- **Where.** `plot_oneway.py`, `make_plot()`; `pdp.py`, `make_pdp()`.
- **Pinned by.** `test_agg_plots.py`: `test_the_volume_bars_follow_the_metrics_own_denominator`.

#### DATA-04 Discrete or continuous axis

- **Decision.** A text or categorical axis, or a numeric one with at most
  `discrete_cutoff` (25) values, is drawn with markers only.
- **Where.** `plot_oneway.py`, `make_plot()`; `pdp.py`, `make_pdp()`;
  `_plot_common.py`, `is_discrete()`.

#### DATA-05 The training data of a model without model$data

- **Decision.** In R, `.model_data()` returns `model$data` when that is a
  data frame, and otherwise evaluates the data the call names in the
  environment where the formula was written; data found that way is
  checked against the model frame, and if a plain column no longer matches,
  the function stops. In Python there is nothing to recover: `fit_glm()`
  keeps the rows it used on the model (`m.data`), so every `GLMFit` has its
  training data and it cannot drift from the fit. The tag marks
  `glm_training_data()`, which reads it.
- **Why.** `MASS::glm.nb()` keeps no `model$data`, and a `glm()` fitted
  without `data =` keeps an environment there. The call still names the
  data, so PDP, A/E and grouped columns can be read. A name that has been
  reassigned since would silently pair other rows with the fit. Storing the
  data with the fit (PY-01) removes both problems at the source.
- **Where.** `_utils.py`, `glm_training_data()`; `model.py`, `fit_glm()`.
- **Pinned by.** `test_pdp.py`: `test_a_glm_nb_model_works_in_make_pdp`;
  `test_rating_table.py`: `test_a_regrouped_column_missing_from_data_is_taken_from_the_model`.

#### DATA-06 A model fitted with na.exclude is read as na.omit

- **Decision.** In R every function that mixes residuals, fitted values or
  influence measures with the model frame reads the model through
  `.as_na_omit()`, which sets the class of its `na.action` to `"omit"` on
  the local copy. Python always fits as with `na.omit` (there is no
  `na.exclude`), so `as_na_omit()` returns the model unchanged; the calls
  are kept at the same places, so the code reads like the R code.
- **Why.** `na.exclude` pads `residuals()`, `fitted()` and `predict()`
  back to the full data with `NA`, while the model frame holds only the
  rows the fit used. Five functions failed on that, and the heatmap and
  the Q-Q plot were silently wrong because the vectors were recycled
  against each other. The fit itself is identical to `na.omit`.
- **Where.** `_utils.py`, `as_na_omit()`; `ae.py`, `plot_glm_predictor()`;
  `diagnostics.py`, `glm_diagnostics()`, `plot_glm_residuals()`;
  `interactions.py`, `plot_residual_heatmap()`, `detect_interactions()`;
  `qq_influence.py`, `plot_glm_qq()`, `plot_glm_influence()`; `pdp.py`,
  `make_pdp()`.
- **Pinned by.** `test_diagnostics.py`: `test_na_rows_give_the_same_result_as_complete_rows`;
  `test_agg_plots.py`: `test_missing_values_in_the_data_do_not_break_the_ae_plot`.

### PY: the port itself

#### PY-01 The package fits the model, so the model can carry its offsets

- **Decision.** Models are fitted with `fit_glm(formula, data, family,
  link=None, weights=None, var_power=1.5, theta=None)`, which returns a
  `GLMFit`, and every function takes a `GLMFit`. Estimation is statsmodels'
  IRLS; `GLMFit` keeps alongside it what an R `glm` keeps and statsmodels
  does not: the formula, the offset expressions, the prior-weight
  expression, the training rows with their original columns and index, the
  factor levels, the terms with their data columns, and the patsy design
  info that holds the spline knots. Rows with a missing value in any
  variable the model uses are left out, as `na.omit` does. Gamma's and the
  inverse Gaussian's default link is log, where R's are the inverse and
  1/mu^2. A negative binomial is fitted the way `MASS::glm.nb()` fits it.
  Prior weights given as an array, rather than a column name, are used for
  the fit, and `refit()` and `glm_newdata_parts()` refuse the model, as
  R's `update()` stops on such weights (`GLMFit.weights_array`).
- **Why.** The R package reads everything off the fitted object (CONV-05):
  the offset expressions to evaluate again on other data (OFF-01, OFF-02),
  the levels to name a value the model never saw (SCORE-01), the data to
  derive grouped columns (MV-03). A statsmodels result keeps the offset
  only as an array of the fitted length, which cannot be evaluated on other
  rows; taking any statsmodels model would make every function guess. A
  Gamma severity in pricing has a log link, so the R default would raise
  OFF-07's warning on every default fit.
- **Where.** `model.py`, `fit_glm()`, `GLMFit`.
- **Evidence.** *Parity* (`parity/test_parity_fit.py`): `fit_glm` against
  `glm()` for Poisson, quasi-Poisson and Gaussian with `cut()`, `ns()`, an
  interaction and two offsets: fitted values and predictions on new data
  2e-14, deviance and AIC 1e-15, standard errors 8e-12. Negative binomial
  against `MASS::glm.nb`: theta 5e-14, fitted values 1e-14. Gamma and
  inverse Gaussian with a log link and prior weights: deviance 1e-15,
  fitted values 2e-8 (a flat optimum; the two IRLS loops stop on
  marginally different points of it).
- **Consequence.** A model fitted elsewhere cannot be passed in; refit it
  with `fit_glm()`. Every function that takes a model checks that it is a
  `GLMFit` and names the argument that is not (`check_models()` in
  `_utils.py` for the rating table, lift, impact and report functions, the
  port of R's `.check_models()`). DATA-05 and DATA-06 are moot. `fit_glm()` cannot fit a
  Gamma model to a constant response (statsmodels refuses the perfect fit,
  where R returns a deviance of 0), which is never a pricing case.
- **History (Python port).** Until the README examples were verified, an
  array of prior weights was silently replaced by weights of 1 in a refit
  (`screen_features()`, a sampled `plot_glm_influence()`) and on new data
  (`plot_glm_predictor(data=...)`); the model is now refused there. The
  inverse Gaussian family was accepted but undocumented and had no AIC; it
  now has R's AIC and a parity test.
- **Pinned by.** `test_core.py`: `test_training_predictions_match_fitted`,
  `test_prediction_reevaluates_offsets`, `test_rhs_vars_include_offset_vars`,
  `test_prior_weights_given_as_an_array_are_not_dropped_on_other_rows`,
  `test_a_model_that_is_not_a_glmfit_is_refused_by_name`,
  `test_inverse_gaussian_is_a_family_with_an_aic`;
  `parity/test_parity_fit.py`: `test_fit_glm_matches_r_on_the_fit`,
  `test_fit_glm_predicts_new_data_as_r_does`,
  `test_the_negative_binomial_theta_matches_glm_nb`;
  `parity/test_parity_diagnostics.py`: `test_glm_diagnostics_table`
  (deviance and AIC against R).

#### PY-02 R formula syntax is translated, not replaced

- **Decision.** Formulas are written as in R. `ParsedFormula` takes every
  `offset()` term out as an expression and refuses a subtracted one;
  `translate()` turns backticks, `^`, `TRUE`/`FALSE`, `c()` and the dotted
  argument names into Python, and the rest goes to patsy with the R
  functions a pricing formula uses under their R names (`R_FUNCTIONS`):
  `offset()`, `factor()`, `cut()` (`r_cut()`, with R's labels), `ns()`,
  `bs()`, `poly()`, `log()`, `exp()`, `sqrt()`, `pmin()`, `pmax()`,
  `ifelse()`, `I()`, `^`, `*`, `:`, `-1`. Not supported: `.`, the separate
  `offset =` argument and `na.exclude`.
- **Why.** The same formula text in both packages keeps them comparable and
  makes the parity tests possible (PY-08). Term labels, level labels and
  messages that quote a term then read the same; level labels such as
  `"(17,24]"` matter because they are tariff levels and lookup keys
  (EXP-01). Offsets kept as expressions are what OFF-01 to OFF-03 rely on.
  The `offset =` argument was left out because one place for offsets is
  simpler than two, and two places is what made OFF-01 necessary in R.
- **Where.** `_formula.py`, `ParsedFormula`, `translate()`, `r_cut()`,
  `eval_expr()`, `expr_vars()`.
- **Pinned by.** `test_core.py`: `test_formula_parts_and_offsets`,
  `test_cut_labels_as_in_r`; `test_rating_table.py`:
  `test_cut_in_the_formula_does_not_give_a_table_full_of_na`,
  `test_i_terms_resolve_to_their_underlying_column`,
  `test_a_column_name_with_a_space_survives_the_whole_rating_table`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`.

#### PY-03 Spline and polynomial bases are R's

- **Decision.** `ns()`, `bs()` and `poly()` reproduce `splines::ns`,
  `splines::bs` and `stats::poly`: knots at type-7 quantiles of the
  training data, the same boundary knots, the same basis, linear (`ns`) or
  Taylor (`bs`) extrapolation beyond the boundary. They are patsy stateful
  transforms, so the knots are learned once on the fit data and reused on
  any data the model scores, as R's `makepredictcall` does.
- **Why.** A basis that spans another space gives another fit and other
  rating factors, and nothing could be compared with R. patsy has no
  `ns()`, and its own spline functions are not written to match R's.
- **Where.** `splines.py`, `ns_basis()`, `bs_basis()`, `_spline_design()`,
  `_NS`, `_BS`, `_Poly`.
- **Evidence.** *Parity:* `parity/test_parity_fit.py` fits `ns()` and
  predicts past its boundary knots, matching R to 2e-14; the rating table
  parity uses `ns()` and matches R to 1e-10 in its factors.
- **Pinned by.** `test_core.py`: `test_training_predictions_match_fitted`
  (the stored knots reproduce the fit on the training rows);
  `parity/test_parity_fit.py`: `test_fit_glm_predicts_new_data_as_r_does`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`;
  `parity/test_parity_diagnostics.py`: `test_glm_diagnostics_table`.

#### PY-04 R's term order and labels over patsy's

- **Decision.** `GLMFit.terms` and `term_labels` follow R's `terms()`: by
  degree, in formula order within a degree, the factors of an interaction
  in their order of first appearance in the formula, labels written back
  the R way (`REGIO:LEEFTIJD`, backticked names, `c()`, `^`, and named
  arguments with R's names and spaces round `=`:
  `ns(LEEFTIJD, knots = c(30, 50), Boundary.knots = c(18, 80))`). The rating
  table reads the terms in the same order. Tables name the family as R does
  (`family_label`: `"Gamma"`, `"Negative Binomial(1.4)"`). The design
  matrix columns and coefficient names stay patsy's (`REGIO[T.West]` where
  R writes `REGIOWest`); `Term.cols` points a term to its columns.
- **Why.** Term labels are what a user reads and types: the `Variable` of
  an interaction row, the `Term` of `glm_collinearity()`, the predictor
  named in `plot_glm_predictor()`. With patsy's order the same formula gave
  other labels and another row order than R, so a table or a script could
  not move between the packages and parity on structure could not be exact.
  Coefficient names stay inside the model object (`m.coef`), so they keep
  patsy's spelling.
- **Where.** `model.py`, `fit_glm()` (the block that sorts `m.terms`),
  `Term`; `_formula.py`, `r_code()`; `rating_table.py`, `_r_terms()`,
  `_r_factor_codes()`.
- **Evidence.** *Parity:* structure, labels and base values of the rating
  table exact.
- **Pinned by.** `test_rating_table.py`: `test_columns_come_in_the_r_order`;
  `test_core.py`: `test_term_labels_with_named_arguments_read_as_in_r`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`;
  `parity/test_parity_diagnostics.py`: `test_glm_collinearity_table`.
- **History (Python port).** Until the README examples were verified, named
  arguments kept patsy's form (`knots=c(...)`, `boundary_knots=`,
  `include_lowest=TRUE`); `r_code()` now writes them as R deparses them.
- **History (Python port).** Added in commit `6bfaeb3` of this repository.

#### PY-05 Randomness through a numpy Generator, never the global state

- **Decision.** Every function with `seed` creates
  `rng = np.random.default_rng(seed)` and passes it to everything random it
  calls: `gini_boot()`, `_quantile_residuals()`, the double-lift
  bootstrap, the simulated interaction null, row sampling and the
  screening split. numpy's global random state is never read or set.
- **Why.** R's `set.seed()` changes the global stream (CONV-08), which a
  caller's own simulation then inherits without knowing. A Generator per
  call makes `seed` reproduce one call and nothing else. R's random streams
  cannot be reproduced in numpy anyway, so matching R draw for draw was
  never an option; the deterministic parts are compared instead (PY-08).
- **Consequence.** For the same seed, random outputs differ from R's:
  bootstrap intervals (RATE-11), the simulated interaction null (INT-03,
  INT-05), randomised quantile residuals of a discrete response (DIAG-04),
  sampled rows (DIAG-05, SCR-06) and the booster. Everything that is not
  random matches R. Tests state properties rather than R's numbers.
- **Where.** `interactions.py`, `detect_interactions()`; `lift.py`,
  `model_lift()`, `double_lift()`; `qq_influence.py`, `plot_glm_qq()`,
  `plot_glm_influence()`; `screen.py`, `screen_features()`.
- **Pinned by.** `test_diagnostics.py`: `test_na_rows_give_the_same_result_as_complete_rows`
  (one seed, the same residuals on two fits);
  `parity/test_parity_impact_lift.py`: `test_double_lift_parity`
  (deterministic with `n_boot=0`).

#### PY-06 Results as Result, DataFrame.attrs and figure attributes

- **Decision.** An R list is returned as `Result`, a dict whose items are
  also attributes (`imp.summary` or `imp["summary"]`; an item wins over a
  dict method of the same name, so `r.values` reads like R's `r$values`).
  An attribute on an R data frame becomes `DataFrame.attrs`
  (`tbl.attrs["intercept_frequency"]`, `base_values`). A table R attaches to
  a plot with `attr(p, "cells")` is an attribute of the plotly figure
  (`fig.cells`, `fig.residuals`, `fig.influential`), set by `attach()`.
  The modebar configuration R sets with `plotly::config()` is kept as
  `fig.pricing_config` by `set_config()`, to pass to
  `fig.show(config=...)`; `pricing_report()` passes it. R data frames are
  pandas DataFrames with the same column names and an index from 0.
- **Why.** The R names and structure carry over, so the examples and a
  user's habits do too: `$summary` becomes `.summary`. A plain dict would
  need `["summary"]` everywhere, and a class per return value would add
  many classes for no gain. `DataFrame.attrs` is pandas' own place for
  metadata. Python's plotly takes the configuration when a figure is shown,
  not on the figure.
- **Consequence.** pandas marks `attrs` as experimental, and not every
  operation carries it over; read the intercepts from the table as
  `make_rating_table()` returns it.
- **Where.** `_core.py`, `Result`, `attach()`, `set_config()`.
- **Pinned by.** `test_interactions.py`:
  `test_plot_residual_heatmap_returns_a_plot_plus_its_cell_table`;
  `test_qq_influence.py`: `test_plot_glm_qq_returns_a_plotly_object_and_its_residuals`,
  `test_plot_glm_influence_ranks_rows_and_returns_them`;
  `test_rating_table.py`: `test_intercept_is_per_unit_of_exposure`;
  `test_screen.py`: `test_screen_features_returns_the_documented_structure`.

#### PY-07 A grouped column the caller supplies keeps its value where the model cannot check it

- **Decision.** In `grouping_column()`, when `data` already holds the
  grouped column and a categorical source value does not occur in the
  model's data (a brand without claims, so absent from the severity data),
  the caller's value is kept for those rows. Every value the model's data
  does know is still checked, and a disagreement is still an error
  (MV-05). Without the column in `data`, such a value is still an error
  (MV-07). The R package does the same since November 2026.
- **Why.** The error without the column advises putting the grouped column
  in `data` yourself, so doing that has to work. The kept value is not a
  guess by the package: the caller states it, and the model's data has
  nothing to contradict it. The rating table then shows the level as an
  empty cell (RT-10), as any level the severity model was not fitted on.
- **Where.** `_utils.py`, `grouping_column()`.
- **Evidence.** *Parity:* table D of the rating-table parity test, a
  grouped column in `data` for a level the severity data never saw, matches
  R to 1e-14 in its factors, the empty severity cell included.
- **Pinned by.** `test_groupings.py`:
  `test_a_grouped_column_already_in_data_is_kept_for_a_value_the_model_never_saw`;
  `test_rating_table.py`:
  `test_a_grouped_column_in_data_keeps_its_value_for_a_level_the_model_never_saw`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`.
- **History.** The port had this first; R stopped even with the column in
  `data`, so its own advice could not be followed. R was fixed to match in
  November 2026.

#### PY-08 Parity with R is measured, not assumed

- **Decision.** `tests/parity/` fits the same formulas on the same data in
  both languages, runs the R function on its own fit and compares every
  deterministic output: structure, labels, base values and counts exactly,
  fitted quantities at a relative tolerance of 1e-6 or tighter (volumes
  1e-9, the fit itself 1e-10). R is
  run through `Rscript` by `run_r()` in `tests/parity/_r.py` (data passed
  as CSV, results back as JSON) and fits with a tight
  `glm.control(epsilon)`, so its own convergence does not dominate. Random
  outputs are switched off (`n_boot=0`) or left out (PY-05). The tests are
  marked `parity` and skipped when `Rscript` is not found; CI runs
  `pytest -m "not parity"`.
- **Why.** The two packages must give the same tariff. The unit tests
  ported from R state properties, which both packages can satisfy while
  their numbers differ; only a run side by side catches another basis,
  another level order or another binning rule. At R's default tolerance a
  Gamma log-link fit stops about 5e-7 from the optimum, which would then be
  measured instead of the package.
- **Where.** `tests/parity/_r.py` and the parity test files; no tag in the
  package.
- **Evidence.** The parity table in section 4.1.
- **Pinned by.** `parity/test_parity_fit.py`: `test_fit_glm_matches_r_on_the_fit`,
  `test_fit_glm_predicts_new_data_as_r_does`,
  `test_the_negative_binomial_theta_matches_glm_nb`;
  `parity/test_parity_rating_table.py`: `test_rating_table_matches_r`,
  `test_rating_table_parity_covers_what_it_should`;
  `parity/test_parity_impact_lift.py`: `test_premium_impact_parity`,
  `test_model_lift_parity`, `test_double_lift_parity`;
  `parity/test_parity_plots.py`: `test_agg_all_parity`,
  `test_plot_glm_predictor_parity`, `test_make_pdp_parity`,
  `test_make_rating_plot_parity`;
  `parity/test_parity_diagnostics.py`: `test_glm_diagnostics_table`,
  `test_glm_collinearity_table`, `test_plot_glm_residuals_bins_and_bands`,
  `test_residual_heatmap_cells`, `test_detect_interactions_deterministic_columns`,
  `test_plot_glm_influence_top_rows`,
  `test_quantile_residuals_and_qq_plot_of_a_gamma_model`;
  `parity/test_parity_screen.py`: `test_screen_features_parity`.

#### PY-09 R's conditions map onto one exception and two warning categories

- **Decision.** `stop()` raises `PricingError`, a `ValueError`;
  `warning()` is `warnings.warn(..., PricingWarning)`; `message()` is
  `PricingMessage`, a warning category of its own that is shown once per
  call site and can be silenced with
  `warnings.filterwarnings("ignore", category=PricingMessage)`. The texts
  are R's English messages with the `fn: ` prefix; examples inside them use
  Python syntax (`at={'BOEKJAAR': 2024}`), and install hints say
  `pip install`. A formula that cannot be read raises `FormulaError`, a
  subclass of `PricingError`, so one `except PricingError` catches every
  refusal; a formula ending in an operator is refused, where patsy would
  drop the operator.
- **Why.** CONV-03 draws the line between an error and a warning on
  whether the output is still correct, and that line has to survive the
  port, so each R condition keeps its kind. A category of its own for
  messages lets a user silence information without silencing the warnings
  that say a result needs care. The same text means one search finds a
  message in both packages.
- **Where.** `_core.py`, `PricingError`, `PricingWarning`,
  `PricingMessage`, `stop()`, `warn()`, `message()`.
- **Pinned by.** `test_groupings.py`: `test_anything_that_would_be_a_guess_is_refused`;
  `test_core.py`: `test_a_formula_that_cannot_be_read_is_a_pricing_error`;
  `test_interactions.py`: `test_the_interaction_scan_is_not_fooled_by_overdispersion`
  (a `PricingMessage`); `test_rating_table.py`: `test_non_log_link_raises_a_warning`.

---

## 4. Differences from the R package

Collected while porting (`docs/deviations.md`), with the decision each one
belongs to.

### 4.1 Parity measured against R

Same data and formulas through both packages; R fitted with a tight
`glm.control(epsilon)` so its own convergence does not dominate (PY-08).

| Area | Largest relative difference |
|---|---|
| `fit_glm` vs `glm()` (Poisson, quasi-Poisson, Gaussian; `cut()`, `ns()`, an interaction, two offsets): fitted values and predictions on new data / deviance and AIC / standard errors | 2e-14 / 1e-15 / 8e-12 |
| `fit_glm` negative binomial vs `MASS::glm.nb`: theta / fitted values | 5e-14 / 1e-14 |
| `fit_glm` Gamma and inverse Gaussian (log link, prior weights) vs `glm()`: deviance / fitted values and predictions | 1e-15 / 2e-8 (a flat optimum: both IRLS loops stop on marginally different points of it) |
| `make_rating_table`: Factor_* / SE_* / Uplift_* / Exposure | 1e-10 / 1e-9 / 1e-13 / 2e-15; structure, labels, base values exact |
| `premium_impact` rates, by_level, rate change | 1e-15 to 2e-9 |
| `model_lift` / `double_lift` tables, Gini, mad | 1e-10 / 4e-14 / 3e-10 |
| `agg_all`, `make_plot` | 5e-15 |
| `plot_glm_predictor`, `make_pdp` | 1e-9 |
| `glm_diagnostics`, `glm_collinearity` | 8e-11, 2e-9 |
| `plot_glm_residuals` (Poisson) | 1e-14; Gamma deviance centring 6e-8 absolute (R's `integrate()` tolerance) |
| heatmap cells, `detect_interactions` (non-random columns) | 1e-14, 8e-10 |
| `plot_glm_influence`, Gamma quantile residuals | 3e-14, 3e-9 |
| `screen_features` | structure, candidates, messages exact; boosting itself not comparable |

### 4.2 General differences

- **Models are fitted by the package** with `fit_glm()` (statsmodels
  underneath) instead of being any R `glm` (PY-01). Gamma's and the inverse
  Gaussian's default link is log. Families: poisson, quasipoisson, gamma,
  gaussian, binomial, inverse.gaussian, tweedie, negbin. Prior weights
  given as an array are refused by a refit and on new data, as R's
  `update()` stops on them; a column name follows the rows.
- **R formula syntax** is supported as listed in PY-02. Not supported: `.`,
  the separate `offset =` argument (write the offset in the formula), and
  `na.exclude` (Python always fits as `na.omit`, so DATA-06 is moot).
- **Term labels and term order follow R** (`REGIO:LEEFTIJD`, named
  arguments as `knots = c(30, 50)`); the design matrix columns and
  coefficient names are patsy's (`REGIO[T.West]` where R writes
  `REGIOWest`) (PY-04).
- **Randomness uses numpy's Generator** (PY-05): bootstrap intervals, the
  simulated interaction null, randomised quantile residuals, sampling and
  boosting give other draws than R's `set.seed()` streams. Everything that
  is not random matches R.
- **Messages keep R's English text** (PY-09); examples inside them use
  Python syntax, install hints say `pip install`.
- **Return values** (PY-06): R lists are `Result` objects; data frame
  attributes are `DataFrame.attrs`; tables attached to a plot are figure
  attributes (`fig.cells`, `fig.residuals`, `fig.influential`); plotly's
  config is kept as `fig.pricing_config`.
- **Row numbers are 0-based** (`premium_impact().policy.Row`, the row
  `add_groupings()` names when a grouped column disagrees with the model,
  MV-03); the influence table's `Row` is the data's index label (DIAG-05).
- **Missing values**: R's `NA` is NaN, None or `pd.NA` depending on the
  column type.

### 4.3 Per function

- `make_plot(display="facet")` is built with plotly subplots, not ggplot2
  and `ggplotly()`: the same panels and values, slightly different legend
  and hover text.
- Category axes get their order set explicitly, where R's plotly takes it
  from the factor (CONV-07).
- `premium_impact(spotlight=)` takes a boolean array or Series, or a string
  with an R-style expression on the data's columns; the caller's own
  variables cannot be used inside the string. `stats.quantiles` is a dict
  (RATE-05).
- `model_lift()` and `double_lift()`: with many exactly tied predictions
  (whole ages, few factors) the order of the ties depends on last-bit
  rounding, which differs between R and numpy, so a bin boundary can fall
  on a different policy. Totals per bin then differ slightly; with
  continuous predictors the bins are identical (RATE-06).
- `screen_features()`: `Gain` is xgboost's `total_gain` normalised to sum
  to 1 (R's `xgb.importance` Gain) (SCR-05). The baseline refit falls back
  to the supplied model, with R's warning, in one case more than R: a
  factor level missing from the training split while two or more levels
  remain. R's `update()` then succeeds and its `predict()` on the
  validation or test split stops with "factor has new levels", outside the
  `tryCatch`; Python treats the predictions as part of the refit attempt
  (SCR-03).
- `glm_collinearity()`: `Flag` is a nullable boolean column (`<NA>` for
  terms in an interaction) (DIAG-02, DIAG-07).
- `fit_glm()` cannot fit a Gamma model to a constant response (statsmodels
  refuses the perfect fit, R returns deviance 0). Never a pricing case
  (PY-01).
- An offset that comes out of a rate is never evaluated, so data to be
  scored does not need its column, where R stops (OFF-03). An offset that
  stays in the rate is evaluated and refused with R's message when it
  cannot be.

### 4.4 Found while porting, fixed in both packages

Three defects of the R package came to light in the port and were fixed in
R and Python alike in November 2026, each pinned by tests on both sides:

- A grouped column the caller puts in `data` keeps its value for a source
  value the model's data never saw; R stopped, although its own message
  advised that column (MV-05, PY-07; parity table D).
- A logical predictor in the rating table is a two-level categorical
  variable with `FALSE`, the model's reference, as base; it was read as
  continuous, with the most frequent value as base, numeric levels in the
  export and a line in the rating plot (RT-14).
- The rating table, lift, impact and report functions check that each
  model is a fitted GLM and name the argument that is not; they failed on
  the first attribute they read (PY-01, R's `.check_models()`).

---

## 5. Known limitations and open points

These are known, deliberate or not yet addressed. Each says what to do
about it today. Items 1 to 12 are the R package's and hold for Python too;
items 13 to 16 are the port's own.

1. **The bootstrap intervals hold the models fixed** (RATE-11). They
   resample policies, not the fit, so an in-sample interval is narrower
   than an honest one. Use a holdout for a Gini or a double lift you want
   to act on.
2. **Heatmap and interaction scan take `exp(offset)` as the volume**
   (OFF-08). With a BM offset, the exposure in the tooltip and
   `MaxAE_ExposureShare` are BM-weighted. The A/E is unaffected.
3. **Same-name columns with two meanings are only caught through
   `merge_vars`** (MV-04). Without a declared source column there is
   nothing to compare the two models' columns against. Declaring the
   relation is the protection.
4. **A measurement with at most 25 distinct values, used only through a
   band, is treated as a code** (MV-06). A value between two observed ones
   then errors instead of taking the nearer band. Make the source a
   continuous term of a model, or put the grouped column in the data.
5. **A spline extrapolates beyond its range in `predict()`** (MV-07). The
   package warns about it in `add_groupings()` but does not cap. Cap the
   values before scoring: `np.clip(x, lo, hi)`, or `pmin(pmax(x, lo), hi)`
   inside a formula.
6. **Only two-way interactions** get a rating-table block or an
   interaction scan (RT-06, INT-03).
7. **With one model, `Factor_Premium` and `intercept_premium` are
   missing** (RT-03). Use that model's own factor and intercept.
8. **The PDP grid is not the rating table's grid** (PDP-04), so the two
   are not point for point comparable.
9. **Only models from `fit_glm()` are supported** (PY-01). GAMs, GBMs,
   mixed models and GLMs fitted with statsmodels directly are not read;
   refit with `fit_glm()`.
10. **The standard errors in the rating table assume the model's
    dispersion** (RT-12). For a Poisson that is 1, so on overdispersed
    claim counts they are too small; fit a quasipoisson for honest ones.
11. **The interaction scan uses one dispersion for the whole book**
    (INT-05). Under a gamma frailty, cells with a large expected count
    carry more overdispersion than the average.
12. **Deviance residuals are centred for the common families only**
    (DIAG-06): Poisson, quasipoisson, negative binomial with unit weights,
    binomial with at most 20 distinct trial counts, Gamma, Gaussian. Other
    families are shown uncentred with a warning; use Pearson residuals
    there.
13. **Random results differ from R's for the same seed** (PY-05). Compare
    random outputs between the packages by their spread, not draw for
    draw.
14. **`DataFrame.attrs` is experimental in pandas** (PY-06) and does not
    survive every operation. Read the intercepts from the table as
    `make_rating_table()` returns it.
15. **Lift bins can differ from R's with many tied predictions** (RATE-06,
    section 4.3). Totals per bin differ slightly; with continuous predictors
    the bins are identical.
16. **A few plot strings hold non-ASCII characters literally** (CONV-06),
    where the porting guide asks for `\u` escapes. Harmless; change them
    when the module is next touched.

---

## 6. Test map

### 6.1 Unit tests

One Python test per R `test_that()` block, named after it in snake case, so
a search for an R test name in snake case finds its port. The R tests of
`test-offset-link.R`, `test-awkward-names.R`, `test-collinearity-spotlight.R`,
`test-data-table-input.R` and `test-review-findings.R` were spread over the
module they concern. Tests state the property the R test states; the data
come from numpy's Generator, so their numbers differ from the R suite's.

| Area | Test file |
|---|---|
| `fit_glm()`, formulas, offsets, shared helpers | `test_core.py` |
| offsets and BM in rates, `premium_impact()`, contribution, spotlight, scoring checks, `at` | `test_impact.py` |
| rating table, `base`, `merge_vars`, grouped columns, awkward names, trimming, the review's rating-table findings | `test_rating_table.py` |
| `add_groupings()` | `test_groupings.py` |
| `model_lift()`, `double_lift()`, BM in the lift, intervals | `test_lift.py` |
| `plot_glm_predictor()`, `agg_all()`, `make_plot()`, `make_rating_plot()`, binned residuals, offsets on the A/E axis | `test_agg_plots.py` |
| `make_pdp()`, PDP with a BM offset | `test_pdp.py` |
| diagnostics, collinearity, deviance centring, `na.omit` | `test_diagnostics.py` |
| Q-Q and influence | `test_qq_influence.py` |
| interactions, overdispersion in the scan | `test_interactions.py` |
| screening | `test_screen.py` |
| export and report | `test_export_report.py` |
| axis ranges | `test_y_range.py` |

Shared fixtures live in `tests/conftest.py`, the port of
`tests/testthat/setup.R`: a 20,000-row simulated portfolio (`dat`), a
Poisson model with `ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR)` and an
exposure offset (`m_freq`), the severity data and a Gamma severity
(`d_sev`, `m_sev`), a model with an interaction (`m_freq2`), two rating
tables (`tbl`, `tbl2`) and the plot helpers `trace` and `trace_y`.

### 6.2 Parity tests

In `tests/parity/`, marked `parity`, skipped without `Rscript` (PY-08).
The R side needs the R package pricingtoolsRmO and `jsonlite` installed;
`RSCRIPT` in the environment points to another `Rscript`.

| File | Compares with R |
|---|---|
| `parity/test_parity_fit.py` | `fit_glm()` against `glm()` (Poisson, quasi-Poisson, Gamma with weights, Gaussian) and `MASS::glm.nb()`: fitted values, deviance, null deviance, AIC, dispersion, standard errors, theta, and predictions on new data with NA from `cut()` and extrapolation past the `ns()` knots |
| `parity/test_parity_rating_table.py` | four rating tables: `ns()`, the latest year as base, interactions of every type, `merge_vars`, a chosen base, a trimmed grid, `base_level="exposure"`, a `cut()` term, a logical predictor, a grouped column in `data` for a level the severity data never saw |
| `parity/test_parity_impact_lift.py` | `premium_impact()`, `model_lift()`, `double_lift()` with an exposure and a BM offset, bootstraps off |
| `parity/test_parity_plots.py` | `agg_all()`, `make_plot()`, `plot_glm_predictor()`, `make_pdp()`, `make_rating_plot()`: the values behind the traces |
| `parity/test_parity_diagnostics.py` | `glm_diagnostics()`, `glm_collinearity()`, `plot_glm_residuals()`, heatmap cells, the deterministic columns of `detect_interactions()`, `plot_glm_influence()`, Gamma quantile residuals |
| `parity/test_parity_screen.py` | `screen_features()`: candidates, skipped columns, `InModel`, near-duplicates, split sizes, messages; ranking and verdict qualitatively |

`parity/test_parity_rating_table.py`: `test_report_max_relative_differences`
prints the largest relative difference per column (`pytest -s`); it asserts
nothing beyond `test_rating_table_matches_r`.

### 6.3 R tests without a Python counterpart

| R test | Why, and what pins the decision instead |
|---|---|
| `test-offset-link.R`: "an offset passed as an argument is neutralised too" | nothing to test: `fit_glm()` has no `offset =` argument (OFF-01) |
| `test-qq-influence.R`: "quantile residuals match statmod, the reference implementation" | statmod is R only; replaced by `test_qq_influence.py`: `test_quantile_residuals_are_the_cdf_transform` |
| `test-review-findings.R`: "na.exclude gives the same result as na.omit" | no `na.exclude` in Python; replaced by `test_diagnostics.py`: `test_na_rows_give_the_same_result_as_complete_rows` |
| `test-data-table-input.R`: the data.table and tibble tests | replaced by the other-table-input tests listed under CONV-02 |
| `test-groupings.R`: "a data.table comes back as one, and the input is left alone" | replaced by `test_groupings.py`: `test_the_input_is_left_alone` |

A few R tests were renamed in the port: "a glm.nb model works in make_pdp
and the A/E plot" became two tests (`test_pdp.py`:
`test_a_glm_nb_model_works_in_make_pdp`, `test_agg_plots.py`:
`test_a_glm_nb_model_works_in_the_ae_plot`), "a second offset does not end
up on the exposure axis" became `_ae` and `_pdp` variants, and "offsets from
the formula and the argument add up" became
`test_impact.py`: `test_offsets_from_the_formula_add_up`. The other
`test-offset-link.R` tests are in `test_impact.py` under their own names;
"an offset that cannot be evaluated per row is refused" is adapted to
formula offsets (OFF-03).

### 6.4 Running the tests

```sh
.venv\Scripts\python.exe -m pytest -q                  # everything, parity included when R is found
.venv\Scripts\python.exe -m pytest -q -m "not parity"  # what CI runs
.venv\Scripts\python.exe -m pytest -q -m parity        # parity with R only
```

CI (`.github/workflows/tests.yml`) runs `pytest -q -m "not parity"` on
every push to `main` and on pull requests, on Python 3.11 and 3.12, each
with pandas 2.x and pandas 3.x.

---

## 7. Glossary

| Term | Meaning |
|---|---|
| `Exposure` | years on risk of a policy record; the default `exposure_col` |
| `AantalClaims` | claim count; the default `claims_col` |
| `SCHADELAST` | total loss amount of the record; the default `loss_col` |
| `BOEKJAAR` | accounting year; the default `year_col` |
| `BM_KORTING` | bonus-malus discount factor, typically carried as `offset(log(BM_KORTING))` |
| `AvgLoss` | average claim, `SCHADELAST / AantalClaims`, the usual severity response, weighted by `AantalClaims` |
| base | the policy every factor is measured against (RT-02) |
| intercept | the base policy's prediction per unit of exposure, offsets removed |
| factor | a prediction relative to the base, everything else at base (RT-01) |
| uplift | an interaction's joint factor over the product of its two main factors |
| thin cell | a level or cell with fewer than `min_claims` claims (RT-07) |
| `SE_Frequency`, `SE_Severity` | standard error of the logarithm of a factor (RT-12) |
| inconclusive | a double-lift verdict whose interval includes 0 (RATE-09) |
| cell-total dispersion | `sum((y - mu)^2) / sum(mu)`, the overdispersion of a sum of rows (INT-05) |
| A/E | actual over expected; for counts `sum(y) / sum(mu)` |
| source / grouped column | a column and a regrouping of it, such as `Stedelijkheid` and `Stedelijkheid_groep` (MV) |
| kept offset | an offset that stays in a comparison rate, such as the BM (OFF-05) |
| dislocation | the per-policy premium change between two tariffs |
| rebase | scaling the new rates to the old total, to separate level from redistribution |
| raking | iterative proportional fitting of expected to actual's margins (INT-03) |
| `GLMFit` | the fitted model object of this package, the counterpart of an R `glm` (PY-01) |
| factor code | patsy's text for one factor of a term, such as `ns(LEEFTIJD, 3)` or `Q("AUTO GEWICHT")`; `term_base_var()` gives the column behind it |
| stateful transform | a patsy function that learns its parameters (spline knots) on the fit data and reuses them on new data (PY-03) |
| aliased column | a design column that is a linear combination of the ones before it; its coefficient is NaN, as R's is `NA` |
| `Result` | a dict whose items are also attributes, the counterpart of an R list (PY-06) |
| Generator | numpy's `np.random.Generator`, created per call from `seed` (PY-05) |
| parity test | a test that runs the same data and formula through both packages and compares the results (PY-08) |
