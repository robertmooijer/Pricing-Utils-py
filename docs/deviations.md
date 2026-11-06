# Where the Python port differs from the R package, and how close it is

Collected while porting; the source for the "Differences from the R package"
sections of README.md and TECHNICAL.md.

## Parity measured against R (tests/parity)

Same data and formulas through both packages; R fitted with a tight
`glm.control(epsilon)` so its own convergence does not dominate.

| Area | Largest relative difference |
|---|---|
| `fit_glm` vs `glm()` (Poisson, quasi-Poisson, Gaussian; `cut()`, `ns()`, an interaction, two offsets): fitted values and predictions on new data / deviance and AIC / standard errors | 2e-14 / 1e-15 / 8e-12 |
| `fit_glm` negative binomial vs `MASS::glm.nb`: theta / fitted values | 5e-14 / 1e-14 |
| `fit_glm` Gamma (log link, prior weights) vs `glm(Gamma("log"))`: deviance / fitted values and predictions | 1e-15 / 1e-8 (a flat optimum: both IRLS loops stop on marginally different points of it) |
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

## Deliberate or unavoidable differences

General
- Models are fitted with `fit_glm()` (statsmodels underneath) instead of
  being any R `glm`. R formula syntax supported: `offset()` (any number),
  `factor()`, `cut()` (R labels), `ns()`, `bs()`, `poly()` (R-compatible
  bases), `log()`, `exp()`, `sqrt()`, `pmin()`, `pmax()`, `ifelse()`, `I()`,
  `^`, `*`, `:`, `-1`, backticked names. Not supported: `.`, the separate
  `offset =` argument (write the offset in the formula), `na.exclude`
  (Python always fits as na.omit, so DATA-06 is moot).
- Gamma's default link is `log` (R's default is the inverse). Families:
  poisson, quasipoisson, gamma, gaussian, binomial, tweedie, negbin.
- Term labels and term order follow R (`REGIO:LEEFTIJD`); the design
  matrix columns and coefficient names are patsy's (`REGIO[T.West]` where R
  writes `REGIOWest`).
- Randomness uses numpy's Generator: bootstrap intervals, the simulated
  interaction null, randomised quantile residuals, sampling and boosting
  give different draws than R's `set.seed()` streams. Everything that is
  not random matches R.
- Messages keep R's English text; examples inside them use Python syntax
  (`at={'BOEKJAAR': 2024}`), install hints say `pip install`.
- R lists are `Result` objects (dict with attribute access); data frame
  attributes are `DataFrame.attrs`; tables attached to a plot are figure
  attributes (`fig.cells`, `fig.residuals`, `fig.influential`); plotly's
  config is kept as `fig.pricing_config` (pass it to `fig.show(config=...)`).
- Row numbers are 0-based (`premium_impact().policy.Row`); the influence
  table's `Row` is the data's index label.

Per function
- `make_plot(display="facet")` is built with plotly subplots, not
  ggplot2 + ggplotly: same panels and values, slightly different legend and
  hover text.
- Category axes get their order set explicitly (R's plotly takes it from
  the factor).
- `premium_impact(spotlight=)` takes a boolean array/Series or a string
  with an R-style expression on the data's columns; the caller's own
  variables cannot be used inside the string. `stats.quantiles` is a dict.
- `model_lift` / `double_lift`: with many exactly tied predictions (whole
  ages, few factors) the order of the ties depends on last-bit rounding,
  which differs between R and numpy, so a bin boundary can fall on a
  different policy. Totals per bin then differ slightly; with continuous
  predictors the bins are identical.
- `add_groupings` / `make_rating_table(merge_vars=)` [MV-05]: a grouped
  column the caller put in the data keeps its value for a source value the
  model's data never saw (a brand without claims). R stops there, which is
  a known defect of the R version.
- `screen_features`: Gain is xgboost's `total_gain` normalised to sum to 1
  (R's `xgb.importance` Gain). [SCR-03] The baseline refit falls back to
  the supplied model, with R's warning, in one case more than R: a factor
  level missing from the training split while two or more levels remain.
  R's `update()` then succeeds and its `predict()` on the validation or
  test split stops with "factor has new levels", outside the `tryCatch`;
  Python treats the predictions as part of the refit attempt.
- `glm_collinearity`: `Flag` is a nullable boolean column (`<NA>` for terms
  in an interaction).
- A logical predictor in the rating table behaves as in R (Type
  "continuous", base = most frequent value); that R behaviour itself looks
  like a defect worth fixing in both.
- `fit_glm` cannot fit a Gamma model to a constant response (statsmodels
  refuses the perfect fit, R returns deviance 0). Never a pricing case.
- [OFF-03] An offset that comes out of a rate is never evaluated, so data
  to be scored does not need its column (R evaluates it in order to
  subtract it, and stops when it cannot). An offset that stays in the rate
  is evaluated and refused with R's message when it cannot be.
