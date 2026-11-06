# pricingtools-rmo

A Python package for **GLM-based non-life insurance pricing analysis**, the
port of the R package
[pricingtoolsRmO](https://github.com/robertmooijer/Pricing-Utils). It covers
the standard workflow around a frequency/severity model pair: one-way
exploration, actual-vs-expected checks, partial dependence plots, model
diagnostics (dispersion, binned residuals, collinearity, interaction
detection), portfolio-level model comparison (lift, Gini, double lift), the
construction of a multiplicative rating table with thin-cell flags,
premium-impact (dislocation) analysis, and deliverables, a formatted Excel
rating workbook and a one-call HTML report, all with interactive plotly
visualisations.

The functions, their arguments, their defaults and their outputs are the R
ones, so a script moves between the two packages almost line by line, and
the numbers agree: every deterministic result is checked against the R
package in the test suite (see [Differences from the R
package](#differences-from-the-r-package) for the measured parity). The one
thing Python does not have is R's `glm()`, so models are fitted with
[`fit_glm()`](#fitting-models-with-fit_glm), which takes an R model formula,
offsets included, and keeps everything the analysis functions read off a
fitted model: the formula, the data, the offset expressions, the prior
weights and the factor levels.

All plots follow a consistent house style (navy/blue/gold palette, exposure
bars on a secondary axis, horizontal legends) and export to PNG via the
plotly mode bar.

---

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data conventions](#data-conventions)
- [Quick start example](#quick-start-example)
- [Fitting models with `fit_glm()`](#fitting-models-with-fit_glm)
- [Working with the results](#working-with-the-results)
- [Function reference](#function-reference)
  - [`agg_all()`](#agg_all)
  - [`make_plot()`](#make_plot)
  - [`make_pdp()`](#make_pdp)
  - [`plot_glm_predictor()`](#plot_glm_predictor)
  - [`glm_diagnostics()`](#glm_diagnostics)
  - [`glm_collinearity()`](#glm_collinearity)
  - [`plot_glm_residuals()`](#plot_glm_residuals)
  - [`plot_glm_qq()`](#plot_glm_qq)
  - [`plot_glm_influence()`](#plot_glm_influence)
  - [`detect_interactions()`](#detect_interactions)
  - [`plot_residual_heatmap()`](#plot_residual_heatmap)
  - [`screen_features()`](#screen_features)
  - [`make_rating_table()`](#make_rating_table)
  - [`add_groupings()`](#add_groupings)
  - [`make_rating_plot()`](#make_rating_plot)
  - [`model_lift()`](#model_lift)
  - [`double_lift()`](#double_lift)
  - [`premium_impact()`](#premium_impact)
  - [`export_rating_table()`](#export_rating_table)
  - [`pricing_report()`](#pricing_report)
  - [Spline bases: `ns()`, `bs()`, `poly()`](#spline-bases-ns-bs-poly)
  - [House-style colours](#house-style-colours)
- [Actuarial methodology](#actuarial-methodology)
  - [Methods at a glance](#methods-at-a-glance), every function in one table
- [Input validation and warnings](#input-validation-and-warnings)
- [Differences from the R package](#differences-from-the-r-package)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Technical documentation](TECHNICAL.md): every assumption and design
  decision, why it was made, and where it lives in the code. The Python code
  carries the same decision tags as the R code (`# [MV-05]`, `# [OFF-03]`,
  ...), so a search for a tag finds the same place in both packages.
- [`docs/deviations.md`](docs/deviations.md) and
  [`docs/porting-guide.md`](docs/porting-guide.md): where the port differs
  from R, and how R constructs map onto Python.

---

## Requirements

- Python >= 3.11 (developed on 3.11; the CI runs 3.11 and 3.12)
- pandas 2.x or 3.x: the code runs on both, and the CI tests both
- Hard dependencies (installed automatically): `numpy`, `pandas`, `scipy`,
  `statsmodels` (the IRLS fit behind `fit_glm()`), `patsy` (the design
  matrices) and `plotly`
- Optional: `openpyxl` (only for `export_rating_table()`, extra `[excel]`);
  `xgboost` (only for `screen_features()`, extra `[screen]`).
  `pricing_report()` needs nothing beyond plotly.
- R is not needed, except to run the parity tests against the R package
  (see [Tests](#tests)).

## Installation

The package is installed from a local clone of this repository. The
distribution is called `pricingtools-rmo`, the import name is
`pricingtools_rmo`.

```sh
git clone https://github.com/robertmooijer/Pricing-Utils-py.git
cd Pricing-Utils-py
python -m venv .venv
.venv\Scripts\activate            # Windows; on Linux/macOS: source .venv/bin/activate

pip install -e .                  # the package and its hard dependencies
pip install -e ".[excel]"         # + openpyxl, for export_rating_table()
pip install -e ".[screen]"        # + xgboost, for screen_features()
pip install -e ".[dev]"           # + pytest, openpyxl and xgboost, to run the tests
```

`-e` installs in editable mode, so a `git pull` takes effect without
reinstalling. Extras combine: `pip install -e ".[excel,screen]"`.

In an analysis script or notebook,

```python
from pricingtools_rmo import *
```

is the counterpart of `library(pricingtoolsRmO)`: it imports exactly the
public API (the package's `__all__`), which is the functions listed above,
`fit_glm()` and its `GLMFit`, the spline bases `ns()`, `bs()` and `poly()`,
the house-style colour constants (`ta_navy`, `ta_blue`, `ta_lightblue`,
`ta_gold`, `ta_muted`, `ta_years_base` and the palette helper
`ta_year_palette(n)`), the condition classes `PricingError`,
`PricingWarning` and `PricingMessage`, and `Result`. The examples below
assume it. If you prefer a namespace, `import pricingtools_rmo as pt` and
`pt.fit_glm(...)` work just as well.

## Data conventions

The functions operate on a policy(-year) level pandas DataFrame with, at
minimum:

| Concept          | Default column name | Role                                                        |
|------------------|---------------------|-------------------------------------------------------------|
| Exposure         | `Exposure`          | earned exposure (policy years)                              |
| Claim count      | `AantalClaims`      | response (y) for frequency **and** weight for severity      |
| Loss amount      | `SCHADELAST`        | response (y) for severity                                   |
| Accounting year  | `BOEKJAAR`          | optional split dimension                                    |

> The defaults reflect the original (Dutch) dataset naming, and are the same
> as in the R package. Every function accepts `exposure_col`, `claims_col`,
> `loss_col` and (where relevant) `year_col` arguments, so datasets with
> different column names work by overriding these. Column presence is
> validated up front with a clear error.

**All functions take raw data** and default to the raw dataset names above;
aggregation happens internally where needed. `agg_all()` is the aggregation
engine behind `make_plot()` and returns English, standardised column names
regardless of the input naming: `Exposure`, `ClaimCount`, `Loss`,
`Frequency`, `Severity` and (when split) `Year`. It remains available
standalone for building your own summary tables.

**Categorical columns** are what R calls factors. A string column is
categorical with its levels in sorted order, so the alphabetically first
level is the reference, as R's `factor()` does with a character vector. A
pandas `Categorical` keeps the order of its categories, so the first
category is the reference: that is how you choose it, as `relevel()` does in
R. A numeric column is continuous unless the formula says otherwise, with
`factor(BOEKJAAR)` for instance.

**Model conventions.** The toolkit assumes the classical frequency/severity
setup:

```python
# Frequency: Poisson (or quasi/negbin) with log link and an exposure offset
m_freq = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + REGIO + offset(log(Exposure))",
                 dat, family="poisson")

# Severity: Gamma with log link on average cost per claim,
# weighted by the number of claims
claims = dat[dat.AantalClaims > 0].copy()
claims["AvgLoss"] = claims.SCHADELAST / claims.AantalClaims
m_sev = fit_glm("AvgLoss ~ GEWICHT + REGIO", claims, family="gamma",
                weights="AantalClaims")
```

Two things matter throughout:

1. **Fit models with `fit_glm()`.** Every function that takes a model reads
   the formula, the training data, the offsets and the prior weights off the
   `GLMFit` it returns. A statsmodels results object does not carry those
   and is not accepted. The R package's first rule, *fit with a `data =`
   argument*, has no Python counterpart: a `GLMFit` always carries the rows
   it was fitted on.
2. **Use a log link** for anything feeding `make_rating_table()`. The
   multiplicative factor decomposition is only valid for log-link models; a
   non-log link triggers a warning. The log link is `fit_glm()`'s default
   for every family except gaussian and binomial, including Gamma, where R's
   default is the inverse link.

## Quick start example

A self-contained run on a simulated motor portfolio: paste it into a script
or a notebook. The portfolio is reused by the examples in the rest of this
README.

```python
import numpy as np
import pandas as pd
from pricingtools_rmo import *

# --- A simulated portfolio, one row per policy-year -------------------
rng = np.random.default_rng(2024)
n = 40_000
dat = pd.DataFrame({
    "LEEFTIJD": rng.integers(18, 81, n),                                  # driver age
    "REGIO": rng.choice(["Noord", "Zuid", "West", "Stad"], n,
                        p=[.35, .25, .2, .2]),
    "GEWICHT": (rng.normal(1300, 200, n).clip(800, 1800) / 100).round() * 100,
    "KILOMETRAGE": (rng.gamma(4, 4000, n) / 100).round() * 100,           # not in the models
    "BOEKJAAR": rng.integers(2021, 2025, n),
    "Exposure": rng.uniform(0.1, 1.0, n).round(3),
})
young_city = (dat.LEEFTIJD < 30) & (dat.REGIO == "Stad")                # a real interaction
eta = (-2.3 + 0.5 * (dat.LEEFTIJD < 25) + 0.005 * np.abs(dat.LEEFTIJD - 45)
       + dat.REGIO.map({"Noord": 0, "Zuid": .1, "West": .2, "Stad": .3})
       + 0.4 * young_city + 0.00002 * (dat.KILOMETRAGE - 16000)
       + 0.04 * (dat.BOEKJAAR - 2021))
dat["AantalClaims"] = rng.poisson(dat.Exposure * np.exp(eta))
avg = rng.gamma(2.0, 1200 * np.exp(0.0004 * (dat.GEWICHT - 1300)) / 2.0, n)
dat["SCHADELAST"] = dat.AantalClaims * avg
dat["PREMIE"] = (dat.Exposure * 150                                       # current premium
                 * dat.REGIO.map({"Noord": 1.0, "Zuid": 1.1, "West": 1.1, "Stad": 1.3})
                 * np.where(dat.LEEFTIJD < 25, 1.4, 1.0)).round(2)

# --- Fit models -------------------------------------------------------
m_freq = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + REGIO + factor(BOEKJAAR)"
                 " + offset(log(Exposure))", dat, family="poisson")

claims = dat[dat.AantalClaims > 0].copy()
claims["AvgLoss"] = claims.SCHADELAST / claims.AantalClaims
m_sev = fit_glm("AvgLoss ~ GEWICHT + REGIO", claims, family="gamma",
                weights="AantalClaims")
```

```python
# --- One-way exploration ----------------------------------------------
# make_plot takes raw rows and aggregates internally (via agg_all):
make_plot(dat, "REGIO", "Frequency", ta_blue,
          y_label="Frequency", display="color", by_year=True).show()
make_plot(dat, "REGIO", "Severity", ta_gold,
          y_label="Severity", display="facet", by_year=True).show()

# --- Model vs data ------------------------------------------------------
plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=50).show()     # actual vs expected
make_pdp(m_freq, dat, "REGIO", metric="Frequency").show()    # partial dependence

# --- Diagnostics ---------------------------------------------------------
print(glm_diagnostics(m_freq, m_sev))   # deviance, AIC, dispersion per model
plot_glm_residuals(m_freq).show()       # binned residuals vs fitted value
plot_glm_residuals(m_freq, "LEEFTIJD").show()   # binned residuals vs a predictor

# --- Rating table --------------------------------------------------------
tbl = make_rating_table(m_freq, m_sev, data=dat,
                        base_level="exposure",      # base = largest level
                        trim=(0.005, 0.995))        # trim outlier tails

print(tbl.attrs["intercept_premium"])   # base premium per unit of exposure
make_rating_plot(tbl, "LEEFTIJD").show()   # factor curves

# --- Deliverables --------------------------------------------------------
export_rating_table(tbl, "rating_table.xlsx")   # formatted Excel workbook ([excel] extra)

impact = premium_impact(dat,                    # dislocation vs current tariff
                        model_freq_new=m_freq, model_sev_new=m_sev,
                        old_premium_col="PREMIE", by="REGIO",
                        at={"BOEKJAAR": dat.BOEKJAAR.max()})   # tariff year
print(impact.summary)
impact.plot.show()

pricing_report(m_freq, m_sev, dat,              # everything in one HTML
               file="pricing_report.html")
```

In a notebook a figure displays itself when it is the last expression of a
cell, so the `.show()` calls are only needed in a script.

---

## Fitting models with `fit_glm()`

```python
fit_glm(formula, data, family="poisson", link=None, weights=None,
        var_power=1.5, theta=None, maxiter=100, tol=1e-10)
```

`fit_glm()` is the counterpart of R's `glm()` and the one function of this
package that has no namesake in the R version. It fits with statsmodels'
IRLS, so coefficients, deviances and standard errors are those of `glm()` to
within the convergence tolerance (measured against R: see the [parity
table](#parity-measured-against-r)), and returns a `GLMFit` that predicts on
new data the way R's `predict()` does: it rebuilds the design with the
training levels and spline knots, and evaluates every offset expression
again on the new rows.

| Argument | Description |
|---|---|
| `formula` | an R model formula as a string, offsets included, see below |
| `data` | a pandas DataFrame (an object with a `to_pandas()` method is converted) |
| `family` | `"poisson"`, `"quasipoisson"`, `"gamma"` (`"Gamma"` is accepted too), `"gaussian"`, `"binomial"`, `"tweedie"` or `"negbin"` |
| `link` | `None` takes the family's default (table below); otherwise `"log"`, `"identity"`, `"inverse"`, `"logit"`, `"sqrt"` or `"probit"` |
| `weights` | prior weights: a column name or an expression on the columns, e.g. `"AantalClaims"` for a severity model on the average claim |
| `var_power` | Tweedie variance power (default 1.5) |
| `theta` | negative binomial: `None` estimates theta as `MASS::glm.nb()` does; a number fixes it, as `MASS::negative.binomial(theta)` |
| `maxiter`, `tol` | IRLS iteration limit and convergence tolerance |

Rows with a missing value in any variable the model uses (response,
predictors, offsets, weights) are left out, as R's default `na.omit` does,
and unused levels of a `Categorical` are dropped, as `model.frame()` drops
them.

### The formula

The formula is written exactly as in R:

```python
m = fit_glm("AantalClaims ~ cut(LEEFTIJD, c(17, 24, 40, 65, 99)) + pmin(GEWICHT, 1600)"
            " + REGIO * factor(BOEKJAAR) + offset(log(Exposure))", dat)
print(m.term_labels)
```

| R syntax | Example | Notes |
|---|---|---|
| main effects and interactions | `REGIO + LEEFTIJD`, `LEEFTIJD * REGIO`, `LEEFTIJD:REGIO`, `(REGIO + LEEFTIJD + factor(BOEKJAAR))^2` | term labels and their order follow R (`REGIO:LEEFTIJD`) |
| no intercept | `REGIO - 1` | |
| offsets | `offset(log(Exposure))`, `offset(log(Exposure)) + offset(log(BM))` | any number of them; see below |
| factors | `factor(BOEKJAAR)` | numeric codes as levels |
| bands | `cut(LEEFTIJD, c(17, 24, 40, 65, 99))` | R's labels, `(17,24]`; `labels =`, `right =`, `include.lowest =` and `dig.lab =` work |
| splines and polynomials | `ns(LEEFTIJD, 4)`, `ns(LEEFTIJD, knots = c(25, 40, 60))`, `bs(LEEFTIJD, df = 5)`, `poly(GEWICHT, 2)` | R-compatible bases, `Boundary.knots =` included; see [Spline bases](#spline-bases-ns-bs-poly) |
| transformations | `log()`, `exp()`, `sqrt()`, `abs()`, `log2()`, `log10()`, `log1p()`, `floor()`, `ceiling()`, `round()`, `pmin()`, `pmax()`, `ifelse()`, `I()` | `I(GEWICHT^2)`, `ifelse(LEEFTIJD < 25, 1, 0)`, `TRUE`/`FALSE` |
| backticked names | ``log(`AUTO GEWICHT`)`` | column names that are not identifiers |

Not supported: `.` (all other columns; name them), the separate `offset =`
argument of R's `glm()` (write the offset in the formula; `fit_glm()` has no
such argument) and `na.action` (a fit is always `na.omit`).

The design matrix itself is built by patsy, so the coefficient names are
patsy's: `REGIO[T.West]` where R writes `REGIOWest`, and
`ns(LEEFTIJD, 4)[0]` where R writes `ns(LEEFTIJD, 4)1`. The term labels, which
is what every function of this package reports, are R's.

### Offsets

An offset is written in the formula, as often as needed, and kept as an
**expression** rather than as a column of numbers. That is the whole point:
every time the model scores data, the expression is evaluated again on
those rows, which is what R's `predict()` does and what the package relies
on to take the exposure out of a prediction.

```python
m_freq.offset_exprs                      # ['log(Exposure)']
new = dat.head(5)
m_freq.predict(new)                      # expected claims, exposure included
m_freq.predict(new, drop_offsets=[0])    # offset 0 taken out: claims per policy-year
m_freq.predict(new, type="link")         # the linear predictor
```

`drop_offsets` takes the positions in `offset_exprs` of the offset
expressions to subtract on the link scale, so with
`offset(log(Exposure)) + offset(log(BM))` the call `drop_offsets=[0]` gives
the rate per policy-year at each policy's own bonus-malus scale, and
`drop_offsets=[0, 1]` the rate at BM = 1. That is exact for any link and
any offset expression; the analysis functions do it for you, and the
[methodology chapter](#deriving-the-offset-and-the-link) explains which part
of the offset each one takes out and why.

### Families and links

| `family` | Default link | Dispersion | R equivalent |
|---|---|---|---|
| `"poisson"` | log | fixed at 1 | `poisson()` |
| `"quasipoisson"` | log | Pearson estimate | `quasipoisson()` |
| `"gamma"` | **log** | Pearson estimate | `Gamma(link = "log")`; R's own default link is the inverse |
| `"gaussian"` | identity | Pearson estimate | `gaussian()` |
| `"binomial"` | logit | fixed at 1 | `binomial()` |
| `"inverse.gaussian"` | log | Pearson estimate | `inverse.gaussian(link = "log")`; R's own default link is 1/mu^2 |
| `"tweedie"` | log | Pearson estimate | `statmod::tweedie(var.power, link.power = 0)`, with `var_power` |
| `"negbin"` | log | fixed at 1, theta estimated or given | `MASS::glm.nb()`, or `MASS::negative.binomial(theta)` with `theta` |

AIC is computed as R computes it; it is `NaN` for the quasi-Poisson and the
Tweedie, where R reports `NA`.

### Weights

```python
m_sev = fit_glm("AvgLoss ~ GEWICHT + REGIO", claims, family="gamma",
                weights="AantalClaims")
```

Pass the prior weights as a **column name** or an expression on the columns.
Like an offset, it is evaluated again on any data the model is held
against: `plot_glm_predictor()` on a later period weights by that period's
claim counts, and a refit weights by the claim counts of its own rows. An
array of weights is accepted for the fit itself, but it cannot be
re-evaluated on other rows, so `refit()`, `plot_glm_predictor(data=...)`
and a sampled `plot_glm_influence()` refuse such a model with a
`PricingError`, as R's `update()` does, and `screen_features()` (which
refits on a training split) falls back to the supplied model with a
warning. So put the weights in a column.

### Predicting and refitting

```python
new = dat.head(3).assign(Exposure=1.0)
m_freq.predict(new)                                  # predict(m, new, type = "response")
m_2023 = m_freq.refit(dat[dat.BOEKJAAR <= 2023])     # update(m, data = ...)
print(m_2023)
```

`predict(newdata=None, type="response", drop_offsets=None)` predicts on the
training rows when `newdata` is `None`, and returns a numpy array. A level
the model was not fitted on is a `PricingError` naming the value, never a
silent `NaN`; the analysis functions that score data
(`premium_impact()`, `model_lift()`, `double_lift()`,
`plot_glm_predictor(data=...)`) check this beforehand and name the column,
the values and the number of rows. `refit(data)` fits the same formula,
family, link and weights on other data.

### What a `GLMFit` carries

The attributes mirror an R `glm` object where the package uses them:

| R | Python (`m` is a `GLMFit`) |
|---|---|
| `formula(m)` | `m.formula` |
| `family(m)$family`, `family(m)$link` | `m.family` (`"gamma"` for R's `"Gamma"`), `m.link`; `m.family_label` is the name as R prints it |
| `m$data` (the rows used) | `m.data`: the rows the fit used, with the original columns and the original index |
| `model.frame(m)[[1]]`, `m$prior.weights`, `m$offset` | `m.y`, `m.prior_weights`, `m.offset` (`None` without an offset) |
| `fitted(m)`, `predict(m, type = "link")` | `m.fitted`, `m.eta` |
| `predict(m, newdata, type = "response")` | `m.predict(newdata)`; `drop_offsets=` takes offset expressions out |
| `coef(m)`, `vcov(m)` | `m.coef` (a Series, `NaN` for an aliased column), `m.vcov` (a DataFrame, estimated columns only) |
| `summary(m)$coefficients` | `m.summary_table()` |
| `deviance(m)`, `m$null.deviance`, `df.residual(m)`, `AIC(m)`, `m$rank`, `nobs(m)` | `m.deviance`, `m.null_deviance`, `m.df_residual`, `m.aic`, `m.rank`, `m.nobs` |
| Pearson dispersion | `m.dispersion`; `m.dispersion_summary` is the one `summary()` uses (1 for Poisson, binomial and negative binomial) |
| `residuals(m, type)` | `m.residuals("deviance" / "pearson" / "response" / "working")` |
| `hatvalues(m)`, `cooks.distance(m)`, `rstandard(m)` | `m.hatvalues()`, `m.cooks_distance()`, `m.rstandard()` |
| `model.matrix(m)` | `m.model_matrix(newdata=None)`: the full design, aliased columns included |
| `attr(terms(m), "term.labels")`, `m$xlevels` | `m.term_labels`, `m.xlevels` (keyed by patsy's code of the term) |
| `m$theta` (glm.nb) | `m.theta` |
| `update(m, data = d)` | `m.refit(d)` |

`m.result` is the underlying statsmodels results object, for anything else
statsmodels offers; its parameter names are generic (`const`, `x1`, ...),
so read coefficients from `m.coef` or `m.summary_table()`.

```python
print(m_freq.summary_table())
print(m_freq.deviance, m_freq.aic, m_freq.dispersion)
```

---

## Working with the results

The R package returns data frames, lists and plotly widgets. Here they are:

- **Tables** are pandas DataFrames with R's column names and a fresh
  `0..n-1` index.
- **A list of results** is a `Result`: a dict whose items can also be read
  as attributes, so `impact$summary` becomes `impact.summary` or
  `impact["summary"]`, and `list(impact)` names what is in it.
- **Attributes on a table**, R's `attr(tbl, "intercept_premium")`, are in
  `DataFrame.attrs`: `tbl.attrs["intercept_premium"]`. pandas does not carry
  `attrs` through every operation, so read them off the table as it was
  returned.
- **Plots** are `plotly.graph_objects.Figure` objects. A table that R
  attaches to a plot is an attribute of the figure: `fig.cells`
  (`plot_residual_heatmap()`), `fig.residuals` and `fig.type`
  (`plot_glm_qq()`), `fig.influential` (`plot_glm_influence()`). Python
  plotly takes the mode-bar settings at render time, so R's
  `plotly::config()` (PNG export, a file name per plot) is kept as
  `fig.pricing_config`:

```python
fig = make_plot(dat, "REGIO", "Frequency", ta_blue, y_label="Frequency",
                by_year=False)
fig.show(config=fig.pricing_config)                     # as R shows it
fig.write_html("oneway_regio.html", config=fig.pricing_config)
```

- **Conditions** keep R's three kinds. R's `stop()` raises a
  `PricingError` (a `ValueError`), `warning()` issues a `PricingWarning`,
  and `message()` issues a `PricingMessage`, a warning category of its own
  that Python shows once per call site. The texts are R's, with the
  examples inside them in Python syntax, so a search finds the same message
  in both packages.

```python
import warnings
warnings.filterwarnings("ignore", category=PricingMessage)   # R's suppressMessages()

try:
    make_rating_table(m_freq, m_sev, data=dat, base={"REGIO": "Oost"})
except PricingError as e:
    print(e)
```

---

## Function reference

What each function does, how to call it and how to read its output. The
examples continue on the quick-start portfolio (`dat`, `claims`, `m_freq`,
`m_sev`). The signatures are the Python ones, taken from the code; every
argument has the R name and the R default, with `None` for `NULL`, a tuple
for `c(lo, hi)` and a dict for a named list. The docstring of each function
(`help(make_rating_table)`) is the full reference, translated from the R
documentation.

### `agg_all()`

Aggregates raw policy rows to one row per level of a grouping column
(optionally per accounting year), with frequency and severity computed on
the aggregate. This is the aggregation engine used internally by
`make_plot()`; call it directly when you want the summary table itself.

```python
agg_all(d, col, by_year,
        exposure_col="Exposure", claims_col="AantalClaims",
        loss_col="SCHADELAST", year_col="BOEKJAAR")
```

| Argument | Description |
|---|---|
| `d` | DataFrame with raw rows |
| `col` | grouping column name (string) |
| `by_year` | `True` = also split by accounting year |
| `*_col` | column-name overrides (see [Data conventions](#data-conventions)) |

**Returns** a DataFrame with `col`, `[Year]`, `Exposure`, `ClaimCount`,
`Loss`, `Frequency` (`ClaimCount / Exposure`, `NaN` when exposure ≤ 0) and
`Severity` (`Loss / ClaimCount`, `NaN` when there are no claims). The groups
come in the order they first occur in the data, as data.table's `by =`
gives them in R; `Year` is a categorical of the year labels.

```python
print(agg_all(dat, "REGIO", by_year=False))
print(agg_all(dat, "REGIO", by_year=True).head())
```

Warnings are raised for missing values in the input columns and for groups
with non-positive exposure, so data-quality issues are visible instead of
silently averaged away:

```python
d = dat.head(2000).copy()
d.loc[[0, 1], "Exposure"] = np.nan
agg_all(d, "REGIO", by_year=False)
# PricingWarning: agg_all: 2 NA value(s) in column 'Exposure' are ignored (na.rm = TRUE).
```

### `make_plot()`

One-way plot of frequency or severity with volume bars on a secondary axis.
Takes **raw policy rows** and aggregates internally via `agg_all()`.

```python
make_plot(data, col, metric="Frequency", color_single=None, y_label=None,
          display="color", by_year=None, metric_fmt=4,
          exposure_col="Exposure", claims_col="AantalClaims",
          loss_col="SCHADELAST", year_col="BOEKJAAR",
          discrete_cutoff=25, y_range=None)
```

- `metric` selects what is plotted: `"Frequency"` (claims / exposure) or
  `"Severity"` (loss / claim), both computed on the internal aggregate.
- **The bars follow the metric's own denominator**: exposure under a
  frequency, the claim count under a severity. Those two routinely run in
  opposite directions, the level with the most exposure is often the one
  with the fewest claims, so exposure bars under a severity line would
  tell the reader the opposite of how solid each point is.
- `display="color"`: one coloured line per year in a single panel. Note:
  the bars then show the **sum over all years**, and the trace is labelled
  "... (all years)" accordingly.
- `display="facet"`: one panel per year. The bars are rescaled per facet to
  the metric range; the tooltip shows true values.
- Categorical X-axes (and numeric columns with ≤ `discrete_cutoff` unique
  values) are drawn as markers only, so categories are never connected by a
  line.
- `by_year`, `y_label` and `color_single` (where it is used) have no
  default in R; leaving them out here is an error that says so, as it is
  there.

**Returns** a plotly figure: the bars, and one line per year or a single
line.

```python
make_plot(dat, "LEEFTIJD", "Frequency", ta_blue, y_label="Claim frequency",
          by_year=False).show()
make_plot(dat, "GEWICHT", "Severity", ta_gold, y_label="Average claim",
          display="facet", by_year=True).show()
```

### `make_pdp()`

**The question it answers:** *what does this one variable do to the price,
holding everything else where it is?*

```python
make_pdp(model, raw_data, pred_var, metric="Frequency", transform=None,
         grid_res=50, y_label=None, metric_fmt=4,
         exposure_col="Exposure", claims_col="AantalClaims",
         loss_col="SCHADELAST", discrete_cutoff=25, y_range=None,
         keep_offsets=True)
```

**What it does.** Take your whole portfolio and pretend, for a moment,
that every single policy is 30 years old. Price them all. Average the
result. Then do the same for 31, 32, and so on. Each point on the line is
"what this book would cost if everyone sat at this value", so the mix of
every *other* rating factor is identical at every point, and the line shows
the effect of age alone.

That is the difference from the observed line drawn beside it. The observed
line groups the policies that really are 30 years old, and those people also
differ in region, vehicle and everything else. The two lines answering
different questions is normal, not a sign of misfit.

**Reading the chart:**

| Element | What it is |
|---|---|
| **Bars** | how much data sits at each value, exposure for frequency, claim count for severity. Short bars mean a thin, unreliable stretch of the line |
| **Observed** (dotted) | what actually happened, per group, mix and all |
| **PDP (model)** (solid) | what the model says this variable does on its own |

The two lines running roughly parallel is the healthy picture. The PDP being
far flatter than the observed line usually means the variable is picking up
something another factor already explains. A PDP that jumps where the bars
are tiny is the model extrapolating, not a real effect.

Prices are always **per unit of exposure**, a full year, so a frequency of
0.12 means 12 claims per 100 policy-years, whatever the actual policy
durations are.

A second offset, a bonus-malus scale carried as `offset(log(BM))`, stays in
at each policy's own value, because the observed line is at each policy's
own BM as well. Taking it out put the PDP at the class without discount: on
a book with an average BM of 0.74 it ran 35% above the observed line at
every level of a variable the model had exactly right. `keep_offsets=False`
takes it out anyway, and the y-axis title names the offsets that stayed in.
The rating table is the place for factors at BM = 1.

`transform` is kept for compatibility with old scripts, as in R: it is
deprecated and ignored, with a warning.

**Performance.** Policies that are identical on every *other* predictor
get the same answer, so they are collapsed into one before predicting. That
turns `grid_res × 1,000,000` predictions into `grid_res × 54,000` on a large
book, with an identical result. If it is still slow, the number of distinct
combinations is the driver: band a very granular continuous variable, or
lower `grid_res`.

**Returns** a plotly figure with the bars, an "Observed" line and a
"PDP (model)" line.

```python
make_pdp(m_freq, dat, "LEEFTIJD", metric="Frequency").show()
make_pdp(m_sev, claims, "GEWICHT", metric="Severity").show()
```

For a severity model, pass the data the observed severity should come from:
`claims` here, where `SCHADELAST` and `AantalClaims` give the observed
`Σloss / Σclaims` per level. Why the averaging is arithmetic rather than
geometric: [why the PDP is computed on the response
scale](#why-the-pdp-is-computed-on-the-response-scale).

### `plot_glm_predictor()`

Actual-vs-expected plot per predictor, on the model's own training data or
on any later data you pass.

```python
plot_glm_predictor(model, predictor, data=None, n_bins=150,
                   weight_var=None, weight_label=None,
                   color=ta_navy, color_pred=ta_gold,
                   title=None, ylab=None, xlab=None,
                   metric_fmt=4, bin_type="quantile",
                   exposure_col="Exposure",
                   ci=True, ci_level=0.95, y_range=None,
                   ratio=False)
```

Mode is detected automatically:

- **Offset model with log link** (typical frequency model): the response and
  predictions are counts; both are divided by the exposure, giving observed
  and predicted **frequency** per bin.
- **Otherwise** (typical severity model): the response is already an average;
  observed and predicted are **prior-weight-weighted means** per bin (e.g.
  claim-count-weighted for severity).
- `weight_var` overrides the weight column explicitly (it must exist in
  the model's data; an unknown name is an error, not silently ignored).

```python
plot_glm_predictor(m_freq, "LEEFTIJD", n_bins=50).show()
plot_glm_predictor(m_sev, "GEWICHT").show()
```

**Monitoring a later period.** `data` evaluates the same comparison on rows
the model never saw, which is the usual way to check whether a tariff still
matches the experience:

```python
m_hist = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + REGIO + offset(log(Exposure))",
                 dat[dat.BOEKJAAR <= 2023])
recent = dat[dat.BOEKJAAR == 2024]
plot_glm_predictor(m_hist, "REGIO", data=recent,
                   title="Frequency - REGIO - 2024").show()
```

The data needs the predictors, the response and any weight or offset
column, since the plot holds the realised figure against the prediction.
Rows with a missing value in a model variable are dropped, as the fit
dropped them, and a level the model never saw is an error that names the
column, the values and how many rows hold them, rather than a quietly
missing row:

```python
plot_glm_predictor(m_freq, "REGIO", data=recent.assign(BOEKJAAR=2025))
# PricingError: plot_glm_predictor: the data holds value(s) the model was not fitted
#   on: 'BOEKJAAR' = 2025 (10046 rows). A model has no factor for a level it never
#   saw. Map those values to a level it knows or leave the rows out.
```

Weights and offsets are evaluated again on the new rows, so a severity
model weighted by the claim count keeps weighting by that period's claim
counts.

One thing reads differently out of sample. In-sample, a categorical term
that is in the model is pinned to the observed level under a canonical
link, so the two lines coincide by construction and tell you nothing. On
other data nothing is pinned, and a gap is evidence that the level has
moved. The dispersion behind the error bars still comes from the fitted
model, because it is a property of the model and not of the period. The
title does not say which data was used, so set `title` yourself when you put
several periods next to each other.

**The A/E ratio itself.** `ratio=True` replaces the two lines with their
ratio: one A/E line and a dashed reference line at 1.

```python
plot_glm_predictor(m_hist, "REGIO", data=recent, ratio=True).show()
```

Two lines show the **level**, which is the view you want when the question
is how the risk runs across a variable. The ratio shows the **error**,
which is the view you want when the question is where the tariff is off: a
level sitting 10% low is obvious at 0.90, and barely visible when two
lines nearly touch on an axis that spans the whole book. Counts give
`Σy / Σμ` per bin, a weighted model the ratio of the two weighted means,
so it is the same A/E that `plot_residual_heatmap()` colours per cell and
`double_lift()` plots per bin. The error bars are the ones on the observed
point divided by the prediction, so a bin whose band covers 1 is not
evidence of anything.

It only says something where observed and predicted are free to differ,
which is the same caveat as above: use it on data the model never saw, on
a variable outside the model, or per bin of a continuous variable. On a
categorical term that is in the model, in-sample, it is a flat line at 1
by construction.

**More than one offset.** A model may carry a second, known relativity as
an offset next to the exposure, a bonus-malus scale that is given rather
than estimated, `offset(log(Exposure)) + offset(log(BM))`. `exp(offset)` is
then `Exposure × BM`, and dividing by it would put the axis in claims per
*BM-adjusted* policy-year, which is not a quantity anyone reads off a chart.
Both series are divided by `exposure_col` instead, so the axis stays in
claims per policy-year and the bars show real exposure. The A/E is untouched
by the choice, because both sides divide by the same thing. `make_pdp()`
weights its average by the same column, so the two halves of that chart
agree. With a single `offset(log(Exposure))` nothing changes: the two are
the same number.

The rating table handles the second offset the way you would want: it
removes **both**, so the intercept and factors are the base tariff *before*
BM, and the premium chain becomes

```text
premie = intercept × Π factoren × BM × Exposure
```

which reconstructs the model prediction exactly:

```python
dat["BM"] = rng.choice([0.6, 0.8, 1.0, 1.2], len(dat))      # a given BM scale
dat["ClaimsBM"] = rng.poisson(m_freq.fitted * dat.BM)       # claims at that scale
m_bm = fit_glm("ClaimsBM ~ ns(LEEFTIJD, 4) + REGIO + factor(BOEKJAAR)"
               " + offset(log(Exposure)) + offset(log(BM))", dat)
tbl_bm = make_rating_table(m_bm, None, data=dat, claims_col="ClaimsBM")

def factor(tbl, var, level):
    rows = tbl[(tbl.Variable == var) & tbl.Group.isna()]
    return rows.loc[rows.Level == str(level), "Factor_Frequency"].item()

pol = dat.iloc[0]
chain = (tbl_bm.attrs["intercept_frequency"]
         * factor(tbl_bm, "LEEFTIJD", pol.LEEFTIJD)
         * factor(tbl_bm, "REGIO", pol.REGIO)
         * factor(tbl_bm, "BOEKJAAR", pol.BOEKJAAR)
         * pol.BM * pol.Exposure)
print(chain, m_bm.predict(dat.iloc[[0]]))                    # the same number
```

**Error bars.** The observed points carry a 95% interval by default, which
turns the plot from "these two lines differ" into "these two lines differ
by more than the noise here". A bin whose bar comfortably spans the
predicted line is not evidence of anything, however far apart the two
markers look, and a bar that is long simply says the bin is thin. Switch
them off with `ci=False` or widen them with `ci_level`.

The interval is on the **observed** point only. The predicted line is the
model; its parameter uncertainty is a different and usually much smaller
quantity. The standard error comes from the fitted family's own variance
function, scaled by the Pearson dispersion where the family estimates one:
counts use `sqrt(φ·Σ V(μ)) / Σexposure`, weighted means the usual
`sqrt(φ·Σ w·V(μ)) / Σw`. Measured coverage on a correctly specified
Poisson is 0.96 for a binned continuous predictor and 0.98 for a
categorical one outside the model.

One case where the bars cannot be read as a test: for a **categorical term
that is in the model** under a canonical link, observed equals predicted at
every level by construction, so the bar always contains the predicted point,
and coverage is 1 by identity, not by fit. The bar length still tells you
how much evidence the level carries, but the absence of a gap there is
arithmetic. That is the blind spot
[`detect_interactions()`](#detect_interactions) exists to cover; see [why a
one-way check cannot see an
interaction](#why-a-one-way-check-cannot-see-an-interaction).

**Binning.** A numeric predictor with at most `n_bins` distinct values is
not binned at all: every value gets its own point, on its exact position.
That matters for variables like vehicle age or number of claims: binning
would merge neighbouring values (ages 0 and 1 into one point at, say,
0.78) and, because equal-width bins span the observed range, a single
outlier would silently shift every point on the plot.

Only when there are more distinct values than `n_bins` does binning kick in:
**quantile bins** by default (each bin holds roughly the same number of
observations, avoiding noisy thin tails), or equal-width bins with
`bin_type="width"`. A binned point then sits at the weight-weighted mean of
the values in its bin, not at the bin edge.

**Shared y-axis across variables.** By default the primary y-axis
auto-scales to the values of that one plot, so two predictors on very
different scales can look equally volatile. Pass `y_range=(lo, hi)` to pin
the axis, and reuse the same value across predictors to make them directly
comparable:

```python
rng_y = (0, 0.3)
plot_glm_predictor(m_freq, "LEEFTIJD", y_range=rng_y).show()
plot_glm_predictor(m_freq, "REGIO", y_range=rng_y).show()
```

Both plots then share one axis, so the stronger driver is visibly the
stronger one. Note the trade-off: a shared axis flattens variables with a
narrow spread of their own, so use auto-scaling when inspecting one variable
in detail and a fixed range when comparing variables side by side.

`y_range` is available on every plotting function in the package
(`make_plot()`, `make_pdp()`, `make_rating_plot()`, `plot_glm_residuals()`,
`plot_glm_qq()`, `plot_glm_influence()`, and `premium_impact()`, which
additionally takes an `x_range` for its change axis; the heatmap's colour
scale has `z_range`). It is `None` everywhere by default, which keeps the
usual auto-scaling.

**Returns** a plotly figure with weight bars and an "Observed" and a
"Predicted" line, or with `ratio=True` an "A/E" line and a reference line
at 1.

### `glm_diagnostics()`

Compact fit summary for a frequency and/or severity model.

```python
glm_diagnostics(model_freq=None, model_sev=None)
```

**Returns** a DataFrame with one row per model: `Model`, `Family`, `Link`,
`N`, `Deviance`, `DFResidual`, `AIC`, `Dispersion` (Pearson χ² / df) and
`DevianceExplained` (1 − deviance / null deviance, a pseudo-R²). `Family`
is written as R writes it (`Gamma`, `Negative Binomial(1.5)`).

```python
print(glm_diagnostics(m_freq, m_sev))
```

For Poisson/binomial families a **dispersion above 1.2 raises an
overdispersion warning**: the point estimates are still consistent, but
standard errors are understated, so consider a quasi-Poisson or negative
binomial family before reading anything into term significance.

### `glm_collinearity()`

**The question it answers:** *are two of my rating factors telling the
model the same thing?*

```python
glm_collinearity(model, threshold=3)
```

**What it does.** If you put both vehicle weight and engine capacity in a
tariff, the model cannot tell which of the two is doing the work: they move
together. It will split the effect between them, and it can split it almost
arbitrarily: a big positive factor on one and a compensating negative one on
the other, giving the right total price for a nonsense pair of factors. This
function measures, for each term, how much of it the other terms already
explain.

**The number to read is `GVIF_scaled`.** It says how many times wider that
term's uncertainty is because of overlap with the rest:

| `GVIF_scaled` | Meaning |
|---|---|
| 1.0 | this term stands on its own |
| 2.0 | its factors are twice as uncertain as they would be alone |
| **3.0 or more** | flagged: the factors are not readable individually |

On the R package's demo portfolio, two near-duplicate weight columns were
flagged at 8.4, while every independent term sat at 1.0. The same situation
on the quick-start portfolio:

```python
dat["GEWICHT_PROXY"] = dat.GEWICHT + rng.normal(0, 20, len(dat))   # a near-duplicate
m_col = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + REGIO + GEWICHT + GEWICHT_PROXY"
                " + offset(log(Exposure))", dat)
print(glm_collinearity(m_col))
# PricingWarning: glm_collinearity: 2 term(s) at or above a scaled GVIF of 3
#   (GEWICHT_PROXY, GEWICHT); their factors are unstable and should not be read
#   individually.
```

**Returns** a DataFrame with `Term`, `DF`, `GVIF`, `GVIF_scaled`,
`InInteraction` and `Flag`, ordered by `GVIF_scaled`. `Flag` is a nullable
boolean column, so a term that is not judged reads `<NA>`.

**The most important thing about this diagnostic:** collinearity damages
**interpretation, not prediction**. The model's premiums can be perfectly
accurate while the individual factors behind them are meaningless. That
matters here because a rating table *is* an interpretation: you are
publishing those factors, so they have to mean something on their own.

The fix is usually to drop one of the pair, or combine them into a single
variable. Which one you keep is a business decision, not a statistical one;
the model has no preference.

**Terms in an interaction are reported, not flagged.** For an interaction,
and for every main effect it contains, the GVIF depends on how the
variables are coded rather than on the data. In a model with
`pmin(LEEFTIJD, 30):STEDELIJKHEID`, writing the age part as
`pmin(LEEFTIJD, 30) - 30` moved the scaled GVIF of `STEDELIJKHEID` from
14.12 to 1.11, with identical predictions. Those terms get
`InInteraction = True` and `Flag = <NA>`, with a warning naming them; judge
them on the stability of the combined effect instead.

> The counterpart of the near-duplicate warning in
> [`screen_features()`](#screen_features): that one catches correlated
> *candidates* before they go in, while this one catches them once both are in.

Why an ordinary VIF will not do here: a plain VIF is per coefficient, and a
pricing term (a spline, a factor) spans several columns. The generalised VIF
of Fox and Monette handles multi-column terms, and `GVIF^(1/(2·DF))` puts
them back on a scale comparable across terms, read like a VIF on the
square-root scale: the default threshold of 3 corresponds to a VIF of about
9.

### `plot_glm_residuals()`

Binned residual plot, the readable alternative to raw residual plots, which
are dominated by the 0/1 claim pattern on policy-level count data.

```python
plot_glm_residuals(model, predictor=None, n_bins=50,
                   residual_type="pearson", y_range=None)
```

Residuals are averaged per quantile bin of the **fitted value** (default) or
of a **predictor** (`predictor="LEEFTIJD"`), and drawn together with a ±2·SE
band (scaled by the estimated dispersion). Under a correctly specified model
roughly 95% of the bin means should fall inside the band; a systematic
pattern outside it (e.g. a curve over a predictor) signals missed structure,
a candidate spline, banding or interaction. Categorical predictors get error
bars per level instead of a ribbon.

The bin means are drawn as **points, not joined into a line**. They are
independent draws, and a connecting line invites the eye to trace a trend
through what is mostly noise; the band is what says whether a point means
anything.

**Deviance residuals are centred first.** A deviance residual is not zero
on average under a correct model: a claim-free policy has `-sqrt(2 mu)`, so
on a frequency model with small means the residuals are negative on
average. On a correct Poisson book of 60,000 rows every one of 50 bins sat
below the band (mean -0.220 against a band of 0.058). With
`residual_type="deviance"` each residual is therefore shown minus its
expectation under the fitted model, and the band follows each bin's own
variance; the axis says "centred". That is exact for the Poisson,
quasi-Poisson, negative binomial, binomial, Gamma and Gaussian families;
anything else is shown uncentred with a warning. Pearson residuals need
none of this.

```python
plot_glm_residuals(m_freq).show()                                  # vs fitted value
plot_glm_residuals(m_freq, "LEEFTIJD", residual_type="deviance").show()
plot_glm_residuals(m_freq, "REGIO").show()                         # error bars per level
```

**Returns** a plotly figure.

### `plot_glm_qq()`

**The question it answers:** *the model gets the average right, but does
it get the spread right?*

```python
plot_glm_qq(model, n_max=5000, band=True, seed=None,
            title=None, y_range=None)
```

**Why you would care.** A model can predict an average claim of €2,000 and
be right on average, while being completely wrong about how claims are
spread around that. If real losses have a much fatter tail than the model
assumes, every premium is fine on average and the capital you hold behind
them is not. Every other plot in this package checks the average. This one
checks the shape.

**What it does.** For each policy it asks: *given what the model predicted
for you, how extreme was your actual outcome?* A policy at the model's 90th
percentile scores 90%. Do that for everyone and, if the model has the shape
right, those scores should be spread perfectly evenly from 0 to 100%. The
plot converts them to a scale where "perfectly even" is a straight diagonal
line.

**Reading the chart:**

| What you see | What it means |
|---|---|
| Points on the diagonal | the family fits, nothing to do |
| Ends curling **up on the right, down on the left** | real outcomes are more extreme than assumed; the tail is fatter than the model thinks |
| Ends bending **towards the middle** | the model assumes more spread than there is |
| An S-shape on a frequency model | over- or underdispersion, the same message [`glm_diagnostics()`](#glm_diagnostics) gives as a number |
| A curving upper tail on a Gamma severity | large losses are heavier than Gamma allows; consider a log-normal, or capping |

The shaded band is where points may fall by chance. It is *pointwise*, so
with several thousand points a handful will always sit outside it even when
the model is right. **Read the shape of the curve, don't count the
exceptions.**

**One thing to know before you trust the standard version.** R's own
`plot(model)` draws this chart too, and for claim counts it is unusable: it
rejects models that are exactly correct. Measured on a Poisson that was
right by construction, 20,000 rows and 87% of them claim-free:

| Method | points outside the band (should be 5%) | verdict on a correct model |
|---|---|---|
| base R `plot(model)` | 8.3% | rejected |
| `boot::glm.diag.plots()` | 1.3% | rejected |
| **this function** | **5.1%** | **accepted** |

The reason is that claim counts are whole numbers, mostly zero, and the
classical residuals were built for continuous data. This function uses
**quantile residuals** (Dunn & Smyth, 1996), which handle that correctly.
The randomisation involved means two calls give slightly different pictures:
pass `seed` if you need the same one twice. A negative binomial
(`family="negbin"`) gets quantile residuals as well, from its own CDF, as
long as it carries no prior weights; any other family falls back to
deviance residuals and says so.

```python
q = plot_glm_qq(m_freq, seed=1)
q.show()
print(q.type, q.residuals[:5])      # "quantile", and the residual of every row
plot_glm_qq(m_sev).show()
```

**Returns** a plotly figure; the residuals of every row are attached as
`fig.residuals`, and `fig.type` says whether they are quantile or deviance
residuals. `n_max` only thins what is drawn (both extremes are always kept).

The R implementation is verified against `statmod::qresid()`, the reference
implementation (identical for Poisson, equal to 6e-14 for Gamma), and this
port against the R package (see the [parity
table](#parity-measured-against-r)). The mathematics and the per-family
formulas: [reading a Q-Q plot](#reading-a-q-q-plot-of-quantile-residuals).

### `plot_glm_influence()`

**The question it answers:** *is one odd policy pulling my whole tariff
around?*

```python
plot_glm_influence(model, n_label=10, data_cols=None, n_max=5000,
                   max_rows=2e5, seed=None, title=None, y_range=None)
```

**What it does.** A model fits every policy at once, but not every policy
gets an equal vote. One record with a vehicle weight of 20 tonnes, because
someone typed kilograms where tonnes were meant, can bend a whole curve
towards itself. This plot finds the records with the loudest votes and shows
you *why* each one is loud.

The chart has two axes, and the combination is the point:

- **Left to right, how unusual the policy is.** Far right means its
  rating factors are unlike anything else in the book.
- **Up and down, how badly the model fits it.** Far from the middle line
  means the actual outcome was nothing like the predicted one.
- **Bubble size, how much the fit actually moves** because of it.

On the R package's demo portfolio, the big bubble on the far right is the
20-tonne car. Note where it sits vertically: almost on the zero line, i.e.
**apparently well fitted**. That is the trap this plot exists to expose, a
record that unusual drags the model towards itself until its own residual
looks innocent. It hides from every residual check while quietly distorting
the curve for everyone else.

The `n_label` strongest records come back as a table you can act on. Implant
the keying error in the quick-start portfolio, on a policy with a claim, and
it comes out on top:

```python
d_inf = dat.copy()
d_inf.loc[7, ["GEWICHT", "AantalClaims"]] = [20_000, 1]   # kilograms typed as tonnes
m_inf = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + REGIO + GEWICHT"
                " + offset(log(Exposure))", d_inf)
p = plot_glm_influence(m_inf, n_label=10)
p.show()
print(p.influential.head(3))   # Row (the index label) 7 first; CooksD, Leverage,
                               # StdResid, Actual, Fitted and the model's variables
```

Give the same record no claim and the fit can push its expected claims, and
with them its weight in the fit, towards zero, and it no longer stands out:
the other half of the story below.

**Two things this plot is not.**

It is a **ranking, not a test**. On 100,000 policies no single record swings
a coefficient much, and the textbook thresholds are useless at that size:
`D > 1` never fires, `D > 4/n` flags thousands of ordinary rows. So it tells
you which records are the most influential *relative to each other*, and you
go and look at them.

And it is **not a misfit check**. The two are not the same thing, and on a
frequency model they can be opposites: a policy with an exposure of 0.001
that has a claim is the worst-fitted record in your book and is completely
invisible here, because a sliver of exposure carries almost no weight in the
fit. Use [`plot_glm_qq()`](#plot_glm_qq) or
[`plot_glm_residuals()`](#plot_glm_residuals) to catch that half.

`data_cols` picks the columns carried into the tooltip and the table
(default: the model's own variables). The leverages come from a QR
decomposition of the weighted design matrix, which grows with rows times
coefficients, so `max_rows` (default 200,000) computes on a random sample of
that size, with a message, rather than letting a large model exhaust
memory; `max_rows=None` uses every row.

Both points, with the measurements behind them: [Cook's distance on a large
book](#cooks-distance-on-a-large-book).

**Returns** a plotly figure, with the top rows attached as
`fig.influential`. Its `Row` is the index label of the row in the model's
data, so `m.data.loc[row]` opens the record.

### `detect_interactions()`

Scans every two-way combination of variables and ranks them by the
interaction structure the GLM has not captured.

```python
detect_interactions(model, vars=None, n_bins=10, min_claims=30,
                    n_sim=200, top_n=None, seed=None)
```

For each pair it builds the two-way table of actual versus expected claims,
matches the row and column margins by **iterative proportional fitting**, so
main-effect misfit (a spline needing another knot, a level that is simply
mispriced) is scaled away first, and takes the remaining deviance as the
statistic. That is the likelihood-ratio test between the additive and the
saturated model on the table.

Because that deviance is unreliable in sparse tables, the reference
distribution comes from **simulation** rather than chi-square asymptotics
whenever the model is a Poisson count model: claims are resampled from the
raked expected values under the additive null, `n_sim` times. Other families
fall back to the dispersion-scaled chi-square (reported in the `Method`
column).

**Overdispersion is taken out first.** Claim counts nearly always vary more
than a Poisson allows, and a Poisson null reads that extra variance as
interaction: on ten books with a dispersion of 1.9 and no interaction at
all, 33 of 60 pairs came out at `P < 0.05` where 3 would be expected. The
deviance is therefore divided by the dispersion of a cell total,
`sum((y - mu)^2) / sum(mu)`, and the simulation draws from a Poisson with
the expected values divided by it. That is not the row-level dispersion of
`glm_diagnostics()`: a cell total adds policies by their expected claims,
so the policies with more expected claims weigh more in it (measured 1.91
against 1.72). Over 100 books per case the false-positive rate is 3.3% at a
dispersion of 1.94 and 6.7% on pure Poisson data, and a 16% uplift in one
corner of a pair is found in 83% of the books on Poisson data and 48% at a
dispersion of 2. The value used is in the `Dispersion` column.

`vars` defaults to the model's own predictors, but you can pass variables
that are *not* in the model, which is how a completely omitted rating factor
surfaces.

```python
print(detect_interactions(m_freq, seed=1))
print(detect_interactions(m_freq, vars=["LEEFTIJD", "REGIO", "KILOMETRAGE"],
                          seed=1))
```

The quick-start portfolio has young drivers in the city simulated as extra
risky, and `m_freq` has no term for it, so `LEEFTIJD` × `REGIO` comes out on
top. On the R package's demo portfolio, missing the same kind of age ×
region interaction, the true pair came out at `Z = 33.9` against 1.6 for the
runner-up, and its worst cell was 27% mispriced on 3% of the exposure.

**Returns** a DataFrame, strongest first: `VarX`, `VarY`, `Cells`,
`Claims`, `Deviance`, `DF`, `Z`, `P`, `Method`, `Dispersion`, `MaxAE`,
`MaxAE_Claims`, `MaxAE_ExposureShare`. The simulation uses numpy's random
generator: pass `seed` for the same `Z` and `P` twice. They will not equal
R's for the same seed.

> The ranking is by statistical signal (`Z`), which is not the same as
> materiality. Judge the two separately: `MaxAE` is the A/E of the worst
> cell that has enough claims, and `MaxAE_ExposureShare` says how much of
> the portfolio sits in it.

### `plot_residual_heatmap()`

The A/E ratio per cell of two variables, the view that shows *where* a model
leaks.

```python
plot_residual_heatmap(model, var_x, var_y, n_bins=20,
                      min_claims=30, z_range=None, title=None)
```

On the quick-start portfolio, `m_freq` has clean main effects for age and
region but no interaction between them, while the simulation made young
drivers in the city extra risky. Every one-way check on this model is
spotless; the heatmap shows the youngest city cells running red
(underpriced):

```python
h = plot_residual_heatmap(m_freq, "LEEFTIJD", "REGIO")
h.show()
print(h.cells[h.cells.gy == "Stad"].sort_values("gx").head(4))   # by age band
```

- Values come from the model's own rows, so they always align with the
  fitted values; both variables must be columns of the data the model was
  fitted on.
- The colour scale is **diverging around 1.0**, blue overpriced, red
  underpriced, neutral grey at break-even. `z_range` fixes it across
  several heatmaps.
- Cells with fewer than `min_claims` claims are **left blank and marked
  with a grey cross** rather than coloured: in a thin cell the A/E is
  mostly noise, and colouring it would make it the loudest thing on the
  plot. Their volume is still in the tooltip. When every cell is thin, on a
  small book, the plot is the crosses alone, with a warning.
- The cell table (`gx`, `gy`, `Actual`, `Expected`, `Exposure`, `Claims`,
  `AE`, `IsThin`) is attached to the returned figure as `fig.cells`.

### `screen_features()`

Fits a boosted challenger with the GLM as an offset, so it can only model
what the GLM leaves behind. Requires `xgboost` (the `[screen]` extra).

```python
screen_features(model, features=None, split=(0.6, 0.2, 0.2),
                max_depth=2, eta=0.05, nrounds=2000,
                early_stopping_rounds=40, n_shap=4000,
                cor_threshold=0.95, max_levels=50, max_rows=math.inf,
                tree_method="hist", nthread=None, seed=None,
                exposure_col="Exposure", claims_col="AantalClaims",
                loss_col="SCHADELAST")
```

The baseline you pass decides which question it answers. With a **minimal
model** it screens candidates *before* you choose what goes into the tariff,
"what does this data add on top of what I already price on". With your
**full model** it asks what the finished tariff still misses. The baseline
is refitted on the training split, so both sides of the comparison are out
of sample. No scorable model is returned.

```python
dat["CONTROL"] = rng.normal(size=len(dat))     # a column you know is noise
scr = screen_features(m_freq, seed=1)
print(scr.summary)
print(scr.features)
scr.plot.show()
```

On the quick-start portfolio, `KILOMETRAGE`, simulated with a real effect
and left out of `m_freq`, should come out on top. On the R package's demo
portfolio the two tables read:

| Stage | Deviance | Change |
|---|---|---|
| baseline GLM | 16720 | |
| + booster, depth 1 (additive only) | 16650 | -0.41% |
| + booster, depth 2 (may use interactions) | 16653 | +0.02% |

| Feature | InModel | PermDeviance | Gain |
|---|---|---|---|
| KILOMETRAGE | False | 66.83 | 0.4291 |
| GEWICHT | False | 16.78 | 0.1460 |
| GEWICHT_PROXY | False | 14.21 | 0.2058 |
| KLEUR | False | 3.04 | 0.0414 |
| LEEFTIJD | True | 0.64 | 0.0381 |
| REGIO | True | -0.14 | 0.0045 |
| RUIS | False | -3.57 | 0.1351 |

The candidates are every usable column of the model's data except the
response, the offset, the prior weights and the outcome columns
`claims_col`, `loss_col` and `exposure_col`; a message names the ones left
out. That matters: offered the loss amount, the booster on a frequency
model finds that claims follow losses, ranks it first with 97% of the
gain, and the verdict is then about leakage rather than risk. A column you
name in `features` yourself stays, with a warning if it is one of those.

Things to read carefully:

- **The verdict is staged.** While the depth-1 step still improves the
  fit, main effects are missing and the interaction ranking cannot be
  trusted. Fix the main effects, then run again. `scr.verdict` says where
  you are, in one sentence.
- **Rank on `PermDeviance`, not `Gain`.** `PermDeviance` is the increase
  in out-of-sample deviance when the feature is shuffled with the baseline
  held fixed, so it is the *incremental* contribution. `Gain` is biased
  towards continuous and high-cardinality features; in the run above it
  gives pure noise (`RUIS`) a 13.5% share while the permutation test
  correctly puts it at −3.57.
- **It is a ranking, not a test.** Zero is not a threshold. The figure is
  a difference between two deviances on one random split, so it carries
  sampling noise, and its scale depends on how large that split is, so a
  value of 5 means nothing on its own. Measured on a 150,000-row book over
  three seeds, pure noise columns ranged from −3.8 to +2.3 while a genuine
  but weak predictor ranged from −1.9 to +2.0. The strong features were
  never in doubt (30 to 48); the marginal ones were not separable at all.
- **Two cheap ways to see where the noise band sits.** Add a column you
  know is noise, as `CONTROL` above, and read every other feature against
  where it lands. And re-run with two or three `seed` values: anything whose
  sign flips between runs is inside the band.
- **Near-duplicates split their importance.** `GEWICHT` and
  `GEWICHT_PROXY` each get roughly half, and which one comes out on top is
  arbitrary. Pairs above `cor_threshold` are listed in `scr.correlated`
  with a warning.

**Returns** a `Result` with `summary` (the staged deviance comparison),
`verdict`, `features` (`Feature`, `InModel`, `PermDeviance`, `Gain`),
`gain_note`, `interactions` (the SHAP ranking of pairs, or `None` with
`n_shap=0`), `correlated` (or `None`), `stats` (the raw numbers) and `plot`.
The plot ranks candidates by incremental value and mutes everything at or
below zero, a useful visual cue, not a verdict. `Gain` is xgboost's
`total_gain` normalised to sum to 1, which is what R's `xgb.importance`
reports as Gain. The sampling, the split and the boosting are random, so
the numbers are not R's for the same `seed`.

**Run time.** `tree_method="hist"` (the default, as in xgboost 2.0 and
later) is the biggest lever on a large book: on 2,000,000 rows with 6
candidates, boosting and permutation took 291.6 s under xgboost's old
default and 108.4 s under `"hist"`, for an identical feature ranking. The
next is `max_rows`: every stage is linear in rows while the ranking settles
early (on a 1.5M-row book the order of the candidates was unchanged from
100,000 rows upwards), so `max_rows=2e5` is a sensible first try. `nthread`
defaults to one fewer than the available cores.

> **Sensitivity.** A booster is *less* sensitive to a two-way interaction
> between known rating factors than
> [`detect_interactions()`](#detect_interactions): the cell-based test
> concentrates the signal into one statistic with a known null, while the
> booster must discover the split structure and pays a variance cost for
> its flexibility. On the same data, a case where the cell test reaches
> Z = 7.7 can leave the booster reporting nothing at all. Use this
> function to screen features and for the global verdict; use
> `detect_interactions()` to hunt interactions.

### `make_rating_table()`

**The question it answers:** *turn my two models into a price list I can
hand to someone.*

**What it does.** A GLM is a formula on a log scale, not something you can
put in a tariff document. This turns it into the multiplicative table
insurers actually quote from: **one base premium, and one factor per
variable that you multiply onto it.**

```text
              base premium                       € 312
  age 24      × 1.41
  region Stad             × 1.22
  fuel diesel                       × 0.94
                                            = € 504
```

Every factor says the same thing: *how many times more expensive than the
base case is this level?* A factor of 1.41 is 41% more expensive, 0.94 is 6%
cheaper, and the base level is 1.00 by definition. Multiply the base premium
by one factor per variable and you get exactly what the model predicts,
verified to machine precision, not approximately.

**The base case** is the policy every factor is measured against: the first
level of each categorical variable (or the largest by exposure, with
`base_level="exposure"`) and the median of each continuous one. It is a
reference point, not a recommendation: moving it changes every factor and
the base premium together, and leaves all quoted prices identical.

The accounting year is the exception: when a model uses `year_col`
(default `BOEKJAAR`), its base is the **latest** year, so the base premium
is today's price level and the year factors say how much cheaper earlier
years were. With the first year as base, the base premium is the price
level of the oldest year in the data. `base` still overrides it, and
`year_col=None` puts the year back under the general rule.

To pick the base yourself, name it per variable with `base`. The model is
not refitted; only the factors and the base premium move:

```python
tbl = make_rating_table(m_freq, m_sev, data=dat,
                        base={"REGIO": "Zuid", "LEEFTIJD": 40})
print(tbl.attrs["base_values"])
```

A categorical base must be an existing level. A continuous base is used
exactly as given, even between grid points, and a value outside the data
gets a warning. Variables you do not name keep the default above.

```python
make_rating_table(model_freq=None, model_sev=None, data=None,
                  grid_res=50,
                  exposure_col="Exposure", claims_col="AantalClaims",
                  base_level="first",
                  trim=(0, 1),
                  min_claims=30, grid_step=None, base=None,
                  merge_vars=None, year_col="BOEKJAAR",
                  loss_col="SCHADELAST")
```

| Argument | Description |
|---|---|
| `model_freq`, `model_sev` | fitted `GLMFit` models (at least one; log link expected) |
| `data` | original dataset, used for base values, grids and exposure |
| `grid_res` | target number of grid points for continuous variables |
| `grid_step` | step size for continuous grids; `None` picks a readable one, see [Readable grids](#readable-grids-for-continuous-variables) |
| `base_level` | `"first"` = first level is the reference; `"exposure"` = the level with the largest exposure is the reference |
| `trim` | quantile range for continuous grids, e.g. `(0.005, 0.995)` to avoid outlier tails and spline extrapolation |
| `min_claims` | thin-cell threshold: levels with fewer claims get `IsThin = True` (categorical thin levels also raise a warning) |
| `base` | your own base per variable, e.g. `{"REGIO": "Zuid", "LEEFTIJD": 40}`; `None` keeps the default base |
| `merge_vars` | two columns that are one rating factor, e.g. `{"Stedelijkheid": "Stedelijkheid_groep"}`; the regrouped column may live in the model's own data rather than in `data`; see [One rating factor, two columns](#one-rating-factor-two-columns) |
| `year_col` | the accounting-year column, whose default base is the latest year; `None` treats it like any other variable |
| `loss_col` | the loss amount, read only to warn when claims without payment make frequency x severity too high |

**Returns** a DataFrame with one row per level/grid point per variable:

| Column | Meaning |
|---|---|
| `Variable` | variable name, or `"A:B"` for an interaction |
| `Type` | `"categorical"` / `"continuous"` (of the X variable) |
| `Level`, `LevelNum` | level as text and (when applicable) numeric |
| `Group`, `GroupVar`, `XVar` | interaction bookkeeping (`None` for main effects, so `tbl.Group.isna()` picks the main-effect rows) |
| `IsBase` | `True` on the base row/cell (factor exactly 1) |
| `Exposure`, `ClaimCount` | data volume per level, and per cell for interactions; a continuous variable is binned onto its own grid first, so the cells are filled whatever the two types are |
| `Factor_Frequency`, `Factor_Severity`, `Factor_Premium` | the multipliers themselves, how many times the base. `Premium = Frequency × Severity`. A variable that is in only one of the two models gets 1.00 in the other. A level the model was not fitted on, such as a level without claims that the severity model never saw, gets `NaN` in that model's column and in `Factor_Premium`, with a warning |
| `Uplift_Frequency`, `Uplift_Severity`, `Uplift_Premium` | interaction rows only, the part the two main effects do **not** already explain. See below. **Absent altogether when the model carries no interaction term**, since they would then be `NaN` in every row |
| `SE_Frequency`, `SE_Severity` | standard error of the **logarithm** of the factor, from the model's own covariance matrix: `exp(log(Factor) ± 1.96 SE)` is a 95% interval. 0 on the base row and for a variable the model does not use; on an interaction row it belongs to the joint factor. See [Thin cells](#thin-cells) |
| `IsThin` | `True` when `ClaimCount < min_claims`: this factor rests on very few claims and is correspondingly unstable. Dimmed in `make_rating_plot()`, greyed out in the Excel export, see [Thin cells](#thin-cells) |

On `tbl.attrs`, R's attributes:

- `intercept_frequency`, `intercept_severity`, `intercept_premium`: the
  model prediction at the base point, **per unit of exposure** (`NaN` for a
  model that was not supplied);
- `base_values`, a dict with the base value used per variable.

```python
tbl = make_rating_table(m_freq, m_sev, data=dat, base_level="exposure")
print(tbl.attrs["intercept_premium"])
print(tbl[tbl.Variable == "REGIO"][["Level", "IsBase", "Exposure", "ClaimCount",
                                     "Factor_Frequency", "Factor_Severity",
                                     "Factor_Premium", "SE_Frequency", "IsThin"]])
```

#### Factor vs Uplift on an interaction row

This is the one column that trips people up, so here it is concretely.
Suppose young drivers cost 1.40× the base, city drivers 1.20×, and you have
both:

| | Meaning | Young **and** city |
|---|---|---|
| main effect, age | how much more a young driver costs | 1.40 |
| main effect, region | how much more a city driver costs | 1.20 |
| **`Factor_*`** | what the model charges this **combination**, all in | 1.85 |
| **`Uplift_*`** | what the combination costs **on top of** the two separately | 1.85 ÷ (1.40 × 1.20) = **1.10** |

So `Factor` is the price and `Uplift` is the surprise. The two main effects
would have predicted 1.68; the model says 1.85; the extra 10% is the
interaction: being young *in the city* is worse than being young and being
in the city added together.

Read them for different purposes:

- **`Factor_*`** is the price of the combination. It already contains
  both main effects, so it *replaces* the two main-effect factors of that
  pair.
- **`Uplift_*`** is what you look at to decide whether the interaction is
  worth having at all. It is exactly `exp(β)` of the interaction
  coefficient: the fitted interaction on its own, with both main effects
  divided out. **An uplift of 1.00 everywhere means there is no
  interaction**: the two variables are simply additive on the log scale
  and the term is buying you nothing. Uplifts scattered between 0.97 and
  1.03 are noise. Uplifts that run in a clear direction, like 1.20 for the
  young in the city falling to 0.95 for the old in the country, are a real
  effect.

**Ignore the reference cells when you read that.** Every cell where at
least one of the two variables sits at its base level has an uplift of
exactly 1.00 *by construction*, not by evidence. On a 3 × 2 interaction
that is four of the six cells; only two carry an estimated coefficient. So
"they are all 1.00" only means something once those are set aside.

**Building a premium: one of the two, never both.** Either take the joint
`Factor` *instead of* the two main effects of that pair (1.85), or take the
two main effects and the `Uplift` *on top* (1.40 × 1.20 × 1.10 = 1.85); both
give the model's prediction exactly. What you must not do is multiply the
joint `Factor` onto the two main effects: that counts them twice, and on a
real tariff it put premiums 16% too high on average and up to 3.4 times.
The Tariff sheet of the [Excel export](#export_rating_table) uses the second
form, which is why its interaction keys hold the uplift.

The three ways side by side, for a 23-year-old in the city:

```python
m_int = fit_glm("AantalClaims ~ LEEFTIJD * REGIO + offset(log(Exposure))", dat)
t_int = make_rating_table(m_int, None, data=dat)

def fac(tbl, var, level, group=None, col="Factor_Frequency"):
    rows = tbl[(tbl.Variable == var)
               & (tbl.Group.isna() if group is None else tbl.Group == group)]
    return rows.loc[rows.Level == str(level), col].item()

i = t_int.attrs["intercept_frequency"]
joint = i * fac(t_int, "LEEFTIJD:REGIO", 23, "Stad")
mains_uplift = (i * fac(t_int, "LEEFTIJD", 23) * fac(t_int, "REGIO", "Stad")
                * fac(t_int, "LEEFTIJD:REGIO", 23, "Stad", "Uplift_Frequency"))
twice = (i * fac(t_int, "LEEFTIJD", 23) * fac(t_int, "REGIO", "Stad")
         * fac(t_int, "LEEFTIJD:REGIO", 23, "Stad"))           # wrong
model = m_int.predict(pd.DataFrame({"LEEFTIJD": [23], "REGIO": ["Stad"],
                                    "Exposure": [1.0]}))
print(joint, mains_uplift, model)   # the same number three times
print(twice)                        # the main effects counted twice
```

#### Readable grids for continuous variables

A tariff lists ages 18, 19, 20 and weights 800, 850, 900, not 19.2653 and
832.65. The grid therefore runs in steps you would quote, chosen per
variable:

1. a whole-number column with at most `2 × grid_res` distinct values is
   listed **on those values**, which is what gives an age per year, and
   what keeps a coded column such as a sum insured, five values spread
   over a million, to five rows;
2. otherwise take the (trimmed) range and aim at roughly `grid_res`
   points;
3. round that raw step to a 1, 2 or 5 times a power of ten, the way an
   axis picks its ticks.

A stepped grid is then laid on whole multiples of the step, so the numbers
read as round values rather than as an offset sequence. On the R package's
demo portfolio that gives:

| Variable | Range | Step | Points |
|---|---|---|---|
| `LEEFTIJD` | 18 to 80 | every value | 63 |
| `GEWICHT` | 800 to 2400 | 50 | 33 |
| `KILOMETRAGE` | 2000 to 86000 | 2000 | 43 |

Rule 1 works off the values, never off a step of 1 across the range. That
distinction is the difference between a five-row grid and a million-row one
for a column whose codes happen to be far apart.

Set your own with `grid_step`, as one number for everything or per variable,
as a dict:

```python
tbl = make_rating_table(m_freq, m_sev, data=dat,
                        grid_step={"LEEFTIJD": 5, "GEWICHT": 200})
print(tbl[tbl.Variable == "GEWICHT"].Level.tolist())
```

Naming only some variables is fine, the rest keep the automatic step. A name
that matches no continuous variable is reported rather than silently
ignored, so a typo does not leave you thinking you set a step you did not:

```python
tbl = make_rating_table(m_freq, m_sev, data=dat, grid_step={"LEEFTIJ": 5})
# PricingWarning: make_rating_table: 'grid_step' names with no continuous variable to
#   apply to, and therefore ignored: LEEFTIJ. Continuous variables in this model:
#   LEEFTIJD, GEWICHT.
```

Two things the step does *not* know about. It is driven by the **range**, so
a single outlier coarsens the grid for everyone, which is what `trim` is
for, and trimming happens before the step is chosen. And it is not density-
or curvature-aware: a spline that bends sharply at young ages gets the same
resolution as the flat stretch above 40.

**Offsets and the base point.** Every variable the model offsets on is held
at 1 in the base row, and the offset the model actually carries is then
subtracted on the link scale, so `intercept × factors` is a rate per unit of
exposure whatever that column is called and whatever form the offset takes,
see [deriving the offset and the link](#deriving-the-offset-and-the-link).
The factors themselves are ratios of two predictions that share the base
row, so the offset cancels there either way; the intercept is the one number
that carries it. If the offset variable differs from `exposure_col`, the
intercepts are still correct but the `Exposure` and `ClaimCount` columns are
summed from a different column, and the function warns. With a second
offset, a bonus-malus scale, the table is the tariff before BM: see the
worked premium chain under
[`plot_glm_predictor()`](#plot_glm_predictor).

**Rows outside the grid.** With `trim`, policies beyond the trimmed range
are excluded from the exposure histogram rather than swept into the outer
bins. The exposure shown therefore adds up to less than the portfolio total,
by design, since the grid does not cover those policies. The three slices of
a continuous group variable in an interaction are not trimmed: they stand
for the whole distribution, so their cells add up to the full exposure.

Inline formula transformations such as `factor(BOEKJAAR)` and
`ns(LEEFTIJD, 4)` are resolved to their underlying columns; terms whose base
variable cannot be found in `data` are skipped with a warning rather than
crashing the call. A term that bands a column, such as
`cut(LEEFTIJD, c(17, 24, 40, 65, 99))`, is reported on its bands, and a
logical column as a categorical variable with the levels `FALSE` and
`TRUE`, `FALSE` as base (the model's reference).

#### One rating factor, two columns

A severity model often uses a regrouped version of a frequency variable.
Urbanisation runs 1 to 5 in the frequency model, but the severity showed
no difference between the top two classes, so there the column is 1, 2, 3,
4-5. The two models now name different columns for one tariff field, and
the table has to be told that they are one, or it reports both.

Everything below is one worked example you can paste and run. It has its
own data, so `dat`, `m_freq` and `m_sev` are not the quick-start ones here.

```python
import numpy as np
import pandas as pd
from pricingtools_rmo import *

rng = np.random.default_rng(7)
n = 30_000
dat = pd.DataFrame({
    "Stedelijkheid": rng.integers(1, 6, n).astype(str),      # urbanisation 1 to 5
    "LEEFTIJD": rng.uniform(18, 80, n).round(),
    "MERK": rng.choice(["BMW", "MERCEDES", "NISSAN", "TOYOTA"], n),
    "Exposure": rng.uniform(0.3, 1, n).round(3),
})

# the regrouped column: 4 and 5 together
dat["Stedelijkheid_groep"] = np.where(dat.Stedelijkheid.isin(["4", "5"]), "4-5",
                                      dat.Stedelijkheid)

dat["AantalClaims"] = rng.poisson(dat.Exposure * np.exp(
    -2.2 + 0.15 * dat.Stedelijkheid.astype(int)
    + dat.MERK.map({"BMW": 0, "MERCEDES": 0, "NISSAN": 0.3, "TOYOTA": 0.3})))
dat["SCHADELAST"] = np.where(dat.AantalClaims > 0, dat.AantalClaims * rng.gamma(
    3, 600 * np.where(dat.Stedelijkheid_groep == "4-5", 1.25, 1)
    * dat.MERK.map({"BMW": 1, "MERCEDES": 1.15, "NISSAN": 1, "TOYOTA": 1.15})), 0.0)

m_freq = fit_glm("AantalClaims ~ Stedelijkheid + LEEFTIJD + offset(log(Exposure))",
                 dat, family="poisson")

d_sev = dat[dat.AantalClaims > 0].copy()
d_sev["AvgLoss"] = d_sev.SCHADELAST / d_sev.AantalClaims
m_sev = fit_glm("AvgLoss ~ Stedelijkheid_groep", d_sev, family="gamma",
                weights="AantalClaims")
```

##### 1. What you get without `merge_vars`

```python
tbl0 = make_rating_table(m_freq, m_sev, data=dat)
# PricingWarning: make_rating_table: one column looks like a regrouping of another,
#   each used by one of the two models, so the table holds two blocks for what is
#   one tariff field. If that is what they are, pass
#   merge_vars={'Stedelijkheid': 'Stedelijkheid_groep'}.
cols = ["Variable", "Level", "Exposure", "ClaimCount",
        "Factor_Frequency", "Factor_Severity"]
print(tbl0[tbl0.Variable.str.startswith("Stedelijkheid")][cols])
```

| Variable | Level | Exposure | ClaimCount | Factor_Frequency | Factor_Severity |
|---|---|---|---|---|---|
| Stedelijkheid | 1 | 3884.551 | 547 | 1.000000 | 1.000000 |
| Stedelijkheid | 2 | 3913.877 | 653 | 1.184502 | 1.000000 |
| Stedelijkheid | 3 | 3760.830 | 724 | 1.366821 | 1.000000 |
| Stedelijkheid | 4 | 3972.480 | 930 | 1.662571 | 1.000000 |
| Stedelijkheid | 5 | 3963.485 | 1126 | 2.016395 | 1.000000 |
| Stedelijkheid_groep | 1 | 3884.551 | 547 | 1.000000 | 1.000000 |
| Stedelijkheid_groep | 2 | 3913.877 | 653 | 1.000000 | 0.955895 |
| Stedelijkheid_groep | 3 | 3760.830 | 724 | 1.000000 | 1.002073 |
| Stedelijkheid_groep | 4-5 | 7935.965 | 2056 | 1.000000 | 1.252052 |

The table is **correct but awkward**. A variable a model does not use gets
a neutral factor of 1 there, so the frequency effect sits in one block and
the severity effect in the other, and a premium needs both. Whoever builds
the tariff from the `Stedelijkheid` block alone drops the severity effect
with nothing to show that something is missing. That is what the warning
is for: it fires when one column is a regrouping of another and each is
used by one of the two models.

##### 2. The same tariff with `merge_vars`

```python
tbl = make_rating_table(m_freq, m_sev, data=dat,
                        merge_vars={"Stedelijkheid": "Stedelijkheid_groep"})
print(tbl[tbl.Variable == "Stedelijkheid"][cols[1:] + ["Factor_Premium"]])
```

| Level | Exposure | ClaimCount | Factor_Frequency | Factor_Severity | Factor_Premium |
|---|---|---|---|---|---|
| 1 | 3884.551 | 547 | 1.000000 | 1.000000 | 1.000000 |
| 2 | 3913.877 | 653 | 1.184502 | 0.955895 | 1.132259 |
| 3 | 3760.830 | 724 | 1.366821 | 1.002073 | 1.369655 |
| 4 | 3972.480 | 930 | 1.662571 | 1.252052 | 2.081624 |
| 5 | 3963.485 | 1126 | 2.016395 | 1.252052 | 2.524631 |

One block, on the five levels a policy actually carries. Levels 4 and 5
carry the severity factor of the group they fall in, the frequency factors
are untouched, and the exposure and claim counts stay on the real levels
rather than on the grouped ones. The lookup runs on the column that is in
the policy record, so nothing has to be mapped by hand.

##### 3. A premium, and what the two-block version costs you

```python
def fac(t, v, l):
    s = t[(t.Variable == v) & t.Group.isna()]
    return s.loc[s.Level == str(l), "Factor_Premium"].item()

i = tbl.attrs["intercept_premium"]
premium = i * fac(tbl, "Stedelijkheid", "5") * fac(tbl, "LEEFTIJD", 40)

policy = pd.DataFrame({"Stedelijkheid": ["5"], "Stedelijkheid_groep": ["4-5"],
                       "LEEFTIJD": [40], "Exposure": [1.0]})
from_models = m_freq.predict(policy) * m_sev.predict(policy)
print(premium, from_models)
```

```text
697.3217872888056 [697.32178729]
```

Take the same policy from the two-block table and use the `Stedelijkheid`
block only, and you get a premium that is too low by exactly the severity
loading of group 4-5, which lived in the other block. The two-block table
is not wrong, it just has to be looked up twice:

```python
one_block = (tbl0.attrs["intercept_premium"]
             * fac(tbl0, "Stedelijkheid", "5") * fac(tbl0, "LEEFTIJD", 40))
both_blocks = (tbl0.attrs["intercept_premium"]
               * fac(tbl0, "Stedelijkheid", "5")
               * fac(tbl0, "Stedelijkheid_groep", "4-5")
               * fac(tbl0, "LEEFTIJD", 40))
print(one_block, both_blocks)    # too low, and correct
```

```text
556.9433502161282 697.3217872888056
```

556.94 against 697.32: the one-block premium is short by exactly the
severity factor of group 4-5 (1.252052).

##### 4. Choosing the base: both columns move together

```python
tbl5 = make_rating_table(m_freq, m_sev, data=dat,
                         merge_vars={"Stedelijkheid": "Stedelijkheid_groep"},
                         base={"Stedelijkheid": "5"})
print(tbl5[tbl5.Variable == "Stedelijkheid"][
    ["Level", "IsBase", "Factor_Frequency", "Factor_Severity", "Factor_Premium"]])
print(tbl5.attrs["intercept_premium"]
      * fac(tbl5, "Stedelijkheid", "5") * fac(tbl5, "LEEFTIJD", 40))   # unchanged
```

| Level | IsBase | Factor_Frequency | Factor_Severity | Factor_Premium |
|---|---|---|---|---|
| 1 | False | 0.495935 | 0.798689 | 0.396098 |
| 2 | False | 0.587435 | 0.763463 | 0.448485 |
| 3 | False | 0.677854 | 0.800345 | 0.542517 |
| 4 | False | 0.824526 | 1.000000 | 0.824526 |
| 5 | True | 1.000000 | 1.000000 | 1.000000 |

Level 4 has a severity factor of exactly 1, because it sits in the same
group as the base. The base row stays a policy that could exist: asking
for urbanisation 5 puts the regrouped column at 4-5 by itself. The same
policy as in step 3 still costs 697.32, as it must, since the base only
moves the factors and the intercept against each other.

Naming the regrouped column in `base` is refused, because after merging it
has no block of its own to be the base of.

##### 5. A band of a continuous variable

The same thing happens when the frequency model uses an age and the
severity model an age class.

```python
dat["LEEFTIJD_KL"] = pd.cut(dat.LEEFTIJD, [17, 30, 50, 80],
                            labels=["18-30", "31-50", "51-80"])
d2 = dat[dat.AantalClaims > 0].copy()
d2["AvgLoss"] = d2.SCHADELAST / d2.AantalClaims

m_f2 = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + offset(log(Exposure))", dat)
m_s2 = fit_glm("AvgLoss ~ LEEFTIJD_KL", d2, family="gamma", weights="AantalClaims")

t2 = make_rating_table(m_f2, m_s2, data=dat,
                       merge_vars={"LEEFTIJD": "LEEFTIJD_KL"},
                       grid_step={"LEEFTIJD": 10})
print(t2[t2.Variable == "LEEFTIJD"][["Level", "Factor_Frequency", "Factor_Severity"]])
```

| Level | Factor_Frequency | Factor_Severity | Band it falls in |
|---|---|---|---|
| 20 | 1.150505 | 0.955142 | 18-30 |
| 30 | 1.117543 | 0.955142 | 18-30 |
| 40 | 1.048293 | 1.000000 | 31-50 |
| 50 | 1.000000 | 1.000000 | 31-50 |
| 60 | 1.032013 | 0.973852 | 51-80 |
| 70 | 1.057487 | 0.973852 | 51-80 |
| 80 | 1.038234 | 0.973852 | 51-80 |

The frequency factor runs per 10 years off the spline, the severity factor
steps per band. A grid point that does not occur in the data takes the
band of the nearest age that does, which is the band it falls in, because
a band covers a run of neighbouring values. The boundaries come out right:
30 is still in 18-30, and 40 and 50 are in 31-50.

##### 6. When the regrouped column is only in the severity data

The grouping is usually made where it is needed, next to the average claim,
so it ends up in the dataset the severity model is fitted on and not in the
portfolio you hand to `make_rating_table()`. That is fine: a fitted model
carries its own data, so the relation is read off the model and the column
is derived here.

```python
portfolio = dat.drop(columns=["Stedelijkheid_groep", "LEEFTIJD_KL"])
print(list(portfolio.columns))       # no Stedelijkheid_groep at all
print("Stedelijkheid_groep" in d_sev.columns)   # d_sev, which m_sev was fitted on, has it

t6 = make_rating_table(m_freq, m_sev, data=portfolio,
                       merge_vars={"Stedelijkheid": "Stedelijkheid_groep"})
print(np.allclose(t6.Factor_Premium, tbl.Factor_Premium))   # the same table as in step 2
```

That is the whole call. The result is identical to adding the column
yourself first, your DataFrame is not changed, and an age band against a
spline works the same way.

One case is an error rather than a guess. A level that occurs in `data` but
in no row of the severity data has no group to derive. Fit the severity
model without the claims of urbanisation 1 to see it:

```python
m_sev_no1 = fit_glm("AvgLoss ~ Stedelijkheid_groep",
                    d_sev[d_sev.Stedelijkheid != "1"], family="gamma",
                    weights="AantalClaims")
make_rating_table(m_freq, m_sev_no1, data=portfolio,
                  merge_vars={"Stedelijkheid": "Stedelijkheid_groep"})
# PricingError: make_rating_table: the value(s) 1 of 'Stedelijkheid' do not occur in
#   the data the severity model was fitted on, so their group in
#   'Stedelijkheid_groep' is unknown. A level with no claims has no group there:
#   give it one in that model's data, or put 'Stedelijkheid_groep' in 'data'
#   yourself.
```

A level with no claims has no severity factor either way, so the fix is in
the model: group it with a level that does have claims.

##### 7. Two groupings, one column name

A grouping is made where it is needed, so it is often made twice, once per
dataset, and the two need not agree. The frequency model may put NISSAN
with TOYOTA, while the severity model puts NISSAN with BMW and MERCEDES. If
both columns are called `MERK_GROEP`, they are two different variables under
one name, and `data` can hold only one of them.

That is refused:

```python
dat["MERK_GROEP"] = np.where(dat.MERK.isin(["BMW", "MERCEDES"]), "G1", "G2")  # frequency data
d_sev["MERK_GROEP"] = np.where(d_sev.MERK == "TOYOTA", "G2", "G1")            # severity data

m_freq_m = fit_glm("AantalClaims ~ MERK_GROEP + LEEFTIJD + offset(log(Exposure))", dat)
m_sev_m = fit_glm("AvgLoss ~ MERK_GROEP", d_sev, family="gamma", weights="AantalClaims")

make_rating_table(m_freq_m, m_sev_m, data=dat, merge_vars={"MERK": "MERK_GROEP"})
# PricingError: make_rating_table: 'MERK_GROEP' does not mean the same thing
#   everywhere: 'NISSAN' of 'MERK' is 'G2' in the data the frequency model was
#   fitted on and 'G1' in the data the severity model was fitted on. Two groupings
#   of one variable cannot share a column name: give them separate names and name
#   both in 'merge_vars', as
#   merge_vars={'MERK': ['MERK_GROEP_freq', 'MERK_GROEP_sev']}.
```

This is worth stating plainly, because it is the one case where the table
could be wrong rather than absent. If the two groupings use different
labels (`A-B` against `B-C`), predicting fails and nothing is produced. If
they happen to use the same labels (`G1`, `G2`), nothing fails: the
severity factors would be read off the frequency grouping and a brand can
end up tens of percent out, on a table that looks perfectly ordinary. The
mapping in `data` is therefore compared with the one each model's own data
defines, before any factor is computed.

The way out is to keep the two apart and declare both, which is what the
message suggests:

```python
dat["MERK_GR_F"] = np.where(dat.MERK.isin(["BMW", "MERCEDES"]), "BMW_MERCEDES", dat.MERK)
d_sev["MERK_GR_S"] = np.where(d_sev.MERK.isin(["BMW", "NISSAN"]), "BMW_NISSAN", d_sev.MERK)

m_freq_m = fit_glm("AantalClaims ~ MERK_GR_F + LEEFTIJD + offset(log(Exposure))", dat)
m_sev_m = fit_glm("AvgLoss ~ MERK_GR_S", d_sev, family="gamma", weights="AantalClaims")

t7 = make_rating_table(m_freq_m, m_sev_m, data=dat,
                       merge_vars={"MERK": ["MERK_GR_F", "MERK_GR_S"]})
print(t7[t7.Variable == "MERK"][
    ["Level", "Factor_Frequency", "Factor_Severity", "Factor_Premium"]])
```

| Level | Factor_Frequency | Factor_Severity | Factor_Premium |
|---|---|---|---|
| BMW | 1.000000 | 1.000000 | 1.000000 |
| MERCEDES | 1.000000 | 1.080880 | 1.080880 |
| NISSAN | 1.452215 | 1.000000 | 1.452215 |
| TOYOTA | 1.385010 | 1.177437 | 1.630762 |

BMW and MERCEDES share a frequency factor, BMW and NISSAN share a severity
factor, and every brand still gets its own premium factor. The same holds
for two bandings of one continuous variable with different cut points: give
them two names and both groupings land in one block, on the real ages.

##### 8. What else is refused

The relation has to be a function of the variable: every value maps to
exactly one group. That is checked, so a column that is not a regrouping is
refused instead of half applied. When the model uses the column, the check
runs against the data that model was fitted on:

```python
make_rating_table(m_freq, m_sev, data=dat,
                  merge_vars={"Stedelijkheid": "LEEFTIJD"})
# PricingError: make_rating_table: 'LEEFTIJD' is not a regrouping of 'Stedelijkheid'
#   in the data the frequency model was fitted on: the value(s) ...
#   occur there with more than one group.

make_rating_table(m_freq, m_sev, data=dat,
                  merge_vars={"Stedelijkheid": "Stedelijkheid_groep"},
                  base={"Stedelijkheid_groep": "4-5"})
# PricingError: make_rating_table: 'Stedelijkheid_groep' is merged into
#   'Stedelijkheid'; set the base on that variable instead.
```

A declaration that neither model uses raises a warning rather than
changing nothing in silence, and a column that is in neither `data` nor a
model is an error:

```python
t8 = make_rating_table(m_freq, m_sev, data=dat, merge_vars={"MERK": "MERK_GR_F"})
# PricingWarning: make_rating_table: 'merge_vars' mentions MERK_GR_F, which neither
#   model uses, so the declaration changes nothing.

make_rating_table(m_freq, m_sev, data=dat, merge_vars={"MERK": "MERK_KLASSE"})
# PricingError: make_rating_table: 'MERK_KLASSE' is not a column of 'data' and no
#   model uses it, so there is nothing to merge.
```

##### 9. The export

```python
import openpyxl

export_rating_table(tbl, "tariff.xlsx")
export_rating_table(tbl0, "tariff_two_blocks.xlsx")
print(openpyxl.load_workbook("tariff.xlsx").sheetnames)
print(openpyxl.load_workbook("tariff_two_blocks.xlsx").sheetnames)
```

```text
with merge_vars : ['Overview', 'Tariff', 'Stedelijkheid', 'LEEFTIJD']
without         : ['Overview', 'Tariff', 'Stedelijkheid', 'LEEFTIJD', 'Stedelijkheid_groep']
```

One sheet and one lookup key (`Stedelijkheid|5`) per tariff field, instead
of two keys that both have to be applied.

##### The rules in short

- Keyed by the variable to keep: `{"Stedelijkheid": "Stedelijkheid_groep"}`.
  One variable may have several regrouped columns, as a list.
- The regrouped column must be a function of the variable it regroups,
  which is checked against `data` and against the data of the model that
  uses it.
- One grouping per column name. When the two models group differently
  under one name, that is an error, not a table; give the two columns
  separate names and name both.
- It does not have to be in `data`. When only the model that uses it has
  it, the relation is read off that model and the column is derived, and
  your DataFrame is left as it was.
- It keeps no block of its own, cannot be named in `base`, and an
  interaction on it is reported under the name of the variable it is merged
  into.
- Nothing about the models changes. Nothing is refitted and every premium
  is the same before and after; only the shape of the table changes.
- The same declaration adds those columns to a dataset you want to score,
  with [`add_groupings()`](#add_groupings).

### `add_groupings()`

**The question it answers:** *my new portfolio has `Stedelijkheid`, `MERK`
and `LEEFTIJD`, but my models were fitted on groupings of those columns.
How do I score it?*

```python
add_groupings(data, model_freq=None, model_sev=None, merge_vars=None)
```

A grouping is made where it is needed: urbanisation 4 and 5 taken together
next to the average claim, brands grouped one way for the frequency and
another way for the severity, an age band for the severity next to a spline
in the frequency model. Those columns then live in the data each model was
fitted on and nowhere else, while `predict()` needs every one of them on the
data it scores. A `GLMFit` carries its own data, so the relation is read
off the model that uses the column, and `add_groupings()` adds the columns
from the source columns your portfolio does have.

It takes the same `merge_vars` as
[`make_rating_table()`](#one-rating-factor-two-columns), and the table runs
on this same code, so one declaration serves both and the two cannot
disagree about which policy sits in which group. Continuing the worked
example above, with models that use all three kinds of grouping:

```python
d_sev["LFT_KL"] = pd.cut(d_sev.LEEFTIJD, [17, 30, 50, 80],
                         labels=["18-30", "31-50", "51-80"])     # severity: age bands
m_freq = fit_glm("AantalClaims ~ ns(LEEFTIJD, 4) + Stedelijkheid + MERK_GR_F"
                 " + offset(log(Exposure))", dat)
m_sev = fit_glm("AvgLoss ~ LFT_KL + Stedelijkheid_groep + MERK_GR_S", d_sev,
                family="gamma", weights="AantalClaims")

mv = {"Stedelijkheid": "Stedelijkheid_groep",       # severity: 4-5 together
      "MERK": ["MERK_GR_F", "MERK_GR_S"],           # grouped differently per model
      "LEEFTIJD": "LFT_KL"}                         # severity: age bands

nieuw = pd.DataFrame({"LEEFTIJD": [22, 30, 31, 50, 64],
                      "Stedelijkheid": ["1", "3", "4", "5", "2"],
                      "MERK": ["BMW", "MERCEDES", "NISSAN", "TOYOTA", "BMW"]})
nieuw = add_groupings(nieuw, m_freq, m_sev, merge_vars=mv)
print(nieuw)

nieuw["Exposure"] = 1.0                             # a year
premium = m_freq.predict(nieuw) * m_sev.predict(nieuw)

tbl = make_rating_table(m_freq, m_sev, data=dat, merge_vars=mv)
chain = [tbl.attrs["intercept_premium"] * fac(tbl, "LEEFTIJD", p.LEEFTIJD)
         * fac(tbl, "Stedelijkheid", p.Stedelijkheid) * fac(tbl, "MERK", p.MERK)
         for p in nieuw.itertuples()]
print(premium, chain)                               # the same premiums
```

On a portfolio holding only the source columns:

| LEEFTIJD | Stedelijkheid | MERK | Stedelijkheid_groep | MERK_GR_F | MERK_GR_S | LFT_KL |
|---|---|---|---|---|---|---|
| 22 | 1 | BMW | 1 | BMW_MERCEDES | BMW_NISSAN | 18-30 |
| 30 | 3 | MERCEDES | 3 | BMW_MERCEDES | MERCEDES | 18-30 |
| 31 | 4 | NISSAN | 4-5 | NISSAN | BMW_NISSAN | 31-50 |
| 50 | 5 | TOYOTA | 4-5 | TOYOTA | TOYOTA | 31-50 |
| 64 | 2 | BMW | 2 | BMW_MERCEDES | BMW_NISSAN | 51-80 |

Every brand gets the group each model put it in, and the band edges land on
the right side: 30 is still in 18-30, 31 already in 31-50. For each of
these policies the premium from `predict()` equals the one built from the
rating table with the same declaration. Each added column gets the type and
categories it has in the model's data, and the input DataFrame is not
modified.

**What is refused.** Anything that would be a guess is an error, not a
column with a value in it:

```python
# urbanisation 6, which the claims data never saw
add_groupings(pd.DataFrame({"LEEFTIJD": [40], "Stedelijkheid": ["6"], "MERK": ["BMW"]}),
              m_freq, m_sev, merge_vars=mv)
# PricingError: add_groupings: the value(s) 6 of 'Stedelijkheid' do not occur in the
#   data the severity model was fitted on, so their group in 'Stedelijkheid_groep'
#   is unknown. A level with no claims has no group there: give it one in that
#   model's data, or put 'Stedelijkheid_groep' in 'data' yourself.

# a brand neither model knows
add_groupings(pd.DataFrame({"LEEFTIJD": [40], "Stedelijkheid": ["2"], "MERK": ["TESLA"]}),
              m_freq, m_sev, merge_vars=mv)
# PricingError: add_groupings: the value(s) TESLA of 'MERK' do not occur in the data
#   the frequency model was fitted on, so their group in 'MERK_GR_F' is unknown.
#   Map them to a value the model knows first, or put 'MERK_GR_F' in 'data'
#   yourself.

# a grouped column already in the data, disagreeing with the model
add_groupings(pd.DataFrame({"LEEFTIJD": [40, 41], "Stedelijkheid": ["2", "2"],
                            "MERK": ["BMW", "MERCEDES"],
                            "MERK_GR_F": ["BMW_MERCEDES", "MERCEDES"]}),
              m_freq, m_sev, merge_vars=mv)
# PricingError: add_groupings: 'MERK_GR_F' does not mean the same thing everywhere:
#   in 1 row(s) of 'data' it disagrees with the data the frequency model was
#   fitted on; row 1 (0-based) has 'MERK' = MERCEDES and 'MERK_GR_F' = MERCEDES,
#   where the model's data says BMW_MERCEDES.
```

A grouped column that is already in your data is checked, never
overwritten, so a wrong mapping in a source system is caught rather than
scored. (The row number in that message counts from 1, as R's does.) One
column name that the two models group differently is refused too, for the
reason in [Two groupings, one column name](#7-two-groupings-one-column-name).
One difference from R: a grouped column that is already in your data keeps
its value for a source value the model's data never saw, since there is
nothing to check it against; the R version refuses that case.

**What is not refused.** A continuous source outside the range the model
was fitted on takes the outer band, with a warning:

```python
x = add_groupings(pd.DataFrame({"LEEFTIJD": [40, 85], "Stedelijkheid": ["2", "2"],
                                "MERK": ["BMW", "BMW"]}),
                  m_freq, m_sev, merge_vars=mv)
# PricingWarning: add_groupings: 'LEEFTIJD' lies outside the range the severity model
#   was fitted on (18 to 80); 1 row above 80 got '51-80', the outer band(s) of
#   'LFT_KL'. 'LEEFTIJD' is also a continuous term of the frequency model, which
#   extrapolates beyond the range it was fitted on; capping the values,
#   pmin(pmax(LEEFTIJD, 18), 80), keeps it from doing so.
```

The outer bands of a banding are open in practice: nobody over 80 in the
claims data does not make 85 a new class. The real risk of going outside
the range sits elsewhere, in a continuous term such as `ns(LEEFTIJD, 4)`,
which `predict()` extrapolates linearly on the log scale beyond its last
knot without a word. No grouping can prevent that, so the warning names it
and the remedy (`pmin()` and `pmax()` work in a `fit_glm()` formula as they
do in R).

**Categorical or continuous** is read off the models: a factor term is
categorical, a numeric term continuous. A source no model uses directly is
categorical unless it is numeric with more than 25 distinct values, so a
numeric code such as urbanisation 1 to 5 is looked up exactly and an unknown
code is an error, never a neighbouring group.

### `make_rating_plot()`

Plots one variable (or interaction) from a rating table.

```python
make_rating_plot(rating_tbl, var, metric_fmt=4, metric=None, y_range=None)
```

- **Main effects**: factor curves for Frequency, Severity and Premium on the
  primary axis (with a dotted reference line at 1.0), exposure bars on the
  secondary axis.
- **Interactions** (`var="A:B"`): one series per level of the group
  variable. Shows one metric at a time, `metric` picks `"Frequency"`,
  `"Severity"` or `"Premium"` (default: Premium when both models are
  present). Exposure bars are included for categorical × categorical
  interactions.
- **Thin cells**: levels flagged `IsThin` in the rating table are drawn with
  dimmed markers and a "low claim volume" note in the tooltip, so factors
  built on few claims are visually distinct from well-supported ones.
- Categorical variables are drawn as markers only, never connected by a
  line.

```python
tbl = make_rating_table(m_freq, m_sev, data=dat)
make_rating_plot(tbl, "LEEFTIJD").show()          # a factor curve
make_rating_plot(tbl, "REGIO").show()             # markers per level

m_int = fit_glm("AantalClaims ~ LEEFTIJD * REGIO + offset(log(Exposure))", dat)
t_int = make_rating_table(m_int, m_sev, data=dat)
make_rating_plot(t_int, "LEEFTIJD:REGIO", metric="Premium").show()   # one curve per region
```

**Returns** a plotly figure.

### `model_lift()`

**The question it answers:** *does my model actually tell cheap risks from
expensive ones?*

```python
model_lift(model_freq=None, model_sev=None, data=None,
           actual_col="SCHADELAST", exposure_col="Exposure",
           claims_col="AantalClaims", weight_col=None,
           n_bins=10, y_range=None, keep_offsets=True,
           n_boot=200, conf_level=0.9, seed=None, at=None)
```

**What it does.** Every other diagnostic here looks at one variable. This
one judges the model as a whole, in three steps:

1. Price the entire book and sort it from cheapest to most expensive.
2. Cut it into 10 slices, each holding the **same amount of exposure**,
   so slice 1 is the tenth of the book the model thinks is safest, and
   slice 10 the tenth it thinks is worst.
3. In each slice, compare what the model predicted with what actually
   happened.

```python
lift = model_lift(m_freq, m_sev, data=dat, seed=1)
print(lift.table)
print(lift.gini, lift.gini_ci)
lift.plot.show()
lift.plot_lorenz.show()
```

**Reading the chart.** The gold line is what the model predicted, the blue
line what really happened.

- **Does the blue line rise?** That is the whole point. If policies the
  model called risky really did cost more, the model separates risk. A
  flat blue line means the model orders nothing, however good its overall
  average is.
- **Do the two lines sit on top of each other?** Then it is also
  *calibrated*, right about how much, not just about the order. A blue
  line that rises but sits above the gold one at both ends means the model
  under-differentiates: it has the ranking right but is too timid about it.

**The numbers behind the chart** are in `lift.table`, one row per slice:

| Column | What it tells you |
|---|---|
| `Bin` | 1 = the tenth the model thinks is safest, 10 = the worst |
| `Exposure`, `ExposureShare` | how much book is in the slice, should be ≈ 0.10 each by construction |
| `ActualRate` | what the slice really cost, per unit of exposure |
| `PredictedRate` | what the model said it would cost |
| `AE` | `ActualRate / PredictedRate`. **1.00 is perfect.** 1.15 means that slice cost 15% more than priced; 0.90 means 10% less |

`AE` is the column to scan. A run of values above 1 at one end and below 1
at the other is a systematic problem, not noise, and it tells you *where*
the tariff is wrong, which is something a single summary number never does.

**The Gini** compresses the ranking into one number between 0 and 1: 0
means the model separates nothing, higher means it separates better. It
measures **ordering only**: a model can have an excellent Gini and be badly
calibrated, which is exactly what the `AE` column reveals. Use it to compare
candidate models on the same book; do not read an absolute value as good or
bad, because it depends heavily on how heterogeneous the book is.
`lift.plot_lorenz` draws the curve it comes from: the further it bows from
the diagonal, the more the model separates.

**How sure is that number?** Less than it looks. A Gini on raw losses
moves with a handful of large claims: on a 16,000-policy holdout the 90%
interval ran from 0.14 to 0.33 around 0.23, wider than any difference you
would want to choose two models on, and the true risk premium scored 0.13
to 0.30 on the same book. So the Gini comes with a bootstrap interval, in
the chart title and in `lift.gini_ci` (`n_boot=200` replicates,
`conf_level=0.9`, `seed` for the same interval twice, `n_boot=0` to switch
it off). When the interval is wide **and** the ten largest records carry
10% or more of the losses, a warning says so: cap the large losses in
`actual_col`, and compare models on the same capped losses. A wide interval
on a small book has no such remedy and is left to the title.

```python
lift_small = model_lift(m_freq, m_sev, data=dat.head(3000), seed=1)
# PricingWarning: model_lift: the 90% interval for the Gini runs from ... so the
#   point value says little on its own. The ten largest records carry ... capping
#   large losses narrows the interval, provided the models are compared on the
#   same capped losses.
```

**Returns** a `Result` with `table`, `gini`, `gini_ci`, `stats` (the
numbers behind the title: `in_sample`, `n_rows`, `n_dropped`,
`offsets_kept`, ...), `plot` and `plot_lorenz`. A severity-only lift is
binned and weighted on the claim count instead of the exposure
(`weight_col` overrides that choice); the `Exposure` columns then hold
claim counts and the axis titles say so.

**Match the units.** Claim counts with a frequency model, loss amounts
with a severity or risk premium model. If the overall A/E lands far from 1
the function warns, because that nearly always means `actual_col` and the
prediction are different quantities:

```python
lf = model_lift(m_freq, data=dat, n_boot=0)        # a frequency model against SCHADELAST
# PricingWarning: model_lift: the overall actual/expected is ... so 'SCHADELAST'
#   and the model predictions are probably not the same quantity.
lf = model_lift(m_freq, data=dat, actual_col="AantalClaims", seed=1)   # the frequency lift
```

> `data=None` scores the model on the rows it was fitted to, flattering,
> and labelled *(in-sample)* on the chart. Pass a holdout for an honest
> answer.

### `double_lift()`

Which of two tariffs is right where they disagree.

```python
double_lift(data, model_freq_new=None, model_sev_new=None,
            model_freq_old=None, model_sev_old=None,
            old_premium_col=None,
            old_premium_basis="amount",
            actual_col="SCHADELAST", exposure_col="Exposure",
            claims_col="AantalClaims", weight_col=None,
            n_bins=10, rebase=True, y_range=None,
            keep_offsets=True, n_boot=200, conf_level=0.9,
            seed=None, at=None)
```

**The question it answers:** *my new tariff and my old one disagree about
some policies, which of them is right?*

**Why a single lift chart cannot answer this.** Two tariffs can both look
convincing on their own lift charts and still price the same policy €400
apart. Their averages hide it. What settles the argument is looking only at
the policies where they disagree, and asking which one the claims sided
with.

**What it does.** For every policy, work out `new price ÷ old price`. Sort
the book by that ratio and cut it into 10 slices. Slice 1 holds the policies
where the new tariff is most *below* the old one; slice 10 where it is most
*above*. The middle slices are where the two roughly agree, uninteresting by
design. Then plot how each tariff did in each slice.

Same arguments as [`premium_impact()`](#premium_impact): either two model
sets, or one model set and a column holding your current premium.

```python
# against the current premium
dl = double_lift(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                 old_premium_col="PREMIE", seed=1)
print(dl.table)
print(dl.stats.winner, dl.stats.mad_diff_ci)
dl.plot.show()

# against an older model pair
m_freq_old = fit_glm("AantalClaims ~ REGIO + factor(BOEKJAAR) + offset(log(Exposure))", dat)
m_sev_old = fit_glm("AvgLoss ~ REGIO", claims, family="gamma", weights="AantalClaims")
dl2 = double_lift(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                  model_freq_old=m_freq_old, model_sev_old=m_sev_old, seed=1)
print(dl2.stats.winner)
```

**Reading the chart.** Both lines show A/E, actual over expected, so
**1.0 is right and the reference line is the target.**

- **Look at the ends, not the middle.** That is where the tariffs disagree
  and where the comparison is decided.
- **The line that stays closer to 1.0 across the range is the better
  tariff.** On the R package's demo portfolio the old tariff drifted from
  0.78 to 1.36: it was charging 22% too little at one end and 36% too much
  at the other, exactly where it disagreed with the candidate. The
  candidate stayed near 1.0, so the disagreements were the candidate being
  right.
- **A line above 1.0** means that tariff under-charged those policies;
  **below 1.0** means it over-charged them.

**The numbers behind the chart**, in `dl.table`:

| Column | What it tells you |
|---|---|
| `Bin` | 1 = new tariff is cheapest relative to old, 10 = most expensive |
| `RateRatio` | the average `new ÷ old` in that slice, 0.85 means the new tariff is 15% cheaper for those policies |
| `ActualRate` | what those policies really cost |
| `AE_New`, `AE_Old` | each tariff's actual-over-expected. **The comparison is between these two columns, row by row** |

And in `dl.stats`:

| Value | What it tells you |
|---|---|
| `mad_new`, `mad_old` | average distance from 1.0 across the slices, lower is better calibrated |
| `mad_diff`, `mad_diff_ci` | `mad_old - mad_new`, positive when the new tariff is closer to 1, and its bootstrap interval |
| `winner` | `"new"` or `"old"` only when that interval excludes 0; `"inconclusive"` when it does not; `"tie"` when the two are equal |
| `rate_level_change` | how much more the new tariff collects overall, measured before any rebasing |
| `weight_col` | what the slices hold equal amounts of |

**A winner only when the data says so.** The two averages are decided by a
few large claims. On a 16,000-policy holdout the point values said "old" on
the losses as recorded (0.543 against 0.405) and "new" on the same losses
capped at 50,000 (0.222 against 0.271). So the difference gets a bootstrap
interval (`n_boot`, `conf_level`, `seed`, as in `model_lift()`), and
`winner` is `"inconclusive"` unless that interval excludes 0; the chart
title shows the interval. When large losses are behind an inconclusive
verdict, the ten largest records carrying 10% or more, a warning says so.
`n_boot=0` goes back to the point values alone. Even with a winner, it is a
summary of the picture: *where* a tariff drifts matters as much as how far.

**The weight follows the models**, as in `model_lift()`: two severity
models are compared per claim (`claims_col`), everything else per unit of
exposure, and an old premium amount is divided by that same weight. A
severity model on one side and a frequency or risk premium on the other is
refused, because the ratio of a rate per claim to a rate per year orders
the book by neither tariff:

```python
double_lift(dat, model_sev_new=m_sev, model_freq_old=m_freq)
# PricingError: double_lift: one side is a severity model and the other is not, so
#   one rate is per claim and the other per unit of exposure. Compare severity with
#   severity, or frequency x severity with frequency x severity.
```

**Rate level is taken out by default** (`rebase=True`), so both sides
collect the same total premium and the chart is purely about
differentiation. Whether the new tariff should collect more in total is a
separate, commercial question: `rate_level_change` reports it either way,
and [`premium_impact()`](#premium_impact) is where that conversation
belongs.

**Returns** a `Result` with `table`, `stats` and `plot`.

### `premium_impact()`

Dislocation analysis: per-policy comparison of premiums under a new model
set against old models or an existing premium column.

```python
premium_impact(data,
               model_freq_new=None, model_sev_new=None,
               model_freq_old=None, model_sev_old=None,
               old_premium_col=None,
               old_premium_basis="amount",
               rebase=True, by=None, n_show=10,
               exposure_col="Exposure", spotlight=None,
               x_range=None, y_range=None, keep_offsets=True,
               at=None, year_col="BOEKJAAR")
```

Premiums are computed as **rates per unit of exposure**: the exposure offset
is taken out, so every policy is priced for a full year, and the premium is
the product of the models supplied. The old side is either a model pair or
`old_premium_col`, a column with the current premium, interpreted as an
amount for the record (`"amount"`, divided by exposure internally) or as a
rate (`"rate"`).

With `rebase=True` (default) the new premiums are scaled so the
exposure-weighted totals match the old ones. That separates the two
questions a dislocation analysis answers: the overall **rate-level change**
(reported in the summary) and the **redistribution** across the portfolio
(the histogram and quantiles).

```python
imp = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                     old_premium_col="PREMIE", by="REGIO",
                     at={"BOEKJAAR": 2024})
print(imp.summary)
print(imp.by_level)
print(imp.largest_increases.head())
imp.plot.show()
```

**A bonus-malus scale as an offset.** A model may carry a known discount
next to the exposure, `offset(log(Exposure)) + offset(log(BM_KORTING))`. The
two are treated differently, because they mean different things: the
exposure says how long a policy was on risk, the BM says how much it pays.
By default only the exposure is taken out, and the BM stays in the rate, so
`NewRate` is what the policy pays for a year, discount included. That is
what a current premium column holds as well, so the comparison is like for
like.

Taking the BM out of the new side only does real damage. Measured with an
old premium that is exactly the new tariff, so that the true change is 0%
for every policy:

| BM class | Discount | Mean change, BM taken out | Mean change, BM kept (default) |
|---|---|---|---|
| 1 | 0% | -20.0% | 0.0% |
| 2 | 10% | -11.1% | 0.0% |
| 3 | 20% | 0.0% | 0.0% |
| 4 | 30% | +14.3% | 0.0% |
| 5 | 40% | +33.3% | 0.0% |

An unchanged tariff would have looked like a large shift onto the
discounted classes. The summary has a line `Offsets kept in the rate`, so
the output says what its rates are on.

`keep_offsets=False` takes every offset out and puts every policy at the
class without discount. Use it only against an old premium that is also
before discount; dividing a premium including discount by the BM yourself
and keeping the default would take the discount out twice. Against old
models with the same offset the choice does not matter for the changes,
which are identical either way. Two cases fall back to taking everything
out: a model whose offset is not on `exposure_col` at all, and one
expression that holds both, `offset(log(Exposure * BM))`, which cannot be
split and is reported in a warning; write it as two terms to keep the BM.

**The accounting year.** With `BOEKJAAR` as a rating factor, every policy is
priced at the level of its own historic year. Against a current premium
that mixes several years of inflation into the dislocation: on the R
package's review book the rate-level change was +11.1% that way and +34.7%
with every policy at 2024. Pass `at` to price every policy at one year, as
above. Without it, a warning points this out:

```python
imp_hist = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                          old_premium_col="PREMIE")
# PricingWarning: premium_impact: the new models use 'BOEKJAAR' as a rating factor,
#   so every policy is priced at the level of its own year. Against a current
#   premium that mixes inflation into the comparison; price every policy at the
#   tariff year with at={'BOEKJAAR': <year>}.
```

`at` changes what the models score, never the old premium column. The
warning also fires when only one of two model sets uses `year_col`; two
model sets that both use it price each policy at the same year on both
sides, which is like for like. `year_col=None` switches it off.

**What the models cannot price is named.** A level or a year the model was
not fitted on stops the function with the column, the values and the
number of rows, rather than an error about the offset:

```python
d_new = dat.copy()
d_new.loc[:4, "REGIO"] = "Oost"
premium_impact(d_new, model_freq_new=m_freq, model_sev_new=m_sev,
               old_premium_col="PREMIE", at={"BOEKJAAR": 2024})
# PricingError: premium_impact: the data holds value(s) the new frequency model was
#   not fitted on: 'REGIO' = Oost (5 rows). A model has no factor for a level it
#   never saw. Map those values to a level it knows or leave the rows out; for an
#   accounting year beyond the data, price at the latest known year with 'at' and
#   index the result yourself.
```

Map those values to a level the model knows, leave the rows out, or for a
later year use `at`. A numeric value outside the fitted range is priced,
with a warning, because a spline extrapolates there without a word.

**Returns** a `Result`: `summary` (display table), `stats` (the same
numbers as a `Result`; `stats.quantiles` is a dict `p5` .. `p95`), `policy`
(per row `OldRate`, `NewRate`, `NewRateRebased`, `ChangePct`; `Row` is the
0-based position of the row in `data`, for `data.iloc`), `by_level` (per
level of the `by` columns: exposure and premium, mean and total change,
contribution), `spotlight` (when given), `largest_increases` /
`largest_decreases` (top-`n_show` dislocations) and `plot`
(exposure-weighted histogram of premium changes with the median marked).

**Who drives the change.** With `by`, each level gets `MeanChangePct` (what
a policy in that level sees on average), `OldPremium`, `NewPremium`,
`PremiumShare` (its share of the old premium), `TotalChangePct` (the
change of its total premium) and `Contribution = PremiumShare ×
TotalChangePct`. The contributions of one variable's levels **sum to the
change in the total premium**, exactly, which answers the question a
pricing committee actually asks: not "how much does this segment move" but
"how much of our +10.5% comes from it". After a rebase they sum to 0. A
share of exposure times a mean percentage does not add up that way, because
a percentage on a cheap policy moves less money than the same percentage on
an expensive one; on the R package's review book it summed to -4.0% for a
total change of 0.0%.

```python
print(imp.by_level.Contribution.sum())     # 0 after the rebase, up to rounding
```

**Spotlighting a subset.** `spotlight` takes a boolean array or Series with
one value per row (taken in row order), or a string with an R-style
expression evaluated against the columns of `data`:

```python
imp = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                     old_premium_col="PREMIE", at={"BOEKJAAR": 2024},
                     spotlight="REGIO == 'Stad' & LEEFTIJD < 25")
print(imp.spotlight.summary)

young_city = ((dat.REGIO == "Stad") & (dat.LEEFTIJD < 25)).to_numpy()
imp = premium_impact(dat, model_freq_new=m_freq, model_sev_new=m_sev,
                     old_premium_col="PREMIE", at={"BOEKJAAR": 2024},
                     spotlight=young_city)                 # the same subset
```

The string is read with R's operator precedence, `%in%` works
(`"REGIO %in% c('Stad', 'West')"`), and it can refer only to columns of
`data` and the R functions a formula knows. R's version also sees the
caller's own variables; here, put the value in the string or pass a boolean
array. Missing values count as `False`.

It is deliberately a **spotlight and not a filter**: the whole book is still
analysed, the subset appears as a second series on the histogram, and
`imp.spotlight` reports its exposure share, mean and median change, and its
contribution to the portfolio total. Filtering would remove exactly the
context you need: that a segment moves +18% while the book moves +2%.

Its statistics are computed **after** the book-level rebase, because the
rebase is a decision about the whole portfolio. Rebasing within the subset
would force every segment to average out at zero and tell you nothing.

> **No mix effect here.** Old and new premiums are compared on the *same*
> policies, so the portfolio composition is identical on both sides and
> there is nothing to decompose into rate versus mix. `Contribution`
> answers which segment drives the change, not how much of it is mix. A
> genuine mix-shift analysis needs two portfolio snapshots and is a
> different calculation.

### `export_rating_table()`

Formatted Excel export of a `make_rating_table()` result (requires
`openpyxl`, the `[excel]` extra).

```python
export_rating_table(rating_tbl, file="rating_table.xlsx",
                    overwrite=True, digits=4)
```

The workbook contains:

- **Overview**: timestamp, the intercepts (per unit of exposure) and the
  base value per variable, written as numbers so a formula can refer to
  them;
- **Tariff**: every factor of every variable under one another with a
  single `Key` column (`Variable|Level`, plus the group for interaction
  rows), so a premium can be assembled with `VLOOKUP` instead of hopping
  between sheets:

  ```text
  =VLOOKUP("LEEFTIJD|"&B2; Tariff!A:H; 8; FALSE)
  ```

  Multiply the intercept from Overview by one factor per variable and you
  have the risk premium per unit of exposure, which reproduces the model
  prediction to machine precision. With an interaction, multiply by its key
  `Variable|Level|Group` as well: on this sheet that key holds the
  **uplift**, the part the two main effects do not already explain, so
  nothing is counted twice (the interaction sheets keep the joint factor).
  Before this was so, rebuilding a premium by the instruction came out 16%
  too high on average and up to 3.4 times. A bonus-malus discount is not a
  factor here and is applied on top;
- **one sheet per main-effect variable**: level, exposure, claim count
  and the factor columns; the base row is highlighted, thin rows are
  greyed out. Continuous levels are written as **numbers**, so Excel does
  not flag them as "number stored as text";
- **one sheet per interaction**: the long table plus a Level × Group
  matrix of the premium factor, ready for tariff implementation.

```python
tbl = make_rating_table(m_freq, m_sev, data=dat)
path = export_rating_table(tbl, "rating_table.xlsx")
print(path)                       # the normalised path of the written file
```

**Returns** the normalised path of the written file. With
`overwrite=False` an existing file is an error rather than replaced.

### `pricing_report()`

One call that bundles the whole analysis into a single HTML report.

```python
pricing_report(model_freq=None, model_sev=None, data=None,
               file="pricing_report.html",
               title="GLM Pricing Report",
               variables=None,
               include=("diagnostics", "oneway", "ae", "pdp", "rating",
                        "interactions"),
               by_year=False, top_interactions=3,
               exposure_col="Exposure", claims_col="AantalClaims",
               loss_col="SCHADELAST", year_col="BOEKJAAR",
               grid_res=50, base_level="first",
               trim=(0, 1), merge_vars=None, base=None,
               grid_step=None)
```

The report contains the diagnostics table with binned residual plots, and,
per variable (default: all base variables of both models), the one-way
observed plot, actual vs expected (frequency and severity), partial
dependence and the rating-factor plot, plus a section with all interaction
plots. The `"interactions"` block additionally runs
[`detect_interactions()`](#detect_interactions) on the frequency model and
plots the strongest `top_interactions` pairs as A/E heatmaps. Individual
plot failures are shown as a note instead of aborting the report.
`merge_vars`, `base` and `grid_step` go to the rating table, so the report
shows the same tariff as a table built with them.

```python
pricing_report(m_freq, m_sev, dat, file="pricing_report.html",
               title="Motor tariff 2025",
               variables=["LEEFTIJD", "REGIO", "GEWICHT"],
               include=("diagnostics", "ae", "rating"))
```

The page itself is written by the package (no pandoc or htmltools needed);
all plots remain fully interactive. Next to the `.html` a `<name>_files/`
folder is written with plotly.js: keep the two together when sharing. The
top of the report holds the title, both model formulas, a table of contents
and the diagnostics block. The one-way severity reads `loss_col` as a
**total** loss amount per row and aggregates it as `Σloss / Σclaims`.

**Returns** the normalised path of the written file, with a message saying
where it went.

### Spline bases: `ns()`, `bs()`, `poly()`

`ns()`, `bs()` and `poly()` are R's `splines::ns()`, `splines::bs()` and
`stats::poly()`: the same knots (type-7 quantiles of the training data), the
same boundary handling, the same linear (`ns`) and Taylor (`bs`)
extrapolation beyond the boundary knots, and the same basis. A model fitted
with `ns(LEEFTIJD, 4)` therefore spans exactly the space R's does, so fitted
values and every rating factor agree with R (see the [parity
table](#parity-measured-against-r)). The arguments are R's: `ns(x, df,
knots, intercept, Boundary.knots)`, `bs(x, df, knots, degree, intercept,
Boundary.knots)`, `poly(x, degree)`.

They are used inside a formula, where the knots are learned once on the
data the model is fitted on and reused whenever the model scores other data,
which is what R's `makepredictcall` does:

```python
m = fit_glm("AantalClaims ~ ns(LEEFTIJD, knots = c(25, 40, 60), Boundary.knots = c(18, 80))"
            " + bs(GEWICHT, df = 5) + REGIO + offset(log(Exposure))", dat)
print(m.term_labels)
m.predict(pd.DataFrame({"LEEFTIJD": [85], "GEWICHT": [1300], "REGIO": ["Noord"],
                        "Exposure": [1.0]}))   # 85 lies beyond the boundary: extrapolated linearly
```

Outside a formula they return the basis matrix of a vector, as in R:

```python
ns(dat.LEEFTIJD.to_numpy(), 4).shape     # (40000, 4)
```

### House-style colours

`ta_navy` `#00365E`, `ta_blue` `#0073AB`, `ta_lightblue` `#A8C8E0`,
`ta_gold` `#D39F27`, `ta_muted` `#6B7A8D`, and `ta_years_base`, a
nine-colour list. `ta_year_palette(n)` returns `n` colours from it,
interpolating beyond nine the way R's `colorRampPalette()` does.

```python
print(ta_year_palette(3))
print(len(ta_year_palette(11)))
```

---

## Actuarial methodology

The [function reference](#function-reference) above covers *how to use* each
function and how to read its output. This section covers *why* things are
computed the way they are, the reasoning you would want before putting a
number in front of a pricing committee. The methods are the R package's,
one to one, and so are the measurements quoted here.

### Methods at a glance

One line per function: what it computes, the expression behind it, and the
assumption it relies on. `y` is the response, `μ` the fitted value, `w` the
exposure or prior weight, `a` and `e` actual and expected claims in a cell.

| Function | Computes | Key expression | Relies on |
|---|---|---|---|
| [`agg_all()`](#agg_all) | frequency and severity per level | `Σclaims / Σexposure` and `Σloss / Σclaims` | none |
| [`make_plot()`](#make_plot) | the observed one-way | as above, per level of one variable | none |
| [`plot_glm_predictor()`](#plot_glm_predictor) | actual vs expected per bin | counts: `Σy/Σw` against `Σμ/Σw`; weighted models: prior-weight-weighted means | log link, to recover exposure as `exp(offset)` |
| [`make_pdp()`](#make_pdp) | the partial effect on the response scale | weighted mean of `m.predict()` per grid point, exposure offset subtracted on the link scale, any other offset kept → [why](#why-the-pdp-is-computed-on-the-response-scale) | offset read from the model → [how](#deriving-the-offset-and-the-link) |
| [`glm_diagnostics()`](#glm_diagnostics) | fit and dispersion | Pearson `χ²/df`; `1 − D/D₀` | → [overdispersion](#overdispersion) |
| [`glm_collinearity()`](#glm_collinearity) | how far each term is explained by the others | generalised VIF, `GVIF^(1/(2·DF))`; terms in an interaction reported, not flagged | terms, not coefficients |
| [`model_lift()`](#model_lift) | risk separation across the book | `Σactual/Σw` against `Σpred/Σw` per equal-exposure bin; Gini on the Lorenz curve, with a bootstrap interval | → [what lift measures](#what-lift-and-gini-do-and-do-not-measure) |
| [`double_lift()`](#double_lift) | which of two tariffs is right where they differ | A/E of each model per bin of the rate ratio `new/old`; a winner only when the interval of `mad_old − mad_new` excludes 0 | rebased to equal totals |
| [`plot_glm_residuals()`](#plot_glm_residuals) | binned residuals | mean residual per bin, band `±2·√(Σvar)/n`; deviance residuals centred on their expectation | the fitted family and `φ` |
| [`plot_glm_qq()`](#plot_glm_qq) | does the response match the family | randomised quantile residual `Φ⁻¹(U(F(y⁻), F(y)))` → [why not the classical residuals](#reading-a-q-q-plot-of-quantile-residuals) | a closed-form CDF for the family |
| [`plot_glm_influence()`](#plot_glm_influence) | which rows move the fit | leverage vs standardised residual, sized by Cook's `D` | → [on a large book](#cooks-distance-on-a-large-book) |
| [`detect_interactions()`](#detect_interactions) | interaction structure the GLM missed | `D = 2·Σ[a·log(a/e*) − (a − e*)]`, where `e*` is `e` raked to `a`'s margins; null by simulation, scaled by the cell-total dispersion → [why](#why-a-one-way-check-cannot-see-an-interaction) | Poisson counts (otherwise `χ²` scaled by `φ`) |
| [`plot_residual_heatmap()`](#plot_residual_heatmap) | A/E per cell | `Σa / Σe` per cell of two variables | none |
| [`screen_features()`](#screen_features) | incremental value of a candidate | increase in out-of-sample deviance when the feature is shuffled, baseline margin held fixed | a holdout split; log link |
| [`make_rating_table()`](#make_rating_table) | multiplicative relativities | `pred(level, rest at base) / pred(base)`; interaction uplift `joint / (mainₓ · main_g)`; standard error `√(dᵀVd)` of each log factor → [why](#multiplicative-rating-and-the-reconstruction-identity) | log link |
| [`add_groupings()`](#add_groupings) | the grouped columns a model was fitted on, for new data | the source-to-group relation in the model's own data, exact for a categorical source, nearest value for a band | the model carries its data (a `GLMFit` always does) |
| [`make_rating_plot()`](#make_rating_plot) | those relativities, plotted | none | → [thin cells](#thin-cells) |
| [`premium_impact()`](#premium_impact) | dislocation against the current tariff | `new/old − 1` per policy, optionally rebased so the totals match; contribution `(new − old) / old total` per level | log link, to price at exposure = 1 |
| [`export_rating_table()`](#export_rating_table) / [`pricing_report()`](#pricing_report) | deliverables | none | none |

Two assumptions run through almost the whole table and are worth stating
once. **The log link** is what makes relativities multiplicative and lets
`exp(offset)` be read back as exposure; a different link makes several of
these numbers something other than what their names suggest, which is why
the functions warn about it. **Exposure weighting** is done by summing
numerator and denominator separately (`Σa / Σe`), never by averaging ratios:
an average of ratios would give a policy with a month of exposure the same
say as one with a full year.

### Multiplicative rating and the reconstruction identity

For log-link GLMs the prediction factorises multiplicatively. The rating
table exploits this with a single consistent **base point**:

- categorical variables → reference level (`base_level`), unless set in `base`,
- the accounting year (`year_col`) → its **latest** year, unless set in `base`,
- continuous variables → the **median** (inserted as an explicit grid point), unless set in `base`,
- the exposure column → **1**.

Every factor is `prediction(level, others at base) / prediction(base)`, and
the intercept attributes are the prediction at the base point itself. For a
log-link model **without interactions** this makes the identity exact:

```text
prediction(x₁, …, xₖ, exposure = 1)
  = intercept × Factor(x₁) × … × Factor(xₖ)
```

(verified to 1e-10 in the R test suite, and asserted in this package's own
tests). With interactions, the joint relativity of a pair decomposes as
`main_x × main_group × Uplift`, so the uplift columns isolate what the
interaction adds on top of the main effects, and the identity holds with
either the joint factor in place of the pair's two main effects or the
uplift on top of them (shown in code under [Factor vs
Uplift](#factor-vs-uplift-on-an-interaction-row)).

Because the intercept is evaluated at exposure = 1, it is a true base
frequency (respectively base severity / base risk premium) per unit of
exposure, not a value at some arbitrary portfolio exposure.

### Why the PDP is computed on the response scale

A naive GLM PDP averages link-scale predictions and back-transforms:
`exp(mean(link))`. That is a *geometric* mean, and with an exposure offset
it also absorbs the average log-exposure. Both effects bias the curve low
relative to observed frequencies (Jensen's inequality), making the
model-vs-data comparison misleading.

`make_pdp()` therefore:

1. neutralises the offset (exposure = 1),
2. predicts on the response scale, and
3. takes the exposure-weighted (frequency) or claim-weighted (severity)
   arithmetic mean,

which is the same estimand as the observed `sum(claims) / sum(exposure)`
line it is plotted against.

### Actual vs expected

`plot_glm_predictor()` compares like with like: for offset models it
aggregates counts and exposure before dividing (`Σ observed / Σ exposure` vs
`Σ predicted / Σ exposure`); for weighted models it uses
prior-weight-weighted means. Quantile binning keeps each bin credible
instead of leaving near-empty tail bins that look like misfit but are noise.

### Why a one-way check cannot see an interaction

This is not a matter of bad luck or too little data; it is structural. The
score equations of a GLM weight each observation by `(dμ/dη) / V(μ)`, and
for a **canonical** link that weight is exactly 1. What is left is `Σ x(y −
μ) = 0` for every column of the design matrix, so for a categorical variable
in the model the fitted total equals the observed total at every level: the
A/E is exactly 1.000, no matter what happens inside the cells. On the
quick-start model, which misses the young-in-the-city interaction:

```python
rows = pd.DataFrame({"REGIO": m_freq.data.REGIO, "y": m_freq.y, "mu": m_freq.fitted})
totals = rows.groupby("REGIO")[["y", "mu"]].sum()
print((totals.y / totals.mu).round(4))      # 1.0 for every region
```

A missing interaction cancels out in the margins by construction. No amount
of one-way plotting will reveal it, and an actual-vs-expected plot per
predictor will look perfect while whole cells of the portfolio are mispriced
by tens of percent. That is precisely the blind spot
[`detect_interactions()`](#detect_interactions) and
[`plot_residual_heatmap()`](#plot_residual_heatmap) exist to cover.

#### Frequency is exactly blind, severity almost

Poisson with a log link is canonical, so a frequency model is blind by
identity. **Gamma with a log link is not**: the canonical link for Gamma is
the inverse, so this argument does not carry over to a severity model
unchanged. It does not disappear either; it changes what gets pinned.

With a log link on a Gamma, the score weight is `μ/μ² = 1/μ`, so the
equations reduce to `Σ x·w·(y/μ − 1) = 0`. What is forced to exactly 1 is
the weighted mean of the **ratio**, not the ratio of the weighted totals.
Measured on 40,000 rows:

| Model | `Σwy / Σwμ` (the A/E) | `Σw(y/μ) / Σw` |
|---|---|---|
| Poisson log (canonical) | **1.0e-14** | 1.3e-02 |
| Gamma log (used here) | 1.2e-04 | **3.3e-09** |
| Gamma inverse (canonical) | **2.6e-10** | 1.2e-04 |

Each link pins something exactly; the canonical one happens to pin the
quantity an A/E plot draws.

So a severity model does leave a residue in the one-way A/E, and it is far
too small to work with. On the same data:

- with no interaction at all: max `|A/E − 1|` = **0.012%**
- with a genuine omitted interaction of 35%: **0.259%**

A real effect is only about twenty times the numerical floor, and a quarter
of a percent is invisible on an axis running in the thousands. So the
practical conclusion is unchanged for both sides of the tariff: the one-way
check does not find interactions. It is exact blindness for frequency and
effective blindness for severity, and the cell-level tools are what cover it
either way.

There is a second route to the same blindness that has nothing to do with
the link: a model **saturated** in a variable fits each of its levels
exactly, so the A/E is 1.000 there whatever the link. Verified with an
identity-link Poisson on a single factor: max `|A/E − 1|` = 9.9e-14.

Two things to keep in mind when acting on what they find:

- **A candidate is a hypothesis, not a conclusion.** It only counts if
  adding the term to the GLM improves out-of-sample fit, the resulting
  factor pattern is explainable, and it holds up in another accounting
  year. The scan proposes; the GLM decides.
- **Beware of proxies and mix.** A detected `A × B` can be standing in for
  an unmodelled `C`, and an "interaction" with the accounting year is
  usually portfolio mix or trend rather than risk structure.

### Thin cells

Levels backed by very few claims get `IsThin = True` (claim count below
`min_claims`, default 30). This matters because a GLM applies **no
shrinkage** to a categorical level: its coefficient comes essentially from
that level's own claims, so a factor built on a handful of them is an
unstable estimate rather than a signal. (Continuous variables fitted with
splines do borrow strength from neighbouring grid points, so a sparse
stretch of the curve is still supported by the data around it.)

The flag is surfaced everywhere the factors are shown: `make_rating_plot()`
dims the markers and adds a "low claim volume" note to the tooltip,
`export_rating_table()` greys out the row, and `make_rating_table()` warns
when a categorical level trips the threshold, so a factor of 1.4 on 12
claims is never read with the same confidence as one on 12,000. What to do
about it (grouping the level with a related one, capping the factor, or
leaving it out of the tariff) stays a judgement call; the package only flags
it.

A claim count is a proxy, though, and the threshold of 30 is a convention.
The precision itself is in `SE_Frequency` and `SE_Severity`, the standard
error of each factor's logarithm from the model's covariance matrix. On the
R package's review book a fuel type with 31 claims passed the flag, with an
estimated factor of 1.17 against a true 1.00 and a standard error of 0.18: a
95% interval of x0.70 to x1.42 around the factor, which says far more than
`IsThin = False`. A Poisson model's covariance assumes a dispersion of 1,
so on overdispersed counts fit a quasi-Poisson (`family="quasipoisson"`) for
honest standard errors.

A level with exposure and **no claims at all** is a case of its own. The
frequency factor is then the fit running towards zero, and the warning
says so; the severity model has never seen the level, so its factor is
`NaN` in the table rather than the whole variable disappearing from it.

### What lift and Gini do and do not measure

A Gini measures **ordering**, nothing else. It asks whether the policies the
model calls expensive really are, relative to the ones it calls cheap. It
says nothing about whether the level is right: multiply every prediction by
three and the Gini is unchanged while the tariff is ruinous. The lift chart
is what catches that, because it puts actual and predicted on the same axis,
per bin.

So read them together: **Gini for discrimination, the lift chart for
calibration.** A model can be excellent at one and poor at the other.

A single lift chart also cannot rank two candidates. Both can look
convincing separately while disagreeing sharply about individual policies,
and the disagreement is the whole question. That is what
[`double_lift()`](#double_lift) isolates: it bins on the ratio of the two
predictions, so the end bins contain exactly the policies where the models
part company, and the A/E lines show who is right there.

Two cautions. Lift on the training data flatters both models, which is why
`data` is exposed for a holdout. It will not stop you, but it will label the
plot in-sample. And `double_lift()` rebases the two sides to the same total
by default: without that, a candidate that is simply 10% cheaper everywhere
would look systematically better, when the comparison is about
differentiation rather than rate level.

And a third: both numbers are noisy on raw losses. A few large claims move
the Gini and decide the double-lift verdict, so both come with a bootstrap
interval, and the double lift names a winner only when its interval
excludes 0. The bootstrap resamples policies with the models held fixed,
so on the training data it is too narrow as well.

### Collinearity

Collinearity does not damage predictions; it damages **interpretation**. A
GLM with two near-duplicate predictors can fit and forecast perfectly well
while splitting the effect between them arbitrarily, so the individual
coefficients, and therefore the rating factors, swing wildly with small
changes in the data.

That matters here specifically because a rating table *is* an
interpretation: [`make_rating_table()`](#make_rating_table) reads each
term's coefficients back out as relativities. If two terms are collinear,
those two columns of the table are not trustworthy even though the model is
fine. [`glm_collinearity()`](#glm_collinearity) is the check, on the
generalised VIF scale because pricing terms span multiple columns.

### Overdispersion

Poisson frequency models on real portfolios are almost always overdispersed
(unobserved heterogeneity). `glm_diagnostics()` estimates the Pearson
dispersion and warns above 1.2: coefficient estimates remain consistent, but
standard errors scale with √dispersion, so term selection based on naive
Poisson p-values is anti-conservative. Refit with `family="quasipoisson"` or
`family="negbin"` when flagged.

The same extra variance would read as interaction in
[`detect_interactions()`](#detect_interactions), whose null is a Poisson
simulation, so the scan divides it out first. It uses the dispersion of a
cell total, `Σ(y − μ)² / Σμ`, rather than the row-level Pearson one: a
cell total adds policies by their expected claims, and under unobserved
heterogeneity the policies with more expected claims carry more of the
extra variance.

### Deriving the offset and the link

Everything needed is on the fitted model, so nothing is assumed. The link
and its inverse come from the `GLMFit` (`m.link`, `m.linkinv()`), and the
offset from the offset expressions in its formula, so the neutralised
prediction is simply

$$\hat\mu_{\text{rate}} = g^{-1}\big(\hat\eta - \text{offset}\big)$$

which is exact for **any** link and **any** offset expression. No family
is a special case and none is unsupported. In code it is
`m.predict(newdata, drop_offsets=[...])`.

**Which part of the offset is taken out depends on what the number is
compared with.** A model may carry more than one offset, typically the
exposure and a known bonus-malus scale,
`offset(log(Exposure)) + offset(log(BM_KORTING))`. The exposure says how
long a policy was on risk; the BM says how much it pays. So the package
subtracts either the whole offset or only its exposure part:

| Function | Offset taken out | So the number is | Why |
|---|---|---|---|
| `make_rating_table()` (intercept) | **all** of it | per unit of exposure, at BM = 1 | a tariff table describes a base policy; the discount is applied on top, per policy |
| `premium_impact()`, `model_lift()`, `double_lift()`, `make_pdp()` | **only the exposure** (default `keep_offsets=True`) | per unit of exposure, at the policy's own BM | they compare with something that includes the discount: the premium a customer pays, the claims that actually happened, the observed line next to the PDP |

The second row is $g^{-1}(\hat\eta - \text{offset}_{\text{exposure}})$: only
the offset terms that mention `exposure_col` are subtracted. Taking the BM
out there as well put every policy at the class without discount, and a
correct model with an in-sample A/E of exactly 1.000 then read 0.800 on a
book with an average discount of 20%, and a PDP at BM = 1 ran 35% above
its own observed line on a book with an average BM of 0.74.
`keep_offsets=False` restores taking everything out. Two cases fall back to
it automatically: a model whose offset is not on `exposure_col` at all, and
one expression holding both, `offset(log(Exposure * BM))`, which cannot be
split and says so in a warning.

`plot_glm_predictor()` does not remove anything: it compares the model's
own fitted values with the observed response, both including every
offset, and only divides by `exposure_col` to put the axis in claims per
policy-year.

That is worth spelling out because the obvious shortcut, set the exposure
column to 1 so `log(1) = 0`, silently is not. It neutralises exactly one
offset and mis-scales the rest. Measured on the same model and data, rate
per unit of exposure:

| Model | column set to 1 | offset subtracted | correct |
|---|---|---|---|
| `offset(log(Exposure))` | 0.13063 | 0.13063 | 0.13063 |
| `offset(Exposure)` | 0.11764 | 0.04328 | 0.04328 |
| `offset(log(Maanden/12))` | 0.01089 | 0.13063 | 0.13063 |

A factor `e` out on the second, because setting the column to 1 leaves the
offset at 1 rather than 0, and a factor 12 out on the third, because it
leaves `log(1/12)`. Where the shortcut was already right the two agree to
6·10⁻¹⁷.

#### Where the offset can hide

In R an offset reaches a `glm` by two routes, the formula and the separate
`offset =` argument, and the R package reads both. `fit_glm()` has only the
first: there is no `offset=` argument, so every offset is an `offset()`
term of the formula, kept as an expression and listed in `m.offset_exprs`.
That also settles the constraint R's argument form carries: an offset
expression is evaluated against the data being scored, so the variables it
names must be **columns of that data**. `offset(log(Exposure))` is fine,
because `Exposure` travels with the data; a vector from your workspace
cannot be written into a formula at all. An offset that does not evaluate to
one value per row of the data is refused rather than recycled into a
confidently wrong number.

#### What the link does and does not decide

The link does not decide whether the offset can be removed, only what the
result **means**:

- **A log link makes the model multiplicative**, so removing the offset
  gives a rate that a second model may be multiplied onto. That is what
  the frequency × severity workflow rests on.
- **Any other link is additive on its own scale.** The offset still comes
  out, and the prediction at zero offset is exactly right, but it is not a
  multiplicative rate and multiplying a severity model onto it means
  nothing. The functions say so rather than let it pass.

One genuinely separate question is whether `exp(offset)` is an exposure,
which is used for weighting rather than for prediction. It is one only
when the offset is a **logarithm**: with `offset(Exposure)` it is
`exp(Exposure)`, not a volume. That is also read off the model, and the
A/E and weighting code falls back to the prior weights and says why.

**Rating-table factors were never affected** by any of this. They are
ratios of two predictions that share the same base row, so whatever the
offset contributes cancels. The intercept is the one number that carried
it.

Removing the offset on the link scale leaves rounding noise of order 10⁻¹⁶
where the shortcut produced bit-identical values, so the tie check in the
lift charts counts distinct predictions at a tolerance rather than bit for
bit.

### Reading a Q-Q plot of quantile residuals

A normal Q-Q plot of the classical residuals is close to useless for a
frequency model, because for counts none of them are normal. What the
standard tools in R actually draw:

| Package | Q-Q of a glm uses |
|---|---|
| base R `plot(model)` | `rstandard(type = "pearson")`, labelled "Std. Pearson resid." |
| `boot::glm.diag.plots()` | standardised deviance, `dev / (sd·√(1−h))` |
| `car::qqPlot()` | declines, *"QQ plot for studentized residuals not available for glm"* |
| `statmod::qresid()` | randomised quantile residuals (what this package uses) |
| `DHARMa` | simulated residuals, Q-Q against a **uniform** |

On a correctly specified Poisson with 20,000 rows, 87% of them claim-free:

| residual | outside a nominal 5% band | KS test vs N(0,1) |
|---|---|---|
| base R, std. Pearson | 8.34% | `p < 2e-16` |
| `boot`, std. deviance | 1.31% | `p < 2e-16` |
| quantile residual | 5.10% | `p = 0.86` |

The two classical ones fail in opposite directions, Pearson too heavy in the
tails, deviance too light, and both reject a model that is exactly right, so
neither can tell you anything about one that is not. Where the fitted mean
varies little they also collapse onto visible bands, because the response
takes only a handful of values.

`plot_glm_qq()` uses the **randomised quantile residual** instead (Dunn &
Smyth, 1996). For a continuous response,

$$r_i = \Phi^{-1}\!\big(F(y_i;\hat\mu_i,\hat\phi)\big)$$

and for a discrete one the CDF jumps, so the value is drawn uniformly inside
the jump:

$$u_i \sim \mathrm{U}\big(F(y_i-1;\hat\mu_i),\,F(y_i;\hat\mu_i)\big),
\qquad r_i = \Phi^{-1}(u_i)$$

Under a correctly specified model these are **exactly** standard normal,
discreteness included, which is what the table above shows. Per family:

| Family | CDF used | Dispersion |
|---|---|---|
| Poisson | the Poisson CDF of `y` at `μ`, with the jump randomised | fixed at 1, that assumption is what the plot tests |
| Gamma | the Gamma CDF with `shape = w/φ`, `scale = φμ/w` | Pearson estimate |
| binomial | the binomial CDF on `y·w`, jump randomised | fixed at 1 |
| gaussian | the normal CDF with mean `μ` and sd `√(φ/w)` | Pearson estimate |
| negative binomial | its own CDF at `μ` and `θ`, jump randomised (no prior weights) | fixed at 1 |

The Gamma parameterisation is the GLM's own assumption written out: mean `μ`
and variance `φμ²/w`, which is exactly what `weights="AantalClaims"` encodes
on an average-claim-size response.

Because the count case is randomised, two calls give slightly different
pictures; pass `seed` for a reproducible one. The draws come from numpy's
generator, so a seed does not reproduce R's picture point for point; the
residuals of a continuous family are not random and agree with the R
package (Gamma to 3e-9, see the [parity table](#parity-measured-against-r)).
The R implementation is verified against `statmod::qresid()`: identical for
Poisson, equal to 6e-14 for Gamma.

The other principled route is a **simulated envelope**, as `DHARMa` and
`hnp` use: simulate from the fitted model and locate each observation in
that distribution. It generalises further (mixed models, zero-inflation) but
costs `n_sim` simulations per row, so the closed-form residual is the better
fit for a large book. Its one advantage here is that the envelope accounts
for the coefficients having been estimated, while the analytic band from the
Beta order statistics treats them as known and is therefore marginally too
narrow. The band is also pointwise, not simultaneous: with several thousand
points a handful fall outside it even when the model is right. Read the
shape, not the exceptions.

### Cook's distance on a large book

Cook's distance was designed for regressions with tens of observations,
where one point can genuinely swing a coefficient:

$$D_i = \frac{r_{P,i}^2\,h_i}{\phi\,p\,(1-h_i)^2}$$

On a book of 100,000 policies no single row does. Every absolute rule of
thumb breaks with it: `D > 1` never triggers, and `D > 4/n`, here `4 ×
10⁻⁵`, flags thousands of perfectly ordinary rows. So `plot_glm_influence()`
is deliberately **relative**: it ranks rows against each other and labels
the worst `n_label`, rather than testing them against a threshold that does
not apply. The only absolute reference kept is the leverage line at `2p/n`,
which still means what it always did.

What the ranking is good for is **data quality**: a vehicle weight of 20
tonnes or an age of 200 lands at the top and is worth opening. It is a much
weaker tool for model structure, where
[`detect_interactions()`](#detect_interactions) and
[`plot_glm_residuals()`](#plot_glm_residuals) have far more power, because
structural misfit is spread over many rows and none of them stands out
individually.

**Influence is not misfit, and on a Poisson the two can be opposites.**
The IRLS weight of a row is its fitted mean, so a policy with a sliver of
exposure carries almost no weight in the design matrix and has almost no
leverage, whatever its response. Measured over three seeds on 15,000-row
books, implanting each classic keying error one at a time:

| Implanted error | rank on Cook's `D` | rank on Pearson size | leverage |
|---|---|---|---|
| exposure 0.001 with claims | 26 / 35 / 43 | **1 / 1 / 1** | 5·10⁻⁷ |
| vehicle weight 20 tonnes | **1 / 1 / 1** | 1691 / 15000 / 2304 | 0.91 |

They are exact complements. The low-exposure row is the worst-fitted row in
the book and Cook's distance cannot see it at all. The extreme-predictor row
tops the Cook ranking and has an *unremarkable* residual: at a leverage of
0.9 the fit bends to meet it, so the point masks itself.

That is why this is a two-axis plot and not an index plot of `D`: it
separates *unusual in the predictors* (far right) from *badly fitted* (far
up or down), and neither axis alone finds both errors. A residual check,
[`plot_glm_qq()`](#plot_glm_qq) or
[`plot_glm_residuals()`](#plot_glm_residuals), is the complement that
catches the low-exposure case.

---

## Input validation and warnings

The functions fail fast with explicit messages rather than producing
silently wrong numbers. An error is a `PricingError` (a `ValueError`), a
warning a `PricingWarning`, and information a `PricingMessage`; see
[Working with the results](#working-with-the-results) for how to catch or
silence them.

- **Errors**: missing columns (named per function); a model that is not a
  `GLMFit`, named by its argument, in every function that takes a model;
  a formula that cannot be read (`.`, a trailing operator, no `~`), as a
  `FormulaError`, which is a `PricingError`; unknown `weight_var`, unmatched predictors,
  invalid `trim`, `y_range`, `n_boot` or `conf_level`; a level or year a
  model was not fitted on, named with its column, values and row count
  (`premium_impact`, `model_lift`, `double_lift`, `plot_glm_predictor` on
  new data); a rating-table base row a model does not know; an `at` value
  that is not valid for its column; a severity model compared with a
  frequency or risk premium in `double_lift`; a `merge_vars` declaration
  that is not a regrouping, or one column name for two groupings
  (`make_rating_table`, `add_groupings`); in `fit_glm`, an unknown family
  or link and a column the formula names that is not in the data. A formula
  that cannot be read, such as one with `.`, raises a `ValueError` from the
  formula parser.
- **Warnings**:
  - missing values in exposure/claims/loss inputs and groups with
    exposure ≤ 0 (`agg_all`);
  - non-log link where multiplicativity or `exp(offset)` is assumed
    (`make_rating_table`, `make_pdp`, `plot_glm_predictor`);
  - an offset that cannot be neutralised because the exposure column is not
    in the training data (`make_pdp`);
  - unweighted PDP averaging when no offset/weights are found;
  - skipped rating-table terms/interactions, with the reason;
  - overdispersion in Poisson/binomial models (`glm_diagnostics`);
  - categorical rating-table levels with fewer than `min_claims` claims,
    naming the ones without any claim (`make_rating_table`);
  - rating-table cells a model has no factor for, left `NaN`
    (`make_rating_table`);
  - a regrouping that is not declared in `merge_vars`, a declaration that
    changes nothing, and `grid_step` or `base` names that match no variable
    (`make_rating_table`);
  - claims without payment, which make frequency x severity too high
    (`make_rating_table`, `model_lift`);
  - numeric values outside the fitted range when scoring
    (`premium_impact`, `model_lift`, `double_lift`, `add_groupings`);
  - the accounting year priced per policy against a current premium
    (`premium_impact`, switched off by `at` or `year_col=None`);
  - a wide Gini interval or an inconclusive double lift when large losses
    are behind it (`model_lift`, `double_lift`);
  - an overall A/E far from 1, which usually means mismatched units
    (`model_lift`, `double_lift`);
  - rows dropped, counted per reason (`premium_impact`, `model_lift`,
    `double_lift`);
  - terms in an interaction, whose GVIF is not flagged, and flagged terms
    (`glm_collinearity`);
  - a heatmap in which every cell is thin (`plot_residual_heatmap`);
  - an outcome or volume column named as a screening candidate
    (`screen_features`);
  - deviance residuals that cannot be centred for the family
    (`plot_glm_residuals`);
  - `theta` estimation that hits its iteration limit (`fit_glm` with
    `family="negbin"`, typically on data that are not overdispersed at
    all), as `MASS::glm.nb()` warns.

The message texts are the R package's, so the R documentation and a search
of either code base find the same message. Where a message shows how to fix
a call, the example is in Python syntax (`at={'BOEKJAAR': 2024}`), and an
install hint says `pip install`.

---

## Differences from the R package

The port is checked against the R package by the parity tests: the same
data and formulas go through both packages and the results are compared.
Everything that is not random matches R; what differs is listed below, with
the reason. The working notes behind this section are in
[`docs/deviations.md`](docs/deviations.md).

### Parity measured against R

R fitted with a tight `glm.control(epsilon)`, so that its own convergence
does not dominate the comparison.

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

### Deliberate or unavoidable differences

**Fitting.**

- Models are fitted with `fit_glm()` (statsmodels underneath) instead of
  being any R `glm`. R formula syntax supported: `offset()` (any number),
  `factor()`, `cut()` (R labels), `ns()`, `bs()`, `poly()` (R-compatible
  bases), `log()`, `exp()`, `sqrt()`, `pmin()`, `pmax()`, `ifelse()`, `I()`,
  `^`, `*`, `:`, `-1` and backticked names. Not supported: `.`, the separate
  `offset =` argument (write the offset in the formula), and `na.exclude`:
  Python always fits as `na.omit`, so the R package's handling of
  `na.exclude` models has no counterpart.
- Gamma's default link is `log` (R's default is the inverse), as is the
  inverse Gaussian's. Families: poisson, quasipoisson, gamma, gaussian,
  binomial, inverse.gaussian, tweedie, negbin.
- Term labels and term order follow R (`REGIO:LEEFTIJD`,
  `ns(LEEFTIJD, knots = c(30, 50))`, `cut(LEEFTIJD, c(17, 40, 99),
  include.lowest = TRUE)`); the design
  matrix columns and coefficient names are patsy's (`REGIO[T.West]` where R
  writes `REGIOWest`).
- `fit_glm()` cannot fit a Gamma model to a constant response (statsmodels
  refuses the perfect fit, where R returns a deviance of 0). Never a
  pricing case.

**Randomness.** Python uses numpy's `Generator`, so the bootstrap
intervals, the simulated interaction null, the randomised quantile
residuals, sampling and boosting give different draws than R's
`set.seed()` streams. A `seed` makes a Python result reproducible, not equal
to R's. Everything that is not random matches R.

**Results.**

- R lists are `Result` objects (a dict with attribute access); data frame
  attributes are `DataFrame.attrs`; tables attached to a plot are figure
  attributes (`fig.cells`, `fig.residuals`, `fig.influential`); plotly's
  config is kept as `fig.pricing_config` (pass it to
  `fig.show(config=...)`).
- Row numbers are 0-based (`premium_impact().policy.Row`, and the row
  `add_groupings()` names when a grouped column disagrees with the model);
  the influence table's `Row` is the data's index label.
- Messages keep R's English text; examples inside them use Python syntax,
  install hints say `pip install`.

**Per function.**

- `make_plot(display="facet")` is built with plotly subplots, not
  ggplot2 + ggplotly: the same panels and values, a slightly different
  legend and hover text. Category axes get their order set explicitly
  (R's plotly takes it from the factor).
- `premium_impact(spotlight=)` takes a boolean array/Series or a string
  with an R-style expression on the data's columns; the caller's own
  variables cannot be used inside the string. `stats.quantiles` is a dict.
- `model_lift()` / `double_lift()`: with many exactly tied predictions
  (whole ages, few factors) the order of the ties depends on last-bit
  rounding, which differs between R and numpy, so a bin boundary can fall
  on a different policy. Totals per bin then differ slightly; with
  continuous predictors the bins are identical.
- `screen_features()`: `Gain` is xgboost's `total_gain` normalised to sum
  to 1 (R's `xgb.importance` Gain). The baseline refit falls back to the
  supplied model, with R's warning, in one case more than R: a factor level
  missing from the training split while two or more levels remain. R's
  refit then succeeds and its prediction on the validation or test split
  stops with "factor has new levels"; Python treats those predictions as
  part of the refit attempt.
- `glm_collinearity()`: `Flag` is a nullable boolean column (`<NA>` for
  terms in an interaction).
- An offset that comes out of a rate is never evaluated, so data to be
  scored does not need its column (R evaluates it in order to subtract it,
  and stops when it cannot). An offset that stays in the rate is evaluated,
  and refused with R's message when it cannot be.

**Found while porting, fixed in both packages.** Three defects of the R
package came to light in the port and were fixed in R and Python alike: a
grouped column the caller puts in the data now keeps its value for a source
value the model's data never saw (R stopped, although its own message
advised that column); a logical predictor in the rating table is a
two-level categorical variable with `FALSE`, the model's reference, as
base (it was read as continuous, with the most frequent value as base); and
the rating table, lift, impact and report functions name a model argument
that is not a fitted GLM.

---

## Tests

The package ships with a pytest suite (`tests/`), the port of the R
package's `testthat` suite: one Python test per R `test_that()`, on a
simulated portfolio with Poisson/Gamma models (`tests/conftest.py`). Among
other things it verifies exact premium reconstruction from the table,
factor = 1 on base rows, uplift = 1 at reference levels, PDP = exposure-
weighted mean response at exposure 1, offset-neutralised intercepts, inline
`factor()` and `cut()` handling, custom column names, the non-log-link and
overdispersion warnings, binned residuals staying within the ±2·SE band for
a correct model, the thin-cell flag, zero dislocation for identical models
and exact rebase behaviour in `premium_impact()`, the Excel workbook
structure and the generated HTML report. For the interaction tools it
simulates a portfolio with a *known* missing interaction and checks that
`detect_interactions()` ranks the true pair first, and that a portfolio
without any interaction produces no signal. The regression tests of the R
package's external review (`test-review-findings.R`) are ported with the
modules they concern.

```sh
pip install -e ".[dev]"
pytest -m "not parity"      # the Python suite; needs no R
pytest -m parity            # the comparison with the R package
pytest                      # both
```

**Parity tests** (`tests/parity/`, marked `parity`) build the data in
Python, fit the same formulas in both languages, run the R function through
`Rscript` and compare: exact for structure, labels and counts, a relative
tolerance of 1e-6 or tighter for fitted quantities (`fit_glm()` itself
against `glm()` and `MASS::glm.nb()`: 1e-10, a severity fit 1e-7). They
need:

- R with `Rscript`, found through the `RSCRIPT` environment variable, then
  on the `PATH`;
- the R package (`library(pricingtoolsRmO)`) and `jsonlite` installed in
  that R.

Without `Rscript` they are skipped, not failed. The GitHub Actions workflow
(`.github/workflows/tests.yml`) runs `pytest -m "not parity"` on every push
to `main` and on pull requests, on Python 3.11 and 3.12, each with pandas
2.x and pandas 3.x.

## Known limitations

Deliberately out of scope (for now), as in the R package:

- **Uncertainty only in places**: the rating factors carry standard errors
  (`SE_Frequency`, `SE_Severity`) and the Gini and the double-lift verdict a
  bootstrap interval, but premiums and PDPs have no interval, and the
  bootstrap holds the fitted models fixed.
- **In-sample by default**: every diagnostic evaluates on the model's own
  rows unless told otherwise. Some functions do more:
  [`model_lift()`](#model_lift), [`double_lift()`](#double_lift) and
  [`plot_glm_predictor()`](#plot_glm_predictor) take other data through
  `data`, and [`screen_features()`](#screen_features) splits internally
  because a booster is meaningless without one. There is no
  cross-validation and no automated backtesting.
- **Frequency × severity only**: `fit_glm()` fits a Tweedie, but a direct
  risk-premium model does not fit the rating-table structure.
- **No rate-making beyond the risk premium**: no on-levelling, trend
  adjustment or loading for expenses, commission and profit.
- **Two-way interactions only**: higher-order interaction terms are skipped
  (with a warning).

Specific to the Python version:

- Every analysis function takes a `GLMFit` from `fit_glm()`; models fitted
  with statsmodels, scikit-learn or anything else directly are not
  accepted.
- Prior weights given as an array rather than a column name are used for
  the fit, but a refit or new data refuses them (see [Weights](#weights)).
- Randomised results (bootstrap intervals, the interaction null, quantile
  residuals of counts, screening) cannot reproduce R's numbers, only their
  distribution.
- The formula language is R's as far as the table under [The
  formula](#the-formula) goes; `.` and the `offset =` argument are not
  supported.

## License

MIT, see [LICENSE](LICENSE).
