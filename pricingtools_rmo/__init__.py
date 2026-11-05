"""pricingtools_rmo: GLM pricing tools for non-life insurance.

The Python port of the R package pricingtoolsRmO. Fit a frequency and a
severity model with :func:`fit_glm` (R-style formulas, offsets included),
then explore, check, tabulate and compare them with the same functions,
arguments and outputs as in R.
"""
from ._core import PricingError, PricingMessage, PricingWarning, Result
from .ae import plot_glm_predictor
from .agg import agg_all
from .colors import (ta_blue, ta_gold, ta_lightblue, ta_muted, ta_navy,
                     ta_year_palette, ta_years_base)
from .diagnostics import glm_collinearity, glm_diagnostics, plot_glm_residuals
from .export import export_rating_table
from .groupings import add_groupings
from .impact import premium_impact
from .interactions import detect_interactions, plot_residual_heatmap
from .lift import double_lift, model_lift
from .model import GLMFit, fit_glm
from .pdp import make_pdp
from .plot_oneway import make_plot
from .qq_influence import plot_glm_influence, plot_glm_qq
from .rating_plot import make_rating_plot
from .rating_table import make_rating_table
from .report import pricing_report
from .screen import screen_features
from .splines import bs, ns, poly

__version__ = "0.1.0"

__all__ = [
    "fit_glm", "GLMFit", "ns", "bs", "poly",
    "agg_all", "make_plot", "make_pdp", "plot_glm_predictor",
    "glm_diagnostics", "glm_collinearity", "plot_glm_residuals",
    "plot_glm_qq", "plot_glm_influence",
    "detect_interactions", "plot_residual_heatmap", "screen_features",
    "make_rating_table", "add_groupings", "make_rating_plot",
    "model_lift", "double_lift", "premium_impact",
    "export_rating_table", "pricing_report",
    "ta_navy", "ta_blue", "ta_lightblue", "ta_gold", "ta_muted",
    "ta_years_base", "ta_year_palette",
    "PricingError", "PricingWarning", "PricingMessage", "Result",
]
