"""pricingtools_rmo: GLM pricing tools for non-life insurance."""
from .model import fit_glm, GLMFit
from .splines import ns, bs, poly
from ._core import PricingError, PricingWarning, PricingMessage, Result
from .colors import (ta_navy, ta_blue, ta_lightblue, ta_gold, ta_muted,
                     ta_years_base, ta_year_palette)

__version__ = "0.1.0"
