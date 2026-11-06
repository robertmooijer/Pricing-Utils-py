"""Errors, warnings, messages and the result container shared by the package.

R's three kinds of condition map onto Python as follows:

* ``stop()``    -> ``raise PricingError`` (a ``ValueError``)
* ``warning()`` -> ``warnings.warn(..., PricingWarning)``
* ``message()`` -> ``PricingMessage``, a warning category of its own that is
  shown once per call site like an R message, and can be silenced with
  ``warnings.filterwarnings("ignore", category=PricingMessage)``.
"""
from __future__ import annotations

import warnings


# [PY-09]
class PricingError(ValueError):
    """An input the package refuses rather than guess at (R's stop())."""


class PricingWarning(UserWarning):
    """A result that is still correct but needs reading with care."""


class PricingMessage(UserWarning):
    """Information, as R's message(): nothing is wrong."""


def stop(*parts):
    raise PricingError("".join(str(p) for p in parts))


def warn(*parts, stacklevel=3):
    warnings.warn("".join(str(p) for p in parts), PricingWarning,
                  stacklevel=stacklevel)


def message(*parts, stacklevel=3):
    warnings.warn("".join(str(p) for p in parts), PricingMessage,
                  stacklevel=stacklevel)


# [PY-06]
class Result(dict):
    """A dict whose items can also be read as attributes, so a result reads
    like R's list: ``imp.summary`` or ``imp["summary"]``."""

    def __getattribute__(self, name):
        # an item wins over a dict method of the same name (values, keys),
        # so r.values reads like R's r$values
        if not name.startswith("__") and dict.__contains__(self, name):
            return dict.__getitem__(self, name)
        return object.__getattribute__(self, name)

    def __getattr__(self, name):
        raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __dir__(self):
        return list(self.keys())


# [PY-06]
def attach(fig, name, value):
    """Attach a data table to a plotly figure, as R's attr(p, name). Read it
    back as ``fig.<name>``."""
    object.__setattr__(fig, name, value)
    return fig


def set_config(fig, filename):
    """The modebar settings R attaches with plotly::config(). Python plotly
    passes config at render time, so it is kept on the figure as
    ``fig.pricing_config`` and used by fig.show / the report:
    ``fig.show(config=fig.pricing_config)``."""
    cfg = {"displayModeBar": True,
           "modeBarButtonsToRemove": ["lasso2d", "select2d"],
           "toImageButtonOptions": {"format": "png", "filename": filename}}
    object.__setattr__(fig, "pricing_config", cfg)
    return fig
