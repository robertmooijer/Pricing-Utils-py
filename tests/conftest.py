"""Shared simulated portfolio and fitted models for all tests (the port of
tests/testthat/setup.R). The data are drawn with numpy's generator, so the
numbers differ from the R suite's; every test states a property, not a
number copied from R."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from pricingtools_rmo.model import fit_glm


def make_dat(n=20000, seed=42):
    rng = np.random.default_rng(seed)
    dat = pd.DataFrame({
        "LEEFTIJD": np.round(rng.uniform(18, 80, n)),
        "REGIO": pd.Categorical(rng.choice(["Noord", "Zuid", "West"], n,
                                           p=[.5, .3, .2]),
                                categories=["Noord", "West", "Zuid"]),
        "BOEKJAAR": rng.integers(2019, 2024, n),
        "Exposure": rng.uniform(0.1, 1, n),
    })
    lin = (-2.3 + 0.015 * (dat.LEEFTIJD - 40)
           + np.where(dat.REGIO == "Zuid", .25,
                      np.where(dat.REGIO == "West", .4, 0)))
    dat["AantalClaims"] = rng.poisson(dat.Exposure * np.exp(lin))
    sev = rng.gamma(2, (1500 * np.exp(0.01 * (dat.LEEFTIJD - 40))) / 2)
    dat["SCHADELAST"] = np.where(dat.AantalClaims > 0, sev * dat.AantalClaims, 0.0)
    return dat


@pytest.fixture(scope="session")
def dat():
    return make_dat()


@pytest.fixture(scope="session")
def m_freq(dat):
    return fit_glm("AantalClaims ~ ns(LEEFTIJD, 3) + REGIO + factor(BOEKJAAR)"
                   " + offset(log(Exposure))", dat, family="poisson")


@pytest.fixture(scope="session")
def d_sev(dat):
    d = dat[dat.AantalClaims > 0].copy()
    d["AvgLoss"] = d.SCHADELAST / d.AantalClaims
    return d


@pytest.fixture(scope="session")
def m_sev(d_sev):
    return fit_glm("AvgLoss ~ LEEFTIJD + REGIO", d_sev, family="gamma",
                   weights="AantalClaims")


@pytest.fixture(scope="session")
def m_freq2(dat):
    return fit_glm("AantalClaims ~ LEEFTIJD * REGIO + offset(log(Exposure))",
                   dat, family="poisson")


@pytest.fixture(scope="session")
def tbl(m_freq, m_sev, dat):
    from pricingtools_rmo.rating_table import make_rating_table
    return make_rating_table(m_freq, m_sev, data=dat)


@pytest.fixture(scope="session")
def tbl2(m_freq2, dat):
    from pricingtools_rmo.rating_table import make_rating_table
    return make_rating_table(m_freq2, None, data=dat)


def trace_y(fig, name):
    """y values of the named trace of a plotly figure."""
    for t in fig.data:
        if t.name == name:
            return np.asarray(t.y, dtype=float)
    raise KeyError(name)


def trace(fig, name):
    for t in fig.data:
        if t.name == name:
            return t
    raise KeyError(name)
