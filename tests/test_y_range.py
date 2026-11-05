"""y_range is available on every plotting function, unused by default: the
port of tests/testthat/test-y-range.R (the premium_impact test lives with
premium_impact)."""
import pytest

from pricingtools_rmo._core import PricingError
from pricingtools_rmo.ae import plot_glm_predictor
from pricingtools_rmo.colors import ta_blue
from pricingtools_rmo.pdp import make_pdp
from pricingtools_rmo.plot_oneway import make_plot
from pricingtools_rmo.rating_plot import make_rating_plot


def _residuals():
    return pytest.importorskip("pricingtools_rmo.diagnostics").plot_glm_residuals


def test_all_plot_functions_auto_scale_the_y_axis_by_default(dat, m_freq, tbl, tbl2):
    assert make_plot(dat, "REGIO", "Frequency", ta_blue, "F", "color",
                     False).layout.yaxis.autorange is True
    assert make_pdp(m_freq, dat, "REGIO").layout.yaxis.autorange is True
    assert make_rating_plot(tbl, "REGIO").layout.yaxis.autorange is True
    assert make_rating_plot(tbl2, "LEEFTIJD:REGIO").layout.yaxis.autorange is True
    assert _residuals()(m_freq).layout.yaxis.autorange is True
    assert plot_glm_predictor(m_freq, "REGIO").layout.yaxis.autorange is True


def test_all_plot_functions_honour_a_fixed_y_range(dat, m_freq, tbl, tbl2):
    rng = (0, 0.6)
    ps = [
        make_plot(dat, "REGIO", "Frequency", ta_blue, "F", "color", False,
                  y_range=rng),
        make_pdp(m_freq, dat, "REGIO", y_range=rng),
        make_rating_plot(tbl, "REGIO", y_range=rng),
        make_rating_plot(tbl2, "LEEFTIJD:REGIO", y_range=rng),
        _residuals()(m_freq, y_range=rng),
        plot_glm_predictor(m_freq, "REGIO", y_range=rng),
    ]
    for p in ps:
        assert list(p.layout.yaxis.range) == list(rng)
        assert p.layout.yaxis.autorange is not True


def test_make_plot_facet_mode_pins_all_facets_to_the_fixed_range(dat):
    p = make_plot(dat, "REGIO", "Frequency", ta_blue, "F", "facet", True,
                  y_range=(0, 0.6))
    lay = p.layout
    # one y axis per facet; each must span the range
    yaxes = [k for k in lay.to_plotly_json() if k.startswith("yaxis")]
    assert len(yaxes) > 1
    for k in yaxes:
        r = lay[k].range
        if r is not None and len(r) == 2:
            assert r[0] <= 0.01
            assert r[1] >= 0.59


def test_an_invalid_range_is_rejected_everywhere(dat, m_freq, tbl):
    with pytest.raises(PricingError, match="y_range"):
        make_plot(dat, "REGIO", "Frequency", ta_blue, "F", "color", False,
                  y_range=(1, 1))
    with pytest.raises(PricingError, match="y_range"):
        make_pdp(m_freq, dat, "REGIO", y_range=[3])
    with pytest.raises(PricingError, match="y_range"):
        make_rating_plot(tbl, "REGIO", y_range=(2, 1))
    with pytest.raises(PricingError, match="y_range"):
        _residuals()(m_freq, y_range=(float("nan"), 1))
