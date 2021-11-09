import itertools

import numpy as np
import pytest
from gen2_analysis.diffuse import AtmosphericNu
from gen2_analysis.effective_areas import effective_area
from gen2_analysis.externals.AtmosphericSelfVeto import AnalyticPassingFraction
from gen2_analysis.util import PDGCode


@pytest.fixture
def aeff():
    edges = [
        np.logspace(3, 6, 31),
        np.linspace(-1, 1, 5),
        np.logspace(3, 6, 31),
        np.array([0, np.pi]),
    ]
    aeff = np.zeros((6,) + tuple(len(bins) - 1 for bins in edges))
    # set diagnonal in e_nu / e_reco
    for i in range(aeff.shape[1]):
        aeff[:, i, :, i, :] = 1.0
    return effective_area(edges, aeff)


@pytest.mark.parametrize("veto_threshold", [None, 1e3])
def test_conventional(aeff, veto_threshold, snapshot):
    component = AtmosphericNu.conventional(
        aeff, livetime=1.0, veto_threshold=veto_threshold
    )
    assert snapshot == component.expectations


@pytest.mark.parametrize("veto_threshold", [None, 1e3])
def test_prompt(aeff, veto_threshold, snapshot):
    component = AtmosphericNu.prompt(aeff, livetime=1.0, veto_threshold=veto_threshold)
    assert snapshot == component.expectations


@pytest.mark.parametrize(
    "ptype",
    [
        getattr(PDGCode, "Nu" + flavor + anti)
        for flavor, anti in itertools.product(("E", "Mu"), ("", "Bar"))
    ],
)
@pytest.mark.parametrize("kind", ["conventional", "charm"])
@pytest.mark.parametrize("veto_threshold", [1e3])
def test_veto_probability(ptype, kind, veto_threshold, snapshot):
    passing_fraction = AnalyticPassingFraction(kind=kind, veto_threshold=veto_threshold)
    energy = np.logspace(3, 6, 5)
    cos_theta = np.linspace(0.2, 1, 5)
    # round result to account for architecture-specific precision in photospline
    assert snapshot == np.round(
        passing_fraction(ptype, energy[:, None], cos_theta[None, :], depth=2e3), 5
    )
