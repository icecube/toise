import pytest
import numpy as np
import os
import json
from gen2_analysis.effective_areas import effective_area
from gen2_analysis import factory
from gen2_analysis.diffuse import AtmosphericNu

@pytest.fixture
def aeff():
    edges = [np.logspace(3,6,31),np.linspace(-1,1,5),np.logspace(3,6,31),np.array([0,np.pi])]
    aeff = np.zeros((6,) + tuple(len(bins)-1 for bins in edges))
    # set diagnonal in e_nu / e_reco
    for i in range(aeff.shape[1]):
        aeff[:,i,:,i,:] = 1.
    return effective_area(edges, aeff)

def test_conventional(aeff, snapshot):
    component = AtmosphericNu.conventional(aeff, livetime=1., veto_threshold=None)
    assert snapshot == component.expectations
