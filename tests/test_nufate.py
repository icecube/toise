from toise.externals import nuFATE
from toise.util import center
import numpy as np


def test_cache():
    """Cross-sections can be instantiated from cache"""
    enu = np.logspace(4, 12, 81)
    ct = np.linspace(-1, 1, 21)
    nodes = np.exp(center(np.log(enu)))

    cascade = nuFATE.NeutrinoCascade(nodes)

    tt = cascade.transfer_matrix(center(ct), depth=1.5)
    assert tt.shape == (6, 6, 20, 80, 80)
