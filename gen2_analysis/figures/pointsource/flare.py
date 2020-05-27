
from gen2_analysis.figures import figure_data, figure

from gen2_analysis import diffuse, multillh, plotting, surface_veto, pointsource, factory, figures_of_merit, util
from gen2_analysis.cache import ecached, lru_cache

from collections import OrderedDict, defaultdict
from scipy import stats, optimize
from copy import copy
import numpy as np
import re
from tqdm import tqdm
from functools import partial
from StringIO import StringIO

@figure_data()
def sensitivity(exposures, decades=1, gamma=-2, emin=0.):
    figures = OrderedDict([
        ('differential_upper_limit', figures_of_merit.DIFF.ul),
        ('differential_discovery_potential', figures_of_merit.DIFF.dp),
        ('upper_limit', figures_of_merit.TOT.ul),
        ('discovery_potential', figures_of_merit.TOT.dp),
    ])
    assert len({exposure for detector, exposure in exposures}) == 1, "exposures are equal"
    meta = {'cos_zenith': factory.default_cos_theta_bins}
    for detector, exposure in exposures:
        dlabel = re.split('[_-]', detector)[0]
        for zi in tqdm(range(20), desc=dlabel):
            fom = figures_of_merit.PointSource({detector: exposure}, zi)
            for flabel, q in figures.items():
                kwargs = {'gamma': gamma, 'decades': decades}
                if not flabel.startswith('differential'):
                    kwargs['emin'] = emin
                val = fom.benchmark(q, **kwargs)
                if not flabel in meta:
                    meta[flabel] = {}
                if not dlabel in meta[flabel]:
                    meta[flabel][dlabel] = {}
                if flabel.startswith('differential'):
                    val = OrderedDict(zip(('e_center', 'flux', 'n_signal', 'n_background'), val))
                else:
                    val = OrderedDict(zip(('flux', 'n_signal', 'n_background'), val))
                meta[flabel][dlabel][str(zi)] = val
    return meta

@figure
def sensitivity(datasets):
    import matplotlib.pyplot as plt
    fig = plt.gcf()
    ax = plt.gca()
    for dataset in datasets:
        cos_theta = np.asarray(dataset['data']['cos_zenith'])
        xc = -util.center(cos_theta)
        for detector in dataset['data']['discovery_potential'].keys():
            yc = np.full(xc.shape, np.nan)
            for k in dataset['data']['discovery_potential'][detector].keys():
                # items are flux, ns, nb
                yc[int(k)] = dataset['data']['discovery_potential'][detector][k]['flux']
            ax.semilogy(xc, yc*1e-12, label=detector)
    ax.set_xlabel(r'$\sin \delta$')
    ax.set_ylabel(r'$E^2 \Phi_{\nu_x + \overline{\nu_x}}$ $(\rm TeV \,\, cm^{-2} s^{-1})$')
