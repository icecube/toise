
import numpy as np
from functools import partial, wraps
import photospline
from . import taudecay
from gen2_analysis.util import data_dir
import os
import inspect
import itertools
from argparse import Namespace
import cPickle as pickle
import hashlib

TOTAL_XSEC_BIAS = 80
DPDX_BIAS = 10
ENU_GRID = np.logspace(1,15,14*2 + 1)
X_GRID = np.linspace(0,1,6*30 + 1)

def _get_cache_dir():
    cache_dir = os.path.join(data_dir, 'cross_sections', 'cteq08')
    if not os.path.isdir(cache_dir):
        os.mkdir(cache_dir)
    return cache_dir

def _stale(target, dependencies):
    try:
        mtime = os.stat(target).st_mtime
    except OSError:
        return True
    for dep in dependencies:
        if os.stat(dep).st_mtime > mtime:
            return True
    return False

def cached_spline(f, dependencies):
    """
    cache the results of photospline.SplineTable-returning function in a file,
    recalculating whenever the source module is updated
    """
    spec = inspect.getargspec(f)
    types = ['nu{}{}'.format(*args) for args in itertools.product(['e', 'mu', 'tau'], ['', 'bar'])]
    name = f.__name__[4:]
    @wraps(f)
    def wrapper(*args, **kwargs):
        label = [
            name,
            types[args[spec.args.index('nutype')]-1],
            args[spec.args.index('channel')],
            args[spec.args.index('target')],
            'fits'
        ]
        if 'final_nutype' in spec.args:
            label.insert(-2, types[args[spec.args.index('final_nutype')]-1])
        cache = os.path.join(_get_cache_dir(), '.'.join(label))
        if _stale(cache, dependencies):
            print('{} is stale'.format(cache))
            spline = f(*args, **kwargs)
            spline.write(cache)
        else:
            spline = photospline.SplineTable(cache)
        return spline
    return wrapper

def pad_knots(knots, order=2):
    """
    Pad knots out for full support at the boundaries
    """
    pre = knots[0] - (knots[1]-knots[0])*np.arange(order, 0, -1)
    post = knots[-1] + (knots[-1]-knots[-2])*np.arange(1, order+1)
    return np.concatenate((pre, knots, post))

@partial(cached_spline, dependencies=[__file__])
def fit_total(nutype, target, channel, smoothness=1e-12):
    import nusigma
    nucross = np.vectorize(partial(nusigma.nucross, nu=nutype, targ=target, int_bn=channel, how=2))
    x = np.log(ENU_GRID)
    knots = pad_knots(x)
    y = np.log(nucross(ENU_GRID)) + TOTAL_XSEC_BIAS
    z, w = photospline.ndsparse.from_data(y, np.ones(y.shape))
    return photospline.glam_fit(z,w,[x],[knots],[2],[smoothness],[2])

@partial(cached_spline, dependencies=[__file__])
def fit_differential(nutype, target, channel):
    """
    Fit log(dp/dx) as a function of log(enu) and x
    """
    import nusigma
    total_spline = fit_total(nutype, target, channel)
    nucrossdiff = np.vectorize(partial(nusigma.nucrossdiffl, nu=nutype, targ=target, int_bn=channel))
    centers = np.log(ENU_GRID), X_GRID
    nknots = [20,50]
    dpdx = np.asarray([nucrossdiff(e, e*X_GRID)*e for e in ENU_GRID])
    log_dpdx = np.log(dpdx/np.exp(total_spline([np.log(ENU_GRID)[...,None]]) - TOTAL_XSEC_BIAS)) + DPDX_BIAS
    knots = map(pad_knots, map(lambda (c,n): np.linspace(min(c), max(c), n), zip(centers,nknots)))
    z, w = photospline.ndsparse.from_data(log_dpdx, np.where(np.isfinite(log_dpdx), 1, 0))
    return photospline.glam_fit(z,w,centers,knots,[2]*2,[1e-16,1e-16],[2]*2)

@partial(cached_spline, dependencies=[__file__, taudecay.__file__])
def fit_differential_secondary(nutype, final_nutype, target, channel, nknots=[20,50], smoothness=[1e-16,1e-16]):
    """
    Fit log(dp/dx) as a function of log(enu) and x
    
    :param total_spline: total CC cross-section spline, as returned by :py:func:`fit_total`
    :param partial_spline: differential CC cross-section spline, as returned by :py:func:`fit_partial`
    :param enu: parent neutrino energies in GeV
    :param x: ratio of final to parent neutrino energy
    :param initial_nutype: type of parent neutrino
    :param final_nutype: type of final neutrino
    :param nknots: number of knots in (enu,x)
    :param smoothness: regularization strength in (enu,x)
    """
    # polarization of V-A tau- production is -1
    polarization = 1 if nutype == 6 else -1
    assert nutype in set(range(1,7))
    assert final_nutype in set(range(1,7))
    assert nutype % 2 == final_nutype % 2, "no lepton-number-changing interactions"
    if final_nutype == nutype:
        crossdiff = taudecay.tau_regen_crossdiff
    else:
        crossdiff = taudecay.tau_secondary_crossdiff
    xsec = DISCrossSection.create(nutype, target, channel)
    centers = np.log(ENU_GRID), X_GRID
    nknots = [20,50]
    dpdx = np.asarray([crossdiff(xsec.differential, e, e*X_GRID, polarization)*e for e in ENU_GRID])/xsec.total(ENU_GRID[...,None])
    log_dpdx = np.log(dpdx) + DPDX_BIAS
    knots = map(pad_knots, map(lambda (c,n): np.linspace(min(c), max(c), n), zip(centers,nknots)))
    z, w = photospline.ndsparse.from_data(log_dpdx, np.where(np.isfinite(log_dpdx), 1, 0))
    return photospline.glam_fit(z,w,centers,knots,[2]*2,[1e-16,1e-16],[2]*2)

class DISCrossSection(object):
    def __init__(self, total_spline, differential_spline):
        self.sigma = total_spline
        self.dpdx = differential_spline
    @classmethod
    def create(cls, nutype, target, channel):
        """
        Create a parameterization of the neutrino-nucleon interaction cross-section
        """
        return cls(
            fit_total(nutype, target, channel),
            fit_differential(nutype, target, channel)
        )
    @classmethod
    def create_secondary(cls, nutype, final_nutype, target, channel):
        """
        Create a parameterization of the tau neutrino-nucleon interaction cross-section
        """
        return cls(
            fit_total(nutype, target, channel),
            fit_differential_secondary(nutype, final_nutype, target, channel)
        )
    def total(self, enu):
        """
        Total cross-section
        
        :param enu: neutrino energy in GeV
        :returns: cross-section in cm^2
        """
        return np.exp(self.sigma([np.log(enu)]) - TOTAL_XSEC_BIAS)
    def differential(self, enu, ef):
        """
        Differential cross-section
        
        :param enu: incoming neutrino energy in GeV
        :param ef: outcoming lepton energy in GeV
        :returns: d\sigma/dE_f in cm^2 GeV^-1
        """
        # dP/dx, normalized to 1
        dpdx = np.where(ef >= enu, 0, np.exp(self.dpdx([np.log(enu), ef/enu]) - DPDX_BIAS))
        # convert to dsigma/dE_f (cm^2 GeV^-1)
        return dpdx/enu*self.total(enu)

class GlashowResonanceCrossSection(object):
    """
    nuebar e- -> W scattering
    """
    RWmu   =  0.1057        # Braching ratio W -> mu + nu_mu
    @classmethod
    def total(cls, energy):
        """
        implementation copied from NeutrinoGenerator
        
        see e.g. arxiv:1108.3163v2, Eq. 2.1
        """
        GF2       =  1.3604656E-10 # G_F*G_F in GeV^{-4}
        M_e       =  5.1099891E-4  # electron mass in GeV
        MW2       =  6467.858929   # M_W*M_W  in GeV^2
        GeV2_MBARN = 0.3893796623  # GeV^2*mbarn conversion coefficient
        GW2       =  6.935717E-4   # (Full width of W boson / mass W boson)^2
        crs0 = GF2*MW2/np.pi*GeV2_MBARN; # Standard cross-section in mbarn
        SW = 2*M_e*energy/MW2;
        # The total width is devided by the lepton (here muon) one 1/RWmu
        # branching ratios RWe, RWmu, RWtau
        sigma = crs0*SW/( (1 - SW)*(1 - SW) + GW2)/cls.RWmu/3; 
        # conversion : mb to cm2
        return sigma * 1e-27
    @classmethod
    def differential(cls, enu, ef):
        """
        differential cross-section for nuebar e- -> nubar l-
        NB: factor of 3 comes from integration over xl**2, not 3 families of leptons
        """
        xl = ef/enu
        return np.where(xl < 1, cls.total(enu)*cls.RWmu*3*xl**2/enu, 0)