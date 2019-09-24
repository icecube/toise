
"""
Flavor ratio fits
"""

from functools import partial

import pandas as pd
import numpy as np
from tqdm import tqdm

from gen2_analysis.figures import figure_data, figure
from gen2_analysis.cache import ecached, lru_cache
from gen2_analysis import diffuse, multillh, plotting, surface_veto, factory, plotting

def make_components(aeffs):
    aeff, muon_aeff = aeffs
    atmo = diffuse.AtmosphericNu.conventional(aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None)
    atmo.prior = lambda v, **kwargs: -(v-1)**2/(2*0.1**2)
    prompt = diffuse.AtmosphericNu.prompt(aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None)
    prompt.min = 0.5
    prompt.max = 3.
    astro = diffuse.DiffuseAstro(aeff, 1)
    astro.seed = 2.
    components = dict(atmo=atmo, prompt=prompt, astro=astro)
    if muon_aeff:
        components['muon'] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    return components

def triangle_product(seq):
    for i in range(len(seq)):
        for j in range(len(seq)-i):
            yield seq[i], seq[j]

def expand(dataframe):
    e_frac = dataframe.index.levels[0].values
    mu_frac = dataframe.index.levels[1].values
    idx = pd.MultiIndex.from_product((e_frac, mu_frac), names=dataframe.index.names)
    return dataframe.reindex(idx)

@lru_cache(maxsize=4)
def create_bundle(exposures):
    return factory.component_bundle(dict(exposures), make_components)

@lru_cache(maxsize=4)
def asimov_llh(exposures, **nominal):
    components = create_bundle(exposures).get_components()
    gamma = nominal.pop('gamma', -2.5)
    components['gamma'] = multillh.NuisanceParam(gamma, 0.5, min=gamma-0.5, max=gamma+0.5)
    components['e_tau_ratio'] = multillh.NuisanceParam((0.93/3)/(1-1.05/3), None, 0, 1)
    components['mu_fraction'] = multillh.NuisanceParam(1.05/3, None, 0, 1)
    if not 'muon' in components:
        components['muon'] = multillh.NuisanceParam(1, None, 0, 1)
    return multillh.asimov_llh(components, **nominal)

@ecached(__name__+'.profile.{exposures}_{nominal}_{steps}_{gamma_step}', timeout=2*24*3600)
def make_profile(exposures, nominal=dict(), steps=100, minimizer_params=dict(epsilon=1e-2), gamma_step=None):
    llh = asimov_llh(exposures, **nominal)
    steps = np.linspace(0, 1, steps+1)
    params = []
    scan = ['mu_fraction', 'e_tau_ratio']
    free = ['astro']
    fixed = {k: llh.components[k].seed for k in llh.components if not k in scan+free}
    if gamma_step is not None:
        fixed['gamma'] = fixed['gamma'] + np.arange(-10, 11)*gamma_step
    for e, mu in tqdm(triangle_product(steps), total=(steps.size+1)*(steps.size)/2, desc='{} {}'.format(__name__,detector_label(exposures))):
        fixed['mu_fraction'] = mu
        fixed['e_tau_ratio'] = e/(1-mu) if mu != 1 else 1
        assert e+mu <= 1
        fit = llh.fit(minimizer_params, **fixed)
        fit['LLH'] = llh.llh(**fit)
        fit['e_fraction'] = e
        params.append(fit)
    return expand(pd.DataFrame(params).set_index(['e_fraction', 'mu_fraction']))

def extract_ts(dataframe):
    e_frac = dataframe.index.levels[0].values
    mu_frac = dataframe.index.levels[1].values
    maxllh = np.nanmax(dataframe['LLH'])
    idx = pd.MultiIndex.from_product((e_frac, mu_frac), names=dataframe.index.names)

    ts = 2*(pd.Series((np.nanmax(dataframe['LLH'].values)*np.ones(len(idx))), index=idx) - dataframe['LLH'])
    return e_frac, mu_frac, ts.values.reshape(e_frac.size, mu_frac.size)

def psi_binning():
    factory.set_kwargs(psi_bins={k: (0, np.pi) for k in ('tracks', 'cascades', 'radio')})

@figure_data(setup=psi_binning)
def confidence_levels(exposures, astro=2.3, gamma=-2.5, steps=100, gamma_step=0.):
    """
    Calculate exclusion confidence levels for alternate flavor ratios, assuming
    Wilks theorem.

    :param astro: per-flavor astrophysical normalization at 100 TeV, in 1e-18 GeV^-2 cm^-2 sr^-1 s^-1
    :param gamma: astrophysical spectral index
    :param steps: number of steps along one axis of flavor scan
    :param gamma_step: granularity of optimization in spectral index. if 0, the spectral index is fixed.
    """
    from scipy import stats
    profile = make_profile(exposures, steps=steps, nominal=dict(gamma=gamma, astro=astro))
    meta = {}
    efrac, mufrac, ts = extract_ts(profile)
    meta['nue_fraction'] = efrac.tolist()
    meta['numu_fraction'] = mufrac.tolist()
    meta['confidence_level'] = (stats.chi2.cdf(ts.T, 2)*100).tolist()
    return meta

def detector_label(exposures):
    """transform (('Gen2-InIce', 10.0), ('IceCube', 15.0)) -> Gen2-InIce+IceCube 10+15 yr"""
    return '{} {} yr'.format(*map('+'.join, map(partial(map, str), zip(*exposures))))

@figure
def triangle(datasets):
    """
    Plot exclusion contours in flavor composition
    """
    from gen2_analysis.externals import ternary
    import matplotlib.pyplot as plt

    ax = ternary.flavor_triangle(grid=True)
    labels = []
    for i, meta in enumerate(datasets):
        labels.append(detector_label(meta['detectors']))
        values = meta['data']
        cs = ax.ab.contour(values['nue_fraction'], values['numu_fraction'], values['confidence_level'], levels=[68,], colors='C{}'.format(i))
    ax.ab.legend(ax.ab.collections[-len(labels):], labels, bbox_to_anchor=(1.3,1.1))

    return ax.figure
