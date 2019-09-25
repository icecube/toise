
from gen2_analysis.figures import figure_data, figure

from gen2_analysis import diffuse, multillh, plotting, surface_veto, pointsource, factory
from gen2_analysis.cache import ecached, lru_cache

from scipy import stats, optimize
from copy import copy
import numpy as np
from tqdm import tqdm
from functools import partial


def make_components(aeffs, emin=1e2, emax=1e11):
    # zero out effective area beyond active range
    aeff, muon_aeff = copy(aeffs)
    ebins = aeff.bin_edges[0]
    mask = (ebins[1:] <= emax) & (ebins[:-1] >= emin)
    aeff.values *= mask[[None] + [slice(None)] + [None]*(aeff.values.ndim-2)]

    atmo = diffuse.AtmosphericNu.conventional(
        aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None)
    atmo.prior = lambda v, **kwargs: -(v-1)**2/(2*0.1**2)
    prompt = diffuse.AtmosphericNu.prompt(
        aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None)
    prompt.min = 0.5
    prompt.max = 3.
    astro = diffuse.DiffuseAstro(aeff, 1)
    astro.seed = 2.
    components = dict(atmo=atmo, prompt=prompt, astro=astro)
    if muon_aeff:
        components['muon'] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    return components


def subdivide(bundle, key='astro', scales=(1.,), gammas=(-2.3,), decades=1, emin=1e2, emax=1e11):
    import toolz
    bundle = copy(bundle)
    bundle.components = copy(bundle.components)
    for k in bundle.components.keys():
        components = copy(bundle.components[k])
        astro = components.pop(key)
        def downsample(f): return toolz.take_nth(1, f)
        for i, (e_center, chunk) in toolz.pipe(astro.differential_chunks(decades, exclusive=True, emin=emin, emax=emax), enumerate):
            if not chunk:
                continue
            chunk.seed = sum([scale*chunk.seed*(e_center/1e5)**(gamma+2)
                              for scale, gamma in zip(scales, gammas)])
            components['{}_{:02d}'.format(key, i)] = chunk
        bundle.components[k] = components
    return bundle


@lru_cache()
def asimov_llh(bundle, astros=(1,), gammas=(-2.,), decades=1./3, emin=1e2, emax=1e11):
    components = bundle.get_components()
    components['gamma'] = multillh.NuisanceParam(gammas[0])
    scales = [astro / components['astro'].seed for astro in astros]
    llh = multillh.asimov_llh(components, astro=astros[0])
    astroc = llh.components['astro']
    for norm, gamma in zip(astros[1:], gammas[1:]):
        dats = astroc.expectations(gamma=gamma)
        for k, v in dats.items():
            llh.data[k] += norm*v

    chunk_llh = multillh.LLHEval(llh.data)
    chunk_llh.components['gamma'] = components['gamma']
    chunk_llh.components.update(subdivide(
        bundle, decades=decades, gammas=gammas, scales=scales, emin=emin, emax=emax).get_components())

    return chunk_llh


def find_limits(llh, key, nom=None, critical_ts=1**2, plotit=False):
    if nom is None:
        nom = {k: v.seed for k, v in llh.components.items()}
    base = llh.llh(**nom)

    def ts_diff(value):
        fixed = dict(nom)
        fixed[key] = value
        alt = fixed
        ts = -2*(llh.llh(**alt) - base) - critical_ts
        return ts
    g0 = nom[key]
    if ts_diff(0) < critical_ts:
        lo = 0
    else:
        lo = optimize.bisect(ts_diff, 0, g0, xtol=5e-3, rtol=1e-4)

    try:
        hi = optimize.bisect(ts_diff, g0, 1e3*g0, xtol=5e-3, rtol=1e-4)
    except ValueError:
        hi = numpy.inf

    energy_range = llh.components[key]._components.values()[0][0].energy_range
    if plotit and energy_range[0] > 1e6:
        x = linspace(0, g0*2, 101)
        energy_range = llh.components[key]._components.values()[
            0][0].energy_range
        line = plot(x, [ts_diff(x_) for x_ in x],
                    label='%.1g-%.1g' % tuple(energy_range))[0]
        axvline(nom[key], color=line.get_color())

    return lo, hi


@lru_cache()
def unfold_llh(chunk_llh):
    import toolz
    fixed = dict(gamma=-2, prompt=1, atmo=1, muon=1)
    fit = chunk_llh.fit(minimizer_params=dict(epsilon=1e-2), **fixed)
    is_astro = partial(filter, lambda s: s.startswith('astro'))
    keys = toolz.pipe(chunk_llh.components.keys(), is_astro, sorted)[4:]
    xlimits = np.array([chunk_llh.components[k]._components.values()[
                       0][0].energy_range for k in keys])
    xcenters = 10**(np.log10(xlimits).sum(axis=1)/2.)
    ylimits = np.array([find_limits(chunk_llh, key, nom=fit)
                        for key in tqdm(keys)])
    ycenters = np.array([chunk_llh.components[k].seed for k in keys])
    ycenters = np.array([fit[k] for k in keys])

    return xlimits, ycenters, ylimits


@ecached(__name__+'.unfolding.{exposures}.{astro}.{gamma}.emax{emax}.{gzk}x{gzk_norm}', timeout=2*24*3600)
def unfold_bundle(exposures, astro, gamma, gzk='vanvliet', gzk_norm=1, emax=1e9):
    bundle = factory.component_bundle(dict(exposures), make_components)
    return unfold_llh(asimov_llh(bundle, (0.5, 1.), (-2, -2.5)))


def psi_binning():
    factory.set_kwargs(psi_bins={k: (0, np.pi)
                                 for k in ('tracks', 'cascades', 'radio')})


@figure_data(setup=psi_binning)
def unfold(exposures, astro=2.3, gamma=-2.5, gzk='vanvliet', gzk_norm=1, emax=1e9, clean=False):
    """
    Calculate exclusion confidence levels for alternate flavor ratios, assuming
    Wilks theorem.

    :param astro: per-flavor astrophysical normalization at 100 TeV, in 1e-18 GeV^-2 cm^-2 sr^-1 s^-1
    :param gamma: astrophysical spectral index
    :param steps: number of steps along one axis of flavor scan
    :param gamma_step: granularity of optimization in spectral index. if 0, the spectral index is fixed.
    """
    if clean:
        unfold_bundle.invalidate_cache_by_key(
            exposures, astro, gamma, gzk, gzk_norm, emax)
    xlimits, ycenters, ylimits = unfold_bundle(
        exposures, astro, gamma, gzk, gzk_norm, emax)

    return {'xlimits': xlimits.tolist(), 'ylimits': ylimits.tolist(), 'ycenters': ycenters.tolist()}


@figure
def flux_error(datasets):
    import matplotlib.pyplot as plt

    ax = plt.gca()

    for dataset in datasets:
        xlimits = np.array(dataset['data']['xlimits'])
        ylimits = np.array(dataset['data']['ylimits'])
        yvalues = np.array(dataset['data']['ycenters'])
        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5*(edges[1:]+edges[:-1])
        ydiff = 0.5*(ylimits[:, 1]-ylimits[:, 0])
        ax.plot(xvalues, 100*ydiff/yvalues,
                label=r'$\Delta\Phi_\nu/\Phi_\nu$', linestyle='-', marker='.')
    ax.loglog()
    ax.set_xlim((1e4, 2e9))
    ax.set_xlabel('Neutrino energy [GeV]')
    ax.set_ylabel('Relative uncertainty [%]')
    ax.legend()
    plt.tight_layout(0.1)
    return ax.figure


@figure
def unfolded_flux(datasets):
    import matplotlib.pyplot as plt

    ax = plt.gca()
    plot_kwargs = dict(linestyle='None', marker='o')
    for dataset in datasets:
        xlimits = np.array(dataset['data']['xlimits'])
        ylimits = np.array(dataset['data']['ylimits'])
        yvalues = np.array(dataset['data']['ycenters'])
        # factor 3 for all-flavor convention!
        unit = 3e-8
        ylimits *= unit
        yvalues *= unit

        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5*(edges[1:]+edges[:-1])
        x, y = xvalues, yvalues
        yerr = abs(ylimits-yvalues[:, None]).T
        art = ax.errorbar(x, y,
                          xerr=None,
                          yerr=yerr,
                          # uplims=m[mask],
                          **plot_kwargs)

    ax.loglog(nonposy='clip')
    ax.set_xlim((1e4, 2e9))
    ax.set_ylim((2e-9, 1e-6))
    ax.set_xlabel('Neutrino energy [GeV]')
    ax.set_ylabel('Relative uncertainty [%]')
    ax.legend()
    plt.tight_layout(0.1)
    return ax.figure
