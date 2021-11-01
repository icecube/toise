
from gen2_analysis.figures import figure_data, figure

from gen2_analysis import diffuse, multillh, plotting, surface_veto, pointsource, factory
from gen2_analysis.cache import ecached, lru_cache

from scipy import stats, optimize
from copy import copy
import numpy as np
from tqdm import tqdm
from functools import partial
from StringIO import StringIO

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
    composite = diffuse.ArbitraryFlux(aeff, 1)
    composite.seed = 0
    components = dict(atmo=atmo, prompt=prompt, astro=astro, composite=composite)
    if muon_aeff:
        components['muon'] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    return components


def powerlaw(energy, gamma=-2., emax=np.inf):
    return 1e-18*np.exp(-energy/emax)*(energy/1e5)**gamma


def subdivide(bundle, seed_flux, key='astro', decades=1, emin=1e2, emax=1e11):
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
            # NB: seed flux is all-flavor, we want per-flavor
            chunk.seed = seed_flux(e_center)/powerlaw(e_center)/3.
            components['{}_{:02d}'.format(key, i)] = chunk
        bundle.components[k] = components
    return bundle


@lru_cache()
def asimov_llh(bundle, seed_flux, decades=1./3, emin=1e2, emax=1e11):
    components = bundle.get_components()
    comp = components['composite']
    for c, _ in comp._components.values():
        c.set_flux_func(seed_flux)
    llh = multillh.asimov_llh(components, astro=0, composite=1)

    chunk_llh = multillh.LLHEval(llh.data)
    chunk_llh.components['gamma'] = multillh.NuisanceParam(-2)
    chunk_llh.components.update(subdivide(
        bundle, seed_flux=seed_flux, decades=decades, emin=emin, emax=emax).get_components())

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
        hi = np.inf

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
    assert gzk == 'vanvliet'
    def seed_flux(energy):
        # NB: ArbitraryFlux assumes all-flavor flux, and divides by 6 to get per-particle
        return 3*(astro*powerlaw(energy, gamma, emax) + gzk_norm*diffuse.VanVlietGZKFlux()(energy))
    result = unfold_llh(asimov_llh(bundle, seed_flux))
    for k, v in diffuse.DiffuseAstro.__dict__.items():
        if hasattr(v, "cache_info"):
            print(k, v.cache_info())
    return result


def psi_binning():
    factory.set_kwargs(psi_bins={k: (0, np.pi)
                                 for k in ('tracks', 'cascades', 'radio')})


@figure_data(setup=psi_binning)
def unfold(exposures, astro=2.3, gamma=-2.5, gzk='vanvliet', gzk_norm=1.0, emax=1e9, clean=False):
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

def get_ic_contour(source='lars'):
    # 99% confidence interval
    if source == 'lars':
        return np.loadtxt(StringIO("""2.164	3.154
                                      2.248	2.749
                                      2.374	2.917
                                      2.572	3.505
                                      2.712	4.620
                                      2.765	6.259
                                      2.748	7.752
                                      2.679	9.366
                                      2.565	10.358
                                      2.419	9.726
                                      2.329	8.413
                                      2.268	6.890
                                      2.217	5.494
                                      2.169	4.274
                                      """))
    elif source == 'aachen':
        dat = np.loadtxt(StringIO("""1.716	0.225
                                     1.751	0.199
                                     1.792	0.203
                                     1.836	0.211
                                     1.897	0.231
                                     1.982	0.266
                                     2.072	0.322
                                     2.158	0.386
                                     2.249	0.467
                                     2.342	0.590
                                     2.396	0.673
                                     2.454	0.804
                                     2.497	0.949
                                     2.513	1.047
                                     2.520	1.131
                                     2.522	1.267
                                     2.518	1.376
                                     2.514	1.479
                                     2.494	1.584
                                     2.465	1.712
                                     2.416	1.818
                                     2.357	1.875
                                     2.294	1.850
                                     2.245	1.798
                                     2.185	1.686
                                     2.128	1.546
                                     2.092	1.445
                                     2.055	1.339
                                     2.023	1.239
                                     1.995	1.150
                                     1.961	1.034
                                     1.921	0.905
                                     1.887	0.801
                                     1.849	0.685
                                     1.743	0.370
                                     1.715	0.291
                                     """))
        # factor of 3 for all-flavor
        dat[:,1] *= 3
        return dat
        
def ic_butterfly(energy, source='lars'):
    ic_contour = get_ic_contour(source)
    flux = energy**2*1e-18*(ic_contour[:,1][:,None]*(energy[None,:]/1e5)**(-ic_contour[:,0][:,None]))
    return flux.min(axis=0), flux.max(axis=0)

@figure
def unfolded_flux(datasets, label='Gen2-InIce+Radio'):
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer

    fig = plt.figure(figsize=(3.5,3))
    ax = plt.gca()

    # plot underlying fluxes
    x = np.logspace(4,10,51)
    # ApJ 2015 best fit
    ax.plot(x, 3*2.3e-8*(x/1e5)**(-2.5+2), ls='--', color='lightgrey', lw=0.75, zorder=0)
    # + hard component (soft component norm reduced to give similar flux at low energy)
    ax.plot(x, 3*(1e-8*(x/1e5)**(-2.5+2) + 0.5e-8*(x/1e5)**(-2+2)), ls=':', color='k', lw=0.75)

    plot_kwargs = dict(linestyle='None', marker='o', markersize=3)
    for dataset in datasets:
        xlimits = np.array(dataset['data']['xlimits'])
        ylimits = np.array(dataset['data']['ylimits'])
        yvalues = np.array(dataset['data']['ycenters'])
        # factor 3 for all-flavor convention!
        unit = 3e-8
        ylimits *= unit
        yvalues *= unit

        years = next(years for detector, years in dataset['detectors'] if detector != 'IceCube')

        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5*(edges[1:]+edges[:-1])
        x, y = xvalues, yvalues
        yerr = abs(ylimits-yvalues[:, None]).T
        art = ax.errorbar(x, y,
                          xerr=None,
                          yerr=yerr,
                          # uplims=m[mask],
                          label='{label} ({years:.0f} years)'.format(label=label, years=years),
                          **plot_kwargs)

    x = np.logspace(np.log10(25e3), np.log10(2.8e6), 101)
    fill = ax.fill_between(x, *ic_butterfly(x, 'lars'), facecolor='lightgrey', edgecolor='None', label='IceCube (ApJ 2015)')
    x = np.logspace(np.log10(1.94e5), np.log10(7.8e6), 101)
    fill = ax.fill_between(x, *ic_butterfly(x, 'aachen'), facecolor='None', edgecolor='#566573', label='IceCube (tracks only, ApJ 2016)')

    ax.loglog(nonposy='clip')
    ax.set_xlim((1e4, 2e9))
    ax.set_ylim((2e-9, 1e-6))
    ax.set_xlabel('Neutrino energy (GeV)')
    ax.set_ylabel(r'$E^2 \Phi_{\nu} \,\, ({\rm GeV \, cm^{-2} \, sr^{-1} \, s^{-1}})$')

    # sort such that error bars go first
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(handles)), key=lambda i: not isinstance(handles[i], ErrorbarContainer))
    ax.legend([handles[i] for i in order], [labels[i] for i in order], fontsize=8, frameon=False)

    plt.tight_layout(0.1)
    return ax.figure

def plot_crs(ax):
    import plot_flux_results_for_jvs as lars_plot
    def apply_style(kwargs):
        style = dict(linestyle='None', markersize=4)
        style.update(kwargs)
        return style
    ax.errorbar(label='Diffuse $\gamma$ (Fermi LAT)', marker='D', color='C1', **apply_style(lars_plot.get_fermi_igrb_2014()))
    ax.errorbar(label='Cosmic rays (Auger)', marker='o', color='C4', **apply_style(lars_plot.get_auger()))
    ax.errorbar(label='Cosmic rays (TA)', marker='s', color='C2', **apply_style(lars_plot.get_ta()))

@figure
def unfolded_flux_multimessenger(datasets, label='Gen2-InIce+Radio'):
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer

    fig = plt.figure(figsize=(7,3.5))
    ax = plt.gca()

    plot_crs(ax)
    
    assert datasets[0]['source'] == 'gen2_analysis.figures.diffuse.spectrum.unfold'
    args = datasets[0]['args']
    # plot underlying fluxes
    x = np.logspace(4,10,51)
    pl = x**2*args['astro']*powerlaw(x, args['gamma'], args['emax'])
    assert args['gzk'] == 'vanvliet'
    gzk = args['gzk_norm']*diffuse.VanVlietGZKFlux()(x)*x**2
    ax.plot(x, 3*pl, ls=':', color='grey')
    ax.plot(x, 3*gzk, ls=':', color='grey')

    plot_kwargs = dict(linestyle='None', marker='o', markersize=3)
    for dataset in datasets:
        xlimits = np.array(dataset['data']['xlimits'])
        ylimits = np.array(dataset['data']['ylimits'])
        yvalues = np.array(dataset['data']['ycenters'])
        # factor 3 for all-flavor convention!
        unit = 3e-8
        ylimits *= unit
        yvalues *= unit

        years = next(years for detector, years in dataset['detectors'] if detector != 'IceCube')

        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5*(edges[1:]+edges[:-1])
        x, y = xvalues, yvalues
        yerr = abs(ylimits-yvalues[:, None]).T
        art = ax.errorbar(x, y,
                          xerr=None,
                          yerr=yerr,
                          # uplims=m[mask],
                          label='{label} ({years:.0f} years)'.format(label=label, years=years),
                          **plot_kwargs)

    x = np.logspace(np.log10(25e3), np.log10(2.8e6), 101)
    fill = ax.fill_between(x, *ic_butterfly(x, 'lars'), facecolor='lightgrey', edgecolor='None', label='IceCube (ApJ 2015)')
    x = np.logspace(np.log10(1.94e5), np.log10(7.8e6), 101)
    fill = ax.fill_between(x, *ic_butterfly(x, 'aachen'), facecolor='None', edgecolor='#566573', label='IceCube (tracks only, ApJ 2016)')

    ax.set_xscale('log')
    ax.set_yscale('log', nonposy='clip')
    ax.set_ylim(5e-11, 5e-5)
    ax.set_xlabel(r'$E\,\,[\mathrm{GeV}]$')
    ax.set_ylabel(r'$E^{2}\times\Phi\,\,[\mathrm{GeV}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{cm}^{-2}]$')

    handles, labels = ax.get_legend_handles_labels()
    order = [2,3,4,0,1,5]
    osorted = lambda items,order: [x for _,x in sorted(zip(order,items))]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
               loc='upper center',
               ncol=2,
              frameon=False,
             fontsize='small')
    ax.yaxis.set_tick_params(which='both')
    ax.yaxis.set_ticks_position('both')
    ax.set_xlim(1e-1, 1e12)
    ax.xaxis.set_ticks(np.logspace(-1,12,14))
#     ax.xaxis.set_major_formatter(plt.NullFormatter())
#     ax.set_xlabel('')

    plt.tight_layout(0.1)
    return ax.figure
