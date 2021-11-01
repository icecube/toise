from gen2_analysis.figures import figure_data, figure

from gen2_analysis import diffuse, pointsource, surface_veto, multillh, factory
from gen2_analysis.cache import ecached, lru_cache

# from scipy import stats, optimize
from copy import copy
import numpy as np
# from tqdm import tqdm
from functools import partial


def make_components(aeffs, galactic_emission=diffuse.FermiGalacticEmission):
    # zero out effective area beyond active range
    aeff, muon_aeff = copy(aeffs)
    ebins = aeff.bin_edges[0]

    atmo = diffuse.AtmosphericNu.conventional(
        aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None)
    atmo.prior = lambda v, **kwargs: -(v-1)**2/(2*0.1**2)
    prompt = diffuse.AtmosphericNu.prompt(
        aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None)
    prompt.min = 0.5
    prompt.max = 3.
    astro = diffuse.DiffuseAstro(aeff, 1)
    astro.seed = 2.3
    galactic = galactic_emission(aeff, 1)
    components = dict(atmo=atmo, prompt=prompt, astro=astro, galactic=galactic)
    if muon_aeff:
        components['muon'] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    return components

@lru_cache(maxsize=4)
def create_bundle(exposures, galactic_emission=diffuse.FermiGalacticEmission):
    return factory.component_bundle(dict(exposures), partial(make_components, galactic_emission=galactic_emission))

def asimov_llh(exposures, astro=2.3, astro_gamma=-2.5, galactic_emission=diffuse.FermiGalacticEmission):
    components = create_bundle(exposures, galactic_emission).get_components()
    components['gamma'] = multillh.NuisanceParam(astro_gamma)
    return multillh.asimov_llh(components, astro=astro, gamma=astro_gamma)

def binning():
    factory.set_kwargs(
        psi_bins={k: (0, np.pi) for k in ('tracks', 'cascades', 'radio')},
        cos_theta=16
    )

@figure_data(setup=binning)
def fermi_pi0(exposures, astro=2.3, astro_gamma=-2.5, galactic_emission='fermi'):
    if galactic_emission == 'fermi':
        galactic_emission=diffuse.FermiGalacticEmission
    elif galactic_emission == 'kra_5':
        galactic_emission=diffuse.KRAGalacticDiffuseEmission
    elif galactic_emission == 'kra_50':
        galactic_emission=partial(diffuse.KRAGalacticDiffuseEmission, cutoff_PeV=50)
    else:
        raise ValueError("unknown model {}".format(galactic_emission))
    llh = asimov_llh(exposures, astro=astro, astro_gamma=astro_gamma, galactic_emission=galactic_emission)
    nominal = {k: v.seed for k,v in llh.components.items()}
    nominal.update({'astro': astro, 'gamma': astro_gamma})
    del nominal['galactic']
    # h0 = llh.fit(galactic=0, gamma=-2.5, minimizer_params=dict(epsilon=1e-2))
    # h1 = llh.fit(gamma=-2.5, minimizer_params=dict(epsilon=1e-2))
    h0 = {k: v.sum(axis=1) for k,v in llh.llh_contributions(galactic=0, **nominal).items()}
    h1 = {k: v.sum(axis=1) for k,v in llh.llh_contributions(galactic=1, **nominal).items()}
    # sum over energies
    meta = {
        'rates': {component: {k: v.sum(axis=1) for k,v in expectations.items()} for component, expectations in multillh.get_expectations(llh, astro=astro, gamma=astro_gamma, galactic=1).items()},
        'ts': {k: 2*(h1[k]-h0[k]) for k in h0.keys()}
    }
    components = create_bundle(exposures, galactic_emission).get_components()
    components['gamma'] = multillh.NuisanceParam(astro_gamma)
    source = components.pop('galactic')
    ul, ns, nb = pointsource.upper_limit(source, components, **nominal)
    meta['upper_limit'] = ul
    
    meta['signal_spectrum'] = {
        'energy': list(source.bin_edges.values())[0][1],
        'events': multillh.get_expectations(llh, astro=astro, gamma=astro_gamma, galactic=1)['galactic']['IceCube-TracksOnly_unshadowed_tracks'].sum(axis=0)
    }
    
    return meta

@figure
def rates(datasets):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    import healpy
    from gen2_analysis import plotting

    dataset = datasets[0]
    # print(dataset['data']['ts'])
    rates = dataset['data']['rates']
    bg = sum([np.asarray(rates[k][channel]) for k in rates if (k != 'galactic' and k != 'muon') for channel in rates[k]])
    sig = sum([np.asarray(rates[k][channel]) for k in rates if k == 'galactic' for channel in rates[k]])
    
    fig, axes = plt.subplots(2,2)
    
    ts = sum([np.asarray(dataset['data']['ts'][k]) for k in dataset['data']['ts'].keys()])
    # print(ts)
    # galactic = sum(map(np.asarray, dataset['data']['ts']['muon'].values()))
    # print(galactic)
    
    plt.sca(axes[0,0])
    healpy.mollview(bg, hold=True, title='Background (without muons): {:.0f} events'.format(bg.sum()), unit='Events')

    plt.sca(axes[0,1])
    northern_sig = np.where(bg > 10, sig, 0)
    healpy.mollview(northern_sig, hold=True, title='Signal: {:.0f} events'.format(northern_sig.sum()), unit='Events')

    plt.sca(axes[1,1])
    healpy.mollview(ts, hold=True, title='Discovery significance: {:.1f} $\sigma$'.format(np.sqrt(ts.sum())), unit=r'$-2\Delta \ln L$')
    # plt.title(r"{:.1f} $\sigma$".format(np.sqrt(ts.sum())))
    
    ax = axes[1,0]
    spec = dataset['data']['signal_spectrum']
    ax.loglog(*plotting.stepped_path(spec['energy'], np.asarray(spec['events'])[::-1].cumsum()[::-1]), nonposy='clip')
    ax.set_ylim(bottom=1e-1)
    ax.set_xlim(right=1e6)
    
    # plt.title(r"{:.1f} $\sigma$".format(np.sqrt(ts.sum())))
    
    return fig