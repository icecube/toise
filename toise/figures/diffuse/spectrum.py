from toise.figures import figure_data, figure

from toise import (
    diffuse,
    multillh,
    plotting,
    surface_veto,
    pointsource,
    factory,
)
from toise.cache import ecached, lru_cache

from scipy import stats, optimize
from copy import copy
import numpy as np
from tqdm import tqdm
from functools import partial
from io import StringIO


def make_components(aeffs, emin=1e2, emax=1e11):
    # zero out effective area beyond active range
    aeff, muon_aeff = copy(aeffs)
    ebins = aeff.bin_edges[0]
    mask = (ebins[1:] <= emax) & (ebins[:-1] >= emin)
    aeff.values *= mask[[None] + [slice(None)] + [None] * (aeff.values.ndim - 2)]

    atmo = diffuse.AtmosphericNu.conventional(
        aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None
    )
    atmo.prior = lambda v, **kwargs: -((v - 1) ** 2) / (2 * 0.1**2)
    prompt = diffuse.AtmosphericNu.prompt(
        aeff, 1, hard_veto_threshold=np.inf, veto_threshold=None
    )
    prompt.min = 0.5
    prompt.max = 3.0
    astro = diffuse.DiffuseAstro(aeff, 1)
    astro.seed = 2.0
    composite = diffuse.ArbitraryFlux(aeff, 1)
    composite.seed = 0
    components = dict(atmo=atmo, prompt=prompt, astro=astro, composite=composite)
    if muon_aeff:
        components["muon"] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    return components


def powerlaw(energy, gamma=-2.0, emax=np.inf):
    return 1e-18 * np.exp(-energy / emax) * (energy / 1e5) ** gamma


def subdivide(bundle, seed_flux, key="astro", decades=1, emin=1e2, emax=1e11):
    import toolz

    bundle = copy(bundle)
    bundle.components = copy(bundle.components)
    for k in bundle.components.keys():
        components = copy(bundle.components[k])
        astro = components.pop(key)

        def downsample(f):
            return toolz.take_nth(1, f)

        for i, (e_center, chunk) in toolz.pipe(
            astro.differential_chunks(decades, exclusive=True, emin=emin, emax=emax),
            enumerate,
        ):
            if not chunk:
                continue
            # NB: seed flux is all-flavor, we want per-flavor
            chunk.seed = seed_flux(e_center) / powerlaw(e_center) / 3.0
            components["{}_{:02d}".format(key, i)] = chunk
        bundle.components[k] = components
    return bundle


@lru_cache()
def asimov_llh(bundle, seed_flux, decades=1.0 / 3, emin=1e2, emax=1e11):
    components = bundle.get_components()
    comp = components["composite"]
    for c, _ in comp._components.values():
        c.set_flux_func(seed_flux)
    llh = multillh.asimov_llh(components, astro=0, composite=1)

    chunk_llh = multillh.LLHEval(llh.data)
    chunk_llh.components["gamma"] = multillh.NuisanceParam(-2)
    chunk_llh.components.update(
        subdivide(
            bundle, seed_flux=seed_flux, decades=decades, emin=emin, emax=emax
        ).get_components()
    )

    return chunk_llh


def find_limits(llh, key, nom=None, critical_ts=1**2, plotit=False):
    if nom is None:
        nom = {k: v.seed for k, v in llh.components.items()}
    base = llh.llh(**nom)

    def ts_diff(value):
        fixed = dict(nom)
        fixed[key] = value
        alt = fixed
        ts = -2 * (llh.llh(**alt) - base) - critical_ts
        return ts

    g0 = nom[key]
    if ts_diff(0) < critical_ts:
        lo = 0
    else:
        lo = optimize.bisect(ts_diff, 0, g0, xtol=5e-3, rtol=1e-4)

    try:
        hi = optimize.bisect(ts_diff, g0, 1e3 * g0, xtol=5e-3, rtol=1e-4)
    except ValueError:
        hi = np.inf

    energy_range = list(llh.components[key]._components.values())[0][0].energy_range
    if plotit and energy_range[0] > 1e6:
        x = linspace(0, g0 * 2, 101)
        energy_range = list(llh.components[key]._components.values())[0][0].energy_range
        line = plot(
            x, [ts_diff(x_) for x_ in x], label="%.1g-%.1g" % tuple(energy_range)
        )[0]
        axvline(nom[key], color=line.get_color())

    return lo, hi


@lru_cache()
def unfold_llh(chunk_llh):
    import toolz

    fixed = dict(gamma=-2, prompt=1, atmo=1, muon=1)
    fit = chunk_llh.fit(minimizer_params=dict(options=dict(epsilon=1e-2)), **fixed)
    is_astro = partial(filter, lambda s: s.startswith("astro"))
    keys = toolz.pipe(list(chunk_llh.components.keys()), is_astro, sorted)[4:]
    xlimits = np.array(
        [
            list(chunk_llh.components[k]._components.values())[0][0].energy_range
            for k in keys
        ]
    )
    xcenters = 10 ** (np.log10(xlimits).sum(axis=1) / 2.0)
    ylimits = np.array([find_limits(chunk_llh, key, nom=fit) for key in tqdm(keys)])
    ycenters = np.array([chunk_llh.components[k].seed for k in keys])
    ycenters = np.array([fit[k] for k in keys])

    return xlimits, ycenters, ylimits


@ecached(
    __name__ + ".unfolding.{exposures}.{astro}.{gamma}.emax{emax}.{gzk}x{gzk_norm}.{astro_model}",
    timeout=2 * 24 * 3600,
)
def unfold_bundle(exposures, astro, gamma, gzk="vanvliet", gzk_norm=1, astro_model="powerlaw",emax=1e9):
    bundle = factory.component_bundle(dict(exposures), make_components)
    
    assert gzk == "vanvliet"
    
    def powerlaw_flux(energy):
        # NB: ArbitraryFlux assumes all-flavor flux, and divides by 6 to get per-particle
        return 3 * (
            astro * powerlaw(energy, gamma, emax)
            +  gzk_norm * diffuse.VanVlietGZKFlux()(energy)
        )
       
    def dip_spectrum(energy):
        # NB: this is the model with a dip at few hundred GeV fom the 6yr cascade paper (https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.125.121104)
        return 3 * ( 
            4.3e-18 * ((energy/1.e5)**-2)*np.exp(-energy/10**5.1)
            + 1.3e-21 * ((energy/1.e6)**-1.25)*np.exp(-(energy/(10**7.3))**0.5)
            +  gzk_norm * diffuse.VanVlietGZKFlux()(energy)
        )

    function_map={
        'powerlaw': powerlaw_flux,
        'cascade_paper_dip_spectrum': dip_spectrum
    }

    assert(astro_model in list(function_map.keys()))
    seed_flux=function_map[astro_model]

    result = unfold_llh(asimov_llh(bundle, seed_flux))
    for k, v in diffuse.DiffuseAstro.__dict__.items():
        if hasattr(v, "cache_info"):
            print((k, v.cache_info()))
    return result


def psi_binning():
    factory.set_kwargs(
        psi_bins={k: (0, np.pi) for k in ("tracks", "cascades", "radio")}
    )


@figure_data(setup=psi_binning)
def unfold(
    exposures,
    astro=2.3,
    gamma=-2.5,
    gzk="vanvliet",
    gzk_norm=1.0,
    astro_model='powerlaw',
    emax=1e9,
    clean=False
):
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
            exposures, astro, gamma, gzk, gzk_norm, astro_model, emax
        )
    xlimits, ycenters, ylimits = unfold_bundle(
        exposures, astro, gamma, gzk, gzk_norm, astro_model, emax
    )

    return {
        "xlimits": xlimits.tolist(),
        "ylimits": ylimits.tolist(),
        "ycenters": ycenters.tolist(),
    }


@figure
def flux_error(datasets):
    import matplotlib.pyplot as plt

    ax = plt.gca()

    for dataset in datasets:
        xlimits = np.array(dataset["data"]["xlimits"])
        ylimits = np.array(dataset["data"]["ylimits"])
        yvalues = np.array(dataset["data"]["ycenters"])
        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5 * (edges[1:] + edges[:-1])
        ydiff = 0.5 * (ylimits[:, 1] - ylimits[:, 0])
        ax.plot(
            xvalues,
            100 * ydiff / yvalues,
            label=r"$\Delta\Phi_\nu/\Phi_\nu$",
            linestyle="-",
            marker=".",
        )
    ax.loglog()
    ax.set_xlim((1e4, 2e9))
    ax.set_xlabel("Neutrino energy [GeV]")
    ax.set_ylabel("Relative uncertainty [%]")
    ax.legend()
    plt.tight_layout(0.1)
    return ax.figure


def get_ic_contour(source="lars",all_flavor=True):
    # 99% confidence interval
    if source == "lars":
        return np.loadtxt(
            StringIO(
                """2.164	3.154
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
                                      """
            )
        )
    elif source == "aachen":
        dat = np.loadtxt(
            StringIO(
                """1.716	0.225
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
                                     """
            )
        )
    elif source == "6y_cascades":
        dat = np.loadtxt(
            StringIO(
                """
        16295.4 5.31484e-08 
        71709.4 2.35803e-08 
        102127  1.99827e-08  
        2.58918e+06     4.64159e-09   
        2.58918e+06     1.82574e-09
        131471  1.06205e-08      
        78008   1.39249e-08  
        16023.3 2.95521e-08 
        16295.4 5.31484e-08 
        """
            )
        )

    elif source == "10y_diffuse":
        dat = np.loadtxt(
            StringIO(
                """
        14715.6 4.40177e-08            
        100072  1.83298e-08       
        220273  1.32939e-08       
        458691  1.01916e-08       
        997280  7.97466e-09       
        5100000     5.10121e-09       
        5100000     2.12424e-09       
        2334730     3.06903e-09       
        997280  4.5521e-09        
        515683  5.9551e-09        
        253824  7.58843e-09       
        99456.8 9.89831e-09       
        33405.1 1.32939e-08       
        14715.6 1.63567e-08       
        14715.6 4.40177e-08  
        """
            )
        )

    else:
        raise RuntimeError("Unknown IceCube result.")

    # factor of 3 for all-flavor
    if all_flavor: dat[:, 1] *= 3
    return dat


def plot_ic_data(ax,source,all_flavor=True,**kwargs):
    if all_flavor: flavf=3.
    else: flavf=1.
    if source == '10y_diffuse':
        dat = np.loadtxt(StringIO("""
#        5874.68 3.09105e-08     2875.48     9361.91      0      0
        44356   2.21612e-08     29730.9 58847.9 7.86045e-09     7.95203e-09     
        276700  1.21079e-08     173496  442646  2.96006e-09     3.05292e-09     
        1.82456e+06     3.33051e-09     1.10522e+06     3.15858e+06     1.73494e-09     2.20532e-09    
#        1.94433e+07     4.04861e-09     1.44499e+07     3.02453e+07     0       0
        """)).transpose()
    if source == 'glashow_nature':
        dat = np.array([[4.62979e+06,6.30957e+06,8.56512e+06],[1.14994e-07,5.67596e-09,7.29325e-07],[620830,916725,1.19692e+06],[787417,1.07256e+06,1.39135e+06],[0,3.2638e-09,0],[0,2.81791e-08,0]])
        for i in [1,4,5]: dat[i] = dat[i]/3. 
    if source == '6y_cascades':
        dat = np.loadtxt(StringIO("""
        6800.99 3.16228e-08     1804.9  3269.35 2.10701e-08     2.24166e-08     
        14703.7 3.37316e-08     4703.74 6765.24 1.08328e-08     9.10109e-09     
        31567.4 4.42379e-08     10098.4 14848.5 7.78917e-09     9.45374e-09     
        68248.7 1.93605e-08     21832.8 32102.4 5.34108e-09     6.69434e-09     
        146523  1.99956e-08     46171.9 68920.5 5.0413e-09      5.89154e-09     
        316782  1.73108e-09     101339  149006  0       0       
        680099  5.52149e-10     217564  319901  4.67236e-10     4.2177e-09      
        1.47037e+06     1.08989e-08     470374  691625  5.50659e-09     7.13437e-09     
        3.17895e+06     4.89463e-09     1.03205e+06     1.46264e+06     0       0       
        6.82487e+06     1.44482e-09     2.18328e+06     3.21024e+06     0       0       
        1.46523e+07     7.69097e-09     4.68728e+06     6.89205e+06     0       0       
        3.16782e+07     1.35741e-08     1.01339e+07     1.49006e+07     0       0       
        6.80099e+07     2.62236e-08     2.17564e+07     3.19901e+07     0       0 
        """)).transpose()
#        dat=dat[:,:-4]
    is_limit= ((dat[4]==0) & (dat[5]==0))
    dat[4]=np.where(is_limit,0.3*dat[1],dat[4])    
    ax.errorbar(dat[0],flavf*dat[1],xerr=dat[2:4],yerr=flavf*dat[4:],uplims=is_limit,linestyle='',capsize=0,**kwargs) 
    return ax


def plot_ic_limit(ax, source, all_flavor=True,**kwargs):
    if source == "ehe_limit":
        # IceCube
        # log (E^2 * Phi [GeV cm^02 s^-1 sr^-1]) : log (E [Gev])
        # Phys Rev D 98 062003 (2018)
        # Numbers private correspondence Shigeru Yoshida
        ice_cube_limit = np.array(
            (
                [
                    #    (6.199999125, -7.698484687),
                    #    (6.299999496, -8.162876678),
                    #    (6.400000617, -8.11395291),
                    #    (6.500000321, -8.063634144),
                    #    (6.599999814, -8.004841781),
                    #    (6.699999798, -7.944960162),
                    #    (6.799999763, -7.924197388),
                    (6.899999872, -7.899315263),
                    (7.299999496, -7.730561153),
                    (7.699999798, -7.670680637),
                    (8.100001583, -7.683379711),
                    (8.500000321, -7.748746801),
                    (8.899999872, -7.703060304),
                    (9.299999496, -7.512907553),
                    (9.699999798, -7.370926525),
                    (10.10000158, -7.134626026),
                    (10.50000032, -6.926516638),
                    (10.89999987, -6.576523031),
                ]
            )
        )

        ice_cube_limit[:, 0] = 10 ** ice_cube_limit[:, 0]
        ice_cube_limit[:, 1] = 10 ** ice_cube_limit[:, 1]
        if not all_flavor: ice_cube_limit[:,1] /= 3.
    ax.errorbar(
        ice_cube_limit[:, 0],
        ice_cube_limit[:, 1],
        xerr=None,
        yerr=ice_cube_limit[:, 1] * 0.3,
        uplims=np.ones_like(ice_cube_limit[:, 1]),
        **kwargs
    )


def ic_butterfly(energy, source="lars"):
    ic_contour = get_ic_contour(source)
    flux = (
        energy**2
        * 1e-18
        * (
            ic_contour[:, 1][:, None]
            * (energy[None, :] / 1e5) ** (-ic_contour[:, 0][:, None])
        )
    )
    return flux.min(axis=0), flux.max(axis=0)


@figure
def unfolded_flux(datasets, label="Gen2-InIce+Radio"):
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer

    fig = plt.figure(figsize=(3.5, 3))
    ax = plt.gca()

    # plot underlying fluxes
    x = np.logspace(4, 10, 51)
    # ApJ 2015 best fit
    ax.plot(
        x,
        3 * 2.3e-8 * (x / 1e5) ** (-2.5 + 2),
        ls="--",
        color="lightgrey",
        lw=0.75,
        zorder=0,
    )
    # + hard component (soft component norm reduced to give similar flux at low energy)
    ax.plot(
        x,
        3 * (1e-8 * (x / 1e5) ** (-2.5 + 2) + 0.5e-8 * (x / 1e5) ** (-2 + 2)),
        ls=":",
        color="k",
        lw=0.75,
    )

    plot_kwargs = dict(linestyle="None", marker="o", markersize=3)
    for dataset in datasets:
        xlimits = np.array(dataset["data"]["xlimits"])
        ylimits = np.array(dataset["data"]["ylimits"])
        yvalues = np.array(dataset["data"]["ycenters"])
        # factor 3 for all-flavor convention!
        unit = 3e-8
        ylimits *= unit
        yvalues *= unit

        years = next(
            years for detector, years in dataset["detectors"] if detector != "IceCube"
        )

        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5 * (edges[1:] + edges[:-1])
        x, y = xvalues, yvalues
        yerr = abs(ylimits - yvalues[:, None]).T
        art = ax.errorbar(
            x,
            y,
            xerr=None,
            yerr=yerr,
            # uplims=m[mask],
            label="{label} ({years:.0f} years)".format(label=label, years=years),
            **plot_kwargs
        )

    x = np.logspace(np.log10(25e3), np.log10(2.8e6), 101)
    fill = ax.fill_between(
        x,
        *ic_butterfly(x, "lars"),
        facecolor="lightgrey",
        edgecolor="None",
        label="IceCube (ApJ 2015)"
    )
    x = np.logspace(np.log10(1.94e5), np.log10(7.8e6), 101)
    fill = ax.fill_between(
        x,
        *ic_butterfly(x, "aachen"),
        facecolor="None",
        edgecolor="#566573",
        label="IceCube (tracks only, ApJ 2016)"
    )

    ax.loglog(nonpositive="clip")
    ax.set_xlim((1e4, 2e9))
    ax.set_ylim((2e-9, 1e-6))
    ax.set_xlabel("Neutrino energy (GeV)")
    ax.set_ylabel(r"$E^2 \Phi_{\nu} \,\, ({\rm GeV \, cm^{-2} \, sr^{-1} \, s^{-1}})$")

    # sort such that error bars go first
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(handles)), key=lambda i: not isinstance(handles[i], ErrorbarContainer)
    )
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout()
    return ax.figure


def plot_crs(ax):
    from . import plot_flux_results_for_jvs as lars_plot

    def apply_style(kwargs):
        style = dict(linestyle="None", markersize=4)
        style.update(kwargs)
        return style

    ax.errorbar(
        label="Diffuse $\gamma$ (Fermi LAT)",
        marker="D",
        color="C1",
        **apply_style(lars_plot.get_fermi_igrb_2014())
    )
    ax.errorbar(
        label="Cosmic rays (Auger)",
        marker="o",
        color="C4",
        **apply_style(lars_plot.get_auger())
    )
    ax.errorbar(
        label="Cosmic rays (TA)",
        marker="s",
        color="C2",
        **apply_style(lars_plot.get_ta())
    )


@figure
def unfolded_flux_multimessenger(datasets, label="Gen2-InIce+Radio"):
    import matplotlib.pyplot as plt
    from matplotlib.container import ErrorbarContainer

    fig = plt.figure(figsize=(7, 3.5),dpi=300)
    ax = plt.gca()

    plot_crs(ax)

    assert datasets[0]["source"] == "toise.figures.diffuse.spectrum.unfold"
    args = datasets[0]["args"]
    # plot underlying fluxes
    x = np.logspace(4, 10, 51)
    pl = x**2 * args["astro"] * powerlaw(x, args["gamma"], args["emax"])
    assert args["gzk"] == "vanvliet"
    gzk = args["gzk_norm"] * diffuse.VanVlietGZKFlux()(x) * x**2
    ax.plot(x, 3 * pl, ls=":", color="grey")
    ax.plot(x, 3 * gzk, ls=":", color="grey")

    plot_kwargs = dict(linestyle="None", marker="o", markersize=3)
    for dataset in datasets:
        xlimits = np.array(dataset["data"]["xlimits"])
        ylimits = np.array(dataset["data"]["ylimits"])
        yvalues = np.array(dataset["data"]["ycenters"])
        # factor 3 for all-flavor convention!
        unit = 3e-8
        ylimits *= unit
        yvalues *= unit

        years = next(
            years for detector, years in dataset["detectors"] if detector != "IceCube"
        )

        edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
        xvalues = 0.5 * (edges[1:] + edges[:-1])
        x, y = xvalues, yvalues
        yerr = abs(ylimits - yvalues[:, None]).T
        art = ax.errorbar(
            x,
            y,
            xerr=None,
            yerr=yerr,
            # uplims=m[mask],
            label="{label} ({years:.0f} years)".format(label=label, years=years),
            **plot_kwargs
        )

    x = np.logspace(np.log10(25e3), np.log10(2.8e6), 101)
    fill = ax.fill_between(
        x,
        *ic_butterfly(x, "lars"),
        facecolor="lightgrey",
        edgecolor="None",
        label="IceCube (ApJ 2015)"
    )
    x = np.logspace(np.log10(1.94e5), np.log10(7.8e6), 101)
    fill = ax.fill_between(
        x,
        *ic_butterfly(x, "aachen"),
        facecolor="None",
        edgecolor="#566573",
        label="IceCube (tracks only, ApJ 2016)"
    )

    ax.set_xscale("log")
    ax.set_yscale("log", nonpositive="clip")
    ax.set_ylim(5e-11, 5e-5)
    ax.set_xlabel(r"$E\,\,[\mathrm{GeV}]$")
    ax.set_ylabel(
        r"$E^{2}\times\Phi\,\,[\mathrm{GeV}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{cm}^{-2}]$"
    )

    handles, labels = ax.get_legend_handles_labels()
    order = [2, 3, 4, 0, 1, 5]
    osorted = lambda items, order: [x for _, x in sorted(zip(order, items))]
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=2,
        frameon=False,
        fontsize="small",
    )
    ax.yaxis.set_tick_params(which="both")
    ax.yaxis.set_ticks_position("both")
    ax.set_xlim(1e-1, 1e12)
    ax.xaxis.set_ticks(np.logspace(-1, 12, 14))
    #     ax.xaxis.set_major_formatter(plt.NullFormatter())
    #     ax.set_xlabel('')

    plt.tight_layout()
    return ax.figure


@figure
def unfolded_flux_plus_sensitivity_mm(
    datasets, sensitivity, label="Gen2-InIce+Radio", plot_elements=None, all_flavor=True, ax=None
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.container import ErrorbarContainer

    _default_plot_elements = [
        "cr",
        "10y_diffuse",
        "6y_cascade",
        "glashow",
        "ehe",
        "gen2_unfolding",
        "gen2_sensitivity",
        "model_flux_powerlaw"
    ]
    if plot_elements is None:
        plot_elements = _default_plot_elements
    group_label_ic, group_label_gen2 = False, False

    if all_flavor: flavor_factor=3
    else: flavor_factor=1
    if ax is None:
        fig = plt.figure(figsize=(6.2, 3.5), dpi=300)
        ax = plt.gca()
    else:
        plt.sca(ax)

    if "cr" in plot_elements:
        plot_crs(ax)

    assert datasets[0]["source"] == "toise.figures.diffuse.spectrum.unfold"
    args = datasets[0]["args"]

    # plot underlying fluxes
 
    if 'lars_butterfly' in plot_elements:
        x = np.logspace(np.log10(25e3), np.log10(2.8e6), 101)
        fill = ax.fill_between(
            x, *ic_butterfly(x, "lars"), facecolor="lightgrey", edgecolor="None"
        )
        group_label_ic = True

    if "10y_diffuse_butterfly" in plot_elements:
        if 'ic+label+color' in plot_elements: legend_entry,plot_color,group_label_ic = 'IceCube tracks (power-law model)','darkblue',False
        else: legend_entry,plot_color,group_label_ic = None,"#566573",True
        if '10y_diffuse' in plot_elements: legend_entry=None
        poly1 = plt.Polygon(
            np.array(get_ic_contour("10y_diffuse",all_flavor=all_flavor)),
            edgecolor=plot_color,
            alpha=1.0,
            fill=False,
            label=legend_entry,
        )
        ax.add_patch(poly1)

    if "6y_cascade_butterfly" in plot_elements:
        if 'ic+label+color' in plot_elements: legend_entry,plot_color,group_label_ic = 'IceCube showers (power-law model)','#1f77b4',False
        else: legend_entry,plot_color,group_label_ic = None,"lightgrey",True
        if '6y_cascade' in plot_elements:legend_entry=None
        poly2 = plt.Polygon(
            np.array(get_ic_contour("6y_cascades",all_flavor=all_flavor)),
            edgecolor=None,
            alpha=0.5,
            color=plot_color,
            label=legend_entry,
        )
        ax.add_patch(poly2,)
    
    if '10y_diffuse' in plot_elements:
        if 'ic+label+color' in plot_elements: legend_entry,plot_color,group_label_ic = 'IceCube tracks, Aartsen et al., ApJ, 2022','darkblue',False
        else: legend_entry,plot_color,group_label_ic = None,"#566573",True
        plot_ic_data(ax,'10y_diffuse',color=plot_color,lw=1,alpha=0.8,marker='o',markersize=4,label=legend_entry,all_flavor=all_flavor)

    if '6y_cascade' in plot_elements:
        if 'ic+label+color' in plot_elements: legend_entry,plot_color,group_label_ic = 'IceCube showers, Aartsen et al., PRL, 2020','#1f77b4',False
        else: legend_entry,plot_color,group_label_ic = None,"grey",True
        plot_ic_data(ax,'6y_cascades',color=plot_color,lw=1,alpha=0.8,marker='s',markersize=4,label=legend_entry,all_flavor=all_flavor)
  
    if 'glashow' in plot_elements:
        if 'ic+label+color' in plot_elements: legend_entry,plot_color,group_label_ic = 'IceCube Glashow, Aartsen et al., Nature, 2021','#3f97F4',False
        else: legend_entry,plot_color,group_label_ic = None,"#768593",True
        plot_ic_data(ax,'glashow_nature',color=plot_color,lw=1,alpha=0.8,marker='P',markersize=4,label=legend_entry,all_flavor=all_flavor)
    
    if 'ehe' in plot_elements:
        if 'ic+label+color' in plot_elements: legend_entry,plot_color,group_label_ic = 'IceCube EHE, Aartsen et al., PRD, 2018','#1f77b4',False
        else: legend_entry,plot_color,group_label_ic = None,"grey",True
        plot_ic_limit(ax,'ehe_limit',color=plot_color,lw=1,alpha=0.5,label=legend_entry,all_flavor=all_flavor) 
    
    poly_handle,poly_label = [],[] 

    print(group_label_ic)
    if group_label_ic:
        poly_handle.append(Patch(alpha=0.5, edgecolor="#566573", facecolor="lightgrey"))
        poly_label.append("IceCube")

    if 'model_flux_powerlaw' in plot_elements:
        x = np.logspace(3.85, 7.15, 51)
        pl = x**2 * args["astro"] * powerlaw(x, args["gamma"], args["emax"])
        ax.plot(x, flavor_factor * pl, ls=":", lw=1, color='darkblue',zorder=200,label='Astrophysical flux model\n(power law, index = -2.5)')

    if 'model_flux_dip_spectrum' in plot_elements:
        x = np.logspace(3.85, 7.15, 51)
        dip = x**2 * ( 4.3e-18 * ((x/1.e5)**-2)*np.exp(-x/10**5.1)
            + 1.3e-21 * ((x/1.e6)**-1.25)*np.exp(-(x/(10**7.3))**0.5) )
        ax.plot(x, flavor_factor * dip, ls=":", lw=1, color='darkblue',zorder=200,label='Astrophysical flux model\n(Aartsen et al., PRL 2020, model E)')
   
    if 'gen2_unfolding' in plot_elements:
        plot_kwargs = dict(linestyle="None", marker="o", markersize=0,color='#1f77b4')
        for dataset in datasets:
            xlimits = np.array(dataset["data"]["xlimits"])
            ylimits = np.array(dataset["data"]["ylimits"])
            yvalues = np.array(dataset["data"]["ycenters"])
            # factor 3 for all-flavor convention!
            unit = flavor_factor* 1e-8
            ylimits *= unit
            yvalues *= unit

            years = next(
                years
                for detector, years in dataset["detectors"]
                if detector != "IceCube"
            )

            edges = np.concatenate((xlimits[:, 0], [xlimits[-1, 1]]))
            xvalues = 0.5 * (edges[1:] + edges[:-1])
            index_start = (np.asarray(xvalues) < 1e4).sum() - 1
            index_stop = (np.asarray(xvalues) < 1e7).sum()
            x, y = xvalues[index_start:index_stop], yvalues[index_start:index_stop]
            yerr = (abs(ylimits - yvalues[:, None]).T)
            yerr_s =np.array([yerr[0,index_start:index_stop],yerr[1,index_start:index_stop]])
            delta_loge=np.log(x[1]/x[0])
            xerr_s=[x*(1-1/np.exp(0.5*delta_loge)),x*(np.exp(0.5*delta_loge)-1)]
    
            for xx,yy,xe0,xe1,ye0,ye1 in zip(x,y,*xerr_s,*yerr_s):
                ax.add_patch(plt.Rectangle((xx-xe0,yy-ye0),xe0+xe1,ye0+ye1, 
                            edgecolor=None, alpha=0.9,facecolor='#1f77b4',linewidth=0,zorder=100))
        group_label_gen2=True

    if 'gen2_sensitivity' in plot_elements:
        energies,flux,ns,nb = sensitivity
        flux=np.array(flux)*unit
        fluxerr=0.3*flux
        index_start= (np.asarray(energies)<1e7).sum()
        index_low=   (np.asarray(energies)<1e4).sum()
        energies_s,flux_s,fluxerr_s = [x[index_start:-1] for x in [energies,flux,fluxerr]]
        energies_l,flux_l,fluxerr_l = [x[index_low:index_start+1] for x in [energies,flux,fluxerr]]
        ax.errorbar(energies_s,flux_s,xerr=None,yerr=fluxerr_s,uplims=np.ones_like(flux_s),color='#1f77b4')
        group_label_gen2=True

    if 'gen2_sensitivity_extension' in plot_elements:
        ax.plot(energies_l,flux_l,ls="--",color='#1f77b4')
        group_label_gen2=True


    if group_label_gen2:        
        poly_handle.append(Patch(edgecolor=None,alpha=0.9,facecolor='#1f77b4',linewidth=0))
        poly_label.append("{label} ({years:.0f} years)".format(label=label, years=years))

    ax.set_xscale("log")
    ax.set_yscale("log", nonpositive="clip")
    ax.set_xlabel(r"$E\,\,[\mathrm{GeV}]$")
    ax.set_ylabel(
        r"$E^{2}\times\Phi\,\,[\mathrm{GeV}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{cm}^{-2}]$"
    )

    handles, labels = ax.get_legend_handles_labels()
    handles += poly_handle
    labels += poly_label
    order, label_idx = [], len(labels) - 1
    if group_label_gen2:
        order.append(label_idx)
        label_idx -= 1
    if group_label_ic:
        order.append(label_idx)
        label_idx -= 1
    order += range(0, label_idx + 1)
    osorted = lambda items, order: [x for _, x in sorted(zip(order, items))]
    if "ic+label+color" in plot_elements and 'cr' not in plot_elements: legend_ncol=1
    else: legend_ncol=2
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        loc="upper center",
        ncol=legend_ncol,
        frameon=False,
        fontsize='small',
    )
    ax.yaxis.set_tick_params(which="both")
    ax.yaxis.set_ticks_position("both")
    if "cr" in plot_elements:
        ax.set_xlim(5e-2, 3e11)
        if all_flavor: ax.set_ylim(5e-11, 1e-4)
        else: ax.set_ylim(1e-11, 3e-5)
        ax.xaxis.set_ticks(np.logspace(-1, 11, 13))
    else:
        ax.set_xlim(3e3, 1e11)
        if all_flavor: ax.set_ylim(1e-10, 1e-5)
        else: ax.set_ylim(3e-11, 3e-6)
        ax.xaxis.set_ticks(np.logspace(3, 11, 9))

    plt.tight_layout()
    return ax.figure


@figure
def unfolded_flux_plus_sensitivity(datasets, sensitivity, label="Gen2-InIce+Radio",plot_elements=None,all_flavor=True,ax=None):
    _default_plot_elements=['10y_diffuse','6y_cascade','glashow','ehe','gen2_unfolding','gen2_sensitivity','model_flux_powerlaw']
    if plot_elements is None: plot_elements=_default_plot_elements
    return unfolded_flux_plus_sensitivity_mm(datasets,sensitivity,label,plot_elements,all_flavor,ax)

