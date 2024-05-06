"""
Flavor ratio fits
"""

from functools import partial

import pandas as pd
import numpy as np
from scipy.optimize import bisect
from tqdm import tqdm

import toolz
import copy

from toise.figures import figure_data, figure
from toise.cache import ecached, lru_cache
from toise import diffuse, multillh, plotting, surface_veto, factory, plotting

from matplotlib.lines import Line2D


def make_components(aeffs, astro_class=diffuse.DiffuseAstro):
    """
    :param muon_damped:
    """
    aeff, muon_aeff = aeffs
    atmo = diffuse.AtmosphericNu.conventional(aeff, 1, veto_threshold=None)
    atmo.prior = lambda v, **kwargs: -((v - 1) ** 2) / (2 * 0.1**2)
    prompt = diffuse.AtmosphericNu.prompt(aeff, 1, veto_threshold=None)
    prompt.min = 0.5
    prompt.max = 3.0
    astro = astro_class(aeff, 1)
    components = dict(atmo=atmo, prompt=prompt, astro=astro)
    if muon_aeff:
        components["muon"] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    return components


def triangle_product(seq):
    for i in range(len(seq)):
        for j in range(len(seq) - i):
            yield seq[i], seq[j]


def expand(dataframe):
    e_frac = dataframe.index.levels[0].values
    mu_frac = dataframe.index.levels[1].values
    idx = pd.MultiIndex.from_product((e_frac, mu_frac), names=dataframe.index.names)
    return dataframe.reindex(idx)


@lru_cache(maxsize=4)
def create_bundle(exposures, **kwargs):
    return factory.component_bundle(dict(exposures), partial(make_components, **kwargs))


@lru_cache(maxsize=4)
def subdivide(bundle, key="astro", esplit=1e6):
    bundle = copy.copy(bundle)
    bundle.components = copy.copy(bundle.components)
    for k in bundle.components.keys():
        components = copy.copy(bundle.components[k])
        astro = components.pop(key)

        def downsample(f):
            return toolz.take_nth(1, f)

        ebins = astro._aeff.bin_edges[0]
        idx = ebins.searchsorted(esplit) - 1

        for i, sl in enumerate((slice(None, idx), slice(idx, None))):
            chunk = copy.copy(astro)
            chunk._invalidate_cache()
            chunk._flux = np.zeros(astro._flux.shape)
            chunk._flux[:, sl, ...] = astro._flux[:, sl, ...]
            chunk._suffix = "_{:02d}".format(i)
            chunk._energy_range = ebins[sl][0], ebins[sl][-1]
            components[key + chunk._suffix] = chunk
        bundle.components[k] = components
    return bundle


@lru_cache(maxsize=4)
def asimov_llh(exposures, **nominal):
    components = create_bundle(exposures).get_components()
    gamma = nominal.pop("gamma", -2.5)
    components["gamma"] = multillh.NuisanceParam(
        gamma, 0.5, min=gamma - 0.5, max=gamma + 0.5
    )
    components["e_tau_ratio"] = multillh.NuisanceParam(
        (0.93 / 3) / (1 - 1.05 / 3), None, 0, 1
    )
    components["mu_fraction"] = multillh.NuisanceParam(1.05 / 3, None, 0, 1)
    if not "muon" in components:
        components["muon"] = multillh.NuisanceParam(1, None, 0, 1)
    return multillh.asimov_llh(components, **nominal)


@lru_cache(maxsize=4)
def muondamped_asimov_llh(exposures, **nominal):

    ecrit = nominal.get("emu_crit", 2e6)
    # generate data from model where each flavor has a different spectrum
    source_components = create_bundle(
        exposures, astro_class=diffuse.MuonDampedDiffuseAstro
    ).get_components()
    source_components["emu_crit"] = multillh.NuisanceParam(ecrit)
    data = multillh.asimov_llh(source_components).data

    atearth = diffuse.IncoherentOscillation.create()(
        *(
            diffuse.MuonDampedDiffuseAstro.pion_decay_flux(
                np.array((ecrit / 1e3, ecrit * 1e3)), ecrit
            ).T
        )
    )
    mufrac = atearth[1, :] / atearth.sum(axis=0)
    e_tau_ratio = atearth[0, :] / (atearth[0, :] + atearth[2, :])

    # fit with a model with the same energy spectrum for each flavor
    components = subdivide(
        create_bundle(
            exposures,
            astro_class=partial(
                diffuse.MuonDampedDiffuseAstro, fixed_flavor_ratio=True
            ),
        ),
        esplit=ecrit / 2,
    ).get_components()
    components["emu_crit"] = multillh.NuisanceParam(ecrit)
    if not "muon" in components:
        components["muon"] = multillh.NuisanceParam(1, None, 0, 1)
    kwargs = dict()
    for i in range(2):
        param = multillh.NuisanceParam(mufrac[i])
        param.min = 0.0
        param.max = 1.0
        components["mu_fraction_{:02d}".format(i)] = param

        param = multillh.NuisanceParam(e_tau_ratio[i])
        param.min = 0.0
        param.max = 1.0
        components["e_tau_ratio_{:02d}".format(i)] = param

    chunk_llh = multillh.asimov_llh(components, **kwargs)
    chunk_llh.data = data

    return chunk_llh


@lru_cache(maxsize=16)
def fit_llh(llh, **fixed):
    return llh.fit(**fixed)


@ecached(
    __name__ + ".profile.astro{suffix}_{exposures}_{nominal}_{steps}_{gamma_step}",
    timeout=2 * 24 * 3600,
)
def make_profile(
    exposures,
    nominal=dict(),
    suffix="",
    steps=100,
    minimizer_params=dict(tol=None, method="Nelder-Mead"),
    gamma_step=None,
):
    if suffix:
        llh = muondamped_asimov_llh(exposures, **nominal)
    else:
        llh = asimov_llh(exposures, **nominal)
    steps = np.linspace(0, 1, steps + 1)
    fixed = dict(atmo=1, prompt=1, muon=1)
    if "emu_crit" in nominal:
        fixed["emu_crit"] = nominal["emu_crit"]
    if gamma_step is None and not suffix:
        fixed["gamma"] = llh.components["gamma"].seed
    bestfit = llh.fit(minimizer_params=minimizer_params, **fixed)
    params = []
    scan = ["mu_fraction" + suffix, "e_tau_ratio" + suffix]
    free = ["astro" + suffix]
    # fixed = {k: llh.components[k].seed for k in llh.components if not k in scan+free}
    fixed = {k: bestfit[k] for k in bestfit if not k in scan + free}
    if gamma_step is not None and "gamma" in fixed:
        fixed["gamma"] = fixed["gamma"] + np.arange(-10, 11) * gamma_step
    for e, mu in tqdm(
        triangle_product(steps),
        total=(steps.size + 1) * (steps.size) / 2,
        desc="{} {}".format(__name__, detector_label(exposures)),
    ):
        fixed["mu_fraction" + suffix] = mu
        fixed["e_tau_ratio" + suffix] = e / (1 - mu) if mu != 1 else 1
        assert e + mu <= 1
        fit = llh.fit(minimizer_params, **fixed)
        fit["LLH"] = llh.llh(**fit)
        fit["e_fraction" + suffix] = e
        params.append(fit)
    return expand(
        pd.DataFrame(params).set_index(["e_fraction" + suffix, "mu_fraction" + suffix])
    )


@ecached(
    __name__
    + ".source_profile.astro{suffix}_{exposures}_{nominal}_{steps}_{gamma_step}",
    timeout=2 * 24 * 3600,
)
def make_source_profile(
    exposures,
    nominal=dict(),
    suffix="",
    steps=100,
    minimizer_params=dict(epsilon=1e-2),
    gamma_step=None,
):
    if suffix:
        llh = muondamped_asimov_llh(exposures, **nominal)
    else:
        llh = asimov_llh(exposures, **nominal)
    steps = np.linspace(0, 1, steps + 1)
    fixed = dict(atmo=1, prompt=1, muon=1)
    if "emu_crit" in nominal:
        fixed["emu_crit"] = nominal["emu_crit"]
    bestfit = fit_llh(llh, **fixed)
    params = []
    scan = ["mu_fraction" + suffix, "e_tau_ratio" + suffix]
    free = ["astro" + suffix]
    fixed = {k: bestfit[k] for k in bestfit if not k in scan + free}
    if gamma_step is not None and "gamma" in fixed:
        fixed["gamma"] = fixed["gamma"] + np.arange(-10, 11) * gamma_step
    oscillate = diffuse.IncoherentOscillation.create()
    for mufrac in tqdm(steps, desc="{} {}".format(__name__, detector_label(exposures))):
        e, mu, tau = oscillate(1.0 - mufrac, mufrac, 0)[0]
        fixed["mu_fraction" + suffix] = mu / (e + mu + tau)
        fixed["e_tau_ratio" + suffix] = e / (e + tau)
        fit = llh.fit(minimizer_params, **fixed)
        fit["LLH"] = llh.llh(**fit)
        fit["source_mu_fraction" + suffix] = mufrac
        params.append(fit)
    return pd.DataFrame(params).set_index("source_mu_fraction" + suffix)


def extract_ts(dataframe):
    if isinstance(dataframe.index, pd.MultiIndex):
        e_frac = dataframe.index.levels[0].values
        mu_frac = dataframe.index.levels[1].values
        maxllh = np.nanmax(dataframe["LLH"])
        idx = pd.MultiIndex.from_product((e_frac, mu_frac), names=dataframe.index.names)

        ts = 2 * (
            pd.Series(
                (np.nanmax(dataframe["LLH"].values) * np.ones(len(idx))), index=idx
            )
            - dataframe["LLH"]
        )
        return e_frac, mu_frac, ts.values.reshape(e_frac.size, mu_frac.size)
    else:
        return dataframe.index.values, 2 * (
            np.nanmax(dataframe["LLH"]) - dataframe["LLH"]
        )


def psi_binning():
    factory.set_kwargs(
        psi_bins={k: (0, np.pi) for k in ("tracks", "cascades", "radio")}
    )


@figure_data(setup=psi_binning)
def confidence_levels(
    exposures,
    astro=2.3,
    gamma=-2.5,
    steps=100,
    gamma_step=0.0,
    clean: bool = False,
    debug: bool = False,
):
    """
    Calculate exclusion confidence levels for alternate flavor ratios, assuming
    Wilks theorem.

    :param astro: per-flavor astrophysical normalization at 100 TeV, in 1e-18 GeV^-2 cm^-2 sr^-1 s^-1
    :param gamma: astrophysical spectral index
    :param steps: number of steps along one axis of flavor scan
    :param gamma_step: granularity of optimization in spectral index. if 0, the spectral index is fixed.
    """
    from scipy import stats

    if clean:
        make_profile.invalidate_cache_by_key(
            exposures, steps=steps, nominal=dict(gamma=gamma, astro=astro)
        )
    profile = make_profile(
        exposures, steps=steps, nominal=dict(gamma=gamma, astro=astro)
    )
    meta = {}
    efrac, mufrac, ts = extract_ts(profile)
    meta["nue_fraction"] = efrac.tolist()
    meta["numu_fraction"] = mufrac.tolist()
    meta["confidence_level"] = (stats.chi2.cdf(ts.T, 2) * 100).tolist()
    if debug:
        for k in profile.columns:
            if k not in meta:
                meta[k] = profile[k].tolist()
    return meta


@figure_data(setup=psi_binning)
def event_counts(exposures, astro=2.3, gamma=-2.5):
    e, mu = 0.93 / 3, 1.05 / 3
    nominal = dict(
        astro=astro,
        gamma=gamma,
        atmo=0,
        prompt=0,
        muon=0,
        e_tau_ratio=e / (1 - mu),
        mu_fraction=mu,
    )
    assert len(exposures) == 1
    llh = asimov_llh(exposures, **nominal)
    flavors = {
        "e": {"mu_fraction": 0, "e_tau_ratio": 1, "astro": e * astro},
        "mu": {"mu_fraction": 1, "e_tau_ratio": 0, "astro": mu * astro},
        "tau": {"mu_fraction": 0, "e_tau_ratio": 0, "astro": (1 - e - mu) * astro},
        "atmospheric": {"astro": 0, "atmo": 1, "prompt": 1, "muon": 1},
    }
    prefix = exposures[0][0]
    energy_thresholds = {
        k[len(prefix) + 1 :]: v[0]._aeff.get_bin_edges("reco_energy")[:-1]
        for k, v in llh.components["astro"]._components.items()
    }
    meta = {"reco_energy_threshold": energy_thresholds, "event_counts": {}}
    for label, values in flavors.items():
        params = dict(nominal)
        params.update(values)
        meta["event_counts"][label] = {
            k[len(prefix) + 1 :]: v.sum(axis=0)[::-1].cumsum()[::-1]
            for k, v in llh.expectations(**params).items()
        }

    return meta


@figure
def event_counts(datasets):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 1)

    for dataset in datasets:
        for i, (channel, energy) in enumerate(
            dataset["data"]["reco_energy_threshold"].items()
        ):
            # ax = axes.flat[i]
            if channel != "double_cascades":
                continue
            else:
                ax = axes
            sig = dataset["data"]["event_counts"]["tau"][channel]
            bkg = np.sum(
                [
                    dataset["data"]["event_counts"][flavor][channel]
                    for flavor in dataset["data"]["event_counts"]
                    if flavor != "tau"
                ],
                axis=0,
            )
            ax.loglog(energy, sig, label="tau", color="C0")
            ax.loglog(energy, bkg, label="not tau", color="C1")

    plt.tight_layout()


@figure_data(setup=psi_binning)
def muon_damping_constraints(exposures, steps=100, emu_crit=2e6, clean=False):
    """
    Calculate exclusion confidence levels for alternate flavor ratios, assuming
    Wilks theorem.

    :param astro: per-flavor astrophysical normalization at 100 TeV, in 1e-18 GeV^-2 cm^-2 sr^-1 s^-1
    :param gamma: astrophysical spectral index
    :param steps: number of steps along one axis of flavor scan
    :param gamma_step: granularity of optimization in spectral index. if 0, the spectral index is fixed.
    :param emu_crit: critical energy at which muons are more likely to reinteract than decay (GeV)
    :param clean: recalculate from scratch, ignoring cached results
    """

    from scipy import stats

    meta = {
        "earth": {"nue_fraction": [], "numu_fraction": [], "confidence_level": []},
        "source": {"numu_fraction": [], "test_statistic": []},
    }
    for i in range(2):
        suffix = "_{:02d}".format(i)
        if clean:
            make_profile.invalidate_cache_by_key(
                exposures, steps=steps, suffix=suffix, nominal=dict(emu_crit=emu_crit)
            )
        profile = make_profile(
            exposures, steps=steps, suffix=suffix, nominal=dict(emu_crit=emu_crit)
        )
        efrac, mufrac, ts = extract_ts(profile)
        meta["earth"]["nue_fraction"].append(efrac.tolist())
        meta["earth"]["numu_fraction"].append(mufrac.tolist())
        meta["earth"]["confidence_level"].append(
            (stats.chi2.cdf(ts.T, 2) * 100).tolist()
        )

        source_profile = make_source_profile(
            exposures, steps=steps, suffix=suffix, nominal=dict(emu_crit=emu_crit)
        )
        mufrac, ts = extract_ts(source_profile)
        meta["source"]["numu_fraction"].append(mufrac.tolist())
        meta["source"]["test_statistic"].append(ts.tolist())
    return meta


def detector_label(exposures):
    """transform (('Gen2-InIce', 10.0), ('IceCube', 15.0)) -> Gen2-InIce+IceCube 10+15 yr"""
    return "{} {} yr".format(
        *list(map("+".join, list(map(partial(map, str), list(zip(*exposures))))))
    )


@figure
def triangle(datasets):
    """
    Plot exclusion contours in flavor composition
    """
    from toise.externals import ternary
    import matplotlib.pyplot as plt

    ax = ternary.flavor_triangle(grid=True)
    labels = []
    handles = []
    e, mu = 0.93 / 3, 1.05 / 3
    for i, meta in enumerate(datasets):
        handles.append(Line2D([0], [0], color=f"C{i}"))
        labels.append(detector_label(meta["detectors"]))
        values = meta["data"]
        cs = ax.ab.contour(
            values["nue_fraction"],
            values["numu_fraction"],
            values["confidence_level"],
            levels=[
                68,
            ],
            colors=handles[-1].get_color(),
        )
    ax.ab.legend(handles, labels, bbox_to_anchor=(1.3, 1.1))

    return ax.figure


def make_error_boxes(
    x, y, xerr, yerr, facecolor="r", edgecolor="None", alpha=0.5, **kwargs
):

    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for xd, yd, xe, ye in zip(x, y, np.asarray(xerr).T, np.asarray(yerr).T):
        rect = Rectangle((xd - xe[0], yd - ye[0]), xe.sum(), ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(
        errorboxes, facecolor=facecolor, alpha=alpha, edgecolor=edgecolor, **kwargs
    )
    return pc


@figure
def muon_damping(datasets, preliminary=False):
    """
    Plot exclusion contours in flavor composition
    """
    from toise.externals import ternary
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from scipy import optimize, interpolate, stats

    fig = plt.figure(figsize=(5, 4))
    w, h = fig.bbox_inches.width, fig.bbox_inches.height
    griddy = plt.GridSpec(
        2,
        2,
        height_ratios=[4, 3],
        left=0.75 / w,
        right=(w - 0.25) / w,
        bottom=0.5 / h,
        top=(h - 0.35) / h,
        hspace=0.35,
    )

    def position_labels(ax):
        fontdict = dict(size="x-large")
        ax.ab.set_xlabel(r"$f_{e}$", fontdict=fontdict).set_position((0.5, -0.12))
        ax.bc.set_xlabel(r"$f_{\mu}$", fontdict=fontdict).set_position((0.5, -0.20))
        ax.ca.set_xlabel(r"$f_{\tau}$", fontdict=fontdict).set_position((0.5, -0.2))

    assert len(datasets) == 1
    meta = datasets[0]["data"]["earth"]
    ecrit = datasets[0]["args"]["emu_crit"]

    def color_with_alpha(color, alpha):
        rgba = list(mcolors.to_rgba(color))
        rgba[-1] = alpha
        return rgba

    for i in range(2):
        ax = ternary.flavor_triangle(fig, subplotspec=griddy[0, i])
        position_labels(ax)
        source_points = ax.ab.collections[:3]
        cs = ax.ab.contourf(
            meta["nue_fraction"][i],
            meta["numu_fraction"][i],
            meta["confidence_level"][i],
            levels=[0, 68, 95],
            colors=[color_with_alpha("C1", 0.7), color_with_alpha("C1", 0.3)],
        )
        cs = ax.ab.contour(
            meta["nue_fraction"][i],
            meta["numu_fraction"][i],
            meta["confidence_level"][i],
            levels=[68, 95],
            colors="C1",
            linestyles="-",
            linewidths=0.5,
        )
    ax = plt.subplot(griddy[1, :])

    e = np.logspace(4, 8, 101)
    flux = diffuse.MuonDampedDiffuseAstro.pion_decay_flux(e, ecrit)
    ax.plot(e, flux[:, 1] / flux.sum(axis=1), label=plotting.format_energy("%d", ecrit))
    ax.semilogx()

    meta = datasets[0]["data"]["source"]
    e = np.logspace(4, 8, 3)

    def get_x(ebins):
        loge = np.log10(ebins)
        xc = 10 ** ((loge[:-1] + loge[1:]) / 2.0)
        xerr = [xc - ebins[:-1], ebins[1:] - xc]
        return xc, xerr

    x, xerr = get_x(np.logspace(4, 8, 3))

    def get_y(numu_frac, ts, crit_ts=1):
        f = interpolate.interp1d(numu_frac, ts, bounds_error=True)
        y0 = optimize.fminbound(f, 0, 1)
        try:
            ylo = optimize.bisect(lambda y: f(y) - crit_ts, 0, y0)
        except ValueError:
            ylo = 0
        try:
            yhi = optimize.bisect(lambda y: f(y) - crit_ts, y0, 1)
        except ValueError:
            yhi = 1
        return y0, [y0 - ylo, yhi - y0]

    for cl, alpha in zip((0.9, 0.68), (0.3, 0.7)):
        y, yerr = list(
            zip(
                *[
                    get_y(
                        meta["numu_fraction"][i],
                        meta["test_statistic"][i],
                        crit_ts=stats.chi2(1).ppf(cl),
                    )
                    for i in range(2)
                ]
            )
        )

        ax.add_collection(
            make_error_boxes(x, y, xerr=xerr, yerr=yerr, facecolor="C1", alpha=alpha)
        )

    ax.set_xlabel(r"$E_{\nu}$ (GeV)")
    ax.set_ylabel(r"$\nu_{\mu}$ fraction at source")
    ax.set_ylim((0.5, 1.1))

    bbox = ax.bbox.transformed(fig.transFigure.inverted())
    leg = fig.legend(
        source_points,
        ["1:2:0", "0:1:0", "1:0:0"],
        loc=(0.42, 0.83),
        numpoints=1,
        handlelength=1,
        title="Flavor ratio at source",
        fontsize="x-small",
        frameon=False,
    )
    leg.get_title().set_multialignment("center")
    leg.get_title().set_fontsize("small")

    if preliminary:
        ax.add_artist(
            plt.Text(
                0.15,
                0.9,
                "IceCube-Gen2\npreliminary",
                color="C3",
                ha="center",
                va="top",
                multialignment="center",
                transform=ax.transAxes,
            )
        )

    return fig
