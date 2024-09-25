# from scipy import stats, optimize
from copy import copy

# from tqdm import tqdm
from functools import partial
from typing import List, Literal, Optional

import numpy as np

from toise import diffuse, factory, multillh, pointsource, surface_veto
from toise.cache import ecached, lru_cache
from toise.figures import figure, figure_data

from .dnnc import (
    AngularSmearing,
    RingAveraging,
    create_dnn_aeff,
    get_dnn_smoothing,
    get_monopod_smoothing,
)

factory.add_configuration(
    "IceCube-DNNC", lambda **kwargs: {"events": (create_dnn_aeff(nside=16), None)}
)


def make_components(
    aeffs,
    galactic_emission=diffuse.FermiGalacticEmission,
    angular_smoothing_fwhm=None,
    null_hypothesis=None,
):
    # zero out effective area beyond active range
    aeff, muon_aeff = copy(aeffs)
    ebins = aeff.bin_edges[0]

    atmo = diffuse.AtmosphericNu.conventional(aeff, 1, veto_threshold=None)
    atmo.prior = lambda v, **kwargs: -((v - 1) ** 2) / (2 * 0.1**2)
    prompt = diffuse.AtmosphericNu.prompt(aeff, 1, veto_threshold=None)
    prompt.min = 0.5
    prompt.max = 3.0
    astro = diffuse.DiffuseAstro(aeff, 1)
    astro.seed = 2.3
    galactic = galactic_emission(aeff, 1)
    if null_hypothesis is None:
        # if None, attempt to fit injected galactic emission with an RA-averaged version
        galactic_bg = RingAveraging(
            galactic_emission(aeff, 1), angular_smoothing_fwhm=None
        )
    else:
        # otherwise, use the provided component to fit
        galactic_bg = null_hypothesis(aeff, 1)
    components = dict(
        atmo=atmo,
        prompt=prompt,
        astro=astro,
        galactic=galactic,
        galactic_bg=galactic_bg,
    )
    if muon_aeff:
        components["muon"] = surface_veto.MuonBundleBackground(muon_aeff, 1)
    if angular_smoothing_fwhm is None:
        return components
    else:
        return {
            k: AngularSmearing(v, angular_smoothing_fwhm) for k, v in components.items()
        }


@lru_cache(maxsize=4)
def create_bundle(
    exposures,
    galactic_emission=diffuse.FermiGalacticEmission,
    angular_smoothing_fwhm=None,
    null_hypothesis=None,
):
    return factory.component_bundle(
        dict(exposures),
        partial(
            make_components,
            galactic_emission=galactic_emission,
            angular_smoothing_fwhm=angular_smoothing_fwhm,
            null_hypothesis=null_hypothesis,
        ),
    )


def asimov_llh(
    exposures,
    astro=2.3,
    astro_gamma=-2.5,
    galactic_emission=diffuse.FermiGalacticEmission,
    angular_smoothing_fwhm=None,
    null_hypothesis=None,
):
    components = create_bundle(
        exposures,
        galactic_emission,
        angular_smoothing_fwhm,
        null_hypothesis,
    ).get_components()
    components["gamma"] = multillh.NuisanceParam(astro_gamma)
    return multillh.asimov_llh(
        components, astro=astro, gamma=astro_gamma, galactic_bg=0, galactic=1
    )


def binning():
    factory.set_kwargs(
        psi_bins={k: (0, np.pi) for k in ("tracks", "cascades", "radio")}, cos_theta=16
    )


def make_galactic_emission(kind: Literal["fermi", "kra_5", "kra_50", "kra_powerlaw"]):
    if kind == "fermi":
        galactic_emission = diffuse.FermiGalacticEmission
    elif kind == "kra_5":
        galactic_emission = diffuse.KRAGalacticDiffuseEmission
    elif kind == "kra_50":
        galactic_emission = partial(diffuse.KRAGalacticDiffuseEmission, cutoff_PeV=50)
    elif kind == "kra_powerlaw":
        galactic_emission = partial(diffuse.KRAGalacticDiffuseEmission, cutoff_PeV=None)
    else:
        raise ValueError("unknown model {}".format(galactic_emission))
    return galactic_emission


@figure_data(setup=binning)
def fermi_pi0(
    exposures,
    astro=2.3,
    astro_gamma=-2.5,
    galactic_norm: float = 1,
    galactic_emission: Literal["fermi", "kra_5", "kra_50", "kra_powerlaw"] = "fermi",
    null_hypothesis: Literal["ra_scrambled", "kra_5"] = "ra_scrambled",
    angular_smoothing: Literal["dnn", "monopod", "none"] = "none",
    angular_resolution_scale: float = 1.0,
    free_components: List[str] = [],
):
    galactic_emission = make_galactic_emission(galactic_emission)
    if angular_smoothing == "dnn":
        angular_smoothing_fwhm = get_dnn_smoothing
    elif angular_smoothing == "monopod":
        angular_smoothing_fwhm = get_monopod_smoothing
    else:
        angular_smoothing_fwhm = None
    if null_hypothesis == "ra_scrambled":
        null_hypothesis = None
    else:
        null_hypothesis = make_galactic_emission(null_hypothesis)
    llh = asimov_llh(
        exposures,
        astro=astro,
        astro_gamma=astro_gamma,
        galactic_emission=galactic_emission,
        angular_smoothing_fwhm=(
            lambda *args: angular_resolution_scale * angular_smoothing_fwhm(*args)
        )
        if angular_smoothing_fwhm
        else None,
        null_hypothesis=null_hypothesis,
    )
    nominal = {k: v.seed for k, v in llh.components.items()}
    nominal.update({"astro": astro, "gamma": astro_gamma, "atmo": 1})
    # h0 = llh.fit(galactic=0, gamma=-2.5, minimizer_params=dict(epsilon=1e-2))
    # h1 = llh.fit(gamma=-2.5, minimizer_params=dict(epsilon=1e-2))

    # NB: galactic_bg is an RA-scrambled version of the galactic template, so
    # that h0 and h1 have the same energy and declination distribution, but
    # different RA distributions
    nullhypo = nominal | {"galactic": 0, "galactic_bg": 1}
    if free_components:
        fixed = nominal | {"galactic": 0, "galactic_bg": 1}
        for k in free_components:
            fixed.pop(k)
        nullhypo = llh.fit(**fixed)
    h0 = {k: v.sum(axis=1) for k, v in llh.llh_contributions(**nullhypo).items()}
    bestfit = nominal | {"galactic": 1, "galactic_bg": 0}
    h1 = {k: v.sum(axis=1) for k, v in llh.llh_contributions(**bestfit).items()}
    # sum over energies
    meta = {
        "rates": {
            component: {k: v.sum(axis=1) for k, v in expectations.items()}
            for component, expectations in multillh.get_expectations(
                llh,
                astro=astro,
                gamma=astro_gamma,
                galactic=galactic_norm,
                galactic_bg=0,
            ).items()
        },
        "ts": {k: 2 * (h1[k] - h0[k]) for k in h0.keys()},
    }
    meta["best_fit"] = bestfit
    meta["null_hypothesis"] = nullhypo
    meta["nominal"] = nominal

    components = create_bundle(exposures, galactic_emission).get_components()
    components["gamma"] = multillh.NuisanceParam(astro_gamma)
    source = components.pop("galactic")
    # FIXME: how to calculate an upper limit with isotropized galactic component?
    # ul, ns, nb = pointsource.upper_limit(source, components, **nominal)
    # meta["upper_limit"] = ul

    exes = multillh.get_expectations(
        llh, astro=astro, gamma=astro_gamma, galactic=galactic_norm
    )

    channel = next(iter(exes["galactic"].keys()))
    meta["spectrum"] = {
        "energy": list(source.bin_edges.values())[0][1],
        "signal": exes["galactic"][channel].sum(axis=0),
        "background": sum(
            exes[source][channel].sum(axis=0)
            for source in set(exes.keys()).difference({"galactic"})
        ),
    }

    return meta


@figure
def rates(datasets):
    import healpy
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from toise import plotting

    dataset = datasets[0]
    # print(dataset['data']['ts'])
    rates = dataset["data"]["rates"]
    bg = sum(
        [
            np.asarray(rates[k][channel])
            for k in rates
            if (k != "galactic" and k != "muon")
            for channel in rates[k]
        ]
    )
    sig = sum(
        [
            np.asarray(rates[k][channel])
            for k in rates
            if k == "galactic"
            for channel in rates[k]
        ]
    )

    fig, axes = plt.subplots(2, 2, figsize=(6, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    ts = sum(
        [np.asarray(dataset["data"]["ts"][k]) for k in dataset["data"]["ts"].keys()]
    )
    # galactic = sum(map(np.asarray, dataset['data']['ts']['muon'].values()))
    # print(galactic)

    plt.sca(axes[0, 0])

    colat, _ = healpy.pix2ang(healpy.npix2nside(len(bg)), np.arange(len(bg)))

    mask = bg > 200
    # select northern hemisphere only
    mask = np.cos(colat) > 0
    mask = slice(None)

    healpy.mollview(
        np.where(mask, bg, np.nan),
        hold=True,
        title="Background: {:.0f} events".format(bg[mask].sum()),
        unit="Events",
        format="%.2f",
    )

    plt.sca(axes[0, 1])
    northern_sig = np.where(mask, sig, np.nan)
    # XXX disable masking
    # northern_sig = sig
    healpy.mollview(
        northern_sig,
        hold=True,
        title="Signal: {:.0f} events".format(np.nansum(northern_sig)),
        unit="Events",
        format="%.2f",
    )

    plt.sca(axes[1, 1])
    total_ts = np.nansum(ts)
    total_ts = np.nansum(ts[mask])
    healpy.mollview(
        np.where(mask, ts, np.nan),
        hold=True,
        title="Discovery significance: {:.1f} $\sigma$".format(np.sqrt(total_ts)),
        unit=r"$-2\Delta \ln L$",
        format="%.2f",
    )
    # plt.title(r"{:.1f} $\sigma$".format(np.sqrt(ts.sum())))

    ax = axes[1, 0]
    spec = dataset["data"]["spectrum"]
    s = np.asarray(spec["signal"])[::-1].cumsum()[::-1]
    b = np.asarray(spec["background"])[::-1].cumsum()[::-1]
    ax.loglog(
        *plotting.stepped_path(spec["energy"], s),
        nonpositive="clip",
        label="signal",
    )
    ax.loglog(
        *plotting.stepped_path(spec["energy"], b),
        nonpositive="clip",
        label="background",
    )
    ax.set_xlabel("E_min")
    ax.set_ylabel("N(E > E_min)")
    ax.set_ylim(bottom=1e-1)
    ax.set_xlim(right=1e6, left=5e2)
    ax.legend(fontsize="x-small")

    ax2 = plt.twinx(ax)
    ax2.semilogx(
        *plotting.stepped_path(spec["energy"], s / np.sqrt(b)),
        nonpositive="clip",
        label="total",
        color="grey",
    )
    ax2.set_ylabel(r"$S/\sqrt{B}$")

    # rejigger mollview for tighter display
    for ax in fig.axes:
        if "colorbar" in ax.get_label():
            t = ax.texts[0]
            t.set_fontsize("small")
            t.set_va("top")
            for l in ax.xaxis.get_ticklabels():
                l.set_fontsize("small")
        elif title := ax.get_title():
            ax.set_title(title, fontsize="medium")

    # plt.title(r"{:.1f} $\sigma$".format(np.sqrt(ts.sum())))

    return fig
