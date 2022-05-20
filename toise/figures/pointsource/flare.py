import re
from collections import OrderedDict
from copy import copy
from functools import partial

import numpy as np
from toise import (
    diffuse,
    factory,
    figures_of_merit,
    multillh,
    pointsource,
    surface_veto,
    util,
)
from toise.figures import figure, figure_data
from tqdm import tqdm
from toise import radio_aeff_generation

try:
    from typing import List
except ImportError:
    List = list


def point_source_binning():
    """set cos_zenith binning to center on the horizon"""
    factory.set_kwargs(
        cos_theta=np.concatenate(([-1], np.linspace(-0.95, 0.95, 20), [1])),
    )


@figure_data(setup=point_source_binning)
def sensitivity(exposures, decades=1, gamma=-2, emin=0.0):
    figures = OrderedDict(
        [
            ("differential_upper_limit", figures_of_merit.DIFF.ul),
            ("differential_discovery_potential", figures_of_merit.DIFF.dp),
            ("upper_limit", figures_of_merit.TOT.ul),
            ("discovery_potential", figures_of_merit.TOT.dp),
        ]
    )
    assert (
        len({exposure for detector, exposure in exposures}) == 1
    ), "exposures are equal"
    meta = {
        "cos_zenith": factory.default_cos_theta_bins #aeff_factory.instance.get_kwargs(exposures[0][0])[
        #    "cos_theta"
        #]
    }
    dlabel = "+".join([detector for detector, _ in exposures])
    for zi in tqdm(list(range(len(meta["cos_zenith"]) - 1)), desc=dlabel):
        fom = figures_of_merit.PointSource(
            {detector: exposure for detector, exposure in exposures}, zi
        )
        for flabel, q in figures.items():
            kwargs = {"gamma": gamma, "decades": decades}
            if not flabel.startswith("differential"):
                kwargs["emin"] = emin
            val = fom.benchmark(q, **kwargs)
            if not flabel in meta:
                meta[flabel] = {}
            if not dlabel in meta[flabel]:
                meta[flabel][dlabel] = {}
            if flabel.startswith("differential"):
                val = OrderedDict(
                    zip(("e_center", "flux", "n_signal", "n_background"), val)
                )
            else:
                val = OrderedDict(zip(("flux", "n_signal", "n_background"), val))
            meta[flabel][dlabel][str(zi)] = val
    return meta


@figure
def sensitivity(datasets):
    import matplotlib.pyplot as plt

    fig = plt.gcf()
    ax = plt.gca()
    for dataset in datasets:
        cos_theta = np.asarray(dataset["data"]["cos_zenith"])
        xc = -util.center(cos_theta)
        for detector in dataset["data"]["discovery_potential"].keys():
            yc = np.full(xc.shape, np.nan)
            for k in dataset["data"]["discovery_potential"][detector].keys():
                # items are flux, ns, nb
                yc[int(k)] = dataset["data"]["discovery_potential"][detector][k]["flux"]
            ax.semilogy(xc, yc * 1e-12, label=detector)
    ax.set_xlabel(r"$\sin \delta$")
    ax.set_ylabel(
        r"$E^2 \Phi_{\nu_x + \overline{\nu_x}}$ $(\rm TeV \,\, cm^{-2} s^{-1})$"
    )
    ax.legend()


@figure_data()
def single_flare_time_to_signficance(
    exposures,
    flux=1.6e-15,
    gamma=-2.1,
    dec=5.69,
    emin=0.0,
    emax=np.inf,
    days: List[float] = [3, 10, 20, 50, 100, 158, 200, 300, 500],
    duration_penalty: bool = True,
):
    """
    :param flux: average flare flux in TXS paper units (TeV^-1 cm^-2 s^-2 at 100 TeV)
    :param dec: source declination in degrees
    """
    from toise.multillh import asimov_llh
    from toise.pointsource import nevents

    # convert to our flux units (1e-12 TeV^-1 cm^-2 s^-2 at 1 TeV)
    flux_norm = flux * (100.0**-gamma) / 1e-12

    cos_theta = np.linspace(-1, 1, 41)
    factory.set_kwargs(
        cos_theta=cos_theta,
        # psi_bins={k: (0, np.radians(1)) for k in ('tracks', 'cascades', 'radio')},
    )
    decs = np.degrees(np.arcsin(-cos_theta)[::-1])
    zi = len(decs) - decs.searchsorted(dec) - 1

    def make_components(zi, aeffs):
        aeff, muon_aeff = aeffs
        atmo = diffuse.AtmosphericNu.conventional(aeff, 1.0, veto_threshold=None)
        atmo.uncertainty = 0.1
        prompt = diffuse.AtmosphericNu.prompt(aeff, 1.0, veto_threshold=None)
        prompt.min = 0.5
        prompt.max = 3
        astro = diffuse.DiffuseAstro(aeff, 1.0)
        astro.seed = 2
        ps = pointsource.SteadyPointSource(aeff, 1, zenith_bin=zi, emin=emin, emax=emax)
        atmo_bkg = atmo.point_source_background(zenith_index=zi)
        prompt_bkg = prompt.point_source_background(zenith_index=zi)
        astro_bkg = astro.point_source_background(zenith_index=zi)

        components = dict(atmo=atmo_bkg, prompt=prompt_bkg, astro=astro_bkg, ps=ps)
        if muon_aeff is not None:
            # optical
            components["muon"] = surface_veto.MuonBundleBackground(
                muon_aeff, 1
            ).point_source_background(zenith_index=zi, psi_bins=aeff.bin_edges[-1][:-1])
            if min(aeff.get_bin_edges("true_energy"))>1e5:
                print("using radio muon background")
                components["muon"] = radio_aeff_generation.MuonBackground(
                      muon_aeff, 1
                ).point_source_background(zenith_index=zi, psi_bins=aeff.bin_edges[-1][:-1])
        return components

    bundle = factory.component_bundle(dict(exposures), partial(make_components, zi))

    def ts(flare_time, total_time):

        # NB: hypotheses are only compared over the flare time. the total exposure
        # time enters via an optional penalty term
        components = bundle.get_components({exposures[0][0]: flare_time})
        flux_levels = {k: v.seed for k, v in components.items()}
        flux_levels["ps"] = flux_norm

        components["gamma"] = multillh.NuisanceParam(-2.1, 0.5, min=-2.7, max=-1.7)
        components["ps_gamma"] = multillh.NuisanceParam(gamma, 0.5, min=-2.7, max=-1.7)
        gamma_kwargs = {
            k: components[k].seed
            for k in set(components.keys()).difference(list(flux_levels.keys()))
        }

        fixed = dict(flux_levels)
        fixed.update(gamma_kwargs)
        allh = asimov_llh(components, **fixed)
        fixed.pop("ps")
        if False:
            fixed["ps_gamma"] = list(np.linspace(-3.5, -1.5, 41))
            print(fixed)
            null = allh.fit(ps=0, **fixed)
            # print(components['ps'].seed)
            # null = dict(fixed)
            # null['ps'] = 0
            alternate = allh.fit(**fixed)
        else:
            null = dict(fixed)
            null["ps"] = 0
            alternate = dict(fixed)
            alternate["ps"] = flux_norm
        print(null)
        print(alternate)
        nb = nevents(allh, **null)
        ns = nevents(allh, **alternate) - nb

        return {
            "ts": -2
            * (
                allh.llh(**null)
                - allh.llh(**alternate)
                - (np.log(flare_time / total_time) if duration_penalty else 0.0)
            ),
            "ns": ns,
            "nb": nb,
        }
        # return -2*(allh.llh(ps=0, **fixed)-allh.llh(ps=flux_norm, **fixed))

    return {"days": days, "ts": [ts(dt / 365.0, exposures[0][1]) for dt in days]}


def get_effective_dof(ts, target_pvalue):
    from scipy import stats
    from scipy.optimize import fsolve

    return fsolve(lambda dof: stats.chi2.sf(ts, dof) - target_pvalue, 4)[0]


@figure
def single_flare_time_to_signficance(datasets):
    """
    :param effective_dof:
    """
    import matplotlib.pyplot as plt
    from scipy import stats

    fig = plt.gcf()
    ax = plt.gca()
    effective_dof = None
    for dataset in datasets:
        x = dataset["data"]["days"]
        ts = np.asarray([item["ts"] for item in dataset["data"]["ts"]])

        # the time-dependent TS, with its penalty term for the flare length, is
        # not necessarily chi^2 distributed with 1 degree of freedom. Find the
        # effective number of degrees of freedom that make the p-value of a 158
        # day box flare (i.e. TXS-ish) 2e-4 as quoted in https://arxiv.org/abs/1807.08794
        # This is inspired by the observation that the test statistic D in Fig.
        # 3a of https://arxiv.org/abs/0912.1572v1 is distributed like a chi^2
        # with ~4 dof (also, shifted left by 5 units).
        # Since only the spatial/energy pdf changes between detector
        # configurations, we use the same TS distribution to translate p-values
        # for all flare durations.
        if effective_dof is None:
            assert dataset["detectors"][0][0].startswith("IceCube")
            baseline = ts[x.index(158)]
            effective_dof = get_effective_dof(baseline, 2e-4)
            print(("--> effective d.o.f.: {:.1f}".format(effective_dof)))
        corrected_ts = stats.chi2(1).isf(stats.chi2.sf(ts, effective_dof))
        # plt.plot(x, np.sqrt(ts), label='raw significance')
        # plt.plot(x, np.sqrt(corrected_ts), label='effective trials correction')
        detector = dataset["detectors"][0][0]
        if detector.startswith("IceCube"):
            label = "IceCube"
        elif detector.startswith("Gen2"):
            label = "IceCube-Gen2"
        mask = corrected_ts > 0
        mask[-1] = False
        line = plt.plot(np.asarray(x)[mask], np.sqrt(corrected_ts)[mask], label=label)[
            0
        ]
    ax.set_xlabel("Flare duration (days)")
    ax.set_ylabel(r"Significance ($\sigma$)")
    ax.set_title("TXS-like flare")
    # ax.axvline(158, color='grey', ls='--', zorder=-1, lw=0.5)
    for level in (3, 5):
        ax.axhline(level, color="grey", zorder=-1, lw=0.5)

    ax.legend()
    return fig
