from enum import Enum
from functools import partial
import numpy
from . import factory, diffuse, surface_veto, pointsource, multillh
from .util import constants
from . import radio_aeff_generation

# Enum has no facility for setting docstrings inline. Do it by hand.
TOT = Enum("TOT", ["ul", "dp", "fc"])
TOT.__doc__ = r"Total: integrate signal over entire energy range"
TOT.ul.__doc__ = r"90% upper limit (Wilks' Theorem)"
TOT.dp.__doc__ = r"5\sigma discover potential"
TOT.fc.__doc__ = r"90% upper limit (Feldman-Cousins construction)"

DIFF = Enum("DIFF", ["ul", "dp", "fc"])
DIFF.__doc__ = r"Differential: return figure of merit in energy bins"
DIFF.ul.__doc__ = r"90% upper limit (Wilks' Theorem)"
DIFF.dp.__doc__ = r"5\sigma discover potential"
DIFF.fc.__doc__ = r"90% upper limit (Feldman-Cousins construction)"


class GZK(object):
    def __init__(self, exposures):
        self.bundle = factory.component_bundle(exposures, self.make_components)

    def benchmark(self, fom, **kwargs):
        components = self.bundle.get_components()
        components["gamma"] = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        components["uhe_gamma"] = multillh.NuisanceParam(-2, 0.5, min=-2.7, max=-1.7)

        gzk = components.pop("gzk")
        uhe = components.pop("uhe")
        if "uhe_gamma" not in kwargs:
            kwargs["uhe_gamma"] = -2.0

        if fom == TOT.ul:
            return pointsource.upper_limit(
                gzk, components, baseline=100, tolerance=1e-4, gamma=-2.3, **kwargs
            )
        elif fom == TOT.dp:
            return pointsource.discovery_potential(
                uhe, components, baseline=100, tolerance=1e-4, gamma=-2.3, **kwargs
            )
        elif fom == TOT.fc:
            return pointsource.fc_upper_limit(gzk, components, gamma=-2.3, **kwargs)
        elif fom == DIFF.ul:
            return pointsource.differential_upper_limit(
                uhe, components, tolerance=1e-4, gamma=-2.3, **kwargs
            )
        elif fom == DIFF.dp:
            return pointsource.differential_discovery_potential(
                uhe, components, tolerance=1e-4, gamma=-2.3, **kwargs
            )
        elif fom == DIFF.fc:
            return pointsource.differential_fc_upper_limit(
                uhe, components, gamma=-2.3, **kwargs
            )
        else:
            raise RuntimeError("No such fom")

    def event_numbers(self, sigma=5):
        """Returns event numbers from ahlers flux needed to reject null
        hypothesis at sigma *sigma*
        """
        scale = self.benchmark(TOT.dp, sigma=sigma)

        components = self.bundle.get_components()
        components.pop("uhe")
        components["gamma"] = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        llh = multillh.asimov_llh(components)

        bin_edges = components["gzk"].bin_edges

        exes = multillh.get_expectations(llh, gzk=scale, gamma=-2.3)
        nb = pointsource.events_above(
            exes["atmo"], bin_edges
        ) + pointsource.events_above(exes["astro"], bin_edges)
        ns = pointsource.events_above(exes["gzk"], bin_edges)
        return dict(ns=ns, nb=nb)

    @staticmethod
    def make_components(aeffs):
        aeff, muon_aeff = aeffs
        energy_threshold = numpy.inf
        atmo = diffuse.AtmosphericNu.conventional(
            aeff, 1.0, hard_veto_threshold=energy_threshold
        )
        atmo.uncertainty = 0.1
        prompt = diffuse.AtmosphericNu.prompt(
            aeff, 1.0, hard_veto_threshold=energy_threshold
        )
        prompt.min = 0.5
        prompt.max = 3
        astro = diffuse.DiffuseAstro(aeff, 1.0)
        astro.seed = 2
        uhe = diffuse.DiffuseAstro(aeff, 1.0, gamma_name="uhe_gamma")
        gzk = diffuse.AhlersGZK(aeff, 1.0)
        return dict(atmo=atmo, prompt=prompt, astro=astro, gzk=gzk, uhe=uhe)
        # return dict(gzk=gzk, uhe=uhe)


class UHEFlux(object):
    """
    Sensitivity to an isotropic, 1:1:1 flux, ignoring background from known
    IceCube diffuse flux
    """

    def __init__(self, exposures):
        self.bundle = factory.component_bundle(exposures, self.make_components)

    def benchmark(self, fom, **kwargs):
        components = self.bundle.get_components()
        components["uhe_gamma"] = multillh.NuisanceParam(-2, 0.5, min=-2.7, max=-1.7)

        uhe = components.pop("uhe")
        if "uhe_gamma" not in kwargs:
            kwargs["uhe_gamma"] = -2.0
        if fom == DIFF.ul:
            return pointsource.differential_upper_limit(
                uhe, components, tolerance=1e-4, **kwargs
            )
        elif fom == DIFF.dp:
            return pointsource.differential_discovery_potential(
                uhe, components, tolerance=1e-4, **kwargs
            )
        elif fom == DIFF.fc:
            return pointsource.differential_fc_upper_limit(uhe, components, **kwargs)
        else:
            raise RuntimeError("No such fom")

    @staticmethod
    def make_components(aeffs):
        aeff, muon_aeff = aeffs
        energy_threshold = numpy.inf
        atmo = diffuse.AtmosphericNu.conventional(
            aeff, 1.0, hard_veto_threshold=energy_threshold
        )
        atmo.uncertainty = 0.1
        prompt = diffuse.AtmosphericNu.prompt(
            aeff, 1.0, hard_veto_threshold=energy_threshold
        )
        prompt.min = 0.5
        prompt.max = 3
        uhe = diffuse.DiffuseAstro(aeff, 1.0, gamma_name="uhe_gamma")
        components = dict(atmo=atmo, prompt=prompt, uhe=uhe)
        if muon_aeff is not None:
            if min(aeff.get_bin_edges("true_energy")) < 1e5:
                components["muon"] = surface_veto.MuonBundleBackground(muon_aeff, 1)
            else:
                components["muon"] = radio_aeff_generation.MuonBackground(muon_aeff, 1)
        return components


class PointSource(object):
    def __init__(self, exposures, zi):
        self.bundle = factory.component_bundle(
            exposures, partial(self.make_components, zi)
        )

    def benchmark(self, fom, gamma=-2.0, diff_gamma=-2.5, **kwargs):
        components = self.bundle.get_components()
        ps = components.pop("ps")

        decades = kwargs.pop("decades", 0.5)
        emin = kwargs.pop("emin", None)
        if len(kwargs) != 0:
            raise ValueError("Can't take kwargs")
        if emin is not None:
            #        if fom not in (TOT.dp, TOT.ul, TOT.fc):
            #            raise ValueError("emin argument not supported for FoM {}".format(fom))
            ecenter, ps = next(ps.differential_chunks(emin=emin, decades=1000))
        # assume all backgrounds known perfectly
        kwargs = {k: v.seed for k, v in components.items()}
        components["gamma"] = multillh.NuisanceParam(
            diff_gamma, 0.5, min=-2.7, max=-1.7
        )
        components["ps_gamma"] = multillh.NuisanceParam(gamma, 0.5, min=-2.7, max=-1.7)

        if fom == TOT.ul:
            ul, ns, nb = pointsource.upper_limit(
                ps,
                components,
                tolerance=1e-4,
                gamma=diff_gamma,
                ps_gamma=gamma,
                **kwargs
            )
            return ul, ns, nb
        elif fom == TOT.dp:
            dp, ns, nb = pointsource.discovery_potential(
                ps,
                components,
                tolerance=1e-4,
                gamma=diff_gamma,
                ps_gamma=gamma,
                **kwargs
            )
            return dp, ns, nb
        elif fom == TOT.fc:
            return pointsource.fc_upper_limit(
                ps, components, gamma=diff_gamma, ps_gamma=gamma, **kwargs
            )
        elif fom == DIFF.ul:
            return pointsource.differential_upper_limit(
                ps,
                components,
                gamma=diff_gamma,
                ps_gamma=gamma,
                tolerance=1e-4,
                decades=decades,
                emin=emin,
                **kwargs
            )
        elif fom == DIFF.dp:
            return pointsource.differential_discovery_potential(
                ps,
                components,
                gamma=gamma,
                ps_gamma=-2,
                tolerance=1e-4,
                decades=decades,
                emin=emin,
                **kwargs
            )
        else:
            raise RuntimeError("No such fom")

    @staticmethod
    def make_components(zi, aeffs):
        aeff, muon_aeff = aeffs
        atmo = diffuse.AtmosphericNu.conventional(aeff, 1.0, veto_threshold=None)
        atmo.uncertainty = 0.1
        prompt = diffuse.AtmosphericNu.prompt(aeff, 1.0, veto_threshold=None)
        prompt.min = 0.5
        prompt.max = 3
        astro = diffuse.DiffuseAstro(aeff, 1.0)
        astro.seed = 2
        ps = pointsource.SteadyPointSource(aeff, 1, zenith_bin=zi)
        atmo_bkg = atmo.point_source_background(zenith_index=zi)
        prompt_bkg = prompt.point_source_background(zenith_index=zi)
        astro_bkg = astro.point_source_background(zenith_index=zi)

        components = dict(atmo=atmo_bkg, prompt=prompt_bkg, astro=astro_bkg, ps=ps)
        if muon_aeff is not None:
            import numpy as np

            print("muon aeff shape", np.shape(muon_aeff.values))
            if min(aeff.get_bin_edges("true_energy")) < 1e5:
                components["muon"] = surface_veto.MuonBundleBackground(
                    muon_aeff, 1
                ).point_source_background(
                    zenith_index=zi, psi_bins=aeff.bin_edges[-1][:-1]
                )
            else:
                print("using radio muon background")
                components["muon"] = radio_aeff_generation.MuonBackground(
                    muon_aeff, 1
                ).point_source_background(
                    zenith_index=zi, psi_bins=aeff.bin_edges[-1][:-1]
                )
        return components
