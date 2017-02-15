from enum import Enum
from functools import partial
import numpy
from . import factory, diffuse, pointsource, multillh

TOT = Enum('TOT', ['ul', 'dp', 'fc'])
DIFF = Enum('DIFF', ['ul', 'dp', 'fc'])

class GZK(object):

    def __init__(self, exposures):
        self.bundle = factory.component_bundle(exposures, self.make_components)

    def benchmark(self, fom, **kwargs):
        components = self.bundle.get_components()
        components['gamma'] =  multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        components['uhe_gamma'] =  multillh.NuisanceParam(-2, 0.5, min=-2.7, max=-1.7)

        gzk = components.pop('gzk')
        uhe = components.pop('uhe')
        if 'uhe_gamma' not in kwargs:
            kwargs['uhe_gamma'] = -2.

        if fom == TOT.ul:
            return pointsource.upper_limit(gzk,
                                           components,
                                           baseline=100,
                                           tolerance=1e-4,
                                           gamma=-2.3,
                                           **kwargs)
        elif fom == TOT.dp:
            return pointsource.discovery_potential(gzk,
                                                   components,
                                                   baseline=100,
                                                   tolerance=1e-4,
                                                   gamma=-2.3,
                                                   **kwargs)
        elif fom == TOT.fc:
            return pointsource.fc_upper_limit(gzk,
                                              components,
                                              gamma=-2.3,
                                              **kwargs)
        elif fom == DIFF.ul:
            return pointsource.differential_upper_limit(uhe,
                                                        components,
                                                        tolerance=1e-4,
                                                        gamma=-2.3,
                                                        **kwargs)
        elif fom == DIFF.dp:
            return pointsource.differential_discovery_potential(uhe,
                                                                components,
                                                                tolerance=1e-4,
                                                                gamma=-2.3,
                                                                **kwargs)
        elif fom == DIFF.fc:
            return pointsource.differential_fc_upper_limit(uhe,
                                                           components,
                                                           gamma=-2.3,
                                                           **kwargs)
        else:
            raise RuntimeError('No such fom')


    def event_numbers(self, sigma=5):
        """ Returns event numbers from ahlers flux needed to reject null
        hypothesis at sigma *sigma*
        """
        scale = self.benchmark(TOT.dp, sigma=sigma)
        
        components = self.bundle.get_components()
        components.pop('uhe')
        components['gamma'] =  multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        llh = multillh.asimov_llh(components)
        
        bin_edges = components['gzk'].bin_edges
        
        exes = multillh.get_expectations(llh, gzk=scale, gamma=-2.3)
        nb = pointsource.events_above(exes['atmo'], bin_edges) + pointsource.events_above(exes['astro'], bin_edges)
        ns = pointsource.events_above(exes['gzk'], bin_edges)
        return dict(ns=ns, nb=nb)


    @staticmethod
    def make_components(aeff):
        energy_threshold = numpy.inf
        atmo = diffuse.AtmosphericNu.conventional(aeff, 1., hard_veto_threshold=energy_threshold)
        atmo.uncertainty = 0.1
        prompt = diffuse.AtmosphericNu.prompt(aeff, 1., hard_veto_threshold=energy_threshold)
        prompt.min = 0.5
        prompt.max = 3
        astro = diffuse.DiffuseAstro(aeff, 1.)
        astro.seed = 2
        uhe = diffuse.DiffuseAstro(aeff, 1., gamma_name='uhe_gamma')
        gzk = diffuse.AhlersGZK(aeff, 1.)
        return dict(atmo=atmo, prompt=prompt, astro=astro, gzk=gzk, uhe=uhe)


class PointSource(object):
    def __init__(self, exposures, zi):
        self.bundle = factory.component_bundle(exposures, partial(self.make_components, zi))


    def benchmark(self, fom, **kwargs):
        components = self.bundle.get_components()
        components['gamma'] =  multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        ps = components.pop('ps')
        if fom == TOT.ul:
            return pointsource.upper_limit(ps,
                                           components,
                                           tolerance=1e-4,
                                           gamma=-2.3,
                                           **kwargs)
        elif fom == TOT.dp:
            return pointsource.discovery_potential(ps,
                                                   components,
                                                   tolerance=1e-4,
                                                   gamma=-2.3,
                                                   **kwargs)
        elif fom == TOT.fc:
            return pointsource.fc_upper_limit(ps,
                                              components,
                                              gamma=-2.3,
                                              **kwargs)
        else:
            raise RuntimeError('No such fom')


    @staticmethod
    def make_components(zi, aeff):
        energy_threshold = numpy.inf
        atmo = diffuse.AtmosphericNu.conventional(aeff, 1., hard_veto_threshold=energy_threshold)
        atmo.uncertainty = 0.1
        prompt = diffuse.AtmosphericNu.prompt(aeff, 1., hard_veto_threshold=energy_threshold)
        prompt.min = 0.5
        prompt.max = 3
        astro = diffuse.DiffuseAstro(aeff, 1.)
        astro.seed = 2
        ps = pointsource.SteadyPointSource(aeff, 1, zenith_bin=zi)
        atmo_bkg = atmo.point_source_background(zenith_index=zi)
        prompt_bkg = prompt.point_source_background(zenith_index=zi)
        astro_bkg = astro.point_source_background(zenith_index=zi)
        return dict(atmo=atmo_bkg, prompt=prompt_bkg, astro=astro_bkg, ps=ps)
