
from . import factory, diffuse, pointsource, multillh
import numpy

class GZK(object):

    def __init__(self, exposures):
        self.bundle = factory.component_bundle(exposures, self.make_components)
        
        
    def sensitivity(self):
        components = self.bundle.get_components()
        components['gamma'] =  multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        gzk = components.pop('gzk')
        
        baseline = 100

        scale = pointsource.discovery_potential(gzk, components,
            baseline=baseline, tolerance=1e-4, gamma=-2.3)

        return scale
    
    def event_numbers(self):
        scale = self.sensitivity()
        
        components = self.bundle.get_components()
        components['gamma'] =  multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)
        llh = multillh.asimov_llh(components)
        
        bin_edges = components['gzk'].bin_edges
        
        def pev_events(observables):
            n = 0
            for k, edges in bin_edges.items():
                pev = numpy.where(edges[1][1:] > 5e7)[0][0]
                n += observables[k].sum(axis=0)[pev:].sum()
                
            return n
        
        exes = multillh.get_expectations(llh, gzk=scale, gamma=-2.3)
        nb = pev_events(exes['atmo']) + pev_events(exes['astro'])
        ns = pev_events(exes['gzk'])
        return dict(ns=ns, nb=nb)
        
    @staticmethod
    def make_components(aeff):
        energy_threshold = numpy.inf
        atmo = diffuse.AtmosphericNu.conventional(aeff, 1., hard_veto_threshold=energy_threshold)
        atmo.prior = lambda v, **kwargs: -(v-1)**2/(2*0.1**2)
        prompt = diffuse.AtmosphericNu.prompt(aeff, 1., hard_veto_threshold=energy_threshold)
        prompt.min = 0.5
        prompt.max = 3
        astro = diffuse.DiffuseAstro(aeff, 1.)
        astro.seed = 2
        gzk = diffuse.AhlersGZK(aeff, 1.)
        return dict(atmo=atmo, prompt=prompt, astro=astro, gzk=gzk)

