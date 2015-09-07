
import numpy
import itertools
from scipy.integrate import quad
from copy import copy
import multillh

class DiffuseNuGen(object):
	def __init__(self, effective_area, edges, flux, livetime=1.):
		"""
		:param effective_area: effective area in m^2
		:param edges: edges of the bins in the effective area histogram
		:param flux: flux in 1/(m^2 yr), integrated over the edges in *edges*
		:param livetime: observation time in years
		"""
		self._aeff = effective_area
		self._livetime = livetime
		self._edges = edges
		# dimensions of flux should be: nu type (6),
		# nu energy, cos(nu zenith), reco energy, cos(reco zenith),
		# signature (cascade/track)
		self._flux = flux
		
		self._solid_angle = 2*numpy.pi*numpy.diff(edges[1])
		
		self.seed = 1.
		self.uncertainty = None
	
	def prior(self, value):
		if self.uncertainty is None:
			return 0.
		else:
			return -(value-self.seed)**2/(2*self.uncertainty**2)
	
	@staticmethod
	def _integrate_flux(edges, flux, passing_fraction=lambda *args, **kwargs: 1.):
		from icecube import dataclasses
		intflux = numpy.empty((6, len(edges[0])-1, len(edges[1])-1))
		for i, (flavor, anti) in enumerate(itertools.product(('E', 'Mu', 'Tau'), ('', 'Bar'))): 
			pt = getattr(dataclasses.I3Particle, 'Nu'+flavor+anti)
			for j in range(len(edges[1])-1):
				ct_hi = edges[1][j+1]
				ct_lo = edges[1][j]
				ct = (ct_lo + ct_hi)/2.
				fluxband = numpy.zeros(len(edges[0])-1)
				for k in range(len(fluxband)):
					fluxband[k] = quad(lambda e: flux(pt, e, ct)*passing_fraction(pt, e, ct, depth=2e3), edges[0][k], edges[0][k+1])[0]
				intflux[i,:,j] = fluxband*2*numpy.pi*(ct_hi-ct_lo)
		# return integrated flux in 1/(m^2 yr)
		return intflux*1e4*(3600*24*365)

class AtmosphericNu(DiffuseNuGen):
	def __init__(self, effective_area, edges, flux_func, passing_fraction, livetime):
		if passing_fraction is not None:
			flux = self._integrate_flux(edges, flux_func.getFlux, passing_fraction)
		else:
			flux = self._integrate_flux(edges, flux_func.getFlux)
		super(AtmosphericNu, self).__init__(effective_area, edges, flux, livetime)
		
		# sum over neutrino flavors, energies, and zenith angles
		total = (self._flux[...,None,None,None]*self._aeff*self._livetime).sum(axis=(0,1,2))
		# dimensions of the keys in expectations are now reconstructed energy, zenith
		self.expectations = dict(cascades=total[...,0], tracks=total[...,1])
	
	def point_source_background(self, psi_bins, zenith_index, livetime=None, with_energy=True):
		"""
		convert to a point source background
		
		:param bin_areas: areas (in sr) of the search bins around the putative source
		:param livetime: if not None, the actual livetime to integrate over in seconds
		"""
		background = copy(self)
		bin_areas = (numpy.pi*numpy.diff(psi_bins**2))[None,...]
		# observation time shorter for triggered transient searches
		if livetime is not None:
			bin_areas *= (livetime/self._livetime/(3600.*24*365.))
		# dimensions of the keys in expectations are now energy, radial bin
		background.expectations = {k: (v[:,zenith_index]/self._solid_angle[zenith_index])[...,None]*bin_areas for k,v in self.expectations.items()}
		if not with_energy:
			# just radial bins
			background.expectations = {k: v.sum(axis=0) for k,v in background.expectations.items()}
		return background
	
	@classmethod
	def conventional(cls, effective_area, edges, livetime, veto_threshold=1e3):
		from icecube import NewNuFlux, AtmosphericSelfVeto
		flux = NewNuFlux.makeFlux('honda2006')
		flux.knee_reweighting_model = 'gaisserH3a_elbert'
		pf = None if veto_threshold is None else AtmosphericSelfVeto.AnalyticPassingFraction(kind='conventional', veto_threshold=veto_threshold)
		return cls(effective_area, edges, flux, pf, livetime)

	@classmethod
	def prompt(cls, effective_area, edges, livetime, veto_threshold=1e3):
		from icecube import NewNuFlux, AtmosphericSelfVeto
		flux = NewNuFlux.makeFlux('sarcevic_std')
		flux.knee_reweighting_model = 'gaisserH3a_elbert'
		pf = None if veto_threshold is None else AtmosphericSelfVeto.AnalyticPassingFraction(kind='charm', veto_threshold=veto_threshold)
		return cls(effective_area, edges, flux, pf, livetime)
 
class DiffuseAstro(DiffuseNuGen):
	def __init__(self, effective_area, edges, livetime):
		# reference flux is E^2 Phi = 1e-8 GeV^2 cm^-2 sr^-1 s^-1
		flux = self._integrate_flux(edges, lambda pt, e, ct: 0.5e-18*(e/1e5)**(-2.))
		super(DiffuseAstro, self).__init__(effective_area, edges, flux, livetime)
	
	def point_source_background(self, psi_bins, zenith_index, livetime=None, with_energy=True):
		"""
		convert to a point source background
		
		:param bin_areas: areas (in sr) of the search bins around the putative source
		:param livetime: if not None, the actual livetime to integrate over in seconds
		"""
		background = copy(self)
		bin_areas = (numpy.pi*numpy.diff(psi_bins**2))[None,None,None,None,:,None]
		# observation time shorter for triggered transient searches
		if livetime is not None:
			bin_areas *= (livetime/self._livetime/(3600.*24*365.))
		# dimensions of the keys in expectations are now energy, radial bin
		
		# cut flux down to a single zenith band
		# dimensions of self._flux are flavor, energy, zenith
		sel = slice(zenith_index, zenith_index+1)
		# dimensions of flux are now 1/m^2 sr
		background._flux = (self._flux[:,:,sel]/self._solid_angle[zenith_index])
		# replace reconstructed zenith with opening angle
		# dimensions of aeff are now m^2 sr
		background._aeff = self._aeff[:,:,sel,:,sel,:]*bin_areas
		
		if not with_energy:
			raise ValueError("Can't disable energy dimension just yet")

		return background
	
	def expectations(self, gamma=-2, **kwargs):
		def intflux(e, gamma):
			return (1e5**(-gamma)/(1+gamma))*e**(1+gamma)
		energy = self._edges[0]
		# specweight = (intflux(energy[1:], gamma)-intflux(energy[:-1], gamma))/(intflux(energy[1:], -2)-intflux(energy[:-1], -2))
		
		centers = 0.5*(energy[1:] + energy[:-1])
		specweight = (centers/1e5)**(gamma+2)
		
		flux = (self._flux*(specweight[None,:,None]))[...,None,None,None]
		
		if 'e_fraction' in kwargs:
			flavor_weight = 3*numpy.ones(6)
			e, mu = kwargs['e_fraction'], kwargs['mu_fraction']
			# assert e+mu <= 1.
			flavor_weight[0:2] *= e
			flavor_weight[2:4] *= mu
			flavor_weight[4:6] *= (1. - e - mu)
			flux *= flavor_weight[:,None,None,None,None,None]
		
		total = (flux*self._aeff*self._livetime).sum(axis=(0,1,2))
		return dict(cascades=total[...,0], tracks=total[...,1])

def starting_diffuse_powerlaw(effective_area, edges, livetime=1.,
    flavor_ratio=False, veto_threshold=1e2):
	"""
	Contruct a likelihood for the istropic 1:1:1 power law hypothesis
	
	:param livetime: observation time, in years
	:param veto_threshold: muon energy threshold for atmospheric self-veto
	"""
	llh = multillh.LLHEval(None)
	llh.add_component('conventional', AtmosphericNu.conventional(effective_area, edges, livetime, veto_threshold=veto_threshold))
	llh.add_component('prompt', AtmosphericNu.prompt(effective_area, edges, livetime, veto_threshold=veto_threshold))
	llh.add_component('astro', DiffuseAstro(effective_area, edges, livetime))
	llh.add_component('gamma', multillh.NuisanceParam(-2, min=-4, max=-1))
	if flavor_ratio:
		# we only need two fractions to describe the flavor content
		for flavor in 'e', 'mu':
			llh.add_component(flavor+'_fraction', multillh.NuisanceParam(1./3, min=0, max=1))
	return llh

def throughgoing_diffuse_powerlaw(effective_area, edges, livetime=1.):
	llh = multillh.LLHEval(None)
	llh.add_component('conventional', AtmosphericNu.conventional(effective_area, edges, livetime, veto_threshold=None))
	llh.add_component('prompt', AtmosphericNu.prompt(effective_area, edges, livetime, veto_threshold=None))
	
	# TODO: add atmospheric muon component
	
	llh.add_component('astro', DiffuseAstro(effective_area, edges, livetime))
	llh.add_component('gamma', multillh.NuisanceParam(-2, min=-4, max=-1))
	return llh