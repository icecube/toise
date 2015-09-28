
import numpy
import itertools
from scipy.integrate import quad
from copy import copy
import multillh
import healpy
import os
import numexpr

from util import *

class DiffuseNuGen(object):
	def __init__(self, effective_area, flux, livetime=1.):
		"""
		:param effective_area: effective area in m^2
		:param edges: edges of the bins in the effective area histogram
		:param flux: flux in 1/(m^2 yr), integrated over the edges in *edges*
		:param livetime: observation time in years
		"""
		self._aeff = effective_area
		self._livetime = livetime
		# dimensions of flux should be: nu type (6),
		# nu energy, cos(nu zenith), reco energy, cos(reco zenith),
		# signature (cascade/track)
		self._flux = flux
		
		# FIXME: account for healpix binning
		self._solid_angle = 2*numpy.pi*numpy.diff(self._aeff.bin_edges[1])
		
		self.seed = 1.
		self.uncertainty = None
	
	@property
	def is_healpix(self):
		return self._aeff.is_healpix
		
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
				intflux[i,:,j] = fluxband
		# return integrated flux in 1/(m^2 yr sr)
		return intflux*1e4*(3600*24*365)

	# apply 3-element multiplication + reduction without creating too many
	# unnecessary temporaries. numexpr allows a single reduction, so do it here
	_reduce_flux = numexpr.NumExpr('sum(aeff*flux*livetime, axis=2)')
	def _apply_flux(self, effective_area, flux, livetime):
		return self._reduce_flux(effective_area.values, flux[...,None,None,None], livetime).sum(axis=(0,1))

class AtmosphericNu(DiffuseNuGen):
	def __init__(self, effective_area, flux, livetime, hard_veto_threshold=None):
		
		if isinstance(flux, tuple):
			flux_func, passing_fraction = flux
			if passing_fraction is not None:
				flux = self._integrate_flux(effective_area.bin_edges, flux_func.getFlux, passing_fraction)
			else:
				flux = self._integrate_flux(effective_area.bin_edges, flux_func.getFlux)
		
			# "integrate" over solid angle
			if effective_area.is_healpix:
				flux *= healpy.nside2pixarea(effective_area.nside)
			else:
				flux *= (2*numpy.pi*numpy.diff(effective_area.bin_edges[1]))[None,None,:]
		else:
			# flux was precalculated
			pass
		
		super(AtmosphericNu, self).__init__(effective_area, flux, livetime)
		
		# sum over neutrino flavors, energies, and zenith angles
		total = self._apply_flux(self._aeff, self._flux, self._livetime)
		
		if hard_veto_threshold is not None:
			e_mu, cos_theta = effective_area.bin_edges[2:4]
			scale = numpy.where(~hard_veto_threshold.veto(*numpy.meshgrid(center(e_mu), center(cos_theta), indexing='ij')), 1, 1e-4)
			total *= scale[...,None]
		
		# up to now we've assumed that everything is azimuthally symmetric and
		# dealt with zenith bins/healpix rings. repeat the values in each ring
		# to broadcast onto a full healpix map.
		if self.is_healpix:
			total = total.repeat(self._aeff.ring_repeat_pattern, axis=1)
		# dimensions of the keys in expectations are now reconstructed energy, sky bin (zenith/healpix pixel)
		self.expectations = dict(cascades=total[...,0], tracks=total[...,1])
	
	def point_source_background(self, psi_bins, zenith_index, livetime=None, with_energy=True):
		"""
		convert to a point source background
		
		:param bin_areas: areas (in sr) of the search bins around the putative source
		:param livetime: if not None, the actual livetime to integrate over in seconds
		"""
		
		assert not self.is_healpix, "Don't know how to make PS backgrounds from HEALpix maps yet"
		
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
	
	_fluxes = dict(conventional=dict(), prompt=dict())
	@classmethod
	def conventional(cls, effective_area, livetime, veto_threshold=1e3, hard_veto_threshold=None):
		from icecube import NewNuFlux, AtmosphericSelfVeto
		cache = cls._fluxes['conventional']
		if veto_threshold in cache and cache[veto_threshold][0].compatible_with(effective_area):
			flux = cache[veto_threshold][1]
		else:
			assert len(cls._fluxes['conventional']) == 0
			flux = NewNuFlux.makeFlux('honda2006')
			flux.knee_reweighting_model = 'gaisserH3a_elbert'
			pf = None if veto_threshold is None else AtmosphericSelfVeto.AnalyticPassingFraction(kind='conventional', veto_threshold=veto_threshold)
			flux = (flux, pf)
		instance = cls(effective_area, flux, livetime, hard_veto_threshold)
		if isinstance(flux, tuple):
			cache[veto_threshold] = (effective_area, instance._flux)
		assert len(cls._fluxes['conventional']) > 0 
		
		return instance
	
	@classmethod
	def prompt(cls, effective_area, livetime, veto_threshold=1e3, hard_veto_threshold=None):
		from icecube import NewNuFlux, AtmosphericSelfVeto
		cache = cls._fluxes['prompt']
		if veto_threshold in cache and cache[veto_threshold][0].compatible_with(effective_area):
			flux = cache[veto_threshold][1]
		else:
			flux = NewNuFlux.makeFlux('sarcevic_std')
			flux.knee_reweighting_model = 'gaisserH3a_elbert'
			pf = None if veto_threshold is None else AtmosphericSelfVeto.AnalyticPassingFraction(kind='charm', veto_threshold=veto_threshold)
			flux = (flux, pf)
		instance = cls(effective_area, flux, livetime, hard_veto_threshold)
		if isinstance(flux, tuple):
			cache[veto_threshold] = (effective_area, instance._flux)
		
		return instance
		
class DiffuseAstro(DiffuseNuGen):
	def __init__(self, effective_area, livetime):
		# reference flux is E^2 Phi = 1e-8 GeV^2 cm^-2 sr^-1 s^-1
		flux = self._integrate_flux(effective_area.bin_edges, lambda pt, e, ct: 0.5e-18*(e/1e5)**(-2.))
		# "integrate" over solid angle
		if effective_area.is_healpix:
			flux *= healpy.nside2pixarea(effective_area.nside)
		else:
			flux *= (2*numpy.pi*numpy.diff(effective_area.bin_edges[1]))[None,None,:]
		super(DiffuseAstro, self).__init__(effective_area, flux, livetime)
		
		self._last_params = dict(gamma=-2)
	
	def point_source_background(self, psi_bins, zenith_index, livetime=None, with_energy=True):
		"""
		convert to a point source background
		
		:param bin_areas: areas (in sr) of the search bins around the putative source
		:param livetime: if not None, the actual livetime to integrate over in seconds
		"""
		assert not self.is_healpix, "Don't know how to make PS backgrounds from HEALpix maps yet"
		
		
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
	
	def calculate_expectations(self, **kwargs):
		
		if all([self._last_params[k] == kwargs[k] for k in self._last_params]):
			return self._last_expectations
		
		def intflux(e, gamma):
			return (1e5**(-gamma)/(1+gamma))*e**(1+gamma)
		energy = self._aeff.bin_edges[0]
		
		centers = 0.5*(energy[1:] + energy[:-1])
		specweight = (centers/1e5)**(kwargs['gamma']+2)
		self._last_params['gamma'] = kwargs['gamma']
		
		flux = (self._flux*(specweight[None,:,None]))[...,None,None,None]
		
		if 'e_fraction' in kwargs:
			flavor_weight = 3*numpy.ones(6)
			e, mu = kwargs['e_fraction'], kwargs['mu_fraction']
			# assert e+mu <= 1.
			flavor_weight[0:2] *= e
			flavor_weight[2:4] *= mu
			flavor_weight[4:6] *= (1. - e - mu)
			flux *= flavor_weight[:,None,None,None,None,None]
			for k in 'e_fraction', 'mu_fraction':
				self._last_params[k] = kwargs[k]
		
		total = self._apply_flux(self._aeff, self._flux, self._livetime)
		# up to now we've assumed that everything is azimuthally symmetric and
		# dealt with zenith bins/healpix rings. repeat the values in each ring
		# to broadcast onto a full healpix map.
		if self.is_healpix:
			total = total.repeat(self._aeff.ring_repeat_pattern, axis=1)
		self._last_expectations = dict(cascades=total[...,0], tracks=total[...,1])
		return self._last_expectations
	
	def expectations(self, gamma=-2, **kwargs):
		return self.calculate_expectations(gamma=gamma, **kwargs)


def transform_map(skymap):
	"""
	Interpolate a galactic skymap into equatorial coords
	"""
	r = healpy.Rotator(coord=('C', 'G'))
	npix = skymap.size
	theta_gal, phi_gal = healpy.pix2ang(healpy.npix2nside(npix), numpy.arange(npix))
	theta_ecl, phi_ecl = r(theta_gal, phi_gal)
	return healpy.pixelfunc.get_interp_val(skymap, theta_ecl, phi_ecl)

class FermiGalacticEmission(DiffuseNuGen):
	def __init__(self, effective_area, livetime=1.):
		assert effective_area.is_healpix
		# differential flux at 1 GeV [1/(GeV cm^2 sr s)]
		map1GeV = numpy.load(os.path.join(data_dir, 'fermi_galactic_emission.npy'))
		# downsample to resolution of effective area map
		flux_constant = healpy.ud_grade(transform_map(map1GeV), effective_area.nside)
		def intflux(e, gamma=-2.71):
			return (e**(1+gamma))/(1+gamma)
		e = effective_area.bin_edges[0]
		# integrate flux over energy and solid angle: 1/GeV sr cm^2 s -> 1/cm^2 s
		flux = (intflux(e[1:]) - intflux(e[:-1]))
		flux *= healpy.nside2pixarea(effective_area.nside) * 1e4 * 365*24*3600
		flux = flux[None,:,None] * flux_constant[None,None,:]
		
		super(FermiGalacticEmission, self).__init__(effective_area, flux, livetime)
		
		# extract the diagonal in true_angle / reco_angle and broadcast it over healpix rings
		aeff = numpy.diagonal(self._aeff.values, 0, 2, 4).transpose([0,1,4,2,3]).repeat(self._aeff.ring_repeat_pattern, axis=2)
		# sum over neutrino flavors and energies
		total = numexpr.NumExpr('sum(aeff*flux*livetime, axis=1)')(aeff, self._flux[...,None,None], self._livetime).sum(axis=0)
		
		# dimensions of the keys in expectations are now reconstructed energy, sky bin (healpix pixel)
		self.expectations = dict(cascades=total[...,0].T, tracks=total[...,1].T)

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