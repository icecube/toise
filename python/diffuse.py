
import numpy
import itertools
from scipy.integrate import quad
from StringIO import StringIO
from copy import copy
import multillh
import healpy
import os
import numexpr
import cPickle as pickle
import logging

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
	_reduce_flux = numexpr.NumExpr('sum(aeff*flux*livetime, axis=1)')
	
	def _apply_flux(self, effective_area, flux, livetime):
		if effective_area.shape[2] > 1:
			return self._reduce_flux(effective_area, flux[...,None,None], livetime).sum(axis=0)
		else:
			return (effective_area*flux[...,None,None]*livetime).sum(axis=(0,1))

def detect(sequence, pred):
	try:
		return next((s for s in sequence if pred(s)))
	except StopIteration:
		return None

class AtmosphericNu(DiffuseNuGen):
	"""
	The diffuse atmospheric neutrino flux. :meth:`.point_source_background`
	returns the corresponding point source background.
	
	The units of the model are scalings of the underlying flux parameterization.
	"""
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
		
		if hard_veto_threshold is not None:
			# reduce the flux in the south
			# NB: assumes that a surface veto has been applied!
			flux = self._flux * numpy.where(center(effective_area.bin_edges[1]) < 0.05, 1, 1e-4)[None,None,:]
		else:
			flux = self._flux
			
		# sum over neutrino flavors, energies, and zenith angles
		total = self._apply_flux(self._aeff.values, flux, self._livetime)

		
		# up to now we've assumed that everything is azimuthally symmetric and
		# dealt with zenith bins/healpix rings. repeat the values in each ring
		# to broadcast onto a full healpix map.
		if self.is_healpix:
			total = total.repeat(self._aeff.ring_repeat_pattern, axis=0)
		# dimensions of the keys in expectations are now reconstructed energy, sky bin (zenith/healpix pixel)
		self.expectations = dict(tracks=total.sum(axis=2))
	
	def point_source_background(self, zenith_index, livetime=None, with_energy=True):
		"""
		Convert flux to a form suitable for calculating point source backgrounds.
		The predictions in **expectations** will be differential in the opening-angle
		bins provided in the effective area instead of being integrated over them.
		
		:param zenith_index: index of the sky bin to use. May be either an integer
		                     (for single point source searches) or a slice (for
		                     stacking searches)
		:param livetime: if not None, the actual livetime to integrate over in seconds
		:param with_energy: if False, integrate over reconstructed energy. Otherwise,
		                    provide a differential prediction in reconstructed energy.
		"""
		assert not self.is_healpix, "Don't know how to make PS backgrounds from HEALpix maps yet"
		
		background = copy(self)
		psi_bins = self._aeff.bin_edges[-1][:-1]
		bin_areas = (numpy.pi*numpy.diff(psi_bins**2))[None,...]
		# observation time shorter for triggered transient searches
		if livetime is not None:
			bin_areas *= (livetime/self._livetime/(3600.*24*365.))
		if isinstance(zenith_index, slice):
			omega = self._solid_angle[zenith_index,None]
		else:
			omega = self._solid_angle[zenith_index]
		# dimensions of the keys in expectations are now energy, radial bin
		background.expectations = {k: (v[zenith_index,:]/omega)[...,None]*bin_areas for k,v in self.expectations.items()}
		if not with_energy:
			# just radial bins
			background.expectations = {k: v.sum(axis=0) for k,v in background.expectations.items()}
		return background
	
	_cache_file = os.path.join(data_dir, 'cache', 'atmospheric_fluxes.pickle')
	if os.path.exists(_cache_file):
		with open(_cache_file) as f:
			_fluxes = pickle.load(f)
	else:
		_fluxes = dict(conventional=dict(), prompt=dict())
	@classmethod
	def conventional(cls, effective_area, livetime, veto_threshold=1e3, hard_veto_threshold=None):
		"""
		Instantiate a conventional atmospheric neutrino flux, using the Honda
		parameterization with corrections for the cosmic ray knee and the fraction
		of atmospheric neutrinos accompanied by muons.
		
		The flux will be integrated over the effective area's energy and zenith
		angle bins the first time this method is called. Depending on the number of
		bins this can take several minutes. Subsequent calls with the same veto
		threshold and an effective area of the same shape will use the cached flux
		and instantiate much more quickly.
			
		:param effective_area: an instance of :py:class:`effective_areas.effective_area`
		:param livetime: observation time, in years
		:param veto_threshold: muon energy, in GeV, above which atmospheric muons
		                       can be vetoed. This will be used to modify the effective
		                       atmospheric neutrino flux.
		:param hard_veto_threshold: if not None, reduce the atmospheric flux to
		                            1e-4 of its nominal value in the southern
		                            hemisphere to model the effect of a surface
		                            veto. This assumes that an energy threshold
		                            has been applied to the effective area. 
		"""
		from icecube import NewNuFlux, AtmosphericSelfVeto
		cache = cls._fluxes['conventional']
		shape_key = effective_area.values.shape[:4]
		flux = detect(cache.get(veto_threshold, []), lambda args: args[0]==shape_key)
		if flux is None:
			flux = NewNuFlux.makeFlux('honda2006')
			flux.knee_reweighting_model = 'gaisserH3a_elbert'
			pf = None if veto_threshold is None else AtmosphericSelfVeto.AnalyticPassingFraction(kind='conventional', veto_threshold=veto_threshold)
			flux = (flux, pf)
		else:
			flux = flux[1]
		instance = cls(effective_area, flux, livetime, hard_veto_threshold)
		if isinstance(flux, tuple):
			
			if not veto_threshold in cache:
				cache[veto_threshold] = list()
			cache[veto_threshold].append((shape_key, instance._flux))
			with open(cls._cache_file, 'w') as f:
				pickle.dump(cls._fluxes, f, protocol=2)
		assert len(cls._fluxes['conventional']) > 0 
		
		return instance
	
	@classmethod
	def prompt(cls, effective_area, livetime, veto_threshold=1e3, hard_veto_threshold=None):
		"""
		Instantiate a prompt atmospheric neutrino flux, using the Enberg
		parameterization with corrections for the cosmic ray knee and the fraction
		of atmospheric neutrinos accompanied by muons.
		
		The parameters have the same meanings as in :meth:`.conventional`
		"""
		from icecube import NewNuFlux, AtmosphericSelfVeto
		cache = cls._fluxes['prompt']
		shape_key = effective_area.values.shape[:4]
		flux = detect(cache.get(veto_threshold, []), lambda args: args[0]==shape_key)
		if flux is None:
			flux = NewNuFlux.makeFlux('sarcevic_std')
			flux.knee_reweighting_model = 'gaisserH3a_elbert'
			pf = None if veto_threshold is None else AtmosphericSelfVeto.AnalyticPassingFraction(kind='charm', veto_threshold=veto_threshold)
			flux = (flux, pf)
		else:
			flux = flux[1]
		instance = cls(effective_area, flux, livetime, hard_veto_threshold)
		if isinstance(flux, tuple):
			if not veto_threshold in cache:
				cache[veto_threshold] = list()
			cache[veto_threshold].append((shape_key, instance._flux))
			with open(cls._cache_file, 'w') as f:
				pickle.dump(cls._fluxes, f, protocol=2)
		
		return instance
		
class DiffuseAstro(DiffuseNuGen):
	r"""
	A diffuse astrophysical neutrino flux. :meth:`.point_source_background`
	returns the corresponding point source background.
	
	The unit is the differential flux per neutrino flavor at 100 TeV,
	in units of :math:`10^{-18} \,\, \rm  GeV^{-1} \, cm^{-2} \, s^{-1} \, sr^{-1}`
	"""
	def __init__(self, effective_area, livetime):
		"""
		:param effective_area: the effective area
		:param livetime: observation time, in years
		"""
		# reference flux is E^2 Phi = 1e-8 GeV^2 cm^-2 sr^-1 s^-1
		flux = self._integrate_flux(effective_area.bin_edges, lambda pt, e, ct: 0.5e-18*(e/1e5)**(-2.))
		# "integrate" over solid angle
		if effective_area.is_healpix:
			flux *= healpy.nside2pixarea(effective_area.nside)
		else:
			flux *= (2*numpy.pi*numpy.diff(effective_area.bin_edges[1]))[None,None,:]
		super(DiffuseAstro, self).__init__(effective_area, flux, livetime)
		self._with_psi = False
		
		self._invalidate_cache()
	
	def _invalidate_cache(self):
		self._last_params = dict()
		self._last_expectations = None
	
	def point_source_background(self, zenith_index, livetime=None, with_energy=True):
		__doc__ = AtmosphericNu.point_source_background.__doc__
		assert not self.is_healpix, "Don't know how to make PS backgrounds from HEALpix maps yet"
		
		
		background = copy(self)
		psi_bins = self._aeff.bin_edges[-1][:-1]
		bin_areas = (numpy.pi*numpy.diff(psi_bins**2))[None,None,None,:]
		# observation time shorter for triggered transient searches
		if livetime is not None:
			bin_areas *= (livetime/self._livetime/(3600.*24*365.))
		# dimensions of the keys in expectations are now energy, radial bin
		
		# cut flux down to a single zenith band
		# dimensions of self._flux are flavor, energy, zenith
		if isinstance(zenith_index, slice):
			sel = zenith_index
		else:
			sel = slice(zenith_index, zenith_index+1)
		# dimensions of flux are now 1/m^2 sr
		background._flux = (self._flux[:,:,sel]/self._solid_angle[zenith_index])
		
		# replace reconstructed zenith with opening angle
		# dimensions of aeff are now m^2 sr
		background._aeff = copy(self._aeff)
		background._aeff.values = self._aeff.values[:,:,sel,:,:].sum(axis=4)[...,None]*bin_areas
		background._with_psi = True
		
		background._invalidate_cache()
		
		# total = self._apply_flux(background._aeff.values, background._flux, self._livetime)
		# print background._aeff.values.sum(axis=tuple(range(0, 5)))
		# print total.sum(axis=tuple(range(0, 2)))
		# print background._aeff.values.shape, background._flux.shape
		# print total[...,0].sum()
		# assert total[...,1].sum() > 0
		
		if not with_energy:
			raise ValueError("Can't disable energy dimension just yet")

		return background
	
	def differential_chunks(self, decades=1):
		"""
		Yield copies of self with the neutrino spectrum restricted to *decade*
		decades in energy
		"""
		# now, sum over decades in neutrino energy
		loge = numpy.log10(self._aeff.bin_edges[0])
		bin_range = int(decades/(loge[1]-loge[0]))+1
		
		for i in range(loge.size-1-bin_range):
			start = i
			stop = start + bin_range
			chunk = copy(self)
			chunk._invalidate_cache()
			# zero out the neutrino flux outside the given range
			chunk._flux = self._flux.copy()
			chunk._flux[:,:start,...] = 0
			chunk._flux[:,stop:,...] = 0
			e_center = 10**(0.5*(loge[start] + loge[stop]))
			yield e_center, chunk
	
	def spectral_weight(self, e_center, **kwargs):
		self._last_params['gamma'] = kwargs['gamma']
		return (e_center/1e5)**(kwargs['gamma']+2)
	
	def calculate_expectations(self, **kwargs):
		if self._last_expectations is not None and all([self._last_params[k] == kwargs[k] for k in self._last_params]):
			return self._last_expectations
		
		energy = self._aeff.bin_edges[0]
		centers = 0.5*(energy[1:] + energy[:-1])
		specweight = self.spectral_weight(centers, **kwargs)
		
		flux = (self._flux*(specweight[None,:,None]))#[...,None,None,None]
		
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
		
		
		total = self._apply_flux(self._aeff.values, flux, self._livetime)
		if not self._with_psi:
			total = total.sum(axis=2)
			assert total.ndim == 2
		else:
			assert total.ndim == 3
		
		# up to now we've assumed that everything is azimuthally symmetric and
		# dealt with zenith bins/healpix rings. repeat the values in each ring
		# to broadcast onto a full healpix map.
		if self.is_healpix:
			total = total.repeat(self._aeff.ring_repeat_pattern, axis=0)
		if total.shape[0] == 1:
			total = total.reshape(total.shape[1:])
		self._last_expectations = dict(tracks=total)
		return self._last_expectations
	
	def expectations(self, gamma=-2, **kwargs):
		r"""
		:param gamma: the spectral index :math:`\gamma`.
		:returns: the observable distributions expected for a flux of
		:math:`10^{-18} \frac{E_\nu}{\rm 100 \, TeV}^{\gamma} \,\, \rm  GeV^{-1} \, cm^{-2} \, s^{-1} \, sr^{-1}`
		per neutrino flavor 
		"""
		return self.calculate_expectations(gamma=gamma, **kwargs)

class AhlersGZK(DiffuseAstro):
	"""
	Minimal GZK neutrino flux, assuming that post-ankle flux in Auger/TA is
	pure protons
	see: http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.083010
	Fig 2. left panel, solid red line (protons with source evolution)
	"""
	def __init__(self, *args, **kwargs):
		from scipy import interpolate
		
		super(AhlersGZK, self).__init__(*args, **kwargs)
		logE, logWeight = numpy.log10(numpy.loadtxt(StringIO(
		    """3.095e5	8.345e-13
		    4.306e5	1.534e-12
		    5.777e5	2.305e-12
		    7.091e5	3.411e-12
		    8.848e5	4.944e-12
		    1.159e6	7.158e-12
		    1.517e6	1.075e-11
		    2.118e6	1.619e-11
		    2.868e6	2.284e-11
		    3.900e6	3.181e-11
		    5.660e6	4.502e-11
		    7.891e6	6.003e-11
		    1.042e7	8.253e-11
		    1.449e7	1.186e-10
		    1.918e7	1.670e-10
		    3.224e7	3.500e-10
		    7.012e7	1.062e-9
		    1.106e8	1.892e-9
		    1.610e8	2.816e-9
		    2.235e8	3.895e-9
		    3.171e8	5.050e-9
		    5.042e8	6.529e-9
		    7.787e8	7.401e-9
		    1.199e9	7.595e-9
		    1.801e9	7.084e-9
		    2.869e9	6.268e-9
		    4.548e9	4.972e-9
		    6.372e9	3.959e-9
		    8.144e9	3.155e-9
		    1.131e10	2.318e-9
		    1.366e10	1.747e-9
		    2.029e10	9.879e-10
		    2.612e10	6.441e-10
		    3.289e10	4.092e-10
		    4.885e10	1.828e-10
		    8.093e10	5.691e-11
		    1.260e11	1.677e-11
		    1.653e11	7.984e-12
		    2.167e11	3.631e-12
		    2.875e11	1.355e-12
		    """))).T

		self._interpolant = interpolate.interp1d(logE, logWeight+8, bounds_error=False, fill_value=-numpy.inf)
		
	def spectral_weight(self, e_center, **kwargs):
		return 10**self._interpolant(numpy.log10(e_center))

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
	"""
	Diffuse emission from the galaxy, modeled as 0.95 times the Fermi
	:math:`\pi^0` map, extrapolated with a spectral index of 2.71.
	"""
	def __init__(self, effective_area, livetime=1.):
		assert effective_area.is_healpix
		# differential flux at 1 GeV [1/(GeV cm^2 sr s)]
		map1GeV = numpy.load(os.path.join(data_dir, 'models', 'fermi_galactic_emission.npy'))
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
		
		# sum over opening angles and broadcast zenith angle bin over healpix rings
		aeff = self._aeff.values.sum(axis=4).repeat(self._aeff.ring_repeat_pattern, axis=2)
		# sum over neutrino flavors and energies
		total = numexpr.NumExpr('sum(aeff*flux*livetime, axis=1)')(aeff, self._flux[...,None], self._livetime).sum(axis=0)
		
		# dimensions of the keys in expectations are now reconstructed energy, sky bin (healpix pixel)
		self.expectations = dict(tracks=total)

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