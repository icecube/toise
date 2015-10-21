
import numpy
from scipy import optimize, stats
import itertools
from copy import copy
from multillh import LLHEval, asimov_llh
from util import *
import logging

def psf_quantiles(point_spread_function, psi_bins, mu_energy, ct):
	mu_energy, ct, psi_bins = numpy.meshgrid(mu_energy, ct, psi_bins, indexing='ij')
	return numpy.diff(point_spread_function(psi_bins, mu_energy, ct), axis=2)

class PointSource(object):
	def __init__(self, effective_area, fluence, zenith_bin, point_spread_function, psi_bins, with_energy=True):

		self._edges = effective_area.bin_edges + (psi_bins,)
		
		effective_area = effective_area.values[...,zenith_bin,:]
		expand = [None]*effective_area.ndim
		expand[1] = slice(None)
		# 1/yr
		rate = fluence[tuple(expand)]*(effective_area*1e4)
		
		assert numpy.isfinite(rate).all()
		
		ct, mu_energy = map(center, self._edges[1:3])
		ct = ct[zenith_bin]
		# dimensions: energy, [zenith,] angular bin
		quantiles = psf_quantiles(point_spread_function, psi_bins, mu_energy, ct)
		if numpy.isscalar(zenith_bin):
			self._psf_quantiles = quantiles[:,0,:]
		else:
			self._psf_quantiles = quantiles
		
		self._use_energies = with_energy
		
		self._rate = rate
		self._last_gamma = None
	
	def expectations(self, ps_gamma=-2, **kwargs):
		
		if self._last_gamma == ps_gamma:
			return self._last_expectations
		
		energy = self._edges[0]
		
		centers = 0.5*(energy[1:] + energy[:-1])
		specweight = (centers/1e3)**(ps_gamma+2)
		
		expand = [None]*(self._rate.ndim)
		expand[1] = slice(None)
		
		total = (self._rate*(specweight[expand])).sum(axis=(0,1,2))
		# FIXME: this assumes the track PSF for both tracks and cascades.
		# also, it neglects the opening angle between neutrino and muon
		total = total[...,None]*self._psf_quantiles[...,None,:]
		
		if not self._use_energies:
			total = total.sum(axis=0)
		
		self._last_expectations = dict(cascades=total[...,0,:], tracks=total[...,1,:])
		self._last_gamma = ps_gamma
		return self._last_expectations
	
	def differential_chunks(self, decades=1):
		"""
		Yield copies of self with the neutrino spectrum restricted to *decade*
		decades in energy
		"""
		# now, sum over decades in neutrino energy
		loge = numpy.log10(self._edges[0])
		bin_range = int(decades/(loge[1]-loge[0]))+1
		
		for i in range(loge.size-1-bin_range):
			start = i
			stop = start + bin_range
			chunk = copy(self)
			# zero out the neutrino flux outside the given range
			chunk._rate = self._rate.copy()
			chunk._rate[:,:start,...] = 0
			chunk._rate[:,stop:,...] = 0
			e_center = 10**(0.5*(loge[start] + loge[stop]))
			yield e_center, chunk

class SteadyPointSource(PointSource):
	def __init__(self, effective_area, livetime, zenith_bin, point_spread_function, psi_bins, with_energy=True):
		# reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
		def intflux(e, gamma):
			return (e**(1+gamma))/(1+gamma)
		tev = effective_area.bin_edges[0]/1e3
		# 1/cm^2 yr
		fluence = 1e-12*(intflux(tev[1:], -2) - intflux(tev[:-1], -2))*livetime*365*24*3600
		
		PointSource.__init__(self, effective_area, fluence, zenith_bin, point_spread_function, psi_bins, with_energy)

def nevents(llh, **hypo):
	"""
	Total number of events predicted by hypothesis *hypo*
	"""
	for k in llh.components:
		if not k in hypo:
			if hasattr(llh.components[k], 'seed'):
				hypo[k] = llh.components[k].seed
			else:
				hypo[k] = 1
	return sum(map(numpy.sum, llh.expectations(**hypo).values()))

def discovery_potential(point_source, diffuse_components, sigma=5., baseline=None, tolerance=1e-2, **fixed):
	critical_ts = sigma**2

	
	components = dict(ps=point_source)
	components.update(diffuse_components)
	def ts(flux_norm):
		"""
		Test statistic of flux_norm against flux norm=0
		"""
		allh = asimov_llh(components, ps=flux_norm)
		if len(fixed) == len(diffuse_components):
			return -2*(allh.llh(ps=0, **fixed)-allh.llh(ps=flux_norm, **fixed))
		else:
			null = allh.fit(ps=0, **fixed)
			alternate = allh.fit(ps=flux_norm, **fixed)
			# print null, alternate, -2*(allh.llh(**null)-allh.llh(**alternate))-critical_ts
			return -2*(allh.llh(**null)-allh.llh(**alternate))
	def f(flux_norm):
		return ts(flux_norm)-critical_ts
	if baseline is None:
		# estimate significance as signal/sqrt(background)
		allh = asimov_llh(components, ps=1, **fixed)
		total = nevents(allh, ps=1, **fixed)
		nb = nevents(allh, ps=0, **fixed)
		ns = total-nb
		baseline = min((1000, numpy.sqrt(critical_ts)/(ns/numpy.sqrt(nb))))/10
		baseline = (numpy.sqrt(critical_ts)/(ns/numpy.sqrt(nb)))/10
		# logging.getLogger().info('total: %.2g ns: %.2g nb: %.2g baseline norm: %.2g' % (total, ns, nb, baseline))
	# baseline = 1000
	if baseline > 1e4:
		return numpy.inf
	else:
		# actual = optimize.bisect(f, 0, baseline, xtol=baseline*1e-2)
		actual = optimize.fsolve(f, baseline, xtol=tolerance)
		allh = asimov_llh(components, ps=actual, **fixed)
		total = nevents(allh, ps=actual, **fixed)
		nb = nevents(allh, ps=0, **fixed)
		ns = total-nb
		logging.getLogger().info("baseline: %.2g actual %.2g ns: %.2g nb: %.2g" % (baseline, actual, ns, nb))
		return actual[0]

def upper_limit(point_source, diffuse_components, cl=0.9, **fixed):
	critical_ts = stats.chi2.ppf(cl, 1)

	
	components = dict(ps=point_source)
	components.update(diffuse_components)
	def ts(flux_norm):
		"""
		Test statistic of flux_norm against flux norm=0
		"""
		allh = asimov_llh(components, ps=0)
		if len(fixed) == len(diffuse_components):
			return -2*(allh.llh(ps=0, **fixed)-allh.llh(ps=flux_norm, **fixed))
		else:
			return -2*(allh.llh(**allh.fit(ps=0, **fixed))-allh.llh(**allh.fit(ps=flux_norm, **fixed)))
	def f(flux_norm):
		# NB: minus sign, because now the null hypothesis is no source
		return -ts(flux_norm)-critical_ts
	# estimate significance as signal/sqrt(background)
	allh = asimov_llh(components, ps=1, **fixed)
	total = nevents(allh, ps=1, **fixed)
	nb = nevents(allh, ps=0, **fixed)
	ns = total-nb
	baseline = numpy.sqrt(critical_ts)/(ns/numpy.sqrt(nb))
	logging.getLogger().debug('total: %.2g ns: %.2g nb: %.2g baseline norm: %.2g' % (total, ns, nb, baseline))
	
	if baseline > 1e4:
		return numpy.inf
	else:
		# actual = optimize.bisect(f, 0, baseline, xtol=baseline*1e-2)
		actual = optimize.fsolve(f, baseline/10, xtol=1e-2)
		logging.getLogger().debug("baseline: %.2g actual %.2g" % (baseline, actual))
		return actual[0]


def differential_discovery_potential(point_source, diffuse_components, sigma=5, **fixed):

	energies = []
	sensitivities = []
	for energy, pschunk in point_source.differential_chunks(decades=0.5):
		energies.append(energy)
		sensitivities.append(discovery_potential(pschunk, diffuse_components, **fixed))
	return energies, sensitivities

