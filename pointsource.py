
import numpy
from scipy import optimize
import itertools
from copy import copy
from multillh import LLHEval, asimov_llh

class PointSource(object):
	def __init__(self, effective_area, edges, livetime, zenith_bin, point_spread_function, psi_bins, with_energy=True):
		# reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
		def intflux(e, gamma):
			return (e**(1+gamma))/(1+gamma)
		tev = edges[0]/1e3
		# 1/cm^2 yr
		fluence = 1e-12*(intflux(tev[1:], -2) - intflux(tev[:-1], -2))*livetime*365*24*3600
		
		effective_area = effective_area[...,zenith_bin,:]
		expand = [None]*effective_area.ndim
		expand[1] = slice(None)
		# 1/yr
		rate = fluence[tuple(expand)]*(effective_area*1e4)
		
		ct, mu_energy = map(center, edges[1:3])
		ct = ct[zenith_bin]
		# dimensions: energy, angular bin
		self._psf_quantiles = numpy.diff(point_spread_function(*numpy.meshgrid(psi_bins, mu_energy, ct, indexing='ij')), axis=0)[...,0].T
		
		self._use_energies = with_energy
		
		self._rate = rate
		self._edges = edges + [psi_bins]
	
	def expectations(self, gamma=-2, **kwargs):
		
		energy = self._edges[0]
		
		centers = 0.5*(energy[1:] + energy[:-1])
		specweight = (centers/1e3)**(gamma+2)
		
		expand = [None]*(self._rate.ndim)
		expand[1] = slice(None)
		
		total = (self._rate*(specweight[expand])).sum(axis=(0,1,2))
		# FIXME: this assumes the track PSF for both tracks and cascades.
		total = total[...,None]*self._psf_quantiles[...,None,:]
		
		if not self._use_energies:
			total = total.sum(axis=0)
		
		return dict(cascades=total[...,0,:], tracks=total[...,1,:])
	
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

def discovery_potential(point_source, diffuse_components, sigma=5):
	critical_ts = sigma**2
	def nevents(llh, **hypo):
		"""
		Total number of events predicted by hypothesis *hypo*
		"""
		for k in llh.components:
			if not k in hypo:
				hypo[k] = 1
		return sum(map(numpy.sum, llh.expectations(**hypo).values()))
	
	components = dict(ps=point_source)
	components.update(diffuse_components)
	def ts(flux_norm):
		"""
		Test statistic of flux_norm against flux norm=0
		"""
		allh = asimov_llh(components, ps=flux_norm)
		return -2*(allh.llh(**allh.fit(ps=0))-allh.llh(**allh.fit(ps=flux_norm)))
	def f(flux_norm):
		return ts(flux_norm)-critical_ts
	# estimate significance as signal/sqrt(background)
	allh = asimov_llh(components, ps=1)
	total = nevents(allh, ps=1)
	nb = nevents(allh, ps=0)
	ns = total-nb
	baseline = numpy.sqrt(critical_ts)/(ns/numpy.sqrt(nb))
	return optimize.fsolve(f, baseline, xtol=1e-2)[0]	 

def differential_discovery_potential(point_source, diffuse_components, sigma=5):

	energies = []
	sensitivities = []
	for energy, pschunk in point_source.differential_chunks(decades=0.5):
		energies.append(energy)
		sensitivities.append(discovery_potential(pschunk, diffuse_components))
	return energies, sensitivities

def center(x):
	return 0.5*(x[1:]+x[:-1])

