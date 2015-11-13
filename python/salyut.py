
"""
Hobo unbinned point source analysis
"""

import numpy
import scipy.optimize
import scipy.stats
import scipy.interpolate
from util import *

from icecube import NewNuFlux

class PSLikelihood(object):
	def __init__(self, psi, sigma, E, cos_theta, signal_energy_pdf, background_energy_pdf):
		pass
		
		self.n_total = float(psi.size)
		
		self.psi = psi
		self.sigma = sigma
		self.E = E
		self.cos_theta = cos_theta
		
		self.signal_energy_pdf = signal_energy_pdf
		self.background_energy_pdf = background_energy_pdf
		
		self._band_solid_angle = 2*numpy.pi
		
		self.with_energy = False
	
	def signal(self, psi, sigma, E, cos_theta):
		sigma = numpy.radians(30.)
		s2 = sigma**2
		pdf = numpy.exp(-(psi**2)/(2*s2))/(2*numpy.pi*s2)
		if self.with_energy:
			pdf *= self.signal_energy_pdf(E, cos_theta)
		return pdf
	
	def background(self, E, cos_theta):
		pdf = 1./self._band_solid_angle
		if self.with_energy:
			pdf *= self.background_energy_pdf(E, cos_theta)
		return pdf
	
	def llh(self, ns):
		S = self.signal(self.psi, self.sigma, self.E, self.cos_theta)
		B = self.background(self.E, self.cos_theta)
		return numpy.log((ns/self.n_total)*S + (1. - ns/self.n_total)*B).sum()
	
	def fit(self):
		def minllh(x):
			return -self.llh(*x)
		seeds = [1]
		bounds = [0, None]
		return scipy.optimize.fmin_l_bfgs_b(minllh, seeds, bounds=bounds, approx_grad=True)

class EnergyPDF(object):
	def __init__(self, E, cos_theta, weights):
		def padded_center(edges):
			widths = numpy.diff(edges)
			widths = numpy.concatenate((widths, [widths[-1]]))
			return numpy.concatenate(([edges[0]-widths[0]/2], edges + widths/2.))
		bins = (numpy.logspace(1, 11, 101), numpy.linspace(-1, 0.1, 21))
		h = numpy.histogramdd((E, cos_theta), bins=bins, weights=weights)[0]
		h /= h.sum(axis=0, keepdims=True)
		h /= numpy.diff(bins[0])[:,None]
		centers = map(padded_center, [numpy.log10(bins[0]), bins[1]])
		y = numpy.empty(tuple(n+2 for n in h.shape))
		y[(slice(1,-1),)*h.ndim] = h
		def hyperslab(idx, dim, null=slice(1, -1)):
			target = [null]*h.ndim
			target[dim] = idx
			return target
		for i in range(h.ndim):
			y[hyperslab(0, i, slice(1, -1))] = h[hyperslab(0, i, slice(None))]
			y[hyperslab(-1, i, slice(1, -1))] = h[hyperslab(-1, i, slice(None))]
			
		self._interpolant = scipy.interpolate.RegularGridInterpolator(centers, y, bounds_error=True, fill_value=-numpy.inf)
	def __call__(self, E, cos_theta):
		return self._interpolant((numpy.log10(E), cos_theta), 'nearest')

def hsin(theta):
	"""haversine"""
	return (1.-numpy.cos(theta))/2.

def opening_angle(z1, z2, a1, a2):
	"""
	Calculate the opening angle between two directions specified by zenith and azimuth
	"""
	return hsin(a2-a1) + numpy.cos(a1)*numpy.cos(a2)*hsin(z2-z1)

def astroflux(pt, e, cos_theta):
	"""
	Benchmark astrophysical flux
	"""
	return 0.5e-8*e**-2

def pull_correction(logE):
	"""
	Pull correction for SplineMPEMuEXDifferential in Aachen IC86 diffuse sample
	"""
	x = logE
	
	a, b, c = -3.840927835538128, 0.8524834665561651, -0.7992029125357616
	pull = numpy.exp(numpy.maximum(c, a + b*x))
	
	return pull

def create_energy_pdfs(query):
	
	atmo_weights = query.getWeights()
	astro_weights = query.getWeights(model=astroflux)
	
	E, cos_theta = query['energy'], numpy.cos(query['zenith'])
	
	return [EnergyPDF(E, cos_theta, w) for w in (atmo_weights, astro_weights)]

def skylab_background(query, livetime=1):
	"""
	Prepare a pseudodata sample for Skylab, assuming only conventional atmospheric
	and a diffuse astrophysical background
	"""
	atmo_weights = query.getWeights() + query.getWeights(model=astroflux)
	
	nb = int(numpy.random.poisson(atmo_weights.sum()*livetime*3600*24*365))
	
	dtype = numpy.dtype([(name, float) for name in ('ra', 'sinDec', 'sigma', 'logE')])
	sample = numpy.empty(nb, dtype=dtype)
	
	atmo_idx = numpy.random.choice(numpy.arange(atmo_weights.size), size=nb,
	    p=atmo_weights/atmo_weights.sum(), replace=True,)
	subquery = query[atmo_idx]
	
	sample['ra'] = numpy.random.uniform(0, 2*numpy.pi, size=nb)
	sample['logE'] = subquery['logE']
	sample['sinDec'] = subquery['sinDec']
	sample['sigma'] = subquery['sigma']*pull_correction(subquery['logE'])
	
	return sample

def oneweight_flux(pt, e, ct):
	"""
	Flux equivalent to OneWeight if the normalization term is per particle
	rather than the average of neutrinos and antineutrinos.
	"""
	return 0.5

def skylab_mc(query):
	"""
	Prepare MC sample for Skylab
	"""
	dtype = numpy.dtype([(name, float) for name in ('ra', 'sinDec', 'sigma', 'logE',
	    'trueRa', 'trueDec', 'trueE', 'ow', 'atmoWeight', #'psi',
		)])
	mc = numpy.empty(query.size, dtype=dtype)
	mc['ra'] = query['raDiff']
	mc['logE'] = query['logE']
	mc['sinDec'] = query['sinDec']
	mc['sigma'] = query['sigma']*pull_correction(query['logE'])
	
	mc['trueRa'] = 0.
	mc['trueDec'] = query['trueDec']
	mc['trueE'] = query['trueE']
	mc['ow'] = query.getWeights(model=oneweight_flux)
	mc['atmoWeight'] = query.getWeights()
	
	return mc

def sample_events(query, ns=0, sindec=0, livetime=1):
	
	atmo_weights = query.getWeights()
	astro_weights = query.getWeights(model=astroflux)
	
	nb = int(numpy.random.poisson(atmo_weights.sum()*livetime*3600*24*365))
	
	dtype = numpy.dtype([(name, float) for name in ('psi', 'sigma', 'energy', 'cos_theta')])
	sample = numpy.empty(nb + ns, dtype=dtype)
	
	atmo_idx = numpy.random.choice(numpy.arange(atmo_weights.size), size=nb,
	    p=atmo_weights/atmo_weights.sum(), replace=True,)
	subquery = query[atmo_idx]
	sample['energy'][:nb] = subquery['energy']
	sample['sigma'][:nb] = subquery['psi']
	sample['cos_theta'][:nb] = numpy.cos(subquery['zenith'])
	
	source_zenith = numpy.arccos(-sindec)
	# calculate the angular distance to the hypothetical source direction,
	# scrambling in azimuth
	sample['psi'][:nb] = opening_angle(source_zenith, query['zenith'][atmo_idx], 0., numpy.random.uniform(0, 2*numpy.pi, size=nb))
	
	print sample['psi'][:nb]
	
	# fill in some signal events near the desired declination
	mask = abs(numpy.cos(source_zenith) - query['neutrino_ct']) < 0.05
	astro_idx = numpy.random.choice(numpy.arange(mask.sum()), size=ns,
	    p=astro_weights[mask]/astro_weights[mask].sum(), replace=True,)
	
	subquery = query[mask][astro_idx]
	sample['energy'][-ns:] = subquery['energy']
	sample['psi'][-ns:] = subquery['psi']
	sample['cos_theta'][-ns:] = numpy.cos(subquery['zenith'])
	
	print sample['psi'][-ns:]
	
	return sample

def llh_realization(query, ns=0, sindec=0, livetime=1):
	
	sample = sample_events(query, ns, sindec, livetime)
	
	bkg_e, sig_e = create_energy_pdfs(query)
	
	return PSLikelihood(sample['psi'], sample['sigma'], sample['energy'], sample['cos_theta'], bkg_e, sig_e)