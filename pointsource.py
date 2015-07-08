
import numpy
from scipy import optimize
import itertools

class PointSource(object):
	def __init__(self, effective_area, edges, livetime):
		# reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
		def intflux(e, gamma):
			return (e**(1+gamma))/(1+gamma)
		tev = edges[0]/1e3
		# 1/cm^2 yr
		fluence = 1e-12*(intflux(tev[1:], -2) - intflux(tev[:-1], -2))*livetime*365*24*3600
		
		expand = [None]*effective_area.ndim
		expand[1] = slice(None)
		# 1/yr
		rate = fluence[tuple(expand)]*(effective_area*1e4)
		
		self._rate = rate
		self._edges = edges
	
	def expectations(self, gamma=-2, **kwargs):
		
		energy = self._edges[0]
		
		centers = 0.5*(energy[1:] + energy[:-1])
		specweight = (centers/1e3)**(gamma+2)
		
		expand = [None]*(self._rate.ndim)
		expand[1] = slice(None)
		
		total = (self._rate*(specweight[expand])).sum(axis=(0,1,2))
		
		return dict(cascades=total[...,0], tracks=total[...,1])
	
	def differential_expectations(self, decades=1):
		
		# first, sum over neutrino flavors and zenith angles
		spectra = self._rate.sum(axis=(0, 2))
		
		# now, sum over decades in neutrino energy
		energy = self._edges[0]
		bin_range = int(decades/numpy.log10(energy[1]/energy[0]))
		kernel = numpy.ones(bin_range)
		spectra = numpy.apply_along_axis(numpy.convolve, 0, spectra, kernel, mode='valid')
		
		bin_centers = energy[bin_range/2:-bin_range/2]
		
		return bin_centers, spectra

def ts(n, ns, nb):
	"""
	Test statistic for expectation ns+nb over nb, given n events
	"""
	return numpy.nansum(2*(n*numpy.log((ns+nb)/nb) - ns))

def discovery_ts(ns, nb):
	"""
	Asimov test statistic for median discovery
	"""
	return ts(ns+nb, ns, nb)

def exclusion_ts(ns, nb):
	"""
	Asimov test statistic for median exclusion
	"""
	return -ts(nb, ns, nb)

def center(x):
	return 0.5*(x[1:]+x[:-1])

class PointSourceSensitivityEstimator(object):
	def __init__(self, angular_resolution, bin_containment=0.9):
		self._angular_resolution = angular_resolution
		self._bin_containment = 0.9
		self.diffuse_components = dict()
		self.point_source = None
	
	def add_diffuse_source(self, name, bkg):
		self.diffuse_components[name] = bkg
	
	def set_point_source(self, sig):
		self.point_source = sig
	
	def get_background_rate(self):
		background = 0
		for bkg in self.diffuse_components.values():
			# TODO: treat cascades too
			density = bkg.expectations['tracks']/bkg._solid_angle
			ct, mu_energy = map(center, bkg._edges[1:3])
			psi = self._angular_resolution.median_opening_angle(*numpy.meshgrid(mu_energy, ct, indexing='ij'))
			rate = numpy.pi*(psi**2*density).sum(axis=0)
			background = background + rate
		return background
	
	def differential_sensitivity(self, decades=0.5, kind='discovery', critical_ts=25):
		# FIXME: treat cascades too
		energies, spectra = self.point_source.differential_expectations(decades=decades)
		backgrounds = self.get_background_rate()

		sensitivities = numpy.nan*numpy.zeros((spectra.shape[0], spectra.shape[2]))

		for i,j in itertools.product(*map(range, sensitivities.shape)):
			ns = spectra[i,:,j,1].sum()*self._bin_containment
			nb = backgrounds[j].sum()
			if ns == 0:
				sensitivities[i,j] = numpy.inf
				continue
			baseline = numpy.sqrt(critical_ts)/(ns/numpy.sqrt(nb))
			if kind == 'discovery':
				f = lambda n: discovery_ts(n*ns, nb) - critical_ts
				sensitivities[i,j] = optimize.bisect(f, baseline/10, baseline*10)
			elif kind == 'exclusion':
				f = lambda n: exclusion_ts(n*ns, nb) - critical_ts
				sensitivities[i,j] = optimize.bisect(f, 0, baseline*10)
			else:
				raise ValueError("Unhandled sensitivity type")
		
		return (energies, center(self.point_source._edges[1])), sensitivities

class BinnedPointSourceSensitivityEstimator(object):
	def __init__(self, point_source, diffuse_components, point_spread_function, psi_bins):
		self.diffuse_components = diffuse_components
		self.point_source = point_source
		


		ct, mu_energy = map(center, self.point_source._edges[1:3])
		# paranoia
		for bkg in self.diffuse_components.values():
			assert (ct == center(bkg._edges[1])).all()
			assert (mu_energy == center(bkg._edges[2])).all()
	
		self._psf = point_spread_function
		self._psi_bins = psi_bins
		self._bin_area = numpy.pi*numpy.diff(psi_bins**2)
		self._psf_quantiles = numpy.diff(self._psf(*numpy.meshgrid(self._psi_bins, mu_energy, ct, indexing='ij')), axis=0)
	
	
	def get_background_rate(self):
		background = 0
		for bkg in self.diffuse_components.values():
			# TODO: treat cascades too
			density = bkg.expectations['tracks']/bkg._solid_angle
			ct, mu_energy = map(center, bkg._edges[1:3])
			rate = density.sum(axis=0)[:,None]*self._bin_area[None,:]
			background = background + rate
		return background
	
	def get_signal_rate(self, spectrum):
		return (spectrum[None,:]*self._psf_quantiles).sum(axis=1)
		
	
	def differential_sensitivity(self, decades=0.5, kind='discovery', critical_ts=25):
		# FIXME: treat cascades too
		energies, spectra = self.point_source.differential_expectations(decades=decades)
		backgrounds = self.get_background_rate()

		sensitivities = numpy.nan*numpy.zeros((spectra.shape[0], spectra.shape[2]))

		ct, mu_energy = map(center, self.point_source._edges[1:3])
		# paranoia
		for bkg in self.diffuse_components.values():
			assert (ct == center(bkg._edges[1])).all()
			assert (mu_energy == center(bkg._edges[2])).all()
		
		# i energy range
		# j declination band
		for i,j in itertools.product(*map(range, sensitivities.shape)):
			
			ns = (spectra[i,:,j,1][None,:]*self._psf_quantiles[...,j]).sum(axis=1)
			nb = backgrounds[j]
			
			if ns.sum() == 0:
				sensitivities[i,j] = numpy.inf
				continue
			baseline = numpy.sqrt(critical_ts)/(ns.sum()/numpy.sqrt(nb.sum()))
			if kind == 'discovery':
				f = lambda n: discovery_ts(n*ns, nb) - critical_ts
				sensitivities[i,j] = optimize.bisect(f, baseline/10, baseline*100)
			elif kind == 'exclusion':
				f = lambda n: exclusion_ts(n*ns, nb) - critical_ts
				sensitivities[i,j] = optimize.bisect(f, 0, baseline*10)
			else:
				raise ValueError("Unhandled sensitivity type")
		
		return (energies, center(self.point_source._edges[1])), sensitivities
	
	


