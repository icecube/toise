
import os
import numpy
import itertools
import healpy

from surfaces import get_fiducial_surface
from energy_resolution import get_energy_resolution
from util import *

def load_jvs_mese():
	"""
	Load the effective areas used in the MESE diffuse analysis (10.1103/PhysRevD.91.022001)
	
	:returns: a tuple (edges, aeff). the 6 dimensions of aeff are: nu type (6),
	          nu energy, cos(nu zenith), reco energy, cos(reco zenith),
	          signature (cascade/track). edges is a list of length 4 with the
	          edges in the inner dimensions.
	"""
	shape = None
	edges = None
	aeff = None
	base = '/Users/jakob/Documents/IceCube/reports/charm_search/supplemental/effective_area.per_bin.nu_{flavor}{anti}.{interaction}.{channel}.txt.gz'
	for i, (flavor, anti) in enumerate(itertools.product(('e', 'mu', 'tau'), ('', '_bar'))):
		for j, channel in enumerate(('cascade', 'track')):
			for interaction in 'cc', 'nc', 'gr':
				try:
					data = numpy.loadtxt(base.format(**locals()))
				except:
					pass
				if shape is None:
					edges = []
					for k in range(4):
						lo = numpy.unique(data[:,k*2])
						hi = numpy.unique(data[:,k*2+1])
						edges.append(numpy.concatenate((lo, [hi[-1]])))
					shape = [len(e)-1 for e in reversed(edges)]
					aeff = numpy.zeros([6] + list(reversed(shape)) + [2])
				aeff[i,...,j] += data[:,-2].reshape(shape).T
	
	return edges, aeff

from scipy import interpolate
import tables, dashi

class MuonSelectionEfficiency(object):
	def __init__(self, filename='aachen_muon_selection_efficiency.npz', energy_threshold=0):
		if not filename.startswith('/'):
			filename = os.path.join(data_dir, 'selection_efficiency', filename)
		if filename.endswith('.npz'):
			f = numpy.load(filename)
			self.interp = interpolate.interp1d(f['log_energy'], f['efficiency'],
			    bounds_error=False, fill_value=0.)
		elif filename.endswith('.hdf5'):
			with tables.open_file(filename) as f:
				generated = dashi.histload(f, '/generated')
				detected = dashi.histload(f, '/detected')
			sp = dashi.histfuncs.histratio(detected.project([0]), generated.project([0]))
	
			edges = numpy.concatenate((sp.x - sp.xerr, [sp.x[-1] + sp.xerr[-1]]))
			loge = 0.5*(numpy.log10(edges[1:]) + numpy.log10(edges[:-1]))
	
			loge = numpy.concatenate((loge, loge + loge[-1] + numpy.diff(loge)[0]))
			v = numpy.concatenate((sp.y, sp.y[-5:].mean()*numpy.ones(sp.y.size)))
			w = 1/numpy.concatenate((sp.yerr, 1e-2*numpy.ones(sp.yerr.size)))
			w[~numpy.isfinite(w)] = 1
	
			self.interp = interpolate.UnivariateSpline(loge, v, w=w)
		self.energy_threshold = energy_threshold

	def __call__(self, muon_energy, cos_theta):
		return numpy.where(muon_energy >= self.energy_threshold, numpy.clip(self.interp(numpy.log10(muon_energy)), 0, 1), 0.)

class ZenithDependentMuonSelectionEfficiency(object):
	def __init__(self, filename='sunflower_200m_bdt0_efficiency.fits', energy_threshold=0):
		from icecube.photospline import I3SplineTable
		if not filename.startswith('/'):
			filename = os.path.join(data_dir, 'selection_efficiency', filename)
		self._spline = I3SplineTable(filename)
		self.eval = numpy.vectorize(self._eval)
		self.energy_threshold = energy_threshold
	def _eval(self, loge, cos_theta):
		return self._spline.eval([loge, cos_theta])
	def __call__(self, muon_energy, cos_theta):
		loge, cos_theta = numpy.broadcast_arrays(numpy.log10(muon_energy), cos_theta)
		return numpy.where(muon_energy >= self.energy_threshold, numpy.clip(self.eval(numpy.log10(muon_energy), cos_theta), 0, 1), 0.)

def get_muon_selection_efficiency(geometry, spacing, energy_threshold=0):
	"""
	:param energy_threshold: artificial energy threshold in GeV
	"""
	if geometry == "IceCube":
		return MuonSelectionEfficiency(energy_threshold=energy_threshold)
	else:
		return ZenithDependentMuonSelectionEfficiency("%s_%dm_bdt0_efficiency.fits" % (geometry, spacing), energy_threshold=energy_threshold)

class StepFunction(object):
	"""
	A zenith-dependent energy threshold, modeling the effect of a perfect
	surface veto whose threshold scales with slant depth
	"""
	def __init__(self, threshold=0):
		self.threshold = threshold
	def accept(self, e_mu, cos_theta=1.):
		"""
		Return True if an event would pass the event selection
		"""
		return numpy.where(cos_theta > 0.05, e_mu > self.threshold/cos_theta, True)
	def veto(self, e_mu, cos_theta=1.):
		"""
		Return True if an atmospheric event would be rejected by the veto
		"""
		return numpy.where(cos_theta > 0.05, e_mu > self.threshold/cos_theta, False)

class MuonEffectiveArea(object):
	"""
	The product of geometric area and selection efficiency
	"""
	def __init__(self, geometry, spacing=125):
		self.geometry = geometry
		self.spacing = spacing
		self._surface = get_fiducial_surface(geometry, spacing)
		if geometry == "IceCube":
			self._efficiency = MuonSelectionEfficiency()
		else:
			self._efficiency = ZenithDependentMuonSelectionEfficiency('%s_%dm_bdt0_efficiency.fits' % (geometry, spacing))
	def __call__(self, muon_energy, cos_theta):
		geo = self._surface.azimuth_averaged_area(cos_theta)
		return geo * self._efficiency(muon_energy, cos_theta)

def _interpolate_muon_production_efficiency(cos_zenith):
	"""
	Get the probability that a muon neutrino of energy E_nu from zenith angle
	cos_theta will produce a muon that reaches the detector with energy E_mu
	
	:returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
	    edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
	    with the same axes.
	"""
	from scipy import interpolate
	with tables.open_file('/Users/jakob/Documents/IceCube/projects/2015/gen2_analysis/data/veto/numu.hdf5') as hdf:
		h = dashi.histload(hdf, '/muon_efficiency')
	edges = [numpy.log10(h.binedges[0]), h.binedges[1], numpy.log10(h.binedges[2])]
	centers = map(center, edges)
	newcenters = centers[0], numpy.clip(cos_zenith, centers[1].min(), centers[1].max()), centers[2]
	y = numpy.where(~(h.bincontent <= 0), numpy.log10(h.bincontent), -numpy.inf)
	assert not numpy.isnan(y).any()
	interpolant = interpolate.RegularGridInterpolator(centers, y, bounds_error=True, fill_value=-numpy.inf)
	
	xi = numpy.vstack(map(lambda x: x.flatten(), numpy.meshgrid(*newcenters, indexing='ij'))).T
	assert numpy.isfinite(xi).all()
	
	v = interpolant(xi, 'linear').reshape(map(lambda x: x.size, newcenters))
	v[~numpy.isfinite(v)] = -numpy.inf
	
	assert not numpy.isnan(v).any()
	
	return (h.binedges[0], None, h.binedges[2]), 10**v

def _ring_range(nside):
	"""
	Return the eqivalent cos(zenith) ranges for the rings of a HEALpix map
	with NSide *nside*.
	"""
	# get cos(colatitude) at the center of each ring, and invert to get
	# cos(zenith). This assumes that the underlying map is in equatorial
	# coordinates.
	centers = -healpy.ringinfo(nside, numpy.arange(1, 4*nside))[2]
	return numpy.concatenate(([-1], 0.5*(centers[1:]+centers[:-1]), [1]))

def get_muon_production_efficiency(ct_edges=None):
	"""
	Get the probability that a muon neutrino of energy E_nu from zenith angle
	cos_theta will produce a muon that reaches the detector with energy E_mu
	
	:param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
	    at the centers of these bins. If an integer, interpret as the NSide of
	    a HEALpix map
	:returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
	    edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
	    with the same axes.
	"""
	if ct_edges is None:
		ct_edges = numpy.linspace(-1, 1, 11)
	elif isinstance(ct_edges, int):
		nside = ct_edges
		ct_edges = _ring_range(nside)
	
	edges, efficiency = _interpolate_muon_production_efficiency(center(ct_edges))
	return (edges[0], ct_edges, edges[2]), efficiency

class effective_area(object):
	"""
	Effective area with metadata
	"""
	def __init__(self, edges, aeff, sky_binning='cos_theta'):
		self.bin_edges = edges
		self.values = aeff
		self.sky_binning = sky_binning
		self.dimensions = ['type', 'true_energy', 'true_zenith_band', 'reco_energy', 'reco_zenith_band', 'reco_signature']
	
	def compatible_with(self, other):
		return self.values.shape == other.values.shape and all(((a==b).all() for a, b in zip(self.bin_edges, other.bin_edges)))
	
	@property
	def is_healpix(self):
		return self.sky_binning == 'healpix'
	
	@property
	def nside(self):
		assert self.is_healpix
		return self.nring/4 + 1
	
	@property
	def nring(self):
		assert self.is_healpix
		return self.values.shape[2]
	
	@property
	def ring_repeat_pattern(self):
		assert self.is_healpix
		return healpy.ringinfo(self.nside, numpy.arange(self.nring)+1)[1]
	
def create_throughgoing_aeff(energy_resolution=get_energy_resolution("IceCube"),
    energy_threshold=StepFunction(),
    selection_efficiency=MuonSelectionEfficiency(),
    surface=get_fiducial_surface("IceCube"),
	cos_theta=None):
	"""
	Create effective areas in the same format as above
	
	:param energy_resolution: a muon energy resolution kernel
	:param selection_efficiency: an energy- and zenith-dependent muon selection efficiency
	:param surface: the fiducial surface surrounding the detector
	:param cos_theta: edges of bins in the cosine of the zenith angle. If None,
	    use the native binning of the efficiency histogram. If an integer,
	    interpret as the NSide of a HEALpix map
	"""
	
	# Ingredients:
	# 1) Muon production efficiency
	# 2) Geometric area
	# 3) Selection efficiency
	# 4) Energy resolution
	
	import tables, dashi
	from scipy.special import erf
	
	nside = None
	if isinstance(cos_theta, int):
		nside = cos_theta
	
	# Step 1: Efficiency for a neutrino to produce a muon that reaches the
	#         detector with a given energy
	(e_nu, cos_theta, e_mu), efficiency = get_muon_production_efficiency(cos_theta)
	
	# Step 2: Geometric muon effective area (no selection effects yet)
	# NB: assumes cylindrical symmetry.
	aeff = efficiency * (numpy.vectorize(surface.average_area)(cos_theta[:-1], cos_theta[1:])[None,:,None])
	
	# Step 3: apply selection efficiency
	selection_efficiency = selection_efficiency(*numpy.meshgrid(center(e_mu), center(cos_theta), indexing='ij')).T
	aeff *= selection_efficiency[None,:,:]
	
	# Step 4: apply smearing for energy resolution
	response = energy_resolution.get_response_matrix(e_mu, e_mu)
	aeff = numpy.apply_along_axis(numpy.inner, 2, aeff, response)
	
	total_aeff = numpy.zeros((6,) + aeff.shape + (aeff.shape[1], 2))
	# For now, we make the following assumptions:
	# a) muon channel is sensitive only to nu_mu, and all events are tracks
	# b) cross-section is identical for neutrino and antineutrino
	# b) angular resolution is perfect
	# in other words, write the effective area into a diagonal in nu_zenith, reco_zenith
	diag = numpy.diagonal(total_aeff[2:4,...,1], 0, 2, 4)
	assert id(diag.base) == id(total_aeff), "numpy.diagonal() must return a view"
	diag.setflags(write=True)
	diag[:] = aeff[None,:].transpose((0,1,3,2))
	
	# Step 5: apply an energy threshold in the southern hemisphere
	# print 
	total_aeff *= energy_threshold.accept(*numpy.meshgrid(center(e_mu), center(cos_theta), indexing='ij'))[None,None,None,...,None]
	
	edges = (e_nu, cos_theta, e_mu, cos_theta)
	
	return effective_area(edges, total_aeff, 'cos_theta' if nside is None else 'healpix')
