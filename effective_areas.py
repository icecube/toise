
import os
import numpy
import itertools

from surfaces import Cylinder

data_dir = os.path.join(os.path.dirname(__file__), 'data')

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
	def __init__(self, filename='aachen_muon_selection_efficiency.npz'):
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


	def __call__(self, muon_energy, cos_theta):
		return numpy.clip(self.interp(numpy.log10(muon_energy)), 0, 1)

class ZenithDependentMuonSelectionEfficiency(object):
	def __init__(self, filename='sunflower_200m_bdt0_efficiency.fits'):
		from icecube.photospline import I3SplineTable
		if not filename.startswith('/'):
			filename = os.path.join(data_dir, 'selection_efficiency', filename)
		self._spline = I3SplineTable(filename)
		self.eval = numpy.vectorize(self._eval)
	def _eval(self, loge, cos_theta):
		return self._spline.eval([loge, cos_theta])
	def __call__(self, muon_energy, cos_theta):
		loge, cos_theta = numpy.broadcast_arrays(numpy.log10(muon_energy), cos_theta)
		return numpy.clip(self.eval(numpy.log10(muon_energy), cos_theta), 0, 1)

def create_throughgoing_aeff(energy_resolution=0.25,
    selection_efficiency=MuonSelectionEfficiency(),
    full_sky=False, energy_threshold_scale=1., surface=Cylinder()):
	"""
	Create effective areas in the same format as above
	"""
	
	# Ingredients:
	# 1) Muon production efficiency
	# 2) Geometric area
	# 3) Selection efficiency
	# 4) Energy resolution
	
	import tables, dashi
	from scipy.special import erf
	
	# Step 1: Efficiency for a neutrino to produce a muon that reaches the
	#         detector with a given energy
	with tables.open_file(os.path.join(data_dir, 'veto', 'numu.hdf5')) as hdf:
		efficiency = dashi.histload(hdf, '/muon_efficiency')
	
	# Step 2: Geometric muon effective area (no selection effects yet)
	# FIXME: assumes cylindrical symmetry.
	aeff = efficiency.bincontent * (numpy.vectorize(surface.average_area)(efficiency.binedges[1][:-1], efficiency.binedges[1][1:])[None,:,None])
	
	# Step 3: apply selection efficiency
	emu, cos_theta = numpy.meshgrid(efficiency.bincenters[2]/energy_threshold_scale, efficiency.bincenters[1], indexing='ij')
	selection_efficiency = selection_efficiency(emu, cos_theta).T
	if not full_sky:
		# only accept events from below the horizon
		selection_efficiency[cos_theta.T > 0] = 0
	aeff *= selection_efficiency[None,:,:]
	
	# Step 4: apply smearing for energy resolution
	binwidth = numpy.log10(efficiency.binedges[2][1]/efficiency.binedges[2][0])
	# distance from each bin edge to the middle of the central bin, in sigmas
	t = numpy.linspace(-9.5, 9.5, 20)*(binwidth/energy_resolution)
	kernel = numpy.diff(erf(t))
	kernel /= kernel.sum()
	aeff = numpy.apply_along_axis(numpy.convolve, 2, aeff, kernel, mode='same')
	
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
	
	edges = efficiency.binedges + [efficiency.binedges[1]]
	
	return edges, total_aeff
	