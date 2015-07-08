
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

class MuonSelectionEfficiency(object):
	def __init__(self):
		f = numpy.load(os.path.join(data_dir, 'veto', 'aachen_muon_selection_efficiency.npz'))
		self.interp = interpolate.interp1d(f['log_energy'], f['efficiency'],
		    bounds_error=False, fill_value=0.)
	def __call__(self, muon_energy):
		return self.interp(numpy.log10(muon_energy))

def create_throughgoing_aeff(energy_resolution=0.25,
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
	# FIXME: assumes cylindrical symmetry. either average out or add explicitly
	aeff = efficiency.bincontent * (surface.area(efficiency.bincenters[1], numpy.nan)[None,:,None])
	
	# Step 3: apply selection efficiency
	selection_efficiency = MuonSelectionEfficiency()
	if full_sky:
		# accept everything
		aeff *= selection_efficiency(efficiency.bincenters[2]/energy_threshold_scale)[None,None,:]
	else:
		# only accept events below the horizon
		eff = numpy.zeros(aeff.shape[1:])
		mask = efficiency.bincenters[1] < 0
		eff[mask,:] = selection_efficiency(efficiency.bincenters[2]/energy_threshold_scale)[None,:]
		# eff[mask,:] = 1.
		aeff *= eff[None,:,:]
	
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
	