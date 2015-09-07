
from scipy import interpolate
import pickle, os, numpy
from scipy.special import erf

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def get_energy_resolution(geometry="Sunflower", spacing=200):
	if geometry == "IceCube":
		fname = "aachen_muon_energy_profile.npz"
	else:
		fname = "%s_%sm_bdt0_muon_energy_profile.npz" % (geometry.lower(), spacing)
	return MuonEnergyResolution(fname)

class MuonEnergyResolution(object):
	"""
	A parameterization of the inherent smearing in muon energy resolution
	"""
	def __init__(self, fname='aachen_muon_energy_profile.npz'):
		"""
		The default is a parameterization of the resolution of MuEx on the Aachen
		IC86 diffuse nu_mu sample.
		"""
		if not fname.startswith('/'):
			fname = os.path.join(data_dir, 'energy_reconstruction', fname)
		f = numpy.load(fname)
		self._loge_range = (f['loge'].min(), f['loge'].max())
		self._bias = interpolate.UnivariateSpline(f['loge'], f['mean'], s=f['smoothing'])
		self._sigma = interpolate.UnivariateSpline(f['loge'], f['std'], s=f['smoothing'])
	def get_response_matrix(self, true_energy, reco_energy):
		"""
		:param true_energy: edges of true muon energy bins
		:param reco_energy: edges of reconstructed muon energy bins
		"""
		loge_true = numpy.log10(true_energy)
		loge_center = numpy.clip(0.5*(loge_true[:-1]+loge_true[1:]), *self._loge_range)
		loge_lo = numpy.log10(reco_energy[:-1])
		loge_hi = numpy.log10(reco_energy[1:])
		
		mu, hi = numpy.meshgrid(self._bias(loge_center), loge_hi, indexing='ij')
		sigma, lo = numpy.meshgrid(self._sigma(loge_center), loge_lo, indexing='ij')
		
		return ((erf((hi-mu)/sigma)-erf((lo-mu)/sigma))/2.).T