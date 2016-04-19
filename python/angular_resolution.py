from scipy import interpolate
import pickle, os, numpy

from .util import data_dir

def get_angular_resolution(geometry="Sunflower", spacing=200, scale=1., channel='muon'):
	if channel == 'cascade':
		return PotemkinCascadePointSpreadFunction()
	if geometry == "IceCube":
		fname = "aachen_psf.fits"
	else:
		fname = "%s_%s_bdt0_psf.fits" % (geometry.lower(), spacing)
	return PointSpreadFunction(fname, scale)

class AngularResolution(object):
    def __init__(self, fname=os.path.join(data_dir, 'veto', 'aachen_angular_resolution.npz')):
        f = numpy.load(fname)
        xd = f['log10_energy']
        yd = f['cos_theta']
        x, y = numpy.meshgrid(xd, yd)
        zd = f['median_opening_angle']
        # extrapolate with a constant
        zd[-8:,:] = zd[-9,:]

        self._spline = interpolate.SmoothBivariateSpline(x.flatten(), y.flatten(), zd.T.flatten())
    
    def median_opening_angle(self, energy, cos_theta):
        loge, ct = numpy.broadcast_arrays(numpy.log10(energy), cos_theta)
        
        mu_reco = self._spline.ev(loge.flatten(), ct.flatten()).reshape(loge.shape)
        
        # dirty hack: add the muon/neutrino opening angle in quadtrature
        return numpy.radians(numpy.sqrt(mu_reco**2 + 0.7**2/(10**(loge-3))))

class PointSpreadFunction(object):
    def __init__(self, fname='aachen_psf.fits', scale=1.):
        """
        :param scale: angular resolution scale. A scale of 0.5 will halve the
                      median opening angle, while 2 will double it.
        """
        if not fname.startswith('/'):
            fname = os.path.join(data_dir, 'psf', fname)
        from icecube.photospline import I3SplineTable
        self._spline = I3SplineTable(fname)
        self._loge_extents, self._ct_extents = self._spline.extents[:2]
        if self._ct_extents == (-1, 0):
            self._mirror = True
        else:
            self._mirror = False
        self._scale = scale
    def __call__(self, psi, energy, cos_theta):
        psi, loge, ct = numpy.broadcast_arrays(numpy.degrees(psi)/self._scale, numpy.log10(energy), cos_theta)
        loge = numpy.clip(loge, *self._loge_extents)
        if self._mirror:
            ct = -numpy.abs(ct)
        ct = numpy.clip(ct, *self._ct_extents)
        
        return numpy.array([self._spline.eval(coords) for coords in zip(loge.flatten(), ct.flatten(), psi.flatten())]).reshape(psi.shape)

class PotemkinCascadePointSpreadFunction(object):
    
    def __init__(self, lower_limit=numpy.radians(5), crossover_energy=1e6):
        self._b = lower_limit
        self._a = self._b*numpy.sqrt(crossover_energy)
    
    def __call__(self, psi, energy, cos_theta):
        
        psi, energy, cos_theta = numpy.broadcast_arrays(psi, energy, cos_theta)
        sigma = self._a/numpy.sqrt(energy) + self._b
        
        return 1 - numpy.exp(-psi**2/(2*sigma**2))