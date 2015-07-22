from scipy import interpolate
import pickle, os, numpy

data_dir = os.path.join(os.path.dirname(__file__), 'data')

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
    def __init__(self, fname='aachen_psf.fits'):
        if not fname.startswith('/'):
            fname = os.path.join(data_dir, 'psf', fname)
        from icecube.photospline import I3SplineTable
        self._spline = I3SplineTable(fname)
        self._loge_extents, self._ct_extents = self._spline.extents[:2]
    def __call__(self, psi, energy, cos_theta):
        psi, loge, ct = numpy.broadcast_arrays(numpy.degrees(psi), numpy.log10(energy), cos_theta)
        loge = numpy.clip(loge, *self._loge_extents)
        ct = numpy.clip(ct, *self._ct_extents)
        
        return numpy.array([self._spline.eval(coords) for coords in zip(loge.flatten(), ct.flatten(), psi.flatten())]).reshape(psi.shape)