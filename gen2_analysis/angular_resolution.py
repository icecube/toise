from scipy import interpolate, stats
import pickle
import os
import numpy

from .util import data_dir, center


def get_angular_resolution(
    geometry="Sunflower", spacing=200, scale=1.0, psf_class=None, channel="muon"
):
    if channel == "cascade":
        return PotemkinCascadePointSpreadFunction()
    elif channel == "radio":
        return PotemkinCascadePointSpreadFunction(
            lower_limit=numpy.radians(2), crossover_energy=0
        )
    if geometry == "IceCube":
        fname = "aachen_psf.fits"
    elif psf_class is not None:
        fname = "%s_%s_kingpsf%d" % (geometry, spacing, psf_class[1])
        # return KingPointSpreadFunction(fname, psf_class=psf_class, scale=scale)
        return SplineKingPointSpreadFunction(fname)
    else:
        fname = "11900_MUONGUN_%s_%sm_recos.fits" % (geometry, spacing)
    return PointSpreadFunction(fname, scale)


class AngularResolution(object):
    def __init__(
        self, fname=os.path.join(data_dir, "veto", "aachen_angular_resolution.npz")
    ):
        f = numpy.load(fname)
        xd = f["log10_energy"]
        yd = f["cos_theta"]
        x, y = numpy.meshgrid(xd, yd)
        zd = f["median_opening_angle"]
        # extrapolate with a constant
        zd[-8:, :] = zd[-9, :]

        self._spline = interpolate.SmoothBivariateSpline(
            x.flatten(), y.flatten(), zd.T.flatten()
        )

    def median_opening_angle(self, energy, cos_theta):
        loge, ct = numpy.broadcast_arrays(numpy.log10(energy), cos_theta)

        mu_reco = self._spline.ev(loge.flatten(), ct.flatten()).reshape(loge.shape)

        # dirty hack: add the muon/neutrino opening angle in quadtrature
        return numpy.radians(numpy.sqrt(mu_reco ** 2 + 0.7 ** 2 / (10 ** (loge - 3))))


class PointSpreadFunction(object):
    def __init__(self, fname="aachen_psf.fits", scale=1.0):
        """
        :param scale: angular resolution scale. A scale of 0.5 will halve the
                      median opening angle, while 2 will double it.
        """
        if not fname.startswith("/"):
            fname = os.path.join(data_dir, "psf", fname)
        from photospline import SplineTable

        self._spline = SplineTable(fname)
        self._loge_extents, self._ct_extents = self._spline.extents[:2]
        if self._ct_extents == (-1, 0):
            self._mirror = True
        else:
            self._mirror = False
        self._scale = scale

    def __call__(self, psi, energy, cos_theta):
        psi, loge, ct = numpy.broadcast_arrays(
            numpy.degrees(psi) / self._scale, numpy.log10(energy), cos_theta
        )
        loge = numpy.clip(loge, *self._loge_extents)
        if self._mirror:
            ct = -numpy.abs(ct)
        ct = numpy.clip(ct, *self._ct_extents)

        evaluates = self._spline.evaluate_simple([loge, ct, psi])
        return numpy.where(numpy.isfinite(evaluates), evaluates, 1.0)


class _king_gen(stats.rv_continuous):
    """
    King function, used to parameterize the PSF in XMM and Fermi

    See: http://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/Cicerone/Cicerone_LAT_IRFs/IRF_PSF.html
    """

    def _argcheck(self, sigma, gamma):
        return (gamma > 1).all() and (sigma > 0).all()

    def _pdf(self, x, sigma, gamma):
        return (
            x
            / (sigma ** 2)
            * (1.0 - 1.0 / gamma)
            * (1 + 1.0 / (2.0 * gamma) * (x / sigma) ** 2) ** -gamma
        )

    def _cdf(self, x, sigma, gamma):
        x2 = x ** 2
        a = 2 * gamma * (sigma ** 2)
        b = 2 * sigma ** 2
        return (1.0 - 1.0 / gamma) / (a - b) * (a - (a + x2) * (x2 / a + 1) ** -gamma)


king = _king_gen(name="king", a=0.0)


class _fm_gen(stats.rv_continuous):
    """
    Fisher-von Mises distribution of cos(alpha), the equivalent of a normal
    distribution for distances between points on a 2-sphere
    """

    def _argcheck(self, kappa):
        return numpy.all(kappa >= 0)

    def _pdf(self, x, kappa):
        return numpy.exp(kappa * x) * kappa / (2 * numpy.pi * numpy.sinh(kappa))


fisher = _fm_gen(name="fisher", a=-1, b=1)


class KingPointSpreadFunctionBase(object):
    def __init__(self, scale=1.0):
        self._scale = scale

    def get_quantile(self, p, energy, cos_theta):
        p, loge, ct = numpy.broadcast_arrays(p, numpy.log10(energy), cos_theta)
        if hasattr(self._scale, "__call__"):
            scale = self._scale(10 ** loge)
        else:
            scale = self._scale
        sigma, gamma = self.get_params(loge, ct)
        return numpy.radians(king.ppf(p, sigma, gamma)) / scale

    def __call__(self, psi, energy, cos_theta):
        psi, loge, ct = numpy.broadcast_arrays(
            numpy.degrees(psi), numpy.log10(energy), cos_theta
        )
        if hasattr(self._scale, "__call__"):
            scale = self._scale(10 ** loge)
        else:
            scale = self._scale
        sigma, gamma = self.get_params(loge, ct)
        return king.cdf(psi / scale, sigma, gamma)


class KingPointSpreadFunction(KingPointSpreadFunctionBase):
    def __init__(
        self, fname="Sunflower_240_kingpsf4", psf_class=(0, 4), scale=1.0, **kwargs
    ):
        super(KingPointSpreadFunction, self).__init__(**kwargs)
        import pandas as pd
        import operator
        from scipy import interpolate

        if not fname.startswith("/"):
            fname = os.path.join(data_dir, "psf", fname)
        params = pd.read_pickle(fname + ".pickle")
        bins = pd.read_pickle(fname + ".bins.pickle")
        params = pd.DataFrame(
            {k: params.apply(operator.itemgetter(k)) for k in ("sigma", "gamma")}
        )
        # remove energy underflow bin
        for k in params.index.levels[0]:
            del params.sigma[k, 0]
            del params.gamma[k, 0]
        x, y = list(map(center, bins))
        key = str(psf_class[0])
        self._sigma = interpolate.RectBivariateSpline(
            x, y, abs(params.sigma[key]).values.reshape(9, 10), s=5e-1
        )
        self._gamma = interpolate.RectBivariateSpline(
            x, y, params.gamma[key].values.reshape(9, 10), s=1e1
        )
        self._scale = scale

    def get_params(self, log_energy, cos_theta):
        """
        Interpolate for sigma and gamma
        """
        return self._sigma(log_energy, cos_theta, grid=False), self._gamma(
            log_energy, cos_theta, grid=False
        )


class SplineKingPointSpreadFunction(KingPointSpreadFunctionBase):
    def __init__(self, fname="Sunflower_240_kingpsf1", **kwargs):
        super(SplineKingPointSpreadFunction, self).__init__(**kwargs)
        from photospline import SplineTable

        if not fname.startswith("/"):
            fname = os.path.join(data_dir, "psf", fname)

        self._splines = dict(
            sigma=SplineTable(fname + ".sigma.fits"),
            gamma=SplineTable(fname + ".gamma.fits"),
        )

    def get_params(self, log_energy, cos_theta):
        """
        Interpolate for sigma and gamma
        """

        sigma = 10 ** self._splines["sigma"].evaluate_simple([log_energy, cos_theta])
        gamma = (
            10 ** self._splines["gamma"].evaluate_simple([log_energy, cos_theta]) + 1
        )

        return sigma, gamma


class PotemkinCascadePointSpreadFunction(object):
    def __init__(self, lower_limit=numpy.radians(5), crossover_energy=1e6):
        self._b = lower_limit
        self._a = self._b * numpy.sqrt(crossover_energy)

    def __call__(self, psi, energy, cos_theta):

        psi, energy, cos_theta = numpy.broadcast_arrays(psi, energy, cos_theta)
        sigma = self._a / numpy.sqrt(energy) + self._b

        evaluates = 1 - numpy.exp(-(psi ** 2) / (2 * sigma ** 2))
        return numpy.where(numpy.isfinite(evaluates), evaluates, 1.0)
