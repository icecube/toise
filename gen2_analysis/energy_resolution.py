from scipy import interpolate
import pickle
import os
import numpy
from scipy.special import erf

from .util import data_dir


def get_energy_resolution(geometry="Sunflower", spacing=200, channel="muon"):
    if channel == "cascade":
        return FictiveCascadeEnergyResolution()
    elif channel == "radio":
        return FictiveCascadeEnergyResolution(
            lower_limit=numpy.log10(1 + 0.2), crossover_energy=1e8
        )
    if geometry == "Fictive":
        return FictiveMuonEnergyResolution()
    elif geometry == "IceCube":
        fname = "aachen_muon_energy_profile.npz"
        # FIXME: we have to stretch the energy resolution for IC86 to get the
        # right shape at threshold. why?
        overdispersion = 1.5
    else:
        fname = "11900_MUONGUN_%s_%sm_recos.hd5_cut.npz" % (geometry, spacing)
        overdispersion = 1
    return MuonEnergyResolution(fname, overdispersion=overdispersion)


class EnergySmearingMatrix(object):
    def __init__(
        self,
        bias=None,
        sigma=None,
        loge_range=(-numpy.inf, numpy.inf),
        overdispersion=1.0,
    ):
        """
        The default is a parameterization of the resolution of MuEx on the Aachen
        IC86 diffuse nu_mu sample.
        """
        self._loge_range = loge_range
        self._bias = bias
        self._sigma = sigma

    def bias(self, loge):
        if self._bias is not None:
            return self._bias(loge)
        else:
            raise NotImplementedError

    def sigma(self, loge):
        if self._sigma is not None:
            return self._sigma(loge)
        else:
            raise NotImplementedError

    def get_response_matrix(self, true_energy, reco_energy):
        """
        :param true_energy: edges of true muon energy bins
        :param reco_energy: edges of reconstructed muon energy bins
        """
        loge_true = numpy.log10(true_energy)
        loge_center = numpy.clip(
            0.5 * (loge_true[:-1] + loge_true[1:]), *self._loge_range
        )
        loge_width = numpy.diff(loge_true)
        loge_lo = numpy.log10(reco_energy[:-1])
        loge_hi = numpy.log10(reco_energy[1:])
        # evaluate at the right edge for maximum smearing on a falling spectrum
        loge_center = loge_hi
        mu, hi = numpy.meshgrid(
            self.bias(loge_center) + loge_width, loge_hi, indexing="ij"
        )
        sigma, lo = numpy.meshgrid(self.sigma(loge_center), loge_lo, indexing="ij")

        return ((erf((hi - mu) / sigma) - erf((lo - mu) / sigma)) / 2.0).T


class MuonEnergyResolution(EnergySmearingMatrix):
    """
    A parameterization of the inherent smearing in muon energy resolution
    """

    def __init__(self, fname="aachen_muon_energy_profile.npz", overdispersion=1.0):
        """
        The default is a parameterization of the resolution of MuEx on the Aachen
        IC86 diffuse nu_mu sample.
        """
        if not fname.startswith("/"):
            fname = os.path.join(data_dir, "energy_reconstruction", fname)
        f = numpy.load(fname)
        loge_range = (f["loge"].min(), f["loge"].max())
        bias = interpolate.UnivariateSpline(f["loge"], f["mean"], s=f["smoothing"])
        sigma = interpolate.UnivariateSpline(
            f["loge"], f["std"] * overdispersion, s=f["smoothing"]
        )
        super(MuonEnergyResolution, self).__init__(
            bias, sigma, loge_range, overdispersion
        )


class FictiveMuonEnergyResolution(EnergySmearingMatrix):
    def bias(self, loge):
        return numpy.log10(10 ** (loge / 1.13) + 500)

    def sigma(self, loge):
        return 0.22 + 0.23 * (1 - numpy.exp(-(10 ** loge) / 5e6))


class FictiveCascadeEnergyResolution(EnergySmearingMatrix):
    def __init__(self, lower_limit=numpy.log10(1.1), crossover_energy=1e6):
        super(FictiveCascadeEnergyResolution, self).__init__()
        self._b = lower_limit
        self._a = self._b * numpy.sqrt(crossover_energy)

    def bias(self, loge):
        return loge

    def sigma(self, loge):
        return self._b + self._a / numpy.sqrt(10 ** loge)
