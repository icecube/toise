import os
import numpy as np
from scipy.stats import multivariate_normal
from scipy import interpolate
from scipy.special import erf
from scipy.stats import cauchy

from .energy_resolution import EnergySmearingMatrix

import logging

logger = logging.getLogger("radio resolution parametrisation")


class RadioPointSpreadFunctionPickled(object):
    """A possible point spread function for radio, consisting two Gaussian terms and a constant term (well, ok, extremely poor reconstruction)"""

    def __init__(
        self,
        filename, selection
    ):
        self.filename = filename
        self.selection = selection
        # read the input data
        data = np.load(self.filename, allow_pickle=True)
        def angular_cdf(reco_errors):
            x = np.sort(reco_errors)
            # reco uncertainties
            xf = np.radians(np.concatenate([np.array([0]), x, np.array([180])]))
            # cumulative distribution for interpolation
            yf = np.concatenate([np.array([0]), np.cumsum(np.ones_like(x))/max(np.cumsum(np.ones_like(x))), np.array([1])])
            cdf_resolution_reconstructed = interpolate.interp1d(xf,yf)
            return cdf_resolution_reconstructed
        self.cdf = angular_cdf(data[f'distribution_{selection}'])

    #def PDF(self, space_angle):
    #    return self.pdf(
    #        space_angle
    #    )

    #def pdf(self, space_angle):
    #    """
    #    const = (
    #        norm_const
    #        * np.heaviside(space_angle, 1)
    #        * np.heaviside(self.max_const - space_angle, 1)
    #        / self.max_const
    #    )
    #    return (
    #        multivariate_normal.pdf(space_angle, mean=0, cov=sigma1**2) * 2 * norm1
    #        + multivariate_normal.pdf(space_angle, mean=0, cov=sigma2**2) * 2 * norm2
    #        + const
    #    )
    #    """

    def CDF(self, space_angle):
        return self.cdf(
            space_angle
        )

    #def scale_well_reconstructed_fraction(self, factor):
    #    print("scale_well_reconstructed_fraction not implemented")
    #    # rescale angles in CDF


    def __call__(self, psi, energy, cos_theta):
        psi, energy, cos_theta = np.broadcast_arrays(psi, energy, cos_theta)
        logger.info(psi)
        logger.info(np.shape(psi))
        evaluates = self.CDF(psi[:][0][0])
        return np.where(np.isfinite(evaluates), evaluates, 1.0)


class RadioEnergyResolution(EnergySmearingMatrix):
    """A 1D energy resolution matrix parameterised by a Cauchy function in log(Erec/Eshower)"""

    def __init__(
        self,
        lower_limit=np.log10(1.1),
        loc=0.01868963,
        scale=0.14255128,
        crossover_energy=1e6,
    ):
        super(RadioEnergyResolution, self).__init__()
        self._loc = loc
        self._scale = scale
        self._b = lower_limit
        self._a = self._b * np.sqrt(crossover_energy)

    def bias(self, loge):
        return loge

    def sigma(self, loge):
        return self._b + self._a / np.sqrt(10**loge)

    def set_params(self, paramdict):
        logger.debug("setting energy resolution parameters: {}".format(paramdict))
        if "loc" in paramdict:
            self._loc = paramdict["loc"]
        if "scale" in paramdict:
            self._scale = paramdict["scale"]
        for p in paramdict:
            if p not in ["loc", "scale"]:
                logger.warning("skipping invalid parameter: {}".format(p))

    def get_response_matrix(self, true_energy, reco_energy):
        """
        :param true_energy: edges of true muon energy bins
        :param reco_energy: edges of reconstructed muon energy bins
        """
        loge_true = np.log10(true_energy)
        loge_center = np.clip(0.5 * (loge_true[:-1] + loge_true[1:]), *self._loge_range)
        loge_width = np.diff(loge_true)
        loge_lo = np.log10(reco_energy[:-1])
        loge_hi = np.log10(reco_energy[1:])

        # evaluate at the right edge for maximum smearing on a falling spectrum
        mu, hi = np.meshgrid(self.bias(loge_center), loge_hi, indexing="ij")
        # do not use sigma for radio
        sigma, lo = np.meshgrid(self.sigma(loge_center), loge_lo, indexing="ij")

        return (
            (
                cauchy.cdf((hi - mu), self._loc, self._scale)
                - cauchy.cdf((lo - mu), self._loc, self._scale)
            )
        ).T


def efficiency_sigmoid(x, eff_low, eff_high, loge_turn, loge_halfmax):
    """sigmoid function in logE for efficiency between max(0, eff_low) and eff_high"""
    logx = np.log10(x)
    # choose factors conveniently
    # loge_halfmax should correspond to units in logE from turnover, where 0.25/0.75 of max are reached
    # = number of orders of magnitude in x between 0.25..0.75*(max-min) range
    b = np.log(3) / loge_halfmax

    eff = ((eff_low - eff_high) / (1 + (np.exp(b * (logx - loge_turn))))) + eff_high
    # do not allow below 0
    eff = np.maximum(0, eff)
    return eff


def bound_efficiency_sigmoid(x, eff_low, eff_high, loge_turn, loge_halfmax):
    """sigmoid function in logE for efficiency between 0 and 1"""
    # hard limits between 0 and 1
    eff = efficiency_sigmoid(x, eff_low, eff_high, loge_turn, loge_halfmax)
    # limit to range between 0 and 1
    eff = np.maximum(0, eff)
    eff = np.minimum(1, eff)
    return eff


def radio_analysis_efficiency(E, minval, maxval, log_turnon_gev, log_turnon_width):
    """
    A sigmoid analysis efficiency curve rising from minval to maxval and bounded by 0 and 1

    :param E: energy values for which to return efficiency values
    :param minval: sigmoid value for -inf limit
    :param maxval: sigmoid value for +inf limit
    :param log_turnon_gev: log10 energy turnon in GeV (the point at which 50% value between minval and maxval is reached)
    :param log_turnon_width: with of transition region (log10 width from 25% to 75% transition of the sigmoid)
    """
    # any3_gtr3
    # [-0.19848465  0.92543898  7.42347294  1.2133977 ]
    # 3phased_2support_gtr3
    # [-0.06101333  0.89062991  8.50399113  0.93591249]
    # 3power_gtr3
    # [-0.16480194  0.76853897  8.46903659  1.03517252]
    # 3power_2support_gtr3
    # [-0.0923469   0.73836631  8.72327879  0.85703575]
    return bound_efficiency_sigmoid(E, minval, maxval, log_turnon_gev, log_turnon_width)
