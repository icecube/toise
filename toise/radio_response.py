import os
import numpy as np
from scipy.stats import multivariate_normal
from scipy import interpolate
from scipy.special import erf
from scipy.stats import cauchy

from .energy_resolution import EnergySmearingMatrix

import logging

logger = logging.getLogger("radio resolution parametrisation")


class RadioPointSpreadFunction(object):
    """A possible point spread function for radio, consisting two Gaussian terms and a constant term (well, ok, extremely poor reconstruction)"""

    def __init__(
        self,
        norm1=0.6192978004334891,
        sigma1=3.841393561107805,
        norm2=0.2130050363262992,
        sigma2=33.95551135597404,
        norm_const=0.1676971632402118,
    ):
        self.sigma1 = abs(sigma1)
        self.sigma2 = abs(sigma2)
        self.norm1 = abs(norm1)
        self.norm2 = abs(norm2)
        self.norm_const = abs(norm_const)
        # arbitrarily use a constant term in the range from 0 to 100 deg
        self.max_const = 100.0

    def PDF(self, space_angle):
        return self.pdf(
            space_angle,
            self.norm1,
            self.sigma1,
            self.norm2,
            self.sigma2,
            self.norm_const,
        )

    def pdf(self, space_angle, norm1, sigma1, norm2, sigma2, norm_const):
        const = (
            norm_const
            * np.heaviside(space_angle, 1)
            * np.heaviside(self.max_const - space_angle, 1)
            / self.max_const
        )
        return (
            multivariate_normal.pdf(space_angle, mean=0, cov=sigma1) * 2 * norm1
            + multivariate_normal.pdf(space_angle, mean=0, cov=sigma2) * 2 * norm2
            + const
        )

    def CDF(self, space_angle):
        return self.cdf(
            space_angle,
            self.norm1,
            self.sigma1,
            self.norm2,
            self.sigma2,
            self.norm_const,
        )

    def cdf(self, space_angle, norm1, sigma1, norm2, sigma2, norm_const):
        const = (
            norm_const
            * np.heaviside(space_angle, 1)
            * np.heaviside(self.max_const - space_angle, 1)
            / self.max_const
            * space_angle
            + np.heaviside(space_angle - self.max_const, 0) * norm_const
        )
        return (
            multivariate_normal.cdf(space_angle, mean=0, cov=sigma1) * 2 * norm1
            - norm1
            + multivariate_normal.cdf(space_angle, mean=0, cov=sigma2) * 2 * norm2
            - norm2
            + const
        )

    def store_result(self, result):
        n1, s1, n2, s2, self.norm_const = result

        if s1 < s2:
            self.sigma1 = s1
            self.norm1 = n1
            self.sigma2 = s2
            self.norm2 = n2
        else:
            self.sigma1 = s2
            self.norm1 = n2
            self.sigma2 = s1
            self.norm2 = n1

        totalNorm = self.norm1 + self.norm2 + self.norm_const
        self.totalNorm = totalNorm
        self.norm1 /= totalNorm
        self.norm2 /= totalNorm
        self.norm_const /= totalNorm
        logger.info("-- stored results:")
        logger.info("   norm1: {}".format(self.norm1))
        logger.info("   sigma1: {}".format(self.sigma1))
        logger.info("   norm2: {}".format(self.norm2))
        logger.info("   sigma2: {}".format(self.sigma2))
        logger.info("   norm_const: {}".format(self.norm_const))

    def set_params(self, paramdict):
        result = [
            paramdict["norm1"],
            paramdict["sigma1"],
            paramdict["norm2"],
            paramdict["sigma2"],
            paramdict["norm_const"],
        ]
        self.store_result(result)

    def scale_well_reconstructed_fraction(self, factor):
        n_good, n_bad1, n_bad2 = self.norm1, self.norm2, self.norm_const
        if n_good * factor > 1:
            logger.warning("limiting fraction to 100%")
            factor = np.minimum(1.0 / n_good, factor)
        factor_bad = (1.0 - factor * n_good) / (1.0 - n_good)
        self.norm1 *= factor
        self.norm2 *= factor_bad
        self.norm_const *= factor_bad

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
