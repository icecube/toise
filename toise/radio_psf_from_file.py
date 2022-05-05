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
            xf = np.radians(np.concatenate([np.array([0]), x, np.array([360])]))
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
            np.radians(space_angle)
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


