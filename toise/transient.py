from enum import Enum
from os import path

import numpy as np
from scipy import interpolate
from scipy.integrate import quad

from .pointsource import PointSource
from .util import PDGCode, constants, data_dir

# Enum has no facility for setting docstrings inline. Do it by hand.
TRANSIENT_MODELS = Enum(
    "TRANSIENT_MODELS",
    [
        "GRB_afterglow",
        "high_state_TDE",
        "blazar_flares",
        "sGRB_NSmerger",
        "BNS_merger_8hours",
        "BNS_merger_3days",
        "BNS_merger_1month",
        "BNS_merger_1year",
    ],
)
# transient models digitized by M. Cataldo, taken from https://doi.org/10.48550/arXiv.1810.09994, Fig. 8
TRANSIENT_MODELS.__doc__ = r"Transient source fluences"
TRANSIENT_MODELS.GRB_afterglow.__doc__ = r"GRB afterglow"
TRANSIENT_MODELS.high_state_TDE.__doc__ = r"high state TDE"
TRANSIENT_MODELS.blazar_flares.__doc__ = r"10x6 month blazar flares"
TRANSIENT_MODELS.sGRB_NSmerger.__doc__ = r"short GRB from BNS merger"
# transient models taken from ApJ 849 153, https://doi.org/10.3847/1538-4357/aa8b6a, Fig.4
TRANSIENT_MODELS.BNS_merger_8hours.__doc__ = (
    r"Fang&Metzger fluence for binary neutron star merger 1e3.5-1e4.5 s after merger"
)
TRANSIENT_MODELS.BNS_merger_3days.__doc__ = (
    r"Fang&Metzger fluence for binary neutron star merger 1e4.5-1e5.5 s after merger"
)
TRANSIENT_MODELS.BNS_merger_1month.__doc__ = (
    r"Fang&Metzger fluence for binary neutron star merger 1e5.5-1e6.5 s after merger"
)
TRANSIENT_MODELS.BNS_merger_1year.__doc__ = (
    r"Fang&Metzger fluence for binary neutron star merger 1e6.5-1e7.5 s after merger"
)

TRANSIENT_MODELS.GRB_afterglow.filename = "models/transient/GRB_afterglow.csv"
TRANSIENT_MODELS.high_state_TDE.filename = "models/transient/highTDE_fluence.csv"
TRANSIENT_MODELS.blazar_flares.filename = "models/transient/10x6_month_blazar_flare.csv"
TRANSIENT_MODELS.sGRB_NSmerger.filename = "models/transient/sGRB-NSmerger.csv"
TRANSIENT_MODELS.BNS_merger_8hours.filename = (
    "models/transient/FangMetz_1e3.5_1e4.5_sec.csv"
)
TRANSIENT_MODELS.BNS_merger_3days.filename = (
    "models/transient/FangMetz_1e4.5_1e5.5_sec.csv"
)
TRANSIENT_MODELS.BNS_merger_1month.filename = (
    "models/transient/FangMetz_1e5.5_1e6.5_sec.csv"
)
TRANSIENT_MODELS.BNS_merger_1year.filename = (
    "models/transient/FangMetz_1e6.5_1e7.5_sec.csv"
)

TRANSIENT_MODELS.GRB_afterglow.duration_sec = (
    24 * 3600
)  # 1 day, conservatively long, cf. https://doi.org/10.1103/PhysRevD.76.123001; https://doi.org/10.1086/432567
TRANSIENT_MODELS.high_state_TDE.duration_sec = 1e5
TRANSIENT_MODELS.blazar_flares.duration_sec = 10 * 6 * 2.628e6  # 10 x 6 months in sec
TRANSIENT_MODELS.sGRB_NSmerger.duration_sec = 2  # 2sec
TRANSIENT_MODELS.BNS_merger_8hours.duration_sec = 10**4.5 - 10**3.5
TRANSIENT_MODELS.BNS_merger_3days.duration_sec = 10**5.5 - 10**4.5
TRANSIENT_MODELS.BNS_merger_1month.duration_sec = 10**6.5 - 10**5.5
TRANSIENT_MODELS.BNS_merger_1year.duration_sec = 10**7.5 - 10**6.5

TRANSIENT_MODELS.GRB_afterglow.distance = 40 * constants.Mpc
TRANSIENT_MODELS.high_state_TDE.distance = 150 * constants.Mpc
TRANSIENT_MODELS.blazar_flares.distance = 2e3 * constants.Mpc
TRANSIENT_MODELS.sGRB_NSmerger.distance = 40 * constants.Mpc
TRANSIENT_MODELS.BNS_merger_8hours.distance = 10 * constants.Mpc
TRANSIENT_MODELS.BNS_merger_3days.distance = 10 * constants.Mpc
TRANSIENT_MODELS.BNS_merger_1month.distance = 10 * constants.Mpc
TRANSIENT_MODELS.BNS_merger_1year.distance = 10 * constants.Mpc


class TransientModelFluence(object):
    def __init__(self, pointsource_model, distance_mpc=None):
        if not isinstance(pointsource_model, TRANSIENT_MODELS):
            raise RuntimeError(f"No such transient model defined: {pointsource_model}")

        self.pointsource_model = pointsource_model
        self.transient_duration = pointsource_model.duration_sec
        self.transient_distance = pointsource_model.distance

        # load the model
        filename = path.join(data_dir, pointsource_model.filename)
        data = np.loadtxt(filename, delimiter=",")

        self.E = data[:, 0]
        self.fluence = data[:, 1:]

        logE = np.log10(self.E)
        logFluence = np.log10(self.fluence)

        # interpolant for all-flavor
        self._interpolant = interpolate.interp1d(
            logE,
            np.log10(np.sum(self.fluence, axis=1)) + 8,
            bounds_error=False,
            fill_value=-np.inf,
        )

        # interpolants per flavor (if given)
        self._has_per_flavor_fluence = False
        flavorcodes = [
            PDGCode.NuE,
            PDGCode.NuEBar,
            PDGCode.NuMu,
            PDGCode.NuMuBar,
            PDGCode.NuTau,
            PDGCode.NuTauBar,
        ]
        if np.shape(logFluence)[1] == 6:  # per flavor weight is given
            self._has_per_flavor_fluence = True
            self._interpolant_per_flavor = {
                flav: interpolate.interp1d(
                    logE, logFluence[:, i] + 8, bounds_error=False, fill_value=-np.inf
                )
                for i, flav in enumerate(flavorcodes)
            }

        # scaling for distance is just a distance squared scaling for surface area on the ball from source
        if distance_mpc is None:
            self.distance_factor = 1
        else:
            self.distance_factor = (
                self.transient_distance**2 / (distance_mpc * constants.Mpc) ** 2
            )

    def get_duration_years(self):
        return self.transient_duration / constants.annum

    def has_per_flavor_fluence(self):
        """flag if provided flux file was all-flavor or per-flavor"""
        return self._has_per_flavor_fluence

    def __call__(self, e_center, flavor=None, *args, **kwargs):
        assert flavor is None or flavor in [
            PDGCode.NuE,
            PDGCode.NuEBar,
            PDGCode.NuMu,
            PDGCode.NuMuBar,
            PDGCode.NuTau,
            PDGCode.NuTauBar,
        ]
        if flavor is None:
            # return all flavor fluence
            interpolant = self._interpolant
        else:
            interpolant = self._interpolant_per_flavor[flavor]

        fluence = 10 ** (interpolant(np.log10(e_center)) - 8) / e_center**2
        scaled_fluence = fluence * self.distance_factor * self.get_duration_years()
        return scaled_fluence


class TransientModel(PointSource):
    r"""
    A transient point source of neutrinos.

    The unit is the differential flux per neutrino flavor at 1 TeV,
    in units of :math:`10^{-12} \,\, \rm  TeV^{-1} \, cm^{-2} \, s^{-1}`

    """

    def __init__(
        self,
        effective_area,
        livetime,
        zenith_bin,
        pointsource_model,
        distance_mpc=None,
        emin=0,
        emax=np.inf,
        with_energy=True,
    ):

        # check if requested livetime is larger than transient timescale
        observation_time_scaling = 1
        fluence_curve = TransientModelFluence(pointsource_model, distance_mpc)
        if livetime > fluence_curve.get_duration_years():
            print(
                f"WARNING: requested longer livetime ({livetime}) than PS duration ({fluence_curve.get_duration_years()})"
            )
        elif livetime < fluence_curve.get_duration_years():
            print(
                f"WARNING: requested shorter livetime ({livetime}) than PS duration ({fluence_curve.get_duration_years()}), scaling down collected fluence assuming constant emission"
            )
            observation_time_scaling = livetime / fluence_curve.get_duration_years()

        energy = effective_area.bin_edges[0]

        # integrate over fluence curve
        if fluence_curve.has_per_flavor_fluence():
            fluence = np.asarray(
                [
                    [
                        quad(fluence_curve, energy[i], energy[i + 1], j)[0]
                        for i, e in enumerate(energy[:-1])
                    ]
                    for j in [
                        PDGCode.NuE,
                        PDGCode.NuEBar,
                        PDGCode.NuMu,
                        PDGCode.NuMuBar,
                        PDGCode.NuTau,
                        PDGCode.NuTauBar,
                    ]
                ]
            )
        else:
            fluence = (
                np.asarray(
                    [
                        quad(fluence_curve, energy[i], energy[i + 1])[0]
                        for i, e in enumerate(energy[:-1])
                    ]
                )
                / 6.0
            )

        # zero out fluence outside energy range
        fluence[(energy[:-1] > emax) | (energy[1:] < emin)] = 0

        # apply potential down-scaling if observation time is shorter than transient duration
        fluence *= observation_time_scaling

        PointSource.__init__(self, effective_area, fluence, zenith_bin, with_energy)
        self._livetime = livetime

    def spectral_weight(self, e_center, **kwargs):
        # we don't have a power law here... so return just ones
        return np.ones_like(e_center)
