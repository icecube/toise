from os import path
from enum import Enum
from .util import data_dir, constants
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import quad
from .pointsource import PointSource

# Enum has no facility for setting docstrings inline. Do it by hand.
TRANSIENT_MODEL = Enum("TRANSIENT_MODEL", ["GRB_afterglow",
                               "high_state_TDE",
                               "blazar_flares",
                               "sGRB_NSmerger",
                               "BNS_merger_8hours",
                               "BNS_merger_3days",
                               "BNS_merger_1month",
                               "BNS_merger_1year"])
TRANSIENT_MODEL.__doc__ = r"Transient source fluences"
TRANSIENT_MODEL.GRB_afterglow.__doc__ = r"GRB afterglow"
TRANSIENT_MODEL.high_state_TDE.__doc__ = r"high state TDE"
TRANSIENT_MODEL.blazar_flares.__doc__ = r"10x6 month blazar flares"
TRANSIENT_MODEL.sGRB_NSmerger.__doc__ = r"short GRB from BNS merger"
TRANSIENT_MODEL.BNS_merger_8hours.__doc__ = r"Fang&Metzger fluence for binary neutron star merger 1e3.5-1e4.5 s after merger"
TRANSIENT_MODEL.BNS_merger_3days.__doc__ = r"Fang&Metzger fluence for binary neutron star merger 1e4.5-1e5.5 s after merger"
TRANSIENT_MODEL.BNS_merger_1month.__doc__ = r"Fang&Metzger fluence for binary neutron star merger 1e5.5-1e6.5 s after merger"
TRANSIENT_MODEL.BNS_merger_1year.__doc__ = r"Fang&Metzger fluence for binary neutron star merger 1e6.5-1e7.5 s after merger"

TRANSIENT_MODEL.GRB_afterglow.filename = "models/transient/GRB_afterglow.csv"
TRANSIENT_MODEL.high_state_TDE.filename = "models/transient/highTDE_fluence.csv"
TRANSIENT_MODEL.blazar_flares.filename = "models/transient/10x6_month_blazar_flare.csv"
TRANSIENT_MODEL.sGRB_NSmerger.filename = "models/transient/sGRB-NSmerger.csv"
TRANSIENT_MODEL.BNS_merger_8hours.filename = "models/transient/FangMetz_1e3.5_1e4.5_sec.csv" 
TRANSIENT_MODEL.BNS_merger_3days.filename = "models/transient/FangMetz_1e4.5_1e5.5_sec.csv"
TRANSIENT_MODEL.BNS_merger_1month.filename = "models/transient/FangMetz_1e5.5_1e6.5_sec.csv"
TRANSIENT_MODEL.BNS_merger_1year.filename = "models/transient/FangMetz_1e6.5_1e7.5_sec.csv"


class TransientPointsourceFluence(object):
    def __init__(self, pointsource_model, distance_mpc=None):

        self.pointsource_model = pointsource_model
        #TODO use units Mpc for distance
        Mpc = 1
        if self.pointsource_model == TRANSIENT_MODEL.GRB_afterglow:
            self.transient_duration = 24 * 3600 # 1 day, conservatively long, cf. https://doi.org/10.1103/PhysRevD.76.123001; https://doi.org/10.1086/432567
            self.transient_distance = 40 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.high_state_TDE:
            self.transient_duration = 1e5
            self.transient_distance = 150 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.sGRB_NSmerger:
            self.transient_duration = 2 # 2sec
            self.transient_distance = 40 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.blazar_flares:
            self.transient_duration = 60 * 2.628e+6 # 10 x 6 months in sec
            self.transient_distance = 2e3 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.BNS_merger_8hours:
            self.transient_duration = 10**4.5 - 10**3.5
            self.transient_distance = 10 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.BNS_merger_3days:
            self.transient_duration = 10**5.5 - 10**4.5
            self.transient_distance = 10 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.BNS_merger_1month:
            self.transient_duration = 10**6.5 - 10**5.5
            self.transient_distance = 10 * Mpc
        elif self.pointsource_model == TRANSIENT_MODEL.BNS_merger_1year:
            self.transient_duration = 10**7.5 - 10**6.5
            self.transient_distance = 10 * Mpc
        else:
            raise RuntimeError(f"No such pointsource model: {self.pointsource_model}")
        filename=path.join(data_dir, pointsource_model.filename)
        data = np.loadtxt(filename, delimiter=",")

        self.E = data[:,0]
        self.fluence = data[:,1]

        self.interp = interp1d(np.log10(self.E), np.log10(self.fluence), bounds_error=False, fill_value="extrapolate")

        # scaling for distance is just a distance squared scaling for surface area on the ball from source
        if distance_mpc is None:
            self.distance_factor = 1
        else:
            self.distance_factor = self.transient_distance**2/distance_mpc**2

    def get_duration_years(self):
        return self.transient_duration / constants.annum

    def __call__(self, E, *args, **kwargs):
        # factor of 3 for flavor, 2 for neutrino/antineutrino
        # 10**-12 standard units for toise
        fluence = 10**self.interp(np.log10(E))/ E**2 / 3 / 2  * self.distance_factor * self.get_duration_years()
        return fluence

class TransientPointSource(PointSource):
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
        distance_mpc = None,
        emin=0,
        emax=np.inf,
        with_energy=True,
    ):

        #check if requested livetime is larger than transient timescale
        observation_time_scaling = 1
        fluence_curve = TransientPointsourceFluence(pointsource_model, distance_mpc)
        if livetime > fluence_curve.get_duration_years():
            print(f"WARNING: requested longer livetime ({livetime}) than PS duration ({fluence_curve.get_duration_years()})")
        elif livetime < fluence_curve.get_duration_years():
            print(f"WARNING: requested shorter livetime ({livetime}) than PS duration ({fluence_curve.get_duration_years()}), scaling down collected fluence assuming constant emission")
            observation_time_scaling = livetime / fluence_curve.get_duration_years()

        energy = effective_area.bin_edges[0]

        # integrate over fluence curve
        fluence = np.asarray(
              [quad(fluence_curve, energy[i], energy[i + 1])[0] for i, e in enumerate(energy[:-1])]
        )

        # zero out fluence outside energy range
        fluence[(energy[:-1] > emax) | (energy[1:] < emin)] = 0

        # apply potential down-scaling if observation time is shorter than transient duration
        fluence *= observation_time_scaling

        PointSource.__init__(self, effective_area, fluence, zenith_bin, with_energy)
        self._livetime = livetime

    def spectral_weight(self, e_center, **kwargs):
        return np.ones_like(e_center)
