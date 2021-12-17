import numpy as np
from scipy import stats, interpolate
from io import StringIO
import warnings
from . import pointsource


class Winter2014GRBFluence(object):
    """
    Neutrino fluence per burst from multiple internal shocks

    See:
    Neutrino and cosmic-ray emission from multiple internal shocks in gamma-ray bursts
    http://arxiv.org/abs/1409.2874
    """

    def __init__(self):
        # quasi-diffuse flux from 667 1e53 erg bursts per year at redshift 2
        # GeV / cm^2 sr s

        energy, diffuse_flux = np.loadtxt(
            StringIO(
                """74.865	3.105e-16
		    185.820 1.270e-15
		    669.845 7.765e-15
		    4902.930	9.101e-14
		    18649.945	3.946e-13
		    73595.469	1.405e-12
		    3.249e5 4.692e-12
		    9.775e5 7.876e-12
		    6.331e6 1.132e-11
		    1.614e7 1.004e-11
		    3.288e7 7.831e-12
		    6.755e7 4.924e-12
		    1.241e8 2.933e-12
		    3.467e8 1.177e-12
		    9.580e8 4.844e-13
		    2.076e9 2.319e-13
		    3.819e9 1.234e-13
		    7.155e9 5.527e-14
		    1.027e10	3.516e-14"""
            )
        ).T

        grbs_per_second = 667.0 / units.year
        # fluence per per burst: GeV/cm^2
        # factor of 2 for neutrino/antineutrino
        fluence = diffuse_flux / 2 * (4 * np.pi) / grbs_per_second

        self.dl = LuminosityDistance.instance()
        self._log_fluence = interpolate.interp1d(
            np.log10(energy), np.log10(fluence), kind="cubic", bounds_error=False
        )

    def __call__(self, E, Eiso=1e53, z=2):
        # closer bursts are brighter
        geometric_scale = (self.dl(2.0) / self.dl(z)) ** 2
        # brighter bursts are...brighter
        luminosity_scale = Eiso / 1e53
        # apply redshift
        energy = E * (1.0 + z) / (1.0 + 2)

        return (
            geometric_scale
            * luminosity_scale
            * (10 ** self._log_fluence(np.log10(energy)))
            / energy ** 2
        )


class WaxmannBahcallFluence(object):
    def __call__(self, E, *args, **kwargs):
        # E. Waxman, Nucl. Phys. B, Proc. Suppl. 118, 353 (2003).
        # fluence from a single GRB, assuming 667 bursts per year over the whole sky
        peryear = 667 / (3600 * 24 * 365 * 4 * np.pi)
        # factor of 3 for flavor, 2 for neutrino/antineutrino
        fluence = 0.9e-8 / 3 / 2 / peryear
        return np.where(
            E < 1e5,
            E ** -1 * fluence / 1e5,
            np.where(E > 1e7, E ** -4 * (fluence * (1e7 ** 2)), E ** -2 * fluence),
        )


class GRBPopulation(pointsource.PointSource):
    def __init__(self, effective_area, z, Eiso, with_energy=True):
        """
        :param z: redshift of bursts
        :param Eiso: isotropic energy output of bursts, in erg
        """

        # perburst() returns a fluence in 1/GeV cm^2 per burst
        perburst = Winter2014GRBFluence()
        energy = effective_area.bin_edges[0]

        def intflux(e, gamma=-2):
            """
            Integrate burst spectrum, assuming that
            """
            return (e ** (1 + gamma)) / (1 + gamma)

        norm = perburst(energy[1:, None], Eiso=Eiso, z=z) * energy[1:, None] ** 2

        # 1/cm^2
        fluence = norm * (intflux(energy[1:, None]) - intflux(energy[:-1, None]))
        fluence[~np.isfinite(fluence)] = 0

        nbands = effective_area.bin_edges[1].size - 1
        # sum over all bursts (assuming that observe for t90 each time)
        pointsource.PointSource.__init__(
            self,
            effective_area,
            fluence.sum(axis=1) * 0.9 / nbands,
            slice(None),
            with_energy,
        )


class LuminosityDistance(object):
    _instance = None

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(
        self,
        zmax=10,
        steps=1000,
        O_M=0.3,
        O_L=0.7,
        w=-1,
        v_dip=[0, 0, 0],
        v_mon=0,
        coords=[0, 0],
        Flat=True,
        EOS=False,
        H_0=70,
        der=False,
        dip=False,
    ):
        """
        Calculate luminosity distance in cm for the given redshift

        authors: ulrich feindt, matthias kerschhaggl

        :param z: redshift
        """
        from scipy import integrate, interpolate

        c = 299792.458  # speed of light in km/s

        if Flat:
            O_L = 1 - O_M
            O_K = 0
        else:
            O_K = 1 - O_M - O_L

        # H_rec = 1/(H(z)/H_0)
        if EOS:

            def H_rec(x):
                return 1 / np.sqrt(
                    O_M * (1 + x) ** 3
                    + (O_K) * (1 + x) ** 2
                    + O_L * (1 + x) ** (3 * (1 + w))
                )

        else:

            def H_rec(x):
                return 1 / np.sqrt(O_M * (1 + x) ** 3 + (O_K) * (1 + x) ** 2 + O_L)

        # integral along line of sight
        z = np.logspace(-12, np.log10(zmax), steps)
        integral = integrate.cumtrapz(H_rec(z), z, initial=0.0)

        if O_K == 0:
            result = c * (1 + z) / H_0 * integral
        elif O_K < 0:
            result = (
                c * (1 + z) / H_0 / np.sqrt(-O_K) * np.sin(np.sqrt(-O_K) * integral)
            )
        else:
            result = c * (1 + z) / H_0 / np.sqrt(O_K) * np.sinh(np.sqrt(O_K) * integral)

        if dip:
            if not Flat:
                warnings.warn("dipole not properly implemented for non-flat cosmology")
            cos_theta = np.sin(coords[0]) * np.sin(v_dip[1]) * np.cos(
                coords[1] - v_dip[2]
            ) + np.cos(coords[0]) * np.cos(v_dip[1])
            result_dip = (1 + z) ** 2 * H_rec(z) / H_0 * (v_mon + v_dip[0] * cos_theta)
            result -= result_dip

        result /= units.cm

        self._z = z
        self._dl = result

        self._interpolant = interpolate.interp1d(z, result)

    def __call__(self, z):
        """:returns: luminosity distance in cm"""
        return self._interpolant(z)


class SwiftTriggerHorizon(object):
    """
    Traced from Fig 6a of:

    Reference: http://arxiv.org/pdf/0912.0709v2.pdf
    D. Wandermann and T. Piran, 2010
    redshift and luminosity distribution calculated from Swift detected GRBs
    """

    def __init__(self):
        from io import StringIO

        logL, z = np.loadtxt(
            StringIO(
                """
		49.722	0.599
		49.952	0.735
		50.096	0.835
		50.276	0.954
		50.408	1.085
		50.591	1.275
		50.739	1.434
		50.869	1.621
		51.063	1.905
		51.167	2.082
		51.270	2.248
		51.373	2.469
		51.502	2.748
		51.722	3.290
		51.809	3.542
		51.900	3.806
		51.981	4.065
		52.054	4.321
		52.144	4.649
		52.205	4.879
		52.266	5.119
		52.326	5.374
		52.438	5.865
		52.511	6.260
		52.564	6.506
		52.612	6.743
		52.648	6.957
		52.692	7.214
		52.769	7.622
		52.806	7.882
		52.857	8.186
		52.898	8.465
		52.946	8.776
		52.976	8.963
		"""
            )
        ).T

        self._interpolant = interpolate.interp1d(
            logL, z, bounds_error=False, fill_value=np.inf
        )

    def __call__(self, peak_luminosity):
        """
        :param peak_luminosity: peak isotropic GRB luminosity, in erg/s
        :returns: maximum redshift z at which a GRB would trigger Swift
        """
        logL = np.log10(peak_luminosity)
        return self._interpolant(logL)


class units:
    cm = 3.24077929e-25  # 1 cm in Mpc
    GeV = 624.15  # 1 erg in GeV
    year = 365 * 24 * 3600


def grb_density(z):
    """
    Redshift distribution of GRBs

    :param z: redshift
    :returns: GRB rate per cubic Mpc per s

    Reference: http://arxiv.org/pdf/0912.0709v2.pdf
    D. Wandermann and T. Piran, 2010
    redshift and luminosity distribution calculated from Swift detected GRBs
    """
    n1 = 2.07
    n2 = -1.36
    z1 = 3.11
    # local GRB rate, per cubic Mpc per s
    R0 = 1.25e-9 / (3600.0 * 24.0 * 365.0)
    return R0 * np.where(z <= z1, (1 + z) ** n1, (1 + z1) ** (n1 - n2) * (1 + z) ** n2)


def density_to_comoving_rate(density_function, z):
    return (density_function(z) * units.cm ** 3) * comoving_volume(z) / (1.0 + z)


def grb_rate(z):
    """
    :returns: GRB rate in sr^-1 s^-1
    """
    return (grb_density(z) * units.cm ** 3) * comoving_volume(z) / (1.0 + z)


def luminosity(L):
    """
    GRB luminosity function
    :returns: peak luminosity in ergs/s

    Reference: http://arxiv.org/pdf/0912.0709v2.pdf
    D. Wandermann and T. Piran, 2010
    redshift and luminosity distribution calculated from Swift detected GRBs
    """
    loglstar = 52.53  # * 624.15 #GeV
    Lstar = 10 ** loglstar
    alpha = 0.17
    beta = 1.44
    return np.where(L <= Lstar, (L / Lstar) ** (-alpha), (L / Lstar) ** (-beta))


def sample_t90(size=1):
    return 10 ** stats.norm.rvs(1.6, 0.5, size=size)


def isotropic_equivalent_energy(l_peak, t90=10.0):
    """
    Convert a peak luminosity to an isotropic equivalent energy
    """
    return l_peak * t90 / -np.log(0.1)


def comoving_volume(z):
    """
    Comoving volume in cm^3/sr at z
    """
    h = 0.7
    DH = (3000 / units.cm) / h  # Hubble distance
    return DH * (1.0 + z) ** 2 * angular_distance(z) ** 2 / scale(z)


def angular_distance(z):
    dl = LuminosityDistance.instance()
    return dl(z) / (1.0 + z) ** 2


def scale(z):
    """
    E(z) is proportional to the time derivative of the scale factor. An
    astronomer working at redshift z would measure the hubble constant as
    H_0 * E(z)

    See the Cosmology Bible: http://ned.ipac.caltech.edu/level5/Hogg/Hogg4.html
    """
    OMm = 0.3
    OMl = 0.7
    return np.sqrt(OMm * (1 + z) ** 3 + OMl)


def rejection_sample(func, xmin=0, xmax=8, steps=10000, size=10, log10=False):
    """
    Generate samples from *func* by rejection sampling.

    :param func: callable that returns a number proportional to the probability density
    :param xmin: minimum value
    :param xmax: maximum value
    :param steps: number of times to evaluate the function between *xmin* and *xmax*
    :param size: number of samples to generate
    :param log10: True if *func* returns dP/d(log10(x)), False if it returns dP/dx
    """
    if log10:
        xarr = np.logspace(xmin, xmax, steps)
    else:
        xarr = np.linspace(xmin, xmax, steps)

    # calculate corresponding y-values and get y-min / y-max
    yarr = func(xarr)
    ymin = 0
    ymax = max(yarr)
    retvals = np.empty(size)

    hits = 0
    chunk = min((10 * size, 65535))

    while hits < size:
        xr = np.random.uniform(xmin, xmax, size=chunk)
        if log10:
            xr = 10 ** xr
        yr = np.random.uniform(ymin, ymax, size=chunk)
        values = xr[yr <= func(xr)]
        if hits + values.size > size:
            values = values[: size - hits - values.size]
        retvals[hits : hits + values.size] = values
        hits += values.size

    return retvals
