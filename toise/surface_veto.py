import os
import pickle
from copy import copy

import healpy
import numpy
from scipy import interpolate
from scipy.optimize import fsolve
from scipy.special import erf, erfc

from toise import effective_areas

from . import surfaces
from .pointsource import is_zenith_weight
from .util import *

# hobo-costing!


def surface_area(theta_max, volume):
    """
    Surface coverage area required so that a track that
    """
    d = 1950 - volume._z_range[0]  # depth of the bottom of the detector
    return (
        numpy.pi
        * (d * numpy.tan(theta_max) + numpy.sqrt(volume.get_cap_area() / numpy.pi)) ** 2
    )


def array_cost(area, fill_factor):
    """
    :param area: area of array, in km^2
    :param fill_factor: fraction of surface that is active
    :returns: cost, in dollars, to build the array
    """
    return 225e3 + 7.5e8 * area * fill_factor


def fill_factor_for_threshold(emu_min, passing_rate=1e-4):
    """
    Use the fact that the passing rate of 100 TeV protons for a 1e-3-packed
    scintillator array is roughly 1e-3, and muons come from primaries of ~20x higher energy.

    :returns: fill factor necessary for *passing_rate* rejection of muons of
    energy *emu_min*
    """
    return (1e5 / (20 * emu_min)) * (1e-3 / passing_rate) * 1e-3


def veto_cost_for_angle(theta_max, emu_min, surface):
    """
    cost, in megabucks, of an array that rejects all atmospheric backgrounds
    above emu_min out to theta_max
    """
    fill = fill_factor_for_threshold(emu_min)
    area = surface_area(numpy.radians(theta_max), surface)
    return array_cost(area / 1e6, fill) / 1e6


def veto_cost(area, emu_min, surface):
    """
    cost, in megabucks, of an array that rejects all atmospheric backgrounds
    above emu_min and covers an area *area* km^2
    """
    fill = fill_factor_for_threshold(emu_min)
    return array_cost(area, fill) / 1e6


def margin_for_area(base_surface, area):
    """
    :param area: desired area, in km^2
    """

    def area_diff(margin):
        return base_surface.expand(margin).get_cap_area() / 1e6 - area

    return fsolve(area_diff, 0)[0]


def get_geometric_coverage_for_area(
    geometry, spacing, area, ct_bins=numpy.linspace(0, 1, 11), nsamples=int(1e4)
):
    """
    Calculate the geometric coverage of a surface veto by Monte Carlo
    :param gcdfile: path to a GCD file defining the geometry of the in-ice detector
    :param area: area of the surface veto, in km^2
    :param ct_bins: bins in cos(zenith) over which to average the coverage
    :param nsamples: number of trials in each bin
    :returns: an array of length len(ct_bins)-1 containing the coverage fraction
              in each bin
    """

    ref_surface = surfaces.get_fiducial_surface(geometry, spacing)
    margin = margin_for_area(ref_surface, area)
    veto_surface = ref_surface.expand(margin)

    coverage = numpy.zeros(ct_bins.size - 1)

    if not area > 0:
        return coverage

    for i, (ct_lo, ct_hi) in enumerate(zip(ct_bins[:-1], ct_bins[1:])):
        # upgoing events can never be vetoed, no matter what
        if ct_hi < 0:
            continue
        inside = 0
        dirs, pos = ref_surface.sample_impact_ray(ct_lo, ct_hi, nsamples)
        # project up the to the surface
        pos += ((1950 - pos[:, -1:]) / dirs[:, -1:]) * dirs
        # catch stupid sign errors
        assert (abs(pos[:, -1] - 1950) < 1).all()
        # did the shower cross the surface array?
        coverage[i] = sum(map(veto_surface.point_in_footprint, pos)) / float(nsamples)
    return coverage


class GeometricVetoCoverage(object):
    cache_file = os.path.join(data_dir, "veto", "geometric_veto_coverage.pickle")

    def __init__(self, geometry="Sunflower", spacing=240, area=10.0):
        self.geometry = geometry
        self.spacing = spacing
        self.area = area
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                self.cache = pickle.load(f)
        else:
            self.cache = dict()

    def __call__(self, ct_bins=numpy.linspace(-1, 1, 11)):
        key = (
            self.geometry,
            self.spacing,
            self.area,
            (ct_bins[0], ct_bins[-1], len(ct_bins)),
        )
        if key in self.cache:
            return self.cache[key]
        else:
            coverage = get_geometric_coverage_for_area(
                self.geometry, self.spacing, self.area, ct_bins, 100000
            )
            self.cache[key] = coverage
            if not os.path.isdir(os.path.dirname(self.cache_file)):
                os.makedirs(os.path.dirname(self.cache_file))
            with open(self.cache_file, "wb") as f:
                pickle.dump(self.cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            return coverage


class EulerVetoProbability(object):
    def __init__(self, fname=os.path.join(data_dir, "veto", "vetoeffs.pickle")):
        vetoeffs = pickle.load(open(fname))
        x, y, v_mu, v_e = (
            vetoeffs["logE"],
            vetoeffs["cosZen"],
            vetoeffs["vetoEff_mu"],
            vetoeffs["vetoEff_e"],
        )
        # assuming that the muoninc and EM components are independent
        z = 1 - (1 - v_mu) * (1 - v_e)
        self.spline = interpolate.RectBivariateSpline(x, y, z)
        # don't trust table beyond its statistical limits
        self.pmax = numpy.nanmax(v_mu[v_mu < 1])
        self.log_emin = x[0]
        self.log_emax = x[-1]
        self.ct_min = y[0]
        self.ct_max = y[-2]

    def __call__(self, energy, cos_theta):
        logE, cos_theta = numpy.broadcast_arrays(
            numpy.log10(energy / 1e3), numpy.clip(cos_theta, -1, self.ct_max)
        )
        p = numpy.clip(self.spline(logE, cos_theta, grid=False), 0, self.pmax)
        p[logE > self.log_emax] = self.pmax
        p[logE < self.log_emin] = 0
        p[cos_theta < self.ct_min] = 0
        return p


class UDelSurfaceVeto(effective_areas.VetoThreshold):
    def __init__(
        self, fname=os.path.join(data_dir, "veto", "udel-veto-passing-fraction.txt")
    ):
        self._order = [1, 1, 1, 1, 1]

        coefficients = list(np.loadtxt(fname))

        polys = []
        for o in self._order:
            target = slice(0, o + 1)
            polys.append(np.polynomial.Polynomial(coefficients[target]))
            del coefficients[target]

        self._coefficients = polys

    @staticmethod
    def _fitfunc(eng, a, g1, g2, eng0, curve):
        """
        "When you work in cosmic rays, everything is a broken power law with a smooth roll-off :see_no_evil:"
        -- A. Coleman
        """
        return (
            a
            * eng**g1
            * (1 + (10**eng / 10**eng0) ** curve) ** ((g2 - g1) / curve)
        )

    def __call__(self, energy, cos_theta):
        st2 = 1 - cos_theta**2
        log_eV = np.log10(energy) + 9

        params = [f(st2) for f in self._coefficients]
        return self._fitfunc(log_eV, *params)

    def accept(self, emu, cos_theta):
        return np.where(cos_theta > 0.05, 1 - self(emu, cos_theta), True)


def overburden(cos_theta, depth=1950, elevation=2400):
    """
    Overburden for a detector buried underneath a flat surface.

    :param cos_theta: cosine of zenith angle (in detector-centered coordinates)
    :param depth:     depth of detector (in meters below the surface)
    :param elevation: elevation of the surface above sea level

    :returns: an overburden [meters]
    """
    # curvature radius of the surface (meters)
    r = 6371315 + elevation
    # this is secrety a translation in polar coordinates
    return (
        numpy.sqrt(2 * r * depth + (cos_theta * (r - depth)) ** 2 - depth**2)
        - (r - depth) * cos_theta
    )


def minimum_muon_energy(distance, emin=1e3):
    """
    Minimum muon energy required to survive the given thickness of ice with at
    least emin GeV 50% of the time.

    :returns: minimum muon energy at the surface [GeV]
    """

    def polynomial(x, coefficients):
        return sum(c * x**i for i, c in enumerate(coefficients))

    coeffs = [[2.793, -0.476, 0.187], [2.069, -0.201, 0.023], [-2.689, 3.882]]
    a, b, c = (polynomial(numpy.log10(emin), c) for c in coeffs)
    return 10 ** polynomial(distance, (a, b / 1e4, c / 1e10))


class ParticleType(object):
    PPlus = 14
    He4Nucleus = 402
    N14Nucleus = 1407
    Al27Nucleus = 2713
    Fe56Nucleus = 5626


def gaisser_flux(energy, ptype):
    """
    Evaluate the [Gaisser]_ H3a parameterization of the cosmic ray flux.

    :param energy: total energy of primary nucleus [GeV]
    :param ptype: particle type code

    :returns: particle flux [particles/(GeV m^2 sr s)]

    .. [Gaisser] T. K. Gaisser. Spectrum of cosmic-ray nucleons, kaon production, and the atmospheric muon charge ratio. Astroparticle Physics, 35(12):801--806, 2012. ISSN 0927-6505. doi: 10.1016/j.astropartphys.2012.02.010.
    """
    if ptype < 100:
        z = 1
    else:
        z = ptype % 100

    codes = sorted(
        [v for v in list(ParticleType.__dict__.values()) if isinstance(v, int)]
    )
    idx = codes.index(ptype)

    # normalizations for each element
    norm = [
        [7860.0, 3550.0, 2200.0, 1430.0, 2120.0],
        [20] * 2 + [13.4] * 3,
        [1.7] * 2 + [1.14] * 3,
    ]
    # spectral indices
    gamma = [[2.66, 2.58, 2.63, 2.67, 2.63], [2.4] * 5, [2.4] * 5]
    # cutoff rigitity
    rigidity = [4e6, 30e6, 2e9]

    return sum(
        n[idx] * energy ** (-g[idx]) * numpy.exp(-energy / (r * z))
        for n, g, r in zip(norm, gamma, rigidity)
    )


def bundle_energy_at_depth(eprim, a=1, cos_theta=1.0, depth=1950.0):
    """
    Mean bundle energy at depth, assuming Elbert yields and energy loss rate
    proportional to total bundle energy

    See: http://www.ppl.phys.chiba-u.jp/research/IceCube/EHE/muon_model/atm_muon_model.pdf

    :param eprim: primary energy
    :param a: primary mass number
    :param cos_theta: cosine of zenith angle
    :param depth: vertical depth of detector
    """
    ob = overburden(cos_theta, depth)
    emin = minimum_muon_energy(ob, 7.5e3)
    # slant depth in g/cm^2
    X = ob * 1e2 * 0.921
    et = 14.5
    alpha = 1.757
    beta = 5.25
    # NB: effective muon energy loss rate reduced by a factor 4 to better model
    # flux just above the horizon
    bmu = 1e-6

    surface_energy = (
        et * a / cos_theta * alpha / (alpha - 1.0) * (1 * emin / eprim) ** (-alpha + 1)
    )
    # assume that constant energy loss term is negligible
    return surface_energy, numpy.exp(-bmu * X)


def bundle_energy_distribution(emu_edges, eprim, a=1, cos_theta=1.0, depth=1950.0):
    """
    Approximate the distribution of bundle energies at depth from a given primary

    :param eprim: primary energy
    :param a: primary mass number
    :param cos_theta: cosine of zenith angle
    :param depth: vertical depth of detector
    """
    surface_energy, mean_loss = bundle_energy_at_depth(eprim, a, cos_theta, depth)
    mu = numpy.log10(mean_loss * surface_energy)
    # don't allow bundles to have more energy at the surface than the parent shower
    hi = numpy.log10(numpy.minimum(emu_edges[1:], surface_energy))
    lo = numpy.log10(emu_edges[:-1])
    # width of log-normal distribution fit to (surface bundle energy / primary
    # energy) See:
    # https://wiki.icecube.wisc.edu/index.php/The_optimization_of_the_empirical_model_(IC22)
    sigma = numpy.minimum(
        1.25 - 0.16 * numpy.log10(eprim) + 0.00563 * numpy.log10(eprim) ** 2, 1.0
    )
    dist = numpy.maximum((erf((hi - mu) / sigma) - erf((lo - mu) / sigma)) / 2.0, 0.0)

    # compensate for truncating the gaussian at the surface energy
    upper_tail = erfc((numpy.log10(surface_energy) - mu) / sigma) / 2.0
    return dist / (1.0 - upper_tail)


def bundle_flux_at_depth(emu, cos_theta):
    """
    Approximate the muon bundle flux at depth

    :param emu: total bundle energy
    :param cos_theta: cosine of zenith angle
    :returns: (flux, primary_energy), where flux is in 1/GeV m^2 sr s and
        primary_energy is the primary energy associated with each emu, for
        H/He/CNO/MgAlSi/Fe
    """
    # make everything an array
    emu, cos_theta = list(map(numpy.asarray, (emu, cos_theta)))
    emu_center = 10 ** (center(numpy.log10(emu)))
    shape = numpy.broadcast(emu_center, cos_theta).shape
    # primary spectrum for each element
    contrib = numpy.zeros(shape + (5, 1000))

    penergy = numpy.logspace(2, 12, 1000)
    if cos_theta <= 0:
        return contrib, penergy

    logstep = numpy.unique(numpy.diff(numpy.log10(penergy)))[0]
    de = 10 ** (numpy.log10(penergy) + logstep / 2.0) - 10 ** (
        numpy.log10(penergy) - logstep / 2.0
    )

    ptypes = [
        getattr(ParticleType, pt)
        for pt in ("PPlus", "He4Nucleus", "N14Nucleus", "Al27Nucleus", "Fe56Nucleus")
    ]
    A = [[pt / 100, 1][pt == ParticleType.PPlus] for pt in ptypes]
    for i, (ptype, a) in enumerate(zip(ptypes, A)):
        # hobo-integrate the flux over primary energy bins
        weights = gaisser_flux(penergy, int(ptype)) * de
        # distribute it across muon bundle energy bins
        c = bundle_energy_distribution(
            emu[:, None], penergy[None, :], a, cos_theta=cos_theta
        )
        contrib[..., i, :] = weights * c
    # convert to differential flux
    contrib /= numpy.diff(emu)[:, None, None]
    return contrib, penergy


def trigger_efficiency(eprim, threshold=10**5.5, sharpness=7):
    """
    Trigger efficiency fit to IT73 data

    https://wiki.icecube.wisc.edu/index.php/IceTop-73_Spectrum_Analysis

    :param threshold: energy at which 50% of events trigger
    :param sharpness: speed of transition. 0 makes a flat line at 0.5, infinity
                      a step function
    """
    return 1 / (
        1 + numpy.exp(-(numpy.log10(eprim) - numpy.log10(threshold)) * sharpness)
    )


def untagged_fraction(eprim, **kwargs):
    """ """
    return 1.0 - trigger_efficiency(eprim, **kwargs)


class MuonBundleBackground(object):
    def __init__(self, effective_area, livetime=1.0):
        self._aeff = effective_area
        self.bin_edges = list(self._aeff.bin_edges)

        emu, cos_theta = effective_area.bin_edges[:2]
        # FIXME: account for healpix binning
        self._solid_angle = 2 * numpy.pi * numpy.diff(self._aeff.bin_edges[1])

        flux = numpy.stack(
            [
                bundle_flux_at_depth(emu, ct)[0][:, :4, :].sum(axis=(1, 2))
                for ct in center(cos_theta)
            ]
        ).T

        # from icecube import MuonGun
        # model = MuonGun.load_model('GaisserH4a_atmod12_SIBYLL')
        # flux, edist = numpy.vectorize(model.flux), numpy.vectorize(model.energy)
        # emuc, ct = numpy.meshgrid(center(cos_theta), center(emu))
        # flux = flux(MuonGun.depth(0), ct, 1)*edist(MuonGun.depth(0), ct, 1, 0, emuc)

        flux *= numpy.diff(emu)[:, None]
        if effective_area.is_healpix:
            flux *= healpy.nside2pixarea(effective_area.nside)
        else:
            flux *= self._solid_angle[None, :]

        total = (flux[..., None] * self._aeff.values).sum(axis=0) * (
            constants.annum * livetime
        )
        self._livetime = livetime

        # up to now we've assumed that everything is azimuthally symmetric and
        # dealt with zenith bins/healpix rings. repeat the values in each ring
        # to broadcast onto a full healpix map.
        if effective_area.is_healpix:
            total = total.repeat(effective_area.ring_repeat_pattern, axis=0)

        self.seed = 1.0
        self.uncertainty = None

        self.expectations = total

    def point_source_background(
        self, zenith_index, psi_bins, livetime=None, n_sources=None, with_energy=True
    ):
        """
        Convert flux to a form suitable for calculating point source backgrounds.
        The predictions in **expectations** will be differential in the opening-angle
        bins `psi` instead of being integrated over them.

        :param zenith_index: index of the sky bin to use. May be either an integer
                             (for single point source searches) or a slice (for
                             stacking searches)
        :param livetime: if not None, the actual livetime to integrate over in seconds
        :param n_sources: number of search windows in each zenith band
        :param with_energy: if False, integrate over reconstructed energy. Otherwise,
                            provide a differential prediction in reconstructed energy.
        """
        assert (
            not self._aeff.is_healpix
        ), "Don't know how to make PS backgrounds from HEALpix maps yet"

        background = copy(self)
        bin_areas = (numpy.pi * numpy.diff(psi_bins**2))[None, ...]
        # observation time shorter for triggered transient searches
        if livetime is not None:
            bin_areas *= livetime / self._livetime / constants.annum
        if is_zenith_weight(zenith_index, self._aeff):
            omega = self._solid_angle[:, None]
        elif isinstance(zenith_index, slice):
            omega = self._solid_angle[zenith_index, None]
            bin_areas = bin_areas[None, ...]
        else:
            omega = self._solid_angle[zenith_index]
        # scale the area in each bin by the number of search windows
        if n_sources is not None:
            expand = [None] * bin_areas.ndim
            expand[0] = slice(None)
            bin_areas = bin_areas * n_sources[tuple(expand)]

        # dimensions of the keys in expectations are now energy, radial bin
        if is_zenith_weight(zenith_index, self._aeff):
            background.expectations = (
                numpy.nansum(
                    (self.expectations * zenith_index[:, None]) / omega, axis=0
                )[..., None]
                * bin_areas
            )
        else:
            background.expectations = (self.expectations[zenith_index, :] / omega)[
                ..., None
            ] * bin_areas
        if not with_energy:
            # just radial bins
            background.expectations = background.expectations.sum(axis=0)
        return background
