"""
Idealized detectors used to estimate sensitivities in the WIS whitepaper
"""

from numpy import vectorize
from toolz import memoize

from . import diffuse, effective_areas, energy_resolution, pointsource, surfaces


def make_cylinder(volume=1.0, aspect=1.0):
    r = numpy.cbrt(2 * volume * aspect / numpy.pi**2)
    h = numpy.pi * r / 2 / aspect
    return surfaces.Cylinder(h * 1e3, r * 1e3)


class FictiveEnergyResolution(energy_resolution.EnergySmearingMatrix):
    def bias(self, loge):
        return loge

    def sigma(self, loge):
        return 0.25


class GaussianPointSpreadFunction(object):
    def __init__(self, median_opening_angle=numpy.radians(0.5)):
        self._sigma = median_opening_angle / numpy.sqrt(2 * numpy.log(2))

    def __call__(self, psi, energy, cos_theta):
        psi, loge, ct = numpy.broadcast_arrays(psi / self._sigma, energy, cos_theta)
        return 1.0 - numpy.exp(-(psi**2) / 2.0)


@memoize
def base_aeff():
    """
    Create a horizontal neutrino effective area for a 1 km^2 detector
    """
    cos_theta = linspace(-1, 1, 40)[19:21]
    (
        e_nu,
        cos_theta,
        e_mu,
    ), efficiency = effective_areas.get_muon_production_efficiency(cos_theta)

    # Step 5: apply smearing for energy resolution
    response = FictiveEnergyResolution().get_response_matrix(e_mu, e_mu)
    efficiency = numpy.apply_along_axis(numpy.inner, 3, efficiency, response)

    total_aeff = numpy.zeros((6,) + efficiency.shape[1:])
    total_aeff[2:4, ...] = efficiency * 1e6  # side-on geometry area: 1 km2

    edges = (e_nu, cos_theta, e_mu)

    return effective_areas.effective_area(edges, total_aeff, "cos_theta")


# @memoize


def get_aeff(angular_resolution=0.5, energy_threshold=1e3):
    """
    Create a neutrino effective area with specific performance

    :param angular_resolution: median opening angle of gaussian PSF (degrees)
    :param energy_threshold: energy threshold for muon selection (true muon energy, GeV)
    """
    base = base_aeff()
    e_nu, cos_theta, e_mu = base.bin_edges

    idx = e_mu.searchsorted(energy_threshold)

    psf = GaussianPointSpreadFunction(numpy.radians(angular_resolution))
    psi_bins = numpy.concatenate(
        (
            sqrt(linspace(0, numpy.radians(angular_resolution * 4) ** 2, 101)),
            [numpy.inf],
        )
    )

    # Step 4: apply smearing for angular resolution
    cdf = psf(psi_bins[:-1], 0, 0)
    smear = numpy.concatenate((diff(cdf), [1.0 - cdf[-1]]))

    expand = [None] * 4 + [slice(None)]

    edges = (e_nu, cos_theta, e_mu[idx:], psi_bins)

    return effective_areas.effective_area(
        edges, base.values[..., idx:, None] * smear[tuple(expand)], "cos_theta"
    )


@vectorize
@memoize
def discovery_potential(angular_resolution=0.5, energy_threshold=1e3, area=1):
    """
    10-year discovery potential at the horizon

    :param angular_resolution: median opening angle of gaussian PSF (degrees)
    :param energy_threshold: energy threshold for muon selection (true muon energy, GeV)
    :param area: horizontal muon effective area (km^2)
    :returns (n_signal, n_background, flux)
    """
    # for steady sources, area and livetime are the same
    livetime = 10.0 * area
    aeff = get_aeff(
        angular_resolution=angular_resolution, energy_threshold=energy_threshold
    )
    atmo_bkg = diffuse.AtmosphericNu.conventional(
        aeff, livetime
    ).point_source_background(0)
    astro_bkg = diffuse.DiffuseAstro(aeff, livetime).point_source_background(0)

    ps = pointsource.SteadyPointSource(aeff, livetime, 0)

    scale = pointsource.discovery_potential(
        ps, dict(atmo=atmo_bkg, astro=astro_bkg), astro=1, atmo=1
    )
    ns = scale * ps.expectations().sum()
    nb = atmo_bkg.expectations.sum()

    return ns, nb, 1e-12 * scale
