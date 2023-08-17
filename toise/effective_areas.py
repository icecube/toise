import dashi
import tables
from scipy import interpolate
import numpy as np

import os
import numpy
import itertools
import healpy
import warnings
import copy

from .surfaces import get_fiducial_surface
from .energy_resolution import get_energy_resolution
from .angular_resolution import get_angular_resolution
from .classification_efficiency import get_classification_efficiency
from .util import *


def load_jvs_mese():
    """
    Load the effective areas used in the MESE diffuse analysis (10.1103/PhysRevD.91.022001)

    :returns: a tuple (edges, aeff). the 6 dimensions of aeff are: nu type (6),
              nu energy, cos(nu zenith), reco energy, cos(reco zenith),
              signature (cascade/track). edges is a list of length 4 with the
              edges in the inner dimensions.
    """
    shape = None
    edges = None
    aeff = None
    base = "/Users/jakob/Documents/IceCube/reports/charm_search/supplemental/effective_area.per_bin.nu_{flavor}{anti}.{interaction}.{channel}.txt.gz"
    for i, (flavor, anti) in enumerate(
        itertools.product(("e", "mu", "tau"), ("", "_bar"))
    ):
        for j, channel in enumerate(("cascade", "track")):
            for interaction in "cc", "nc", "gr":
                try:
                    data = numpy.loadtxt(base.format(**locals()))
                except:
                    pass
                if shape is None:
                    edges = []
                    for k in range(4):
                        lo = numpy.unique(data[:, k * 2])
                        hi = numpy.unique(data[:, k * 2 + 1])
                        edges.append(numpy.concatenate((lo, [hi[-1]])))
                    shape = [len(e) - 1 for e in reversed(edges)]
                    aeff = numpy.zeros([6] + list(reversed(shape)) + [2])
                aeff[i, ..., j] += data[:, -2].reshape(shape).T

    return edges, aeff


class MuonSelectionEfficiency(object):
    def __init__(
        self, filename="aachen_muon_selection_efficiency.npz", energy_threshold=0
    ):
        if not filename.startswith("/"):
            filename = os.path.join(data_dir, "selection_efficiency", filename)
        if filename.endswith(".npz"):
            f = numpy.load(filename)

            loge = f["log_energy"]
            eff = f["efficiency"]

            self.interp = interpolate.interp1d(
                loge, eff, bounds_error=False, fill_value=0.0
            )
        elif filename.endswith(".hdf5"):
            with tables.open_file(filename) as f:
                generated = dashi.histload(f, "/generated")
                detected = dashi.histload(f, "/detected")
            sp = dashi.histfuncs.histratio(
                detected.project([0]), generated.project([0])
            )

            edges = numpy.concatenate((sp.x - sp.xerr, [sp.x[-1] + sp.xerr[-1]]))
            loge = 0.5 * (numpy.log10(edges[1:]) + numpy.log10(edges[:-1]))

            loge = numpy.concatenate((loge, loge + loge[-1] + numpy.diff(loge)[0]))
            v = numpy.concatenate((sp.y, sp.y[-5:].mean() * numpy.ones(sp.y.size)))
            w = 1 / numpy.concatenate((sp.yerr, 1e-2 * numpy.ones(sp.yerr.size)))
            w[~numpy.isfinite(w)] = 1

            self.interp = interpolate.UnivariateSpline(loge, v, w=w)
        if energy_threshold is None:
            self.energy_threshold = 0.0
        else:
            self.energy_threshold = energy_threshold

    def __call__(self, muon_energy, cos_theta):
        return numpy.where(
            muon_energy >= self.energy_threshold,
            numpy.clip(self.interp(numpy.log10(muon_energy)), 0, 1),
            0.0,
        )


class ZenithDependentMuonSelectionEfficiency(object):
    def __init__(
        self,
        filename="sunflower_200m_bdt0_efficiency.fits",
        energy_threshold=0,
        scale=1.0,
    ):
        from photospline import SplineTable

        if not filename.startswith("/"):
            filename = os.path.join(data_dir, "selection_efficiency", filename)
        self._spline = SplineTable(filename)
        self._scale = scale
        # cut off no lower than 500 GeV
        self.energy_threshold = max((energy_threshold, 5e2))

    def _eval(self, loge, cos_theta):
        return self._spline.eval([loge, cos_theta])

    def __call__(self, muon_energy, cos_theta):
        loge, cos_theta = numpy.broadcast_arrays(numpy.log10(muon_energy), cos_theta)
        if hasattr(self._scale, "__call__"):
            scale = self._scale(10**loge)
        else:
            scale = self._scale
        return numpy.where(
            muon_energy >= self.energy_threshold,
            numpy.clip(
                scale
                * self._spline.evaluate_simple([numpy.log10(muon_energy), cos_theta]),
                0,
                1,
            ),
            0.0,
        )


class FictiveMuonSelectionEfficiency:
    """
    A selection efficiency for muon tracks, with features that could be
    expected from a sparse vertical string detector:
    - Soft threshold, asymptotically constant efficiency
    - Higher threshold in vertical directions
    """

    def __call__(self, muon_energy, cos_theta):
        # soft turn-on at threshold energy (higher threshold in vertical directions)
        threshold = 10 ** (2 * (3 - np.exp(-np.abs(cos_theta))))
        return 1 - np.exp(-muon_energy / threshold)


class HECascadeSelectionEfficiency(object):
    """
    Imitate the efficiency one would get from a HESE-like selection.

    This is functionally a high-energy cascade (all-flavor) selection.
    This provides a efficiency for a cascade to pass the analysis.
    The efficiency parameterization is a logistic function.
    The user can tune the energy threshold where the logistic function
    turns over by adjusting 'energy_threshold'.
    """

    def __init__(self, geometry="IceCube", spacing=125, energy_threshold=1e5):
        from . import surfaces

        outer = get_fiducial_surface(geometry, spacing)
        side_padding = spacing / 2.0
        top_padding = 180  # top + dust layer exclusion
        fiducial = surfaces.get_inner_volume(geometry, spacing)

        self._fiducial_surface = fiducial

        self._threshold = energy_threshold
        self._fiducial_volume = fiducial.get_cap_area() * (
            fiducial.length + 2 * side_padding - top_padding
        )
        self._efficiency = self._fiducial_volume / outer.volume()
        self._outer_volume = outer.volume()

    def __call__(self, deposited_energy, cos_theta):
        return (
            0.75
            * self._efficiency
            / (1 + numpy.exp(-2.5 * numpy.log(deposited_energy / self._threshold)))
        )


def get_muon_selection_efficiency(geometry, spacing, energy_threshold=0, scale=1.0):
    """
    :param energy_threshold: artificial energy threshold in GeV
    """
    if geometry == "Fictive":
        return FictiveMuonSelectionEfficiency()
    elif geometry == "IceCube":
        return MuonSelectionEfficiency(energy_threshold=energy_threshold)
    else:
        return ZenithDependentMuonSelectionEfficiency(
            "11900_MUONGUN_%s_%sm_efficiency_cut.fits" % (geometry, spacing),
            energy_threshold=energy_threshold,
            scale=scale,
        )


class VetoThreshold(object):
    """
    A braindead model of an event selection with a surface veto.
    """

    def accept(self, e_mu, cos_theta=1.0):
        """
        Return True if an event would pass the event selection
        """
        raise NotImplementedError

    def veto(self, e_mu, cos_theta=1.0):
        """
        Return True if an atmospheric event would be rejected by the veto
        """
        raise NotImplementedError


class StepFunction(VetoThreshold):
    """
    A zenith-dependent energy threshold, modeling the effect of a perfect
    surface veto whose threshold scales with slant depth
    """

    def __init__(self, threshold=0, maximum_inclination=60):
        self.max_inclination = numpy.cos(numpy.radians(maximum_inclination))
        self.threshold = threshold

    def accept(self, e_mu, cos_theta=1.0):

        return numpy.where(
            cos_theta > 0.05,
            (e_mu > self.threshold) & (cos_theta >= self.max_inclination),
            True,
        )

    def veto(self, e_mu, cos_theta=1.0):
        """
        Return True if an atmospheric event would be rejected by the veto
        """
        return numpy.where(
            cos_theta > 0.05,
            (e_mu > self.threshold) & (cos_theta >= self.max_inclination),
            False,
        )


class MuonEffectiveArea(object):
    """
    The product of geometric area and selection efficiency
    """

    def __init__(self, geometry, spacing=125):
        self.geometry = geometry
        self.spacing = spacing
        self._surface = get_fiducial_surface(geometry, spacing)
        self._efficiency = get_muon_selection_efficiency(geometry, spacing)

    def __call__(self, muon_energy, cos_theta):
        geo = self._surface.azimuth_averaged_area(cos_theta)
        return geo * self._efficiency(muon_energy, cos_theta)


def _interpolate_production_efficiency(
    cos_zenith, fname="muon_efficiency.hdf5", flavors=["mu"]
):
    """
    Get the probability that a muon neutrino of energy E_nu from zenith angle
    cos_theta will produce a muon that reaches the detector with energy E_mu

    :returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
        edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
        with the same axes.
    """
    from scipy import interpolate

    efficiencies = []
    with tables.open_file(os.path.join(data_dir, "cross_sections", fname)) as hdf:
        for family, anti in itertools.product(flavors, ("", "_bar")):
            h = dashi.histload(hdf, "/nu" + family + anti)
            edges = [numpy.log10(h.binedges[0]), h.binedges[1]] + list(
                map(numpy.log10, h.binedges[2:])
            )
            centers = list(map(center, edges))
            newcenters = [
                centers[0],
                numpy.clip(cos_zenith, centers[1].min(), centers[1].max()),
            ] + centers[2:]
            with numpy.errstate(divide="ignore"):
                y = numpy.where(
                    ~(h.bincontent <= 0), numpy.log10(h.bincontent), -numpy.inf
                )

            assert not numpy.isnan(y).any()
            interpolant = interpolate.RegularGridInterpolator(
                centers, y, bounds_error=True, fill_value=-numpy.inf
            )

            xi = numpy.vstack(
                [x.flatten() for x in numpy.meshgrid(*newcenters, indexing="ij")]
            ).T
            assert numpy.isfinite(xi).all()

            # NB: we use nearest-neighbor interpolation here because
            # n-dimensional linear interpolation has the unfortunate side-effect
            # of dropping the highest-energy muon energy bin in each neutrino
            # energy bin, in turn because the next-highest-energy bin is zero
            # (-inf in log space). Ignoring that bin significantly
            # underestimates the muon flux from steeply falling neutrino spectra.
            v = interpolant(xi, "nearest").reshape([x.size for x in newcenters])

            v[~numpy.isfinite(v)] = -numpy.inf

            assert not numpy.isnan(v).any()

            efficiencies.append(10**v)

    return (h.binedges[0], None,) + tuple(
        h.binedges[2:]
    ), numpy.array(efficiencies)


def _ring_range(nside):
    """
    Return the eqivalent cos(zenith) ranges for the rings of a HEALpix map
    with NSide *nside*.
    """
    # get cos(colatitude) at the center of each ring, and invert to get
    # cos(zenith). This assumes that the underlying map is in equatorial
    # coordinates.
    centers = -healpy.ringinfo(nside, numpy.arange(1, 4 * nside))[2]
    return numpy.concatenate(([-1], 0.5 * (centers[1:] + centers[:-1]), [1]))


def get_muon_production_efficiency(ct_edges=None):
    """
    Get the probability that a muon neutrino of energy E_nu from zenith angle
    cos_theta will produce a muon that reaches the detector with energy E_mu

    :param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
        at the centers of these bins. If an integer, interpret as the NSide of
        a HEALpix map
    :returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
        edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
        with the same axes.
    """
    if ct_edges is None:
        ct_edges = numpy.linspace(-1, 1, 11)
    elif isinstance(ct_edges, int):
        nside = ct_edges
        ct_edges = _ring_range(nside)

    edges, efficiency = _interpolate_production_efficiency(center(ct_edges))
    return (edges[0], ct_edges, edges[2]), efficiency


def get_starting_event_efficiency(ct_edges=None):
    """
    Get the probability that a muon neutrino of energy E_nu from zenith angle
    cos_theta will produce a muon that reaches the detector with energy E_mu

    :param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
        at the centers of these bins. If an integer, interpret as the NSide of
        a HEALpix map
    :returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
        edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
        with the same axes.
    """
    if ct_edges is None:
        ct_edges = numpy.linspace(-1, 1, 11)
    elif isinstance(ct_edges, int):
        nside = ct_edges
        ct_edges = _ring_range(nside)

    edges, efficiency = _interpolate_production_efficiency(
        center(ct_edges), "starting_event_efficiency.hdf5", ["e", "mu", "tau"]
    )
    return (edges[0], ct_edges, edges[2], edges[3]), efficiency


def get_cascade_production_density(ct_edges=None):
    """
    Get the probability that a muon neutrino of energy E_nu from zenith angle
    cos_theta will produce a muon that reaches the detector with energy E_mu

    :param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
        at the centers of these bins. If an integer, interpret as the NSide of
        a HEALpix map
    :returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
        edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
        with the same axes.
    """
    if ct_edges is None:
        ct_edges = numpy.linspace(-1, 1, 11)
    elif isinstance(ct_edges, int):
        nside = ct_edges
        ct_edges = _ring_range(nside)

    edges, efficiency = _interpolate_production_efficiency(
        center(ct_edges), "cascade_efficiency.hdf5", ["e", "mu", "tau"]
    )
    return (edges[0], ct_edges, edges[2]), efficiency


def calculate_cascade_production_density(ct_edges, energy_edges, depth=0.5):
    from toise.externals import nuFATE

    cc = nuFATE.NeutrinoCascadeToShowers(np.exp(center(np.log(energy_edges))))
    return (energy_edges, ct_edges, energy_edges), cc.transfer_matrix(
        center(ct_edges), depth
    )


def get_doublebang_production_density(ct_edges=None):
    """
    Get the probability that a muon neutrino of energy E_nu from zenith angle
    cos_theta will produce a muon that reaches the detector with energy E_mu

    :param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
        at the centers of these bins. If an integer, interpret as the NSide of
        a HEALpix map
    :returns: a tuple edges, efficiency. *edges* is a 3-element tuple giving the
        edges in E_nu, cos_theta, and E_mu, while *efficiency* is a 3D array
        with the same axes.
    """
    if ct_edges is None:
        ct_edges = numpy.linspace(-1, 1, 11)
    elif isinstance(ct_edges, int):
        nside = ct_edges
        ct_edges = _ring_range(nside)

    edges, efficiency = _interpolate_production_efficiency(
        center(ct_edges), "doublebang_efficiency.hdf5", ["e", "mu", "tau"]
    )
    return (edges[0], ct_edges, edges[2]), efficiency


class effective_area(object):
    """
    Effective area with metadata
    """

    def __init__(self, edges, aeff, sky_binning="cos_theta", source="neutrino"):
        self.bin_edges = edges
        self.values = aeff
        self.sky_binning = sky_binning
        if aeff.ndim == 3:
            self.dimensions = ["true_energy", "true_zenith_band", "reco_energy"]
        elif aeff.ndim == 5:
            self.dimensions = [
                "type",
                "true_energy",
                "true_zenith_band",
                "reco_energy",
                "reco_psi",
            ]
        else:
            raise ValueError(
                "Effective area table must have either 3 dimensions (muon bundle) or 5 (neutrinos). Got: {}".format(
                    aeff.shape
                )
            )

    def get_bin_edges(self, dim_name):
        if len(self.dimensions) == 3:
            return self.bin_edges[self.dimensions.index(dim_name)]
        else:
            return self.bin_edges[self.dimensions.index(dim_name) - 1]

    def get_bin_centers(self, dim_name):
        return center(self.get_bin_edges(dim_name))

    def compatible_with(self, other):
        return self.values.shape == other.values.shape and all(
            ((a == b).all() for a, b in zip(self.bin_edges, other.bin_edges))
        )

    def restrict_energy_range(self, emin, emax):

        # find bins with lower edge >= emin and upper edge <= emax
        mask = (self.bin_edges[0][1:] <= emax) & (self.bin_edges[0][:-1] >= emin)
        idx = numpy.arange(mask.size)[mask][[0, -1]]

        reduced = copy.copy(self)
        reduced.bin_edges = list(reduced.bin_edges)
        reduced.bin_edges[0] = reduced.bin_edges[0][idx[0] : idx[1] + 2]
        reduced.bin_edges = tuple(reduced.bin_edges)

        reduced.values = self.values[:, idx[0] : idx[1] + 1, ...]

        return reduced

    def truncate_energy_range(self, emin, emax):

        # find bins with lower edge >= emin and upper edge <= emax
        mask = (self.bin_edges[0][1:] <= emax) & (self.bin_edges[0][:-1] >= emin)
        idx = numpy.arange(mask.size)[mask][[0, -1]]

        reduced = copy.copy(self)

        reduced.values[:, 0 : idx[0], ...] *= 0
        reduced.values[:, idx[1] + 1 :, ...] *= 0

        return reduced

    @property
    def is_healpix(self):
        return self.sky_binning == "healpix"

    @property
    def is_neutrino(self):
        return self.source == "neutrino"

    @property
    def nside(self):
        assert self.is_healpix
        return self.nring / 4 + 1

    @property
    def nring(self):
        assert self.is_healpix
        return self.values.shape[self.dimensions.index("true_zenith_band")]

    @property
    def ring_repeat_pattern(self):
        assert self.is_healpix
        return healpy.ringinfo(self.nside, numpy.arange(self.nring) + 1)[1]


def eval_psf(point_spread_function, mu_energy, ct, psi_bins):
    ct, mu_energy, psi_bins = numpy.meshgrid(ct, mu_energy, psi_bins, indexing="ij")
    return point_spread_function(psi_bins, mu_energy, ct)


def create_bundle_aeff(
    energy_resolution=defer(get_energy_resolution, "IceCube"),
    veto_efficiency: VetoThreshold = StepFunction(numpy.inf),
    veto_coverage=lambda ct: numpy.zeros(len(ct) - 1),
    selection_efficiency=defer(MuonSelectionEfficiency),
    surface=defer(get_fiducial_surface, "IceCube"),
    cos_theta=None,
    **kwargs,
):
    """
    Create an effective area for atmospheric muon bundles

    :param selection_efficiency: an energy- and zenith-dependent muon selection efficiency
    :type: MuonSelectionEfficiency

    :param surface: the fiducial surface surrounding the detector
    :type surface: surfaces.UprightSurface

    :param veto_coverage: a callable f(cos_theta), returning the fraction of
        the fiducial area that is in the shadow of a surface veto
    :type veto_coverate: surface_veto.GeometricVetoCoverage

    :param energy_threshold: the energy-dependent veto passing fraction
    :type energy_threshold: VetoThreshold

    :param energy_resolution: the muon energy resolution for events that pass the selection
    :type energy_resolution: energy_resolution.MuonEnergyResolution

    :param cos_theta: sky binning to use. If cos_theta is an integer,
        bin in a HEALpix map with this NSide, otherwise bin in cosine of
        zenith angle. If None, use the native binning of the muon production
        efficiency histogram.

    :returns: a tuple of effective_area objects
    """
    # Ingredients:
    # 1) Geometric area
    # 2) Selection efficiency
    # 3) Veto coverage

    import tables
    import dashi
    from scipy.special import erf

    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: Get binning
    (e_nu, cos_theta, e_mu), efficiency = get_muon_production_efficiency(cos_theta)

    # Step 2: Geometric muon effective area (no selection effects yet)
    # NB: assumes cylindrical symmetry.
    aeff = numpy.vectorize(surface.average_area)(cos_theta[:-1], cos_theta[1:])[None, :]

    # Step 3: apply selection efficiency
    # selection_efficiency = selection_efficiency(*numpy.meshgrid(center(e_mu), center(cos_theta), indexing='ij')).T
    selection_efficiency = selection_efficiency(
        *numpy.meshgrid(e_mu[1:], center(cos_theta), indexing="ij")
    )
    aeff = aeff * selection_efficiency

    # Step 4: apply smearing for energy resolution
    response = energy_resolution.get_response_matrix(e_mu, e_mu)
    aeff = numpy.apply_along_axis(
        numpy.inner,
        2,
        aeff[..., None] * numpy.eye(response.shape[0])[:, None, :],
        response,
    )

    # Step 5.1: split the geometric area in the southern hemisphere into a
    #           portion shadowed by the surface veto (if it exists) and one that
    #           is not
    shadowed_fraction = veto_coverage(cos_theta)[None, :]

    # Step 5.2: apply suppression from surface veto
    veto_suppression = 1 - veto_efficiency.accept(
        *numpy.meshgrid(center(e_mu), center(cos_theta), indexing="ij")
    )

    # combine into an energy- and zenith-dependent acceptance for muon bundles
    weights = [shadowed_fraction * veto_suppression, 1 - shadowed_fraction]

    edges = (e_mu, cos_theta, e_mu)

    return [
        effective_area(
            edges,
            aeff * w[..., None],
            "cos_theta" if nside is None else "healpix",
            source="muon",
        )
        for w in weights
    ]


def create_throughgoing_aeff(
    energy_resolution=defer(get_energy_resolution, "IceCube"),
    veto_coverage=lambda ct: numpy.zeros(len(ct) - 1),
    selection_efficiency=defer(MuonSelectionEfficiency),
    surface=defer(get_fiducial_surface, "IceCube"),
    psf=defer(get_angular_resolution, "IceCube"),
    psi_bins=numpy.sqrt(numpy.linspace(0, numpy.radians(2) ** 2, 100)),
    cos_theta=None,
):
    """
    Create an effective area for neutrino-induced, incoming muons

    :param selection_efficiency: an energy- and zenith-dependent muon selection efficiency
    :type: MuonSelectionEfficiency

    :param surface: the fiducial surface surrounding the detector
    :type surface: surfaces.UprightSurface

    :param veto_coverage: a callable f(cos_theta), returning the fraction of
        the fiducial area that is in the shadow of a surface veto
    :type veto_coverate: surface_veto.GeometricVetoCoverage

    :param energy_threshold: the energy-dependent veto passing fraction
    :type energy_threshold: VetoThreshold

    :param energy_resolution: the muon energy resolution for events that pass the selection
    :type energy_resolution: energy_resolution.MuonEnergyResolution

    :param psf: the muon point spread function for events that pass the selection
    :type psf: angular_resolution.PointSpreadFunction

    :param cos_theta: sky binning to use. If cos_theta is an integer,
        bin in a HEALpix map with this NSide, otherwise bin in cosine of
        zenith angle. If None, use the native binning of the muon production
        efficiency histogram.
    :param psi_bins: edges of bins in muon/reconstruction opening angle (radians)

    :returns: an effective_area object
    """
    # Ingredients:
    # 1) Muon production efficiency
    # 2) Geometric area
    # 3) Selection efficiency
    # 4) Point spread function
    # 5) Energy resolution

    import tables
    import dashi
    from scipy.special import erf

    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: Efficiency for a neutrino to produce a muon that reaches the
    #         detector with a given energy
    (e_nu, cos_theta, e_mu), efficiency = get_muon_production_efficiency(cos_theta)

    # Step 2: Geometric muon effective area (no selection effects yet)
    # NB: assumes cylindrical symmetry.
    aeff = efficiency * (
        numpy.vectorize(surface.average_area)(cos_theta[:-1], cos_theta[1:])[
            None, None, :, None
        ]
    )

    # Step 3: apply selection efficiency
    # selection_efficiency = selection_efficiency(*numpy.meshgrid(center(e_mu), center(cos_theta), indexing='ij')).T
    selection_efficiency = selection_efficiency(
        *numpy.meshgrid(e_mu[1:], center(cos_theta), indexing="ij")
    ).T

    # Explicit energy threshold disabled for now; let muon background take over
    # at whatever energy it drowns out the signal
    # selection_efficiency *= energy_threshold.accept(*numpy.meshgrid(e_mu[1:], center(cos_theta), indexing='ij')).T
    aeff *= selection_efficiency[None, None, :, :]

    # Step 4: apply smearing for angular resolution
    # Add an overflow bin if none present
    if numpy.isfinite(psi_bins[-1]):
        psi_bins = numpy.concatenate((psi_bins, [numpy.inf]))
    cdf = eval_psf(psf, center(e_mu), center(cos_theta), psi_bins[:-1])

    total_aeff = numpy.zeros((6,) + aeff.shape[1:] + (psi_bins.size - 1,))
    # expand differential contributions along the opening-angle axis
    total_aeff[2:4, ..., :-1] = aeff[..., None] * numpy.diff(cdf, axis=2)[None, ...]
    # put the remainder in the overflow bin
    total_aeff[2:4, ..., -1] = aeff * (1 - cdf[..., -1])[None, None, ...]

    # Step 5: apply smearing for energy resolution
    response = energy_resolution.get_response_matrix(e_mu, e_mu)
    total_aeff = numpy.apply_along_axis(numpy.inner, 3, total_aeff, response)

    # Step 6: split the effective area in into a portion shadowed by the
    #         surface veto (if it exists) and one that is not
    shadowed_fraction = veto_coverage(cos_theta)[None, None, :, None, None]
    weights = [shadowed_fraction, 1 - shadowed_fraction]

    edges = (e_nu, cos_theta, e_mu, psi_bins)

    return [
        effective_area(
            edges, total_aeff * w, "cos_theta" if nside is None else "healpix"
        )
        for w in weights
    ]


def create_cascade_aeff(
    channel="cascade",
    energy_resolution=defer(get_energy_resolution, channel="cascade"),
    energy_threshold=StepFunction(numpy.inf),
    veto_coverage=lambda ct: numpy.zeros(len(ct) - 1),
    selection_efficiency=defer(HECascadeSelectionEfficiency),
    surface=defer(get_fiducial_surface, "IceCube"),
    psf=defer(get_angular_resolution, "IceCube", channel="cascade"),
    psi_bins=numpy.sqrt(numpy.linspace(0, numpy.radians(20) ** 2, 10)),
    cos_theta=None,
):
    """
    Create an effective area for neutrinos interacting inside the volume

    :returns: an effective_area object
    """

    # Ingredients:
    # 1) Final state production efficiency
    # 2) Geometric area
    # 3) Selection efficiency
    # 4) Point spread function
    # 5) Energy resolution

    import tables
    import dashi
    from scipy.special import erf

    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: Density of final states per meter
    warnings.warn("Only treating cascades at the moment")
    if channel == "cascade":
        (e_nu, cos_theta, e_shower), aeff = get_cascade_production_density(cos_theta)
    elif channel == "doublebang":
        (e_nu, cos_theta, e_shower), aeff = get_doublebang_production_density(cos_theta)

    # Step 2: Geometric effective area (no selection effects yet)
    aeff *= surface.volume()

    warnings.warn("Reconstruction quantities are made up for now")

    # Step 3: apply selection efficiency
    selection_efficiency = selection_efficiency(
        *numpy.meshgrid(e_shower[1:], center(cos_theta), indexing="ij")
    ).T
    aeff *= selection_efficiency[None, None, ...]

    # Step 4: apply smearing for angular resolution
    # Add an overflow bin if none present
    if numpy.isfinite(psi_bins[-1]):
        psi_bins = numpy.concatenate((psi_bins, [numpy.inf]))
    cdf = eval_psf(psf, center(e_shower), center(cos_theta), psi_bins[:-1])

    total_aeff = numpy.empty(aeff.shape + (psi_bins.size - 1,))
    # expand differential contributions along the opening-angle axis
    total_aeff[..., :-1] = aeff[..., None] * numpy.diff(cdf, axis=2)[None, ...]
    # put the remainder in the overflow bin
    total_aeff[..., -1] = aeff * (1 - cdf[..., -1])[None, None, ...]

    # Step 5: apply smearing for energy resolution
    response = energy_resolution.get_response_matrix(e_shower, e_shower)
    total_aeff = numpy.apply_along_axis(numpy.inner, 3, total_aeff, response)

    edges = (e_nu, cos_theta, e_shower, psi_bins)

    return effective_area(
        edges, total_aeff, "cos_theta" if nside is None else "healpix"
    )


def create_starting_aeff(
    energy_resolution=defer(get_energy_resolution, channel="cascade"),
    energy_threshold=StepFunction(numpy.inf),
    veto_coverage=lambda ct: numpy.zeros(len(ct) - 1),
    selection_efficiency=defer(HECascadeSelectionEfficiency),
    classification_efficiency=defer(get_classification_efficiency, "IceCube"),
    surface=defer(get_fiducial_surface, "IceCube"),
    psf=defer(get_angular_resolution, "IceCube", channel="cascade"),
    psi_bins=numpy.sqrt(numpy.linspace(0, numpy.radians(20) ** 2, 10)),
    neutrino_energy=numpy.logspace(4, 12, 81),
    cos_theta=numpy.linspace(-1, 1, 21),
):
    """
    Create an effective area for neutrinos interacting inside the volume

    :returns: an effective_area object
    """

    # Ingredients:
    # 1) Final state production efficiency
    # 2) Geometric area
    # 3) Selection efficiency
    # 4) Point spread function
    # 5) Energy resolution

    import tables
    import dashi
    from scipy.special import erf

    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: Density of final states per meter
    (e_nu, cos_theta, e_shower), aeff = calculate_cascade_production_density(
        cos_theta, neutrino_energy, depth=1.5
    )

    # Step 2: Geometric effective area (no selection effects yet)
    aeff *= surface.volume()

    warnings.warn("Reconstruction quantities are made up for now")

    # Step 3: apply overall selection efficiency
    selection_efficiency = selection_efficiency(
        *numpy.meshgrid(e_shower[1:], center(cos_theta), indexing="ij")
    ).T
    aeff *= selection_efficiency[None, None, ...]
    # Step 3.5: calculate channel selection efficiency, padding the shape for
    # the dimensions neutrino energy, zenith angle, and angular error
    weights = {}
    e_shower_center = np.exp(center(np.log(e_shower)))
    for event_class in classification_efficiency.classes:
        weights[event_class] = numpy.array(
            [
                classification_efficiency(nutype, event_class, e_shower_center)[
                    None, None, :, None
                ]
                for nutype in range(aeff.shape[0])
            ]
        )

    # Step 4: apply smearing for angular resolution
    # Add an overflow bin if none present
    if numpy.isfinite(psi_bins[-1]):
        psi_bins = numpy.concatenate((psi_bins, [numpy.inf]))
    cdf = eval_psf(psf, center(e_shower), center(cos_theta), psi_bins[:-1])

    total_aeff = numpy.empty(aeff.shape + (psi_bins.size - 1,))
    # expand differential contributions along the opening-angle axis
    total_aeff[..., :-1] = aeff[..., None] * numpy.diff(cdf, axis=2)[None, ...]
    # put the remainder in the overflow bin
    total_aeff[..., -1] = aeff * (1 - cdf[..., -1])[None, None, ...]

    # Step 5: apply smearing for energy resolution
    response = energy_resolution.get_response_matrix(e_shower, e_shower)
    total_aeff = numpy.apply_along_axis(numpy.inner, 3, total_aeff, response)

    edges = (e_nu, cos_theta, e_shower, psi_bins)

    return {
        event_class: effective_area(
            edges, total_aeff * w, "cos_theta" if nside is None else "healpix"
        )
        for event_class, w in weights.items()
    }


def _interpolate_ara_aeff(ct_edges=None, depth=200, nstations=37):
    """
    Get the aeff for a neutrino of energy E_nu from zenith angle
    ct_edges for ARA (values from mlu ARAsim). Assumes flavor-independence.

    :param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
        at the centers of these bins. If an integer, interpret as the NSide of
        a HEALpix map

    :returns: a tuple edges, aeff. *edges* is a 2-element tuple giving the
        edges in E_nu, cos_theta, while *aeff* is a 2D array
        with the same axes.
    """
    from scipy import interpolate

    if ct_edges is None:
        ct_edges = numpy.linspace(-1, 1, 11)
    elif isinstance(ct_edges, int):
        nside = ct_edges
        ct_edges = _ring_range(nside)
    # interpolate to a grid compatible with the IceCube/Gen2 effective areas
    loge_edges = numpy.linspace(2, 12, 101)

    fpath = os.path.join(data_dir, "aeff", "cosZenDepAeff_z{}.half.txt".format(depth))

    with open(fpath) as fara:
        # parse file and strip out empty lines
        lines = filter(None, (line.rstrip() for line in fara))

        energy = []
        cos_theta = []
        aeff = []
        paeff = []  # partial aeff over a single energy range
        for fline in lines:
            if "EXPONENT" in fline:
                energy.append(float(fline.split("=")[-1]))
                if paeff:
                    paeff.reverse()
                    aeff.append(paeff)
                    paeff = []
                    cos_theta = []
            else:
                cos_theta.append(float(fline.split()[0]))
                paeff.append(float(fline.split()[1]))
        # ara aeffs have zenith pointing in neutrino direction
        paeff.reverse()
        aeff.append(paeff)

    aeff = numpy.asarray(aeff) * nstations

    # convert energy from exponent to GeV
    # energy = 10**edge(numpy.asarray(energy))*1e-9

    centers = (numpy.asarray(energy) - 9, numpy.asarray(cos_theta))

    edges = numpy.array([energy, cos_theta])
    # centers = map(center, edges)
    newcenters = [
        center(loge_edges),
        numpy.clip(center(ct_edges), centers[1].min(), centers[1].max()),
    ]
    xi = numpy.vstack(
        [x.flatten() for x in numpy.meshgrid(*newcenters, indexing="ij")]
    ).T
    assert numpy.isfinite(xi).all()

    interpolant = interpolate.RegularGridInterpolator(
        centers, aeff, bounds_error=False, fill_value=0
    )
    # NB: we use nearest-neighbor interpolation here because
    # n-dimensional linear interpolation has the unfortunate side-effect
    # of dropping the highest-energy muon energy bin in each neutrino
    # energy bin, in turn because the next-highest-energy bin is zero
    # (-inf in log space). Ignoring that bin significantly
    # underestimates the muon flux from steeply falling neutrino spectra.
    v = interpolant(xi, method="nearest").reshape([x.size for x in newcenters])

    # assume flavor-independence for ARA by extending same aeff across all flavors
    return (10**loge_edges, ct_edges), numpy.repeat(v[None, ...], 6, axis=0)


def create_ara_aeff(
    depth=200,
    nstations=37,
    cos_theta=None,
):
    """
    Create an effective area for ARA

    :param depth: depth in m (100 or 200)
    :type: int

    :param cos_theta: sky binning to use. If cos_theta is an integer,
        bin in a HEALpix map with this NSide, otherwise bin in cosine of
        zenith angle. If None, use the native binning of the muon production
        efficiency histogram.
    :param psi_bins: edges of bins in muon/reconstruction opening angle (radians)

    :returns: an effective_area object
    """
    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: ARA aeff for a neutrino to produce a muon that reaches the
    #         detector with a given energy
    (e_nu, cos_theta), aeff = _interpolate_ara_aeff(cos_theta, depth, nstations)

    # Step 2: for now, assume no energy resolution
    # Note that it doesn't matter which energy distribution we use, just as
    # long as it's identical for all neutrino energy bins
    # e_reco = numpy.copy(e_nu)
    # aeff = numpy.repeat(aeff[...,None], aeff.shape[1], axis=-1)
    # aeff /= aeff.shape[1]
    e_reco = numpy.array([e_nu[0], e_nu[-1]])
    aeff = aeff[..., None]

    # Step 3: dummy angular resolution smearing
    psi_bins = numpy.asarray([0, numpy.inf])
    total_aeff = numpy.zeros(aeff.shape + (psi_bins.size - 1,))
    # put everything in first psi_bin for no angular resolution
    total_aeff[..., 0] = aeff[...]

    edges = (e_nu, cos_theta, e_reco, psi_bins)

    return effective_area(
        edges, total_aeff, "cos_theta" if nside is None else "healpix"
    )


def _load_radio_veff(filename):
    """
    :returns: a tuple (edges, veff). veff has units of m^3
    """
    import pandas as pd
    import json

    if not filename.startswith("/"):
        filename = os.path.join(data_dir, "aeff", filename)
    with open(filename) as f:
        dats = json.load(f)
    index = []
    arrays = {"veff": [], "err": []}
    for zenith, values in dats.items():
        for selection, items in values.items():
            for energy, veff, err in zip(
                items["energies"], items["Veff"], items["Veff_error"]
            ):
                index.append((selection, energy, np.cos(float(zenith))))
                arrays["veff"].append(veff)
                arrays["err"].append(err)
    veff = pd.DataFrame(
        arrays,
        index=pd.MultiIndex.from_tuples(
            index, names=["selection", "energy", "cos_zenith"]
        ),
    )
    veff.sort_index(level=[0, 1, 2], inplace=True)
    # add right-hand bin edges
    energy = veff.index.levels[1].values / 1e9
    energy = np.concatenate([energy, [energy[-1] ** 2 / energy[-2]]])
    # left-hand bin edges were specified in zenith, so add in reverse
    cos_zenith = veff.index.levels[2].values
    cos_zenith = np.concatenate(([2 * cos_zenith[0] - cos_zenith[1]], cos_zenith))
    omega = 2 * np.pi * np.diff(cos_zenith)
    return (energy, cos_zenith), veff["veff"].unstack(level=-1).values.reshape(
        (energy.size - 1, cos_zenith.size - 1)
    ) / omega[None, :]


def _interpolate_radio_veff(
    energy_edges, ct_edges=None, filename="nu_e_Gen2_100m_1.5sigma.json"
):
    from scipy import interpolate

    edges, veff = _load_radio_veff(filename)
    # NB: occasionally there are NaN effective volumes. intepolate through them

    def interp_masked(arr, x, xp):
        valid = ~np.ma.masked_invalid(arr).mask
        return np.interp(x, xp[valid], arr[valid], left=-np.inf)

    veff = np.exp(
        np.apply_along_axis(
            interp_masked,
            0,
            np.log(veff),
            center(np.log10(energy_edges)),
            center(np.log10(edges[0])),
        )
    )
    if ct_edges is None:
        return (energy_edges, edges[1]), veff
    else:
        interp = interpolate.interp1d(
            center(edges[1]), veff, "nearest", axis=1, bounds_error=False, fill_value=0
        )
        return (energy_edges, ct_edges), interp(center(ct_edges))


def create_radio_aeff(
    nstations=305,
    energy_resolution=get_energy_resolution(channel="radio"),
    psf=get_angular_resolution(channel="radio"),
    psi_bins=numpy.sqrt(numpy.linspace(0, numpy.radians(20) ** 2, 10)),
    veff_filename=dict(
        e="nu_e_Gen2_100m_1.5sigma.json", mu="nu_mu_Gen2_100m_1.5sigma.json"
    ),
    cos_theta=np.linspace(-1, 1, 21),
    neutrino_energy=np.logspace(6, 12, 61),
):
    """
    Create an effective area for a nameless radio array
    """
    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: Density of final states per meter
    (e_nu, cos_theta, e_shower), aeff = calculate_cascade_production_density(
        cos_theta, neutrino_energy
    )

    # Step 2: Effective volume in terms of shower energy
    # NB: this includes selection efficiency (usually step 3)
    edges_e, veff_e = _interpolate_radio_veff(
        e_shower, cos_theta, filename=veff_filename["e"]
    )
    edges_mu, veff_mu = _interpolate_radio_veff(
        e_shower, cos_theta, filename=veff_filename["mu"]
    )
    aeff[0:2, ...] *= (veff_e.T)[None, None, ...] * nstations  # electron neutrino
    aeff[2:4, ...] *= (veff_mu.T)[None, None, ...] * nstations  # muon neutrino
    aeff[4:6, ...] *= (veff_mu.T)[None, None, ...] * nstations  # tau neutrino
    total_aeff = aeff

    # Step 4: apply smearing for angular resolution
    # Add an overflow bin if none present
    if numpy.isfinite(psi_bins[-1]):
        psi_bins = numpy.concatenate((psi_bins, [numpy.inf]))
    cdf = eval_psf(psf, center(e_shower), center(cos_theta), psi_bins[:-1])

    total_aeff = numpy.empty(aeff.shape + (psi_bins.size - 1,))
    # expand differential contributions along the opening-angle axis
    total_aeff[..., :-1] = aeff[..., None] * numpy.diff(cdf, axis=2)[None, ...]
    # put the remainder in the overflow bin
    total_aeff[..., -1] = aeff * (1 - cdf[..., -1])[None, None, ...]

    # Step 5: apply smearing for energy resolution
    response = energy_resolution.get_response_matrix(e_shower, e_shower)
    total_aeff = numpy.apply_along_axis(numpy.inner, 3, total_aeff, response)

    edges = (e_nu, cos_theta, e_shower, psi_bins)

    return effective_area(
        edges, total_aeff, "cos_theta" if nside is None else "healpix"
    )


def _interpolate_gen2_ehe_aeff(ct_edges=None):
    """
    Get the aeff for a neutrino of energy E_nu from zenith angle
    ct_edges for Gen2 EHE analysis assuming pDOMs and Sunflower geometry.

    :param ct_edges: edges of *cos_theta* bins. Efficiencies will be interpolated
        at the centers of these bins. If an integer, interpret as the NSide of
        a HEALpix map

    :returns: a tuple edges, aeff. *edges* is a 2-element tuple giving the
        edges in E_nu, cos_theta, while *aeff* is a 2D array
        with the same axes.
    """
    from scipy import interpolate

    if ct_edges is None:
        ct_edges = numpy.linspace(-1, 1, 11)
    elif isinstance(ct_edges, int):
        nside = ct_edges
        ct_edges = _ring_range(nside)

    # interpolate to a grid compatible with the IceCube/Gen2 effective areas
    loge_edges = numpy.linspace(2, 12, 101)

    flavors = ["NuE", "NuMu", "NuTau"]
    filenames = [
        os.path.join(data_dir, "aeff", f"Gen2_EHE_{flavor}_effective_area.npz")
        for flavor in flavors
    ]

    aeffs = []
    for filename in filenames:
        data = numpy.load(filename)
        cos_theta_bins = data["cos_theta_bins"]
        energy_bins = data["energy_bins"]
        energy_bins_at_det = data["energy_bins_at_det"]
        aeffs.append(data["area_in_sqm"])

    e_center = center(energy_bins)
    cos_theta_center = center(cos_theta_bins)
    e_center_at_det = center(energy_bins_at_det)

    new_centers = [
        center(loge_edges),
        numpy.clip(center(ct_edges), cos_theta_center.min(), cos_theta_center.max()),
        center(loge_edges),
    ]

    xi = numpy.vstack(
        [x.flatten() for x in numpy.meshgrid(*new_centers, indexing="ij")]
    )

    vs = []
    for aeff in aeffs:
        interpolant = interpolate.RegularGridInterpolator(
            (numpy.log10(e_center), cos_theta_center, numpy.log10(e_center_at_det)),
            np.moveaxis(aeff, -1, 0),
            bounds_error=False,
            fill_value=0,
        )
        v = interpolant(xi.T, method="nearest").reshape([x.size for x in new_centers])
        vs.append(v)

    # Assume Nu/NuBar symmetry for effective areas
    return (
        (10**loge_edges, ct_edges, 10**loge_edges),
        numpy.concatenate([numpy.repeat(v[None, ...], 2, axis=0) for v in vs]),
    )


def create_gen2_ehe_aeff(
    cos_theta=None,
):
    """
    Create an effective area for a Gen2 EHE analysis prototype

    :param cos_theta: sky binning to use. If cos_theta is an integer,
        bin in a HEALpix map with this NSide, otherwise bin in cosine of
        zenith angle. If None, use the native binning of the muon production
        efficiency histogram.

    :returns: an effective_area object
    """
    nside = None
    if isinstance(cos_theta, int):
        nside = cos_theta

    # Step 1: Aeff for a neutrino to produce a muon that reaches the
    #         detector with a given energy
    (e_nu, cos_theta, e_det), aeff = _interpolate_gen2_ehe_aeff(cos_theta)

    # Step 2: for now, assume _no_ energy resolution
    e_det = np.array([e_det[0], e_det[-1]])
    aeff = aeff.sum(axis=-1, keepdims=True)

    # Step 3: dummy angular resolution smearing
    psi_bins = numpy.array([0, numpy.inf])
    total_aeff = numpy.zeros(aeff.shape + (psi_bins.size - 1,))
    # put everything in first psi_bin for no angular resolution
    total_aeff[..., 0] = aeff[...]

    edges = (e_nu, cos_theta, e_det, psi_bins)

    return effective_area(
        edges, total_aeff, "cos_theta" if nside is None else "healpix"
    )
