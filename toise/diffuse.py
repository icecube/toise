import warnings
import numpy as np
import itertools
from scipy.integrate import quad
from io import StringIO
from copy import copy
from . import multillh
import healpy
import os
import numexpr
import pickle as pickle
import logging
from functools import partial
from enum import Enum
import nuflux

from .util import *
from .pointsource import is_zenith_weight

try:
    from functools import lru_cache
except ImportError:
    from backports.functools_lru_cache import lru_cache

# Enums for diffuse flux models, extracted datapoints provided by Bustamante & Valera
DIFFUSE_MODELS = Enum(
    "DIFFUSE_MODELS",
    [
        "fang_murase",
        "fang_pulsar",
        "heinze_zmax_1",
        "muzio_2019",
        "muzio_2021",
        "padovani_2015",
        "rodrigues_bench_cosmo",
        "rodrigues_bench_source",
        "rodrigues_hlbllacs_cosmo",
        "van_vliet_ta",
        # "AGNJetMax"
    ],
)

DIFFUSE_MODELS.fang_murase.filename = "diffuse/fang_murase.txt"
DIFFUSE_MODELS.fang_pulsar.filename = "diffuse/fang_pulsar.txt"
DIFFUSE_MODELS.heinze_zmax_1.filename = "diffuse/heinze_zmax_1.txt"
DIFFUSE_MODELS.muzio_2019.filename = "diffuse/muzio_2019.txt"
DIFFUSE_MODELS.muzio_2021.filename = "diffuse/muzio_2021.txt"
DIFFUSE_MODELS.padovani_2015.filename = "diffuse/padovani_2015.txt"
DIFFUSE_MODELS.rodrigues_bench_cosmo.filename = "diffuse/rodrigues_bench_cosmo.txt"
DIFFUSE_MODELS.rodrigues_bench_source.filename = "diffuse/rodrigues_bench_source.txt"
DIFFUSE_MODELS.rodrigues_hlbllacs_cosmo.filename = (
    "diffuse/rodrigues_hlbllacs_cosmo.txt"
)
DIFFUSE_MODELS.van_vliet_ta.filename = "diffuse/van_vliet_ta.txt"
# DIFFUSE_MODELS.AGNJetMax.filename                = "diffuse/AGN_jet_max_allowed_Rodrigues_et_al_PRL_126_191101.csv"

# diffuse digitized models provided by V. Valera & M. Bustamante, cf. https://arxiv.org/abs/2204.04237
DIFFUSE_MODELS.fang_murase.__doc__ = "Fang & Murase, cosmic-ray reservoirs (cosmogenic + source), Nature Phys. 14, 396 (2018)"
DIFFUSE_MODELS.fang_pulsar.__doc__ = (
    "Fang et al., newborn pulsars (source), Phys. Rev. D 90, 103005 (2014)"
)
DIFFUSE_MODELS.heinze_zmax_1.__doc__ = (
    "Heinze et al., fit to Auger UHECRs (cosmogenic), Astrophys. J. 873, 88 (2019)"
)
DIFFUSE_MODELS.muzio_2019.__doc__ = "Muzio et al., maximum extra p component (cosmogenic + source), Phys. Rev. D 100, 103008 (2019)"
DIFFUSE_MODELS.muzio_2021.__doc__ = "Muzio et al., cosmic ray-gas interactions (cosmogenic + source), Phys. Rev. D 105, 023022 (2022)"
DIFFUSE_MODELS.padovani_2015.__doc__ = (
    "Padovani et al. BL Lacs (source), Mon. Not. Roy. Astron. Soc. 452, 1877 (2015)"
)
DIFFUSE_MODELS.rodrigues_bench_cosmo.__doc__ = (
    "Rodrigues et al., all AGN (cosmogenic), Phys. Rev. Lett. 126, 191101 (2021)"
)
DIFFUSE_MODELS.rodrigues_bench_source.__doc__ = (
    "Rodrigues et al., all AGN (source), Phys. Rev. Lett. 126, 191101 (2021)"
)
DIFFUSE_MODELS.rodrigues_hlbllacs_cosmo.__doc__ = (
    "Rodrigues et al., HL BL Lacs (cosmogenic), Phys. Rev. Lett. 126, 191101 (2021)"
)
DIFFUSE_MODELS.van_vliet_ta.__doc__ = (
    "Bergman & van Vliet, fit to TA UHECRs (cosmogenic), cf. arXiv:2004.09841, Fig. 3"
)
# DIFFUSE_MODELS.AGNJetMax.__doc__                = "AGN Jets, maximum allowed source, Rodrigues et al., Phys. Rev. Lett. 126, 191101 (2021)"


class NullComponent(object):
    """
    A flux component that predicts zero events. This is useful for padding out
    components that predict events in only a subset of channels (e.g. penetrating
    atmospheric muons).
    """

    def __init__(self, aeff):
        self.seed = 1
        self.uncertainty = None
        i, j = aeff.dimensions.index("true_zenith_band"), aeff.dimensions.index(
            "reco_energy"
        )
        self.expectations = np.zeros((aeff.values.shape[i], aeff.values.shape[j]))


class DiffuseNuGen(object):
    def __init__(self, effective_area, flux, livetime=1.0):
        """
        :param effective_area: effective area in m^2
        :param edges: edges of the bins in the effective area histogram
        :param flux: flux in 1/(m^2 yr), integrated over the edges in *edges*
        :param livetime: observation time in years
        """
        self._aeff = effective_area
        idx = [
            self._aeff.dimensions.index(k) - 1
            for k in ("true_zenith_band", "reco_energy")
        ]
        self.bin_edges = [self._aeff.bin_edges[i] for i in idx]
        i = self._aeff.dimensions.index("true_energy") - 1
        self.energy_range = [self._aeff.bin_edges[i][0], self._aeff.bin_edges[i][-1]]
        self._livetime = livetime
        # dimensions of flux should be: nu type (6),
        # nu energy, cos(nu zenith), reco energy, cos(reco zenith),
        # signature (cascade/track)
        self._flux = flux

        # FIXME: account for healpix binning
        self._solid_angle = 2 * np.pi * np.diff(self._aeff.bin_edges[1])

        self.seed = 1.0
        self.uncertainty = None

    @property
    def is_healpix(self):
        return self._aeff.is_healpix

    def prior(self, value, **kwargs):
        if self.uncertainty is None:
            return 0.0
        else:
            return -((value - self.seed) ** 2) / (2 * self.uncertainty**2)

    @staticmethod
    def _integrate_flux(
        edges,
        flux,
        passing_fraction=lambda *args, **kwargs: 1.0,
        energy_range=(-np.inf, np.inf),
    ):
        from .util import PDGCode

        intflux = np.empty((6, len(edges[0]) - 1, len(edges[1]) - 1))
        for i, (flavor, anti) in enumerate(
            itertools.product(("E", "Mu", "Tau"), ("", "Bar"))
        ):
            pt = getattr(PDGCode, "Nu" + flavor + anti)
            # nuflux 2.0.3 raises an exception on unhandled particle types
            try:
                flux(pt, 0, 0)
            except RuntimeError:
                intflux[i, ...] = 0.0
                continue
            for j in range(len(edges[1]) - 1):
                ct_hi = edges[1][j + 1]
                ct_lo = edges[1][j]
                ct = (ct_lo + ct_hi) / 2.0
                fluxband = np.zeros(len(edges[0]) - 1)

                def f(e):
                    if e < energy_range[0] or e > energy_range[1]:
                        return 0
                    else:
                        return flux(pt, e, ct) * passing_fraction(pt, e, ct, depth=2e3)

                for k in range(len(fluxband)):
                    fluxband[k] = (
                        quad(
                            f,
                            edges[0][k],
                            edges[0][k + 1],
                        )[0]
                        if (
                            edges[0][k] >= energy_range[0],
                            edges[0][k + 1] <= energy_range[1],
                        )
                        else 0.0
                    )
                intflux[i, :, j] = fluxband
        if intflux.sum() == 0:
            warnings.warn(f"{flux} integrated to 0")
        # return integrated flux in 1/(m^2 yr sr)
        return intflux * constants.cm2 * constants.annum

    # apply 3-element multiplication + reduction without creating too many
    # unnecessary temporaries. numexpr allows a single reduction, so do it here
    _reduce_flux = numexpr.NumExpr("sum(aeff*flux*livetime, axis=1)")

    def _apply_flux(self, effective_area, flux, livetime):
        if effective_area.shape[2] > 1:
            return self._reduce_flux(
                effective_area, flux[..., None, None], livetime
            ).sum(axis=0)
        else:
            return (effective_area * flux[..., None, None] * livetime).sum(axis=(0, 1))


def detect(sequence, pred):
    try:
        return next((s for s in sequence if pred(s)))
    except StopIteration:
        return None


class AtmosphericNu(DiffuseNuGen):
    """
    The diffuse atmospheric neutrino flux. :meth:`.point_source_background`
    returns the corresponding point source background.

    The units of the model are scalings of the underlying flux parameterization.
    """

    def __init__(self, effective_area, flux, livetime, hard_veto_threshold=None):
        if isinstance(flux, tuple):
            flux_func, passing_fraction = flux
            if passing_fraction is not None:
                flux = self._integrate_flux(
                    effective_area.bin_edges,
                    flux_func.getFlux,
                    passing_fraction,
                    flux_func.energy_range,
                )
            else:
                flux = self._integrate_flux(
                    effective_area.bin_edges,
                    flux_func.getFlux,
                    flux_func.energy_range,
                )

            # "integrate" over solid angle
            if effective_area.is_healpix:
                flux *= healpy.nside2pixarea(effective_area.nside)
            else:
                flux *= (2 * np.pi * np.diff(effective_area.bin_edges[1]))[
                    None, None, :
                ]
        else:
            # flux was precalculated
            pass

        super(AtmosphericNu, self).__init__(effective_area, flux, livetime)

        if hard_veto_threshold is not None:
            # reduce the flux in the south
            # NB: assumes that a surface veto has been applied!
            flux = (
                self._flux
                * np.where(center(effective_area.bin_edges[1]) < 0.05, 1, 1e-4)[
                    None, None, :
                ]
            )
        else:
            flux = self._flux

        # sum over neutrino flavors, energies, and zenith angles
        total = self._apply_flux(self._aeff.values, flux, self._livetime)

        # up to now we've assumed that everything is azimuthally symmetric and
        # dealt with zenith bins/healpix rings. repeat the values in each ring
        # to broadcast onto a full healpix map.
        if self.is_healpix:
            total = total.repeat(self._aeff.ring_repeat_pattern, axis=0)
        # dimensions of the keys in expectations are now reconstructed energy, sky bin (zenith/healpix pixel)
        self.expectations = total.sum(axis=2)

    def point_source_background(
        self, zenith_index, livetime=None, n_sources=None, with_energy=True
    ):
        """
        Convert flux to a form suitable for calculating point source backgrounds.
        The predictions in **expectations** will be differential in the opening-angle
        bins provided in the effective area instead of being integrated over them.

        :param zenith_index: index of the sky bin to use. May be either an integer
                             (for single point source searches) or a slice (for
                             stacking searches)
        :param livetime: if not None, the actual livetime to integrate over in seconds
        :param n_sources: number of search windows in each zenith band
        :param with_energy: if False, integrate over reconstructed energy. Otherwise,
                            provide a differential prediction in reconstructed energy.
        """
        assert (
            not self.is_healpix
        ), "Don't know how to make PS backgrounds from HEALpix maps yet"

        background = copy(self)
        psi_bins = self._aeff.bin_edges[-1][:-1]
        bin_areas = (np.pi * np.diff(psi_bins**2))[None, ...]
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
            bin_areas = bin_areas * n_sources[expand]

        # dimensions of the keys in expectations are now energy, radial bin
        if is_zenith_weight(zenith_index, self._aeff):
            background.expectations = (
                np.nansum((self.expectations * zenith_index[:, None]) / omega, axis=0)[
                    ..., None
                ]
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

    _cache_file = os.path.join(data_dir, "cache", "atmospheric_fluxes.pickle")
    if os.path.exists(_cache_file):
        with open(_cache_file, "rb") as f:
            _fluxes = pickle.load(f)
    else:
        _fluxes = dict(conventional=dict(), prompt=dict())

    @classmethod
    def conventional(
        cls, effective_area, livetime, veto_threshold=1e3, hard_veto_threshold=None
    ):
        """
        Instantiate a conventional atmospheric neutrino flux, using the Honda
        parameterization with corrections for the cosmic ray knee and the fraction
        of atmospheric neutrinos accompanied by muons.

        The flux will be integrated over the effective area's energy and zenith
        angle bins the first time this method is called. Depending on the number of
        bins this can take several minutes. Subsequent calls with the same veto
        threshold and an effective area of the same shape will use the cached flux
        and instantiate much more quickly.

        :param effective_area: an instance of :py:class:`effective_areas.effective_area`
        :param livetime: observation time, in years
        :param veto_threshold: muon energy, in GeV, above which atmospheric muons
                               can be vetoed. This will be used to modify the effective
                               atmospheric neutrino flux.
        :param hard_veto_threshold: if not None, reduce the atmospheric flux to
                                    1e-4 of its nominal value in the southern
                                    hemisphere to model the effect of a surface
                                    veto. This assumes that an energy threshold
                                    has been applied to the effective area.
        """
        from .externals import AtmosphericSelfVeto

        cache = cls._fluxes["conventional"]
        shape_key = effective_area.values.shape[:4]
        flux = detect(cache.get(veto_threshold, []), lambda args: args[0] == shape_key)
        if flux is None:
            flux = nuflux.makeFlux("honda2006")
            flux.knee_reweighting_model = "gaisserH3a_elbert"
            pf = (
                (lambda *args, **kwargs: 1.0)
                if veto_threshold is None
                else (
                    AtmosphericSelfVeto.AnalyticPassingFraction(
                        kind="conventional", veto_threshold=veto_threshold
                    )
                )
            )
            flux = (flux, pf)
        else:
            flux = flux[1]
        instance = cls(effective_area, flux, livetime, hard_veto_threshold)
        if isinstance(flux, tuple):

            if not veto_threshold in cache:
                cache[veto_threshold] = list()
            cache[veto_threshold].append((shape_key, instance._flux))
            with open(cls._cache_file, "wb") as f:
                pickle.dump(cls._fluxes, f, protocol=pickle.HIGHEST_PROTOCOL)
        assert len(cls._fluxes["conventional"]) > 0

        return instance

    @classmethod
    def prompt(
        cls, effective_area, livetime, veto_threshold=1e3, hard_veto_threshold=None
    ):
        """
        Instantiate a prompt atmospheric neutrino flux, using the Enberg
        parameterization with corrections for the cosmic ray knee and the fraction
        of atmospheric neutrinos accompanied by muons.

        The parameters have the same meanings as in :meth:`.conventional`
        """
        from .externals import AtmosphericSelfVeto

        cache = cls._fluxes["prompt"]
        shape_key = effective_area.values.shape[:4]
        flux = detect(cache.get(veto_threshold, []), lambda args: args[0] == shape_key)
        if flux is None:
            flux = nuflux.makeFlux("sarcevic_std")
            flux.knee_reweighting_model = "gaisserH3a_elbert"
            pf = (
                (lambda *args, **kwargs: 1.0)
                if veto_threshold is None
                else (
                    AtmosphericSelfVeto.AnalyticPassingFraction(
                        kind="charm", veto_threshold=veto_threshold
                    )
                )
            )
            flux = (flux, pf)
        else:
            flux = flux[1]
        instance = cls(effective_area, flux, livetime, hard_veto_threshold)
        if isinstance(flux, tuple):
            if not veto_threshold in cache:
                cache[veto_threshold] = list()
            cache[veto_threshold].append((shape_key, instance._flux))
            with open(cls._cache_file, "wb") as f:
                pickle.dump(cls._fluxes, f, protocol=pickle.HIGHEST_PROTOCOL)

        return instance


class DiffuseAstro(DiffuseNuGen):
    r"""
    A diffuse astrophysical neutrino flux. :meth:`.point_source_background`
    returns the corresponding point source background.

    The unit is the differential flux per neutrino flavor at 100 TeV,
    in units of :math:`10^{-18} \,\, \rm  GeV^{-1} \, cm^{-2} \, s^{-1} \, sr^{-1}`
    """

    def __init__(self, effective_area, livetime, flavor=None, gamma_name="gamma"):
        """
        :param effective_area: the effective area
        :param livetime: observation time, in years
        """
        flux = self._integral_flux(effective_area)[None, :, None]

        if isinstance(flavor, int):
            for i in range(flux.shape[0]):
                if i < 2 * flavor or i > 2 * flavor + 1:
                    flux[i, ...] = 0

        # "integrate" over solid angle
        if effective_area.is_healpix:
            flux *= healpy.nside2pixarea(effective_area.nside)
        else:
            flux = flux * (
                (2 * np.pi * np.diff(effective_area.bin_edges[1]))[None, None, :]
            )
        super(DiffuseAstro, self).__init__(effective_area, flux, livetime)
        self._with_psi = False

        self._gamma_name = gamma_name
        self._suffix = ""
        self._with_energy = True

    @staticmethod
    def _integral_flux(aeff, gamma=-2):
        # reference flux is E^2 Phi = 1e-8 GeV cm^-2 sr^-1 s^-1
        def intflux(e, gamma):
            return ((1e5**-gamma) / (1 + gamma)) * e ** (1 + gamma)

        enu = aeff.bin_edges[0]
        # 1 / m^2 yr
        return (0.5e-18 * constants.cm2 * constants.annum) * (
            intflux(enu[1:], gamma) - intflux(enu[:-1], gamma)
        )

    def _invalidate_cache(self):
        for attr in dir(self):
            clear = getattr(getattr(self, attr), "cache_clear", None)
            if clear:
                clear()

    def point_source_background(
        self, zenith_index, livetime=None, n_sources=None, with_energy=True
    ):
        __doc__ = AtmosphericNu.point_source_background.__doc__
        assert (
            not self.is_healpix
        ), "Don't know how to make PS backgrounds from HEALpix maps yet"

        background = copy(self)
        psi_bins = self._aeff.bin_edges[-1][:-1]
        expand = [None] * 5
        expand[-1] = slice(None)
        bin_areas = (np.pi * np.diff(psi_bins**2))[expand]
        # observation time shorter for triggered transient searches
        if livetime is not None:
            background._livetime = livetime / constants.annum
        # dimensions of the keys in expectations are now energy, radial bin

        # cut flux down to a single zenith band
        # dimensions of self._flux are flavor, energy, zenith
        if is_zenith_weight(zenith_index, self._aeff) or isinstance(
            zenith_index, slice
        ):
            sel = zenith_index
        else:
            sel = slice(zenith_index, zenith_index + 1)

        # scale the area in each bin by the number of search windows
        if n_sources is not None:
            expand = [None] * bin_areas.ndim
            expand[2] = slice(None)
            bin_areas = bin_areas * n_sources[expand]

        # dimensions of flux are now 1/m^2 sr
        if isinstance(sel, slice):
            background._flux = self._flux[:, :, sel] / self._solid_angle[zenith_index]
        else:
            assert (
                abs(self._flux[:, :, :1] - self._flux[:, :, 1:]) < 1e-12
            ).all(), "Diffuse flux must be independent of zenith angle for this weighting to work out"
            background._flux = self._flux[:, :, :1] / self._solid_angle[0]

        # replace reconstructed zenith with opening angle
        # dimensions of aeff are now m^2 sr
        background._aeff = copy(self._aeff)
        if isinstance(sel, slice):
            background._aeff.values = (
                self._aeff.values[:, :, sel, :, :].sum(axis=4)[..., None] * bin_areas
            )
        else:
            background._aeff.values = (
                (self._aeff.values * sel[None, None, :, None, None])
                .sum(axis=2, keepdims=True)
                .sum(axis=4)
            )[..., None] * bin_areas
        background._with_psi = True

        background._with_energy = with_energy
        background._invalidate_cache()

        # total = self._apply_flux(background._aeff.values, background._flux, self._livetime)
        # print background._aeff.values.sum(axis=tuple(range(0, 5)))
        # print total.sum(axis=tuple(range(0, 2)))
        # print background._aeff.values.shape, background._flux.shape
        # print total[...,0].sum()
        # assert total[...,1].sum() > 0

        return background

    def differential_chunks(
        self, decades=1, emin=-np.inf, emax=np.inf, exclusive=False
    ):
        """
        Yield copies of self with the neutrino spectrum restricted to *decade*
        decades in energy
        """
        # now, sum over decades in neutrino energy
        ebins = self._aeff.bin_edges[0]
        loge = np.log10(ebins)
        dloge = loge[1] - loge[0]
        bin_range = int(round(decades / dloge))
        if np.isfinite(emin):
            lo = int(round((np.log10(emin) - loge[0]) / dloge))
        else:
            # when emin is "equal" to an edge in ebins
            # searchsorted sometimes returns inconsistent indices
            # (wrong side). subtract a little fudge factor to ensure
            # we're on the correct side
            lo = ebins.searchsorted(emin - 1e-4)
        if np.isfinite(emax):
            hi = (loge.size - 1) - int(round((loge[-1] - np.log10(emax)) / dloge))
        else:
            hi = min((ebins.searchsorted(emax - 1e-4) + 1, loge.size))

        if exclusive:
            bins = list(range(lo, hi - 1, bin_range))
        else:
            bins = list(range(lo, hi - 1))

        for i in bins:
            start = i
            stop = min((start + bin_range, hi - 1))
            bounds = np.asarray([loge[0] + start * dloge, loge[0] + stop * dloge])
            e_center = 10 ** np.mean(bounds)
            if stop < 0 or start > ebins.size - 1:
                yield e_center, None
                continue
            chunk = copy(self)
            chunk._invalidate_cache()
            # zero out the neutrino flux outside the given range
            chunk._flux = self._flux.copy()
            chunk._flux[:, : max((start, 0)), ...] = 0
            chunk._flux[:, min((stop, loge.size - 1)) :, ...] = 0
            chunk.energy_range = 10**bounds

            yield e_center, chunk

    @property
    def spectral_weight_params(self):
        return (self._gamma_name,)

    def spectral_weight(self, e_center, **kwargs):
        return self._integral_flux(
            self._aeff, kwargs[self._gamma_name]
        ) / self._integral_flux(self._aeff)

    def _apply_flux(self, effective_area, flux, livetime):
        """apply flux without summing over flavor"""
        if effective_area.shape[2] > 1:
            return self._reduce_flux(effective_area, flux[..., None, None], livetime)
        else:
            return (effective_area * flux[..., None, None] * livetime).sum(axis=1)

    @lru_cache(maxsize=512)
    def _apply_flux_weights(self, **kwargs):
        """
        :returns: expectations by flavor (shape 6 x expectations)
        """
        energy = self._aeff.bin_edges[0]
        centers = 0.5 * (energy[1:] + energy[:-1])
        specweight = self.spectral_weight(centers, **kwargs)
        if specweight.ndim == 1:
            specweight = specweight[None, :, None]
        elif specweight.ndim == 2:
            specweight = specweight[..., None]

        flux = self._flux * specweight
        # sum over neutrino energies
        total = self._apply_flux(self._aeff.values, flux, self._livetime)
        if not self._with_psi:
            total = total.sum(axis=3)
            assert total.ndim == 3
        else:
            assert total.ndim == 4

        # up to now we've assumed that everything is azimuthally symmetric and
        # dealt with zenith bins/healpix rings. repeat the values in each ring
        # to broadcast onto a full healpix map.
        if self.is_healpix:
            total = total.repeat(self._aeff.ring_repeat_pattern, axis=1)
        # FIXME is dim 1 still the angular dimension?
        if total.shape[1] == 1:
            total = np.squeeze(total, axis=1)

        if not self._with_energy:
            total = total.sum(axis=2)

        return total

    @lru_cache(maxsize=512)
    def _apply_flavor_weights(self, **kwargs):
        # peel off kwargs we consume
        def param(k):
            return k + self._suffix

        flavor_keys = {
            param(k) for k in ("mu_fraction", "e_tau_ratio", "pgamma_fraction")
        }
        flavor_kwargs = {k: kwargs[k] for k in kwargs.keys() if k in flavor_keys}

        # pass remainder upstream
        spec_kwargs = {k: kwargs[k] for k in kwargs.keys() if k not in flavor_keys}
        expectations_by_flavor = self._apply_flux_weights(**spec_kwargs)

        if param("mu_fraction") in kwargs or param("pgamma_fraction") in flavor_kwargs:
            flavor_weight = 3 * np.ones((6,) + (1,) * (expectations_by_flavor.ndim - 1))
            if param("mu_fraction") in flavor_kwargs:
                eratio, mu = (
                    flavor_kwargs[param("e_tau_ratio")],
                    flavor_kwargs[param("mu_fraction")],
                )
                e = eratio * (1 - mu)
                # assert e+mu <= 1.
                flavor_weight[0:2, ...] *= e
                flavor_weight[2:4, ...] *= mu
                flavor_weight[4:6, ...] *= 1.0 - e - mu
            # See
            # The Glashow resonance at IceCube: signatures, event rates and pp vs. p-gamma interactions
            # Bhattacharya et al
            # http://arxiv.org/abs/1108.3163
            if param("pgamma_fraction") in flavor_kwargs:
                pgamma_fraction = flavor_kwargs[param("pgamma_fraction")]
                assert (
                    param("mu_fraction") not in flavor_kwargs
                ), "flavor fit and pp/pgamma are mutually exclusive"
                assert pgamma_fraction >= 0 and pgamma_fraction <= 1
                flavor_weight[0, ...] = 1 - pgamma_fraction * (1 - 0.78 / 0.5)
                flavor_weight[1, ...] = 1 - pgamma_fraction * (1 - 0.22 / 0.5)
                flavor_weight[2::2, ...] = 1 - pgamma_fraction * (1 - 0.61 / 0.5)
                flavor_weight[3::2, ...] = 1 - pgamma_fraction * (1 - 0.39 / 0.5)
            return (expectations_by_flavor * flavor_weight).sum(axis=0)
        else:
            return expectations_by_flavor.sum(axis=0)

    def calculate_expectations(self, **kwargs):
        # peel off kwargs we consume
        def param(k):
            return k + self._suffix

        keys = {param(k) for k in ("mu_fraction", "e_tau_ratio", "pgamma_fraction")}
        keys.update(self.spectral_weight_params)
        return self._apply_flavor_weights(
            **{k: kwargs[k] for k in kwargs.keys() if k in keys}
        )

    def expectations(self, gamma=-2, **kwargs):
        r"""
        :param gamma: the spectral index :math:`\gamma`.
        :returns: the observable distributions expected for a flux of
        :math:`10^{-18} \frac{E_\nu}{\rm 100 \, TeV}^{\gamma} \,\, \rm  GeV^{-1} \, cm^{-2} \, s^{-1} \, sr^{-1}`
        per neutrino flavor
        """
        return self.calculate_expectations(gamma=gamma, **kwargs)


class MuonDampedDiffuseAstro(DiffuseAstro):
    def __init__(self, *args, **kwargs):
        self._fixed_flavor_ratio = kwargs.pop("fixed_flavor_ratio", False)
        super(MuonDampedDiffuseAstro, self).__init__(*args, **kwargs)
        self._oscillate = IncoherentOscillation.create()

    @staticmethod
    def pion_decay_flux(
        e_nu,
        ecrit_mu=1.0,
    ):
        """
        effective parameterization of the neutrino flux from muon-damped pion decay
        :param e_nu: neutrino energy
        :param ecrit_mu: critical energy at which the muon decay time
                                         and cooling time are equal (see PRL 95, 181101 (2005))
        :returns: an e_nu.size x 3 array containing the ratio of neutrino fluxes
                          with and without muon cooling at the source
        """

        # muon synchrotron cooling (and later, pion cooling) steepens the flux
        # by two powers above the critical energy
        # parameterize this slope change like Hoerandel (2003), neglecting the
        # [probably specious] pile-up effects at the spectral break
        e_knee = 0.3 * ecrit_mu
        epsilon = 5.0
        delta_gamma = 2.0

        def knee_flux(e, e_knee):
            return (1 + (e / e_knee) ** epsilon) ** (-delta_gamma / epsilon)

        flux = np.zeros(e_nu.shape + (3,))

        flux[:, :2] = knee_flux(e_nu, e_knee)[:, None]
        flux[:, 1] += knee_flux(e_nu, 15 * e_knee)

        return flux

    @property
    def spectral_weight_params(self):
        return ("emu_crit", self._gamma_name)

    def spectral_weight(self, e_center, **kwargs):
        emu_crit = kwargs["emu_crit"]
        specweight = self._oscillate(
            *(self.pion_decay_flux(e_center, kwargs["emu_crit"]).T)
        )
        if self._fixed_flavor_ratio:
            avg = specweight.sum(axis=0, keepdims=True) / 3.0
            specweight = avg.repeat(3, 0)
        specweight *= ((e_center / 1e5) ** (kwargs[self._gamma_name] + 2))[None, :]
        return np.repeat(specweight, 2, axis=0)


class AhlersGZKFlux(object):
    """
    Minimal GZK neutrino flux, assuming that post-ankle flux in Auger/TA is
    pure protons
    see: http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.083010
    Fig 2. left panel, solid red line (protons with source evolution)
    """

    def __init__(self):
        from scipy import interpolate

        logE, logWeight = np.log10(
            np.loadtxt(
                StringIO(
                    """3.095e5	8.345e-13
		    4.306e5	1.534e-12
		    5.777e5	2.305e-12
		    7.091e5	3.411e-12
		    8.848e5	4.944e-12
		    1.159e6	7.158e-12
		    1.517e6	1.075e-11
		    2.118e6	1.619e-11
		    2.868e6	2.284e-11
		    3.900e6	3.181e-11
		    5.660e6	4.502e-11
		    7.891e6	6.003e-11
		    1.042e7	8.253e-11
		    1.449e7	1.186e-10
		    1.918e7	1.670e-10
		    3.224e7	3.500e-10
		    7.012e7	1.062e-9
		    1.106e8	1.892e-9
		    1.610e8	2.816e-9
		    2.235e8	3.895e-9
		    3.171e8	5.050e-9
		    5.042e8	6.529e-9
		    7.787e8	7.401e-9
		    1.199e9	7.595e-9
		    1.801e9	7.084e-9
		    2.869e9	6.268e-9
		    4.548e9	4.972e-9
		    6.372e9	3.959e-9
		    8.144e9	3.155e-9
		    1.131e10	2.318e-9
		    1.366e10	1.747e-9
		    2.029e10	9.879e-10
		    2.612e10	6.441e-10
		    3.289e10	4.092e-10
		    4.885e10	1.828e-10
		    8.093e10	5.691e-11
		    1.260e11	1.677e-11
		    1.653e11	7.984e-12
		    2.167e11	3.631e-12
		    2.875e11	1.355e-12
		    """
                )
            )
        ).T

        self._interpolant = interpolate.interp1d(
            logE, logWeight + 8, bounds_error=False, fill_value=-np.inf
        )

    def __call__(self, e_center):
        return 10 ** (self._interpolant(np.log10(e_center)) - 8) / e_center**2


class VanVlietGZKFlux(object):
    """
    A. Van Vliet, R Alves Batista, J. R. Hoerandel, Phys. Rev. D 100, 021302 (2019)
    See Fig. 1 of https://arxiv.org/pdf/1901.01899.pdf
    alpha = 2.0, Emax = 100 EeV, pure proton
    """

    def __init__(self):
        from scipy import interpolate

        logE, logWeight = np.log10(
            np.loadtxt(
                StringIO(
                    """
			2.387e6	5.206e-11
			3.370e6	7.387e-11
			4.973e6	9.571e-11
			7.536e6	1.156e-10
			2.271e7	1.486e-10
			2.924e7	1.595e-10
			3.708e7	1.937e-10
			4.906e7	2.665e-10
			6.551e7	3.957e-10
			9.479e7	6.559e-10
			1.376e8	1.093e-9
			1.984e8	1.697e-9
			2.735e8	2.248e-9
			4.320e8	3.149e-9
			6.267e8	3.794e-9
			1.087e9	4.135e-9
			2.021e9	3.550e-9
			3.383e9	2.817e-9
			4.993e9	1.974e-9
			6.766e9	1.373e-9
			8.985e9	1.022e-9
			1.192e10	6.419e-10
			1.681e10	3.493e-10
			2.333e10	1.811e-10
			3.044e10	9.534e-11
			3.520e10	6.377e-11
			"""
                )
            )
        ).T

        self._interpolant = interpolate.interp1d(
            logE, logWeight + 8, bounds_error=False, fill_value=-np.inf
        )

    def __call__(self, e_center):
        return 10 ** (self._interpolant(np.log10(e_center)) - 8) / e_center**2


class ReasonableGZKFlux(object):
    """
    Neutrino flux for alpha = 2.5, Emax = 10^20 eV, m = 3.4 and 10% protons at Ecr = 10^19.6 eV
    A. Van Vliet, R Alves Batista, J. R. Hoerandel, Phys. Rev. D 100, 021302 (2019)
    See Fig. 1 of https://arxiv.org/pdf/1901.01899.pdf
    Flux file copied from NuRadioMC
    """

    def __init__(self):
        from scipy import interpolate

        E, Weight = np.loadtxt(data_dir + "/models/ReasonableNeutrinos1.txt")
        logE = np.log10(E)
        logWeight = np.log10(Weight)  # flux is expected for all-flavour

        self._interpolant = interpolate.interp1d(
            logE, logWeight + 8, bounds_error=False, fill_value=-np.inf
        )

    def __call__(self, e_center):
        return 10 ** (self._interpolant(np.log10(e_center)) - 8) / e_center**2


class DiffuseModelFlux(object):
    """A tabulated flux. Can be either 2 columns or 7 columns, first is energy, others are either summed E**2*flux or E**2*flux per flavor"""

    def __init__(self, flux_model):
        from scipy import interpolate

        # check if string, else see if it is a defined flux
        if isinstance(flux_model, DIFFUSE_MODELS):
            diff_data = np.loadtxt(
                data_dir + "/models/" + flux_model.filename, delimiter=","
            )
        elif isinstance(flux_model, str):
            if not os.path.isfile(flux_model):
                raise RuntimeError(
                    f"Trying to read diffuse flux from text file, {flux_model} not found."
                )
            # custom flux model provided via text file
            diff_data = np.loadtxt(flux_model, delimiter=",")
        else:
            raise RuntimeError(f"No such diffuse model defined: {flux_model}")
        E = diff_data[:, 0]
        Weight = diff_data[:, 1:]
        logE = np.log10(E)
        logWeight = np.log10(Weight)

        # interpolant for all-flavor
        self._interpolant = interpolate.interp1d(
            logE,
            np.log10(np.sum(Weight, axis=1)) + 8,
            bounds_error=False,
            fill_value=-np.inf,
        )

        # interpolants per flavor (if given)
        self._has_per_flavor_flux = False
        flavorcodes = [
            PDGCode.NuE,
            PDGCode.NuEBar,
            PDGCode.NuMu,
            PDGCode.NuMuBar,
            PDGCode.NuTau,
            PDGCode.NuTauBar,
        ]
        if np.shape(logWeight)[1] == 6:  # per flavor weight is given
            self._has_per_flavor_flux = True
            self._interpolant_per_flavor = {
                flav: interpolate.interp1d(
                    logE, logWeight[:, i] + 8, bounds_error=False, fill_value=-np.inf
                )
                for i, flav in enumerate(flavorcodes)
            }

    def has_per_flavor_flux(self):
        """flag if provided flux file was all-flavor or per-flavor"""
        return self._has_per_flavor_flux

    def __call__(self, e_center, flavor=None):
        assert flavor is None or flavor in [
            PDGCode.NuE,
            PDGCode.NuEBar,
            PDGCode.NuMu,
            PDGCode.NuMuBar,
            PDGCode.NuTau,
            PDGCode.NuTauBar,
        ]
        if flavor is None:
            # return all flavor flux
            return 10 ** (self._interpolant(np.log10(e_center)) - 8) / e_center**2
        # return per-flavor flux
        return (
            10 ** (self._interpolant_per_flavor[flavor](np.log10(e_center)) - 8)
            / e_center**2
        )


def atmos_flux(enu, model):
    """Returns the atmospheric diff flux at enu, averaged
    over zenith for all flavors
    """
    flux = nuflux.makeFlux(model)
    flux.knee_reweighting_model = "gaisserH3a_elbert"
    cos_theta = np.linspace(-1, 1, 100)
    fluxes = []
    for ct in center(cos_theta):
        fluxes.append(
            sum(
                [
                    flux.getFlux(getattr(PDGCode, "".join(combo)), enu, ct)
                    for combo in itertools.product(
                        ("NuE", "NuMu", "NuTau"), ("", "Bar")
                    )
                ]
            )
        )

    return np.mean(fluxes, axis=0)


def astro_flux(enu, norm=4.11e-6, spec=-2.46, cutoff=3e6):
    """Returns the all-flavor astro diff flux at enu, for some
    normalization and spectral index with cutoff at 3PeV. Default
    values from: http://dx.doi.org/10.1088/0954-3899/43/8/084001
    """
    return 3 * norm * enu**spec * np.exp(-enu / cutoff)


def astro_gzk_flux(enu, norm=4.11e-6, spec=-2.46, cutoff=3e6):
    """returns the all-flavor differential flux summed over atmos, astro, and ahlers gzk"""
    ahlers_flux = AhlersGZKFlux()
    return astro_flux(enu, norm, spec, cutoff) + ahlers_flux(enu)


def total_flux(enu):
    """returns the all-flavor differential flux summed over atmos, astro, and ahlers gzk"""
    ahlers_flux = AhlersGZKFlux()
    return (
        atmos_flux(enu, "honda2006")
        + atmos_flux(enu, "sarcevic_std")
        + astro_powerlaw_cutoff(enu)
        + ahlers_flux(enu)
    )


class ArbitraryFlux(DiffuseAstro):
    def __init__(self, *args, **kwargs):
        """Base class for defining arbitrary diffuse fluxes which can be spectral-weighted"""
        super(ArbitraryFlux, self).__init__(*args, **kwargs)
        self._flux_func = None

    def set_flux_func(self, flux_func):
        """Set the *all-flavor* flux as a function of enu"""
        self._flux_func = flux_func
        self._invalidate_cache()

    @property
    def spectral_weight_params(self):
        return tuple()

    def spectral_weight(self, e_center, **kwargs):
        enu = self._aeff.bin_edges[0]
        integrated = np.asarray(
            [
                quad(self._flux_func, enu[i], enu[i + 1])[0]
                for i, e in enumerate(enu[:-1])
            ]
        )
        # Ahlers flux is for all flavors when we want the flux per flavor
        return (
            integrated
            * constants.cm2
            * constants.annum
            / (6 * self._integral_flux(self._aeff))
        )


class DiffuseModel(ArbitraryFlux):
    def __init__(self, model, *args, **kwargs):
        super(DiffuseModel, self).__init__(*args, **kwargs)
        self._flux_func = DiffuseModelFlux(model)

    @property
    def spectral_weight_params(self):
        return tuple()

    def spectral_weight(self, e_center, **kwargs):
        enu = self._aeff.bin_edges[0]
        if self._flux_func.has_per_flavor_flux():
            integrated = np.asarray(
                [
                    [
                        quad(self._flux_func, enu[i], enu[i + 1], j)[0]
                        for i, e in enumerate(enu[:-1])
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
            integrated = (
                np.asarray(
                    [
                        quad(self._flux_func, enu[i], enu[i + 1])[0]
                        for i, e in enumerate(enu[:-1])
                    ]
                )
                / 6.0
            )
        # flux is already for per flavor
        return (
            integrated
            * constants.cm2
            * constants.annum
            / (self._integral_flux(self._aeff))
        )


class AhlersGZK(ArbitraryFlux):
    """
    Minimal GZK neutrino flux, assuming that post-ankle flux in Auger/TA is
    pure protons
    see: http://journals.aps.org/prd/abstract/10.1103/PhysRevD.86.083010
    Fig 2. left panel, solid red line (protons with source evolution)
    """

    def __init__(self, *args, **kwargs):

        super(AhlersGZK, self).__init__(*args, **kwargs)
        self._flux_func = AhlersGZKFlux()


class VanVlietGZK(ArbitraryFlux):
    def __init__(self, *args, **kwargs):
        super(VanVlietGZK, self).__init__(*args, **kwargs)
        self._flux_func = VanVlietGZKFlux()


class ReasonableGZK(ArbitraryFlux):
    """
    Neutrino flux for gamma = 2.5, Emax = 10^20 eV, m = 3.4 and 10% protons at Ecr = 10^19.6 eV
    cf.: A. van Vliet, R. Alves Batista, and J. R. Hoerandel, Phys. Rev. D 100 no. 2, (2019) 021302.
    """

    def __init__(self, *args, **kwargs):
        super(ReasonableGZK, self).__init__(*args, **kwargs)
        self._flux_func = ReasonableGZKFlux()


def transform_map(skymap):
    """
    Interpolate a galactic skymap into equatorial coords
    """
    r = healpy.Rotator(coord=("C", "G"))
    npix = skymap.size
    theta_gal, phi_gal = healpy.pix2ang(healpy.npix2nside(npix), np.arange(npix))
    theta_ecl, phi_ecl = r(theta_gal, phi_gal)
    return healpy.pixelfunc.get_interp_val(skymap, theta_ecl, phi_ecl)


class FermiGalacticEmission(DiffuseNuGen):
    r"""
    Diffuse emission from the galaxy, modeled as 0.95 times the Fermi
    :math:`\pi^0` map, extrapolated with a spectral index of 2.71.
    """

    def __init__(self, effective_area, livetime=1.0):
        assert effective_area.is_healpix
        # differential all-flavor flux at 1 GeV [1/(GeV cm^2 sr s)]
        map1GeV = np.load(
            os.path.join(data_dir, "models", "fermi_galactic_emission.npy")
        )
        # downsample to resolution of effective area map, and divide by 6 to
        # convert from all-neutrino flux to flux per particle
        flux_constant = (
            healpy.ud_grade(transform_map(map1GeV), effective_area.nside) / 6
        )

        def intflux(e, gamma=-2.71):
            return (e ** (1 + gamma)) / (1 + gamma)

        e = effective_area.bin_edges[0]
        # integrate flux over energy and solid angle: 1/GeV sr cm^2 s -> 1/cm^2 s
        flux = intflux(e[1:]) - intflux(e[:-1])
        flux *= (
            healpy.nside2pixarea(effective_area.nside) * constants.cm2 * constants.annum
        )
        flux = flux[None, :, None] * flux_constant[None, None, :]

        super(FermiGalacticEmission, self).__init__(effective_area, flux, livetime)

        # sum over opening angles and broadcast zenith angle bin over healpix rings
        aeff = self._aeff.values.sum(axis=4).repeat(
            self._aeff.ring_repeat_pattern, axis=2
        )
        # sum over neutrino flavors and energies
        total = numexpr.NumExpr("sum(aeff*flux*livetime, axis=1)")(
            aeff, self._flux[..., None], self._livetime
        ).sum(axis=0)

        # dimensions of the keys in expectations are now reconstructed energy, sky bin (healpix pixel)
        self.expectations = total


class KRAGalacticFlux(object):
    """
    See Fig. 2 of arXiv:1504.00227
    """

    def __init__(self, cutoff_PeV=5):
        from scipy import interpolate

        if cutoff_PeV == 5:
            logE, logWeight = np.log10(
                np.loadtxt(
                    StringIO(
                        """
            0.011	1.224e-5
            0.020	8.812e-6
            0.042	6.029e-6
            0.098	4.125e-6
            0.201	3.123e-6
            0.378	2.425e-6
            1.000	1.659e-6
            2.643	1.135e-6
            8.998	7.018e-7
            29.369	4.564e-7
            84.451	2.752e-7
            173.194	1.577e-7
            386.506	6.505e-8
            669.404	3.045e-8
            899.764	1.836e-8
            1261.588	9.751e-9
            1625.583	5.879e-9
            """
                    )
                )
            ).T
        elif cutoff_PeV == 50:
            logE, logWeight = np.log10(
                np.loadtxt(
                    StringIO(
                        """
            0.011	1.224e-5
            0.020	8.812e-6
            0.042	6.029e-6
            0.098	4.125e-6
            0.201	3.123e-6
            0.378	2.424e-6
            1.000	1.659e-6
            2.643	1.164e-6
            8.998	7.570e-7
            29.369	5.448e-7
            84.451	3.921e-7
            173.194	2.822e-7
            386.506	1.701e-7
            826.858	1.000e-7
            1432.067	6.342e-8
            2007.947	4.450e-8
            3784.265	2.191e-8
            7131.993	7.764e-9
            9189.729	5.050e-9
            """
                    )
                )
            ).T
        else:
            raise ValueError("can't handle cutoff {}".format(cutoff_PeV))

        self._interpolant = interpolate.interp1d(
            logE, logWeight + 8, bounds_error=False, fill_value=-np.inf
        )

    def __call__(self, e_center):
        # NB: flux is given as all-particle, here we return per-particle (/6)
        return (
            10 ** (self._interpolant(np.log10(e_center) - 3) - 8) / e_center**2 / 6.0
        )


class KRAGalacticDiffuseEmission(DiffuseNuGen):
    r"""
    Diffuse emission from the galaxy as modeled in

      D.~Gaggero, D.~Grasso, A.~Marinelli, A.~Urbano and M.~Valli,
      %``The gamma-ray and neutrino sky: A consistent picture of Fermi-LAT, Milagro, and IceCube results,''
      Astrophys.\ J.\  {\bf 815}, no. 2, L25 (2015)
      doi:10.1088/2041-8205/815/2/L25
      [arXiv:1504.00227 [astro-ph.HE]].
      %%CITATION = doi:10.1088/2041-8205/815/2/L25;%%
      %99 citations counted in INSPIRE as of 09 Jan 2020
    """

    def __init__(self, effective_area, livetime=1.0, cutoff_PeV=5):
        assert effective_area.is_healpix
        # differential all-flavor flux at 1 GeV [1/(GeV cm^2 sr s)]
        map1GeV = np.load(
            os.path.join(data_dir, "models", "fermi_galactic_emission.npy")
        )
        # average over inner galactic plane to get flux normalization compared to figure 2
        lon, lat = healpy.pix2ang(
            healpy.npix2nside(map1GeV.size), np.arange(map1GeV.size), lonlat=True
        )
        flux_unit = map1GeV[(abs(lat) < 4) & ((lon < 30) | (lon > 330))].mean()

        # downsample to resolution of effective area map, and normalize to the
        # patch shown in fig. 2
        flux_constant = (
            healpy.ud_grade(transform_map(map1GeV), effective_area.nside) / flux_unit
        )

        enu = effective_area.bin_edges[0]
        # integrate flux over energy and solid angle: 1/GeV sr cm^2 s -> 1/cm^2 s
        flux_func = KRAGalacticFlux(cutoff_PeV)
        # integrate to mean of patch shown in fig. 2
        flux = np.asarray(
            [quad(flux_func, enu[i], enu[i + 1])[0] for i, e in enumerate(enu[:-1])]
        )
        flux *= (
            healpy.nside2pixarea(effective_area.nside) * constants.cm2 * constants.annum
        )
        flux = flux[None, :, None] * flux_constant[None, None, :]

        super(KRAGalacticDiffuseEmission, self).__init__(effective_area, flux, livetime)

        # sum over opening angles and broadcast zenith angle bin over healpix rings
        aeff = self._aeff.values.sum(axis=4).repeat(
            self._aeff.ring_repeat_pattern, axis=2
        )
        # sum over neutrino flavors and energies
        total = numexpr.NumExpr("sum(aeff*flux*livetime, axis=1)")(
            aeff, self._flux[..., None], self._livetime
        ).sum(axis=0)

        # dimensions of the keys in expectations are now reconstructed energy, sky bin (healpix pixel)
        self.expectations = total


def pmns_matrix(theta12, theta23, theta13, delta):
    """
    Construct a 3-flavor PMNS mixing matrix, given 3 angles and a CP-violating phase
    """

    def comps(angle):
        return (np.sin(angle), np.cos(angle))

    s12, c12 = comps(theta12)
    s13, c13 = comps(theta13)
    s23, c23 = comps(theta23)
    phase = np.exp(complex(0, delta))
    U = np.matrix(
        [
            [c12 * c13, s12 * c13, s13 / phase],
            [
                -s12 * c23 - c12 * s13 * s23 * phase,
                c12 * c23 - s12 * s13 * s23 * phase,
                c13 * s23,
            ],
            [
                s12 * s23 - c12 * s13 * c23 * phase,
                -c12 * s23 - s12 * s13 * c23 * phase,
                c13 * c23,
            ],
        ]
    )
    return U


def transfer_matrix(U):
    """
    Construct a matrix that transforms a flavor composition at the source to one at Earth
    """

    def prob(alpha, beta):
        return sum(abs(U[alpha, i]) ** 2 * abs(U[beta, i]) ** 2 for i in range(3))

    return np.matrix([[prob(i, j) for j in range(3)] for i in range(3)])


class IncoherentOscillation(object):
    """
    Functor to apply astronomical-baseline oscillations
    """

    @classmethod
    def create(cls, label="nufit_inverted"):
        # taken from NuFit 2.0
        # http://arxiv.org/abs/1409.5439
        if label.lower() == "nufit_inverted":
            params = (33.48, 49.5, 8.51, 254)
        elif label.lower() == "nufit_normal":
            params = (33.48, 42.3, 8.50, 306)
        else:
            raise ValueError("Unknown oscillation parameters '{}'".format(label))
        return cls(*map(np.radians, params))

    def __init__(self, theta12, theta23, theta13, delta):
        self.P = transfer_matrix(pmns_matrix(theta12, theta23, theta13, delta))

    def __call__(self, e, mu, tau):
        original = np.array(np.broadcast_arrays(e, mu, tau), dtype=float)
        oscillated = np.asarray(np.dot(self.P, original))
        return oscillated
