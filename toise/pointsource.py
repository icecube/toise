import logging
from copy import copy
from functools import partial
from io import StringIO

import numpy as np
from scipy import interpolate, optimize, stats
from scipy.optimize import bisect

from .multillh import asimov_llh, get_expectations
from .util import *


def is_zenith_weight(zenith_weight, aeff):
    zenith_dim = aeff.dimensions.index("true_zenith_band")
    # print issubclass(np.asarray(zenith_weight).dtype.type, np.floating), len(zenith_weight), aeff.values.shape[zenith_dim]
    return (
        issubclass(np.asarray(zenith_weight).dtype.type, np.floating)
        and len(zenith_weight) == aeff.values.shape[zenith_dim]
    )


class PointSource(object):
    def __init__(self, effective_area, fluence, zenith_selection, with_energy=True):
        """

        :param effective_area: an effective area
        :param fluence: flux integrated over the energy bins of `effective_area`, in 1/cm^2 s
        :param zenith_selection: if an integer or slice, select only these zenith bins. If a
            sequence of floats of the same length as the number of zenith angle bins, average
            over zenith bins with the given weights (for example, to account for motion of a
            source in local coordinates when the detector is not at Pole)
        """

        self._edges = effective_area.bin_edges
        energy_bins = effective_area.get_bin_edges("true_energy")
        self.energy_range = (energy_bins[0], energy_bins[-1])
        self.bin_edges = self._edges

        if is_zenith_weight(zenith_selection, effective_area):
            zenith_dim = effective_area.dimensions.index("true_zenith_band")
            expand = [None] * effective_area.values.ndim
            expand[zenith_dim] = slice(None)
            effective_area = (
                effective_area.values[..., :-1] * zenith_selection[tuple(expand)]
            ).sum(axis=zenith_dim)
        else:
            effective_area = effective_area.values[..., zenith_selection, :, :-1]
        expand = [None] * effective_area.ndim
        expand[1] = slice(None)
        if len(fluence.shape) > 1 and fluence.shape[1] > 1:
            expand[2] = slice(None)
        # 1/yr
        rate = fluence[tuple(expand)] * (effective_area * 1e4)

        assert np.isfinite(rate).all()

        self._use_energies = with_energy

        self._rate = rate
        self._invalidate_cache()

    def _invalidate_cache(self):
        self._last_params = dict()
        self._last_expectations = None

    def spectral_weight(self, e_center, **kwargs):
        gamma_name = "ps_gamma"
        self._last_params[gamma_name] = kwargs[gamma_name]
        return (e_center / 1e3) ** (kwargs[gamma_name] + 2)

    def expectations(self, **kwargs):

        if self._last_expectations is not None and all(
            [self._last_params[k] == kwargs[k] for k in self._last_params]
        ):
            return self._last_expectations
        energy = self._edges[0]
        centers = 0.5 * (energy[1:] + energy[:-1])
        specweight = self.spectral_weight(centers, **kwargs)

        expand = [None] * (self._rate.ndim)
        expand[1] = slice(None)

        # FIXME: this still neglects the opening angle between neutrino and muon
        total = (self._rate * (specweight[tuple(expand)])).sum(axis=(0, 1))
        # assert total.ndim == 2

        if not self._use_energies:
            total = total.sum(axis=0)

        self._last_expectations = total
        return self._last_expectations

    def get_chunk(self, emin=-np.inf, emax=np.inf):
        ebins = self._edges[0]
        start, stop = ebins.searchsorted((emin, emax))
        start = max((0, start - 1))
        chunk = copy(self)
        chunk._invalidate_cache()
        # zero out the neutrino flux outside the given range
        chunk._rate = self._rate.copy()
        chunk._rate[:, :start, ...] = 0
        chunk._rate[:, stop:, ...] = 0
        chunk.energy_range = (ebins[start], ebins[stop])
        return chunk

    def differential_chunks(
        self, decades=1, emin=-np.inf, emax=np.inf, exclusive=False
    ):
        """
        Yield copies of self with the neutrino spectrum restricted to *decade*
        decades in energy
        """
        # now, sum over decades in neutrino energy
        ebins = self._edges[0]
        loge = np.log10(ebins)
        bin_range = int(round(decades / (loge[1] - loge[0])))

        # when emin is "equal" to an edge in ebins
        # searchsorted sometimes returns inconsistent indices
        # (wrong side). subtract a little fudge factor to ensure
        # we're on the correct side
        lo = ebins.searchsorted(emin - 1e-4)
        hi = min((ebins.searchsorted(emax - 1e-4) + 1, loge.size))
        if exclusive:
            bins = list(range(lo, hi - 1, bin_range))
        else:
            bins = list(range(lo, hi - 1 - bin_range))

        for i in bins:
            start = i
            stop = min((start + bin_range, loge.size - 1))
            chunk = copy(self)
            chunk._invalidate_cache()
            # zero out the neutrino flux outside the given range
            chunk._rate = self._rate.copy()
            chunk._rate[:, :start, ...] = 0
            chunk._rate[:, stop:, ...] = 0
            e_center = 10 ** (0.5 * (loge[start] + loge[stop]))
            chunk.energy_range = (10 ** loge[start], 10 ** loge[stop])
            yield e_center, chunk


class SteadyPointSource(PointSource):
    r"""
    A stead point source of neutrinos.

    The unit is the differential flux per neutrino flavor at 1 TeV,
    in units of :math:`10^{-12} \,\, \rm  TeV^{-1} \, cm^{-2} \, s^{-1}`

    """

    def __init__(
        self,
        effective_area,
        livetime,
        zenith_bin,
        emin=0,
        emax=np.inf,
        with_energy=True,
    ):
        # reference flux is E^2 Phi = 1e-12 TeV cm^-2 s^-1
        # remember: fluxes are defined as neutrino + antineutrino, so the flux
        # per particle (which we need here) is .5e-12
        def intflux(e, gamma):
            return (e ** (1 + gamma)) / (1 + gamma)

        energy = effective_area.bin_edges[0]
        tev = energy / 1e3
        # 1/cm^2 yr
        fluence = (
            0.5e-12
            * (intflux(tev[1:], -2) - intflux(tev[:-1], -2))
            * livetime
            * 365
            * 24
            * 3600
        )
        # zero out fluence outside energy range
        fluence[(energy[:-1] > emax) | (energy[1:] < emin)] = 0

        PointSource.__init__(self, effective_area, fluence, zenith_bin, with_energy)
        self._livetime = livetime


class WBSteadyPointSource(PointSource):
    def __init__(self, effective_area, livetime, zenith_bin, with_energy=True):
        # reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
        # remember: fluxes are defined as neutrino + antineutrino, so the flux
        # per particle (which we need here) is .5e-12
        def intflux(e, gamma):
            return (e ** (1 + gamma)) / (1 + gamma)

        tev = effective_area.bin_edges[0] / 1e3
        # 1/cm^2 yr
        fluence = (
            0.5e-12
            * (intflux(tev[1:], -2) - intflux(tev[:-1], -2))
            * livetime
            * 365
            * 24
            * 3600
        )

        # scale by the WB GRB fluence, normalized to the E^-2 flux between 100 TeV and 10 PeV
        from .grb import WaxmannBahcallFluence

        norm = (
            WaxmannBahcallFluence()(effective_area.bin_edges[0][1:])
            * effective_area.bin_edges[0][1:] ** 2
        )
        norm /= norm.max()
        fluence *= norm

        PointSource.__init__(self, effective_area, fluence, zenith_bin, with_energy)
        self._livetime = livetime


class NSNSMerger(PointSource):
    def __init__(self, effective_area, livetime, zenith_bin, with_energy=True):
        # reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
        # remember: fluxes are defined as neutrino + antineutrino, so the flux
        # per particle (which we need here) is .5e-12
        def intflux(e, gamma):
            return (e ** (1 + gamma)) / (1 + gamma)

        tev = effective_area.bin_edges[0] / 1e3
        # 1/cm^2 yr
        fluence = (
            0.5e-12
            * (intflux(tev[1:], -2) - intflux(tev[:-1], -2))
            * livetime
            * 365
            * 24
            * 3600
        )

        # scale by the WB GRB fluence, normalized to the E^-2 flux between 100 TeV and 10 PeV
        from .nsns import NSNS

        norm = (
            NSNS()(effective_area.bin_edges[0][1:])
            * effective_area.bin_edges[0][1:] ** 2
        )
        norm /= norm.max()
        fluence *= norm

        PointSource.__init__(self, effective_area, fluence, zenith_bin, with_energy)
        self._livetime = livetime


class TruncatedSteadyPointSource(PointSource):
    def __init__(self, effective_area, livetime, zenith_bin, with_energy=True):
        # reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
        # remember: fluxes are defined as neutrino + antineutrino, so the flux
        # per particle (which we need here) is .5e-12
        def intflux(e, gamma):
            return (e ** (1 + gamma)) / (1 + gamma)

        tev = effective_area.bin_edges[0] / 1e3
        # 1/cm^2 yr
        fluence = (
            0.5e-12
            * (intflux(tev[1:], -2) - intflux(tev[:-1], -2))
            * livetime
            * 365
            * 24
            * 3600
        )
        # scale by the WB GRB fluence, normalized to the E^-2 flux between 100
        # TeV and 10 PeV
        from .grb import WaxmannBahcallFluence

        norm = (
            WaxmannBahcallFluence()(effective_area.bin_edges[0][1:])
            * effective_area.bin_edges[0][1:] ** 2
        )
        norm /= norm.max()
        fluence *= norm

        PointSource.__init__(self, effective_area, fluence, zenith_bin, with_energy)
        self._livetime = livetime


# An astrophysics-style powerlaw, with a positive lower limit, no upper limit,
# and a negative index


class powerlaw_gen(stats.rv_continuous):
    def _argcheck(self, gamma):
        return gamma > 1

    def _pdf(self, x, gamma):
        return (gamma - 1) * x**-gamma

    def _cdf(self, x, gamma):
        return 1.0 - x ** (1.0 - gamma)

    def _ppf(self, p, gamma):
        return (1.0 - p) ** (1.0 / (1.0 - gamma))


powerlaw = powerlaw_gen(name="powerlaw", a=1.0)


class StackedPopulation(PointSource):
    @staticmethod
    def draw_source_strengths(n_sources):
        # draw relative source strengths
        scd = powerlaw(gamma=2.5)
        strengths = scd.rvs(n_sources)
        # scale strengths so that the median of the maximum is at 1
        # (the CDF of the maximum of N iid samples is the Nth power of the individual CDF)
        strengths /= scd.ppf(0.5 ** (1.0 / n_sources))
        return strengths

    @staticmethod
    def draw_sindec(n_sources):
        return np.random.uniform(-1, 1, n_sources)

    def __init__(self, effective_area, livetime, fluxes, sindecs, with_energy=True):
        """
        :param n_sources: number of sources
        :param weighting: If 'flux', distribute the fluxes according to an SCD
                          where the median flux from the brightest source is
                          1e-12 TeV^2 cm^-2 s^-1. Scaling the normalization of
                          the model scales this median linearly. If 'equal',
                          assume the same flux from all sources.
        :param source_sindec: sin(dec) of each of the N sources. If None, draw
                              isotropically
        """

        # scatter sources through the zenith bands isotropically
        zenith_bins = effective_area.bin_edges[1]
        self.sources_per_band = np.histogram(-sindecs, bins=zenith_bins)[0]
        self.flux_per_band = np.histogram(-sindecs, bins=zenith_bins, weights=fluxes)[0]

        # reference flux is E^2 Phi = 1e-12 TeV^2 cm^-2 s^-1
        # remember: fluxes are defined as neutrino + antineutrino, so the flux
        # per particle (which we need here) is .5e-12
        def intflux(e, gamma):
            return (e ** (1 + gamma)) / (1 + gamma)

        tev = effective_area.bin_edges[0] / 1e3
        # 1/cm^2 yr
        fluence = (
            0.5e-12
            * (intflux(tev[1:], -2) - intflux(tev[:-1], -2))
            * livetime
            * 365
            * 24
            * 3600
        )
        fluence = np.outer(fluence, self.flux_per_band)

        super(StackedPopulation, self).__init__(
            effective_area, fluence, slice(None), with_energy
        )


def source_to_local_zenith(declination, latitude, ct_bins):
    """
    Return the fraction of the day that a source at a given declination spends in each zenith bin

    :param declination: source declination in degrees
    :param latitude: observer latitude in degrees
    :param ct_bins: edges of bins in cos(zenith), in ascending order
    """
    dec = np.radians(declination)
    lat = np.radians(latitude)

    def offset(hour_angle, ct=0):
        "difference between source elevation and bin edge at given hour angle"
        return (
            np.cos(hour_angle) * np.cos(dec) * np.cos(lat) + np.sin(dec) * np.sin(lat)
        ) - ct

    # find minimum and maximum elevation
    lo = np.searchsorted(ct_bins[1:], offset(np.pi))
    hi = np.searchsorted(ct_bins[:-1], offset(0))
    hour_angle = np.empty(len(ct_bins))
    # source never crosses the band
    hour_angle[: lo + 1] = np.pi
    hour_angle[hi:] = 0
    # source enters or exits
    hour_angle[lo + 1 : hi] = list(
        map(partial(bisect, offset, 0, np.pi), ct_bins[lo + 1 : hi])
    )

    return abs(np.diff(hour_angle)) / np.pi


def nevents(llh, **hypo):
    """
    Total number of events predicted by hypothesis *hypo*
    """
    for k in llh.components:
        if not k in hypo:
            if hasattr(llh.components[k], "seed"):
                hypo[k] = llh.components[k].seed
            else:
                hypo[k] = 1
    return sum(map(np.sum, llh.expectations(**hypo).values()))


def discovery_potential(
    point_source, diffuse_components, sigma=5.0, baseline=None, tolerance=1e-2, **fixed
):
    r"""
    Calculate the scaling of the flux in *point_source* required to discover it
    over the background in *diffuse_components* at *sigma* sigma in 50% of
    experiments.

    :param point_source: an instance of :class:`PointSource`
    :param diffuse components: a dict of diffuse components. Each of the values
        should be a point source background component, e.g. the return value of
        :meth:`diffuse.AtmosphericNu.point_source_background`, along with any
        nuisance parameters required to evaluate them.
    :param sigma: the required significance. The method will scale
        *point_source* to find a median test statistic of :math:`\sigma**2`
    :param baseline: a first guess of the correct scaling. If None, the scaling
        will be estimated from the baseline number of signal and background
        events as :math:`\sqrt{n_B} \sigma / n_S`.
    :param tolerance: tolerance for the test statistic
    :param fixed: values to fix for the diffuse components in each fit.

    :returns: a tuple (norm, ns, nb) giving the flux normalization, the
        number of signal events corresponding to that normalization and the
        total number of background events
    """
    critical_ts = sigma**2

    components = dict(ps=point_source)
    components.update(diffuse_components)

    def ts(flux_norm):
        """
        Test statistic of flux_norm against flux norm=0
        """
        allh = asimov_llh(components, ps=flux_norm, **fixed)
        if len(fixed) == len(diffuse_components):
            return -2 * (allh.llh(ps=0, **fixed) - allh.llh(ps=flux_norm, **fixed))
        else:
            null = allh.fit(ps=0, **fixed)
            alternate = allh.fit(ps=flux_norm, **fixed)
            # print null, alternate, -2*(allh.llh(**null)-allh.llh(**alternate))-critical_ts
            return -2 * (allh.llh(**null) - allh.llh(**alternate))

    def f(flux_norm):
        return ts(flux_norm) - critical_ts

    if baseline is None:
        # estimate significance as signal/sqrt(background)
        allh = asimov_llh(components, ps=1, **fixed)
        total = nevents(allh, ps=1, **fixed)
        nb = nevents(allh, ps=0, **fixed)
        ns = total - nb
        baseline = min((1000, np.sqrt(critical_ts) / (ns / np.sqrt(nb)))) / 10
        baseline = max(((np.sqrt(critical_ts) / (ns / np.sqrt(nb))) / 10, 0.3 / ns))
        # logging.getLogger().warn('total: %.2g ns: %.2g nb: %.2g baseline norm: %.2g ts: %.2g' % (total, ns, nb, baseline, ts(baseline)))
    # baseline = 1000
    if not np.isfinite(baseline):
        return np.inf, np.inf, np.inf
    else:
        # actual = optimize.bisect(f, 0, baseline, xtol=baseline*1e-2)
        actual = optimize.fsolve(f, baseline, xtol=tolerance, factor=1, epsfcn=1)
        allh = asimov_llh(components, ps=actual, **fixed)
        total = nevents(allh, ps=actual, **fixed)
        nb = nevents(allh, ps=0, **fixed)
        ns = total - nb
        logging.getLogger().info(
            "baseline: %.2g actual %.2g ns: %.2g nb: %.2g ts: %.2g"
            % (baseline, actual[0], ns, nb, ts(actual))
        )
        return actual[0], ns, nb


def events_above(observables, edges, ecutoff):
    n = 0
    for k, edge_k in edges.items():
        # print(ecutoff, edges)
        # print(np.shape(edges))
        # print(np.shape(observables[k]))
        if len(edge_k) > 2:
            edge_k = [edge_k[1], edge_k[2]]
        cut = np.where(edge_k[1][1:] > ecutoff)[0][0]
        n += observables[k].sum(axis=0)[cut:].sum()

    return n


def fc_upper_limit(point_source, diffuse_components, ecutoff=0, cl=0.9, **fixed):
    components = dict(ps=point_source)
    components.update(diffuse_components)

    llh = asimov_llh(components, ps=1, **fixed)

    exes = get_expectations(llh, ps=1, **fixed)
    ntot = sum(
        [
            events_above(exes[k], components[k].bin_edges, ecutoff)
            for k in list(exes.keys())
        ]
    )
    ns = events_above(exes["ps"], components["ps"].bin_edges, ecutoff)
    nb = ntot - ns

    logging.getLogger().info("ns: %.2g, nb: %.2g" % (ns, nb))

    if cl != 0.9:
        raise ValueError("I can only handle 90% CL")
    try:
        return fc_upper_limit.table(nb) / ns
    except ValueError:
        raise ValueError(
            "nb=%.2g is too large for the FC construction to be useful" % nb
        )


# Average 90% upper limit for known background
# taken from Table XII of Feldman & Cousins (1998)
fc_upper_limit.table = interpolate.interp1d(
    np.loadtxt(StringIO("0 0.5 1 1.5 2 2.5 3 3.5 4 5 6 7 8 9 10 11 12 13 14 15")),
    np.loadtxt(
        StringIO(
            "2.44 2.86 3.28 3.62 3.94 4.20 4.42 4.63 4.83 5.18 5.53 5.90 6.18 6.49 6.76 7.02 7.28 7.51 7.75 7.99"
        )
    ),
)


def upper_limit(
    point_source, diffuse_components, cl=0.9, baseline=None, tolerance=1e-2, **fixed
):
    """
    Calculate the median upper limit on *point_source* given the background
    *diffuse_components*.

    :param cl: desired confidence level for the upper limit
    :returns: a tuple (norm, ns, nb) giving the flux normalization, the
        number of signal events corresponding to that normalization and the
        total number of background events

    The remaining arguments are the same as :func:`discovery_potential`
    """
    critical_ts = stats.chi2.ppf(cl, 1)

    components = dict(ps=point_source)
    components.update(diffuse_components)

    def ts(flux_norm):
        """
        Test statistic of flux_norm against flux norm=0
        """
        allh = asimov_llh(components, ps=0, **fixed)
        if len(fixed) == len(diffuse_components):
            return -2 * (allh.llh(ps=0, **fixed) - allh.llh(ps=flux_norm, **fixed))
        else:
            return -2 * (
                allh.llh(**allh.fit(ps=0, **fixed))
                - allh.llh(**allh.fit(ps=flux_norm, **fixed))
            )

    def f(flux_norm):
        # NB: minus sign, because now the null hypothesis is no source
        return -ts(flux_norm) - critical_ts

    if baseline is None:
        # estimate significance as signal/sqrt(background)
        allh = asimov_llh(components, ps=1, **fixed)
        total = nevents(allh, ps=1, **fixed)
        nb = nevents(allh, ps=0, **fixed)
        ns = total - nb
        baseline = min((1000, np.sqrt(critical_ts) / (ns / np.sqrt(nb)))) / 10
        baseline = max(((np.sqrt(critical_ts) / (ns / np.sqrt(nb))) / 10, 0.3 / ns))
        logging.getLogger().debug(
            "total: %.2g ns: %.2g nb: %.2g baseline norm: %.2g"
            % (total, ns, nb, baseline)
        )
        # logging.getLogger().warn('total: %.2g ns: %.2g nb: %.2g baseline norm: %.2g ts: %.2g' % (total, ns, nb, baseline, ts(baseline)))
    # baseline = 1000
    if not np.isfinite(baseline):
        return np.inf, np.inf, np.inf
    else:
        # actual = optimize.bisect(f, 0, baseline, xtol=baseline*1e-2)
        actual = optimize.fsolve(f, baseline, xtol=tolerance, factor=1, epsfcn=1)
        logging.getLogger().debug("baseline: %.2g actual %.2g" % (baseline, actual[0]))
        allh = asimov_llh(components, ps=actual, **fixed)
        total = nevents(allh, ps=actual, **fixed)
        nb = nevents(allh, ps=0, **fixed)
        ns = total - nb
        logging.getLogger().info("ns: %.2g nb: %.2g" % (ns, nb))
        return actual[0], ns, nb


def differential_discovery_potential(
    point_source,
    diffuse_components,
    sigma=5,
    baseline=None,
    tolerance=1e-2,
    decades=0.5,
    emin=-np.inf,
    emax=np.inf,
    **fixed
):
    """
    Calculate the discovery potential in the same way as :func:`discovery_potential`,
    but with the *decades*-wide chunks of the flux due to *point_source*.

    :returns: a tuple (energies, sensitivities, ns, nb) giving the central
        energy, flux normalization, number of signal and number of background
        events for each neutrino energy range
    """
    energies = []
    sensitivities = []
    ns = []
    nb = []
    for energy, pschunk in point_source.differential_chunks(
        decades=decades, emin=emin, emax=emax
    ):
        energies.append(energy)
        norm, _ns, _nb = discovery_potential(
            pschunk, diffuse_components, sigma, baseline, tolerance, **fixed
        )
        sensitivities.append(norm)
        ns.append(_ns)
        nb.append(_nb)
    return tuple(map(np.asarray, (energies, sensitivities, ns, nb)))


def differential_upper_limit(
    point_source,
    diffuse_components,
    cl=0.9,
    baseline=None,
    tolerance=1e-2,
    decades=0.5,
    emin=-np.inf,
    emax=np.inf,
    **fixed
):
    """
    Calculate the discovery potential in the same way as :func:`discovery_potential`,
    but with the *decades*-wide chunks of the flux due to *point_source*.

    :returns: a tuple (energies, sensitivities, ns, nb) giving the central
        energy, flux normalization, number of signal and number of background
        events for each neutrino energy range
    """
    energies = []
    sensitivities = []
    ns = []
    nb = []
    for energy, pschunk in point_source.differential_chunks(
        decades=decades, emin=emin, emax=emax
    ):
        energies.append(energy)
        norm, _ns, _nb = upper_limit(
            pschunk, diffuse_components, cl, baseline, tolerance, **fixed
        )
        sensitivities.append(norm)
        ns.append(_ns)
        nb.append(_nb)
    return tuple(map(np.asarray, (energies, sensitivities, ns, nb)))


def differential_fc_upper_limit(
    point_source,
    diffuse_components,
    ecutoff=0,
    cl=0.9,
    decades=0.5,
    emin=-np.inf,
    emax=np.inf,
    **fixed
):
    energies = []
    sensitivities = []
    for energy, pschunk in point_source.differential_chunks(
        decades=decades, emin=emin, emax=emax
    ):
        energies.append(energy)
        sensitivities.append(
            fc_upper_limit(pschunk, diffuse_components, ecutoff, cl, **fixed)
        )
    return np.asarray(energies), np.asarray(sensitivities)
