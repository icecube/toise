import logging
from functools import cache, lru_cache
from pathlib import Path

import healpy
import numpy
import numpy as np
import pandas as pd
from scipy import interpolate, optimize

from toise import angular_resolution, effective_areas, util

notebook_dir = Path(__file__).parents[3] / "notebooks" / "galactic plane"

logging.getLogger("healpy").setLevel("WARN")

# @cache
def create_dnn_aeff(nside: int = 16, scale=1):
    base = notebook_dir / "DNNC_effa"
    # The dict keys are the minimum declination in the range
    edges = np.load(base / "effa_bins.npy", allow_pickle=True).item()
    values = np.load(base / "effa_values.npy", allow_pickle=True).item()

    assert edges.keys() == values.keys()

    d_dec = 30.0
    dec_edges = list(edges.keys())
    dec_edges.append(dec_edges[-1] + d_dec)
    # convert declination to cos(zenith)
    ct_edges = -(np.sin(np.radians(np.array(dec_edges))))[::-1]

    loge_edges = None
    for item in edges.values():
        if loge_edges is not None:
            assert np.all(loge_edges == item[0])
        loge_edges = item[0]

    # transpose to (energy, zenith), and reverse to match order of cos(zenith) edges
    values = np.array(list(values.values())).T[:, ::-1]

    # snap zenith bands to rings of a healpix map
    pixel_centers = -healpy.ringinfo(nside, numpy.arange(1, 4 * nside))[2]
    # broadcast effective area into each ring
    values = values[:, np.digitize(pixel_centers, ct_edges) - 1]
    # reset zenith bands to the bounds of the healpix rings
    ct_edges = effective_areas._ring_range(nside)

    # normalize final state density over final state energies to get a probability
    final_state_density = effective_areas.calculate_cascade_production_density(
        ct_edges, 10**loge_edges, depth=1.5
    )[1]
    norm = final_state_density.sum(axis=3, keepdims=True)
    energy_distribution = np.divide(final_state_density, norm, where=norm != 0)

    # energy_distribution = energy_distribution.sum(axis = 3, keepdims=True)
    # print(energy_distribution.shape)
    # # TODO: add reco energy resolution

    # reco_edges = 10**np.asarray([loge_edges[0], loge_edges[-1]])

    reco_edges = 10**loge_edges

    # add dimensions for neutrino type, ..., final state energy, opening angle
    # effective area is sum of all flavors
    # -> divide by 3 to get average area per flavor (nu + antinu)
    # -> divide by 2 to get average area per particle type (nu or antinu)
    aeff = ((scale / 3 / 2 * np.ones((6, 1, 1))) * values)[..., None, None]

    return effective_areas.effective_area(
        [10**loge_edges, ct_edges, reco_edges, np.array([0, np.pi])],
        aeff * energy_distribution[..., None],
        sky_binning="healpix",
    )


def load_dnn_quantiles(aeff=3):
    df = pd.read_csv(notebook_dir / "DNNC_effa/psf_quantiles.csv", header=[0, 1])
    quantiles = {}
    for q in df.columns.levels[0]:
        dataset = df[q]
        dataset = dataset[~np.isnan(df[q, "Y"])]
        quantiles[float(q)] = dataset
    return quantiles


@cache
def load_dnn_psf():
    df = pd.read_csv(notebook_dir / "DNNC_effa/psf_quantiles.csv", header=[0, 1])
    quantiles = {}
    for q in df.columns.levels[0]:
        dataset = df[q]
        dataset = dataset[~np.isnan(df[q, "Y"])]
        quantiles[float(q)] = interpolate.interp1d(
            dataset["X"], dataset["Y"], fill_value="extrapolate"
        )
    return quantiles


def get_dnn_smoothing(loge_nodes, target_quantile=0.5):
    psf = load_dnn_psf()

    def get_concentration_parameter(psi_deg, quantile):
        cos_alpha = np.cos(np.radians(psi_deg))
        k = 10 ** optimize.bisect(
            lambda logc: angular_resolution.fisher(10**logc).sf(cos_alpha) - quantile,
            -2,
            np.log10(200),
        )
        return k

    k = np.asarray(
        [
            get_concentration_parameter(psf[target_quantile](loge), target_quantile)
            for loge in loge_nodes
        ]
    )
    # for reasons that i don't entirely understand, the "gaussian beam" that
    # healpy.smoothing(map, sigma) convolves a healpy map with looks quite a lot
    # like a fisher-von mises distribution with concentration 2*pi/sigma**2.
    # also, the median of the convolution kernel seems to be always ~half the
    # fwhm parameter. (original idea triggered by
    # https://sourceforge.net/p/healpix/mailman/message/35267249/)
    return np.sqrt(2 * np.pi / k)


def concentration_to_fwhm(k):
    return np.sqrt(2 * np.pi / k)


def smoothing_parameter_from_opening_angles(psi_deg, qcut=0.25):
    """
    Estimate a parameter for healpy.smoothing from a sample of opening angles (in degrees)
    """
    b = np.cos(np.radians(psi_deg))
    mask = b > np.quantile(b, qcut)
    bmin = np.quantile(b, qcut)
    core = angular_resolution.fisher.fit(b[mask], floc=0, fscale=1)
    tail = angular_resolution.fisher.fit(b[~mask], floc=0, fscale=1)
    # XXX optimistically assume that a future NN-based reco will remove outliers
    # -> estimate fisher parameters from best-reconstructed 1-qcut
    # in compensation, this slightly underestimates the core of well-reconstructed events
    return concentration_to_fwhm(core[0])
    # return [(1-qcut, concentration_to_sigma(core[0])), (qcut, concentration_to_sigma(tail[0]))]


def load_monopod_angular_errors(aeff=3):
    e_reco, psi = np.loadtxt(notebook_dir / "gen2-cascade-reco" / f"d{aeff}a.txt").T
    return pd.DataFrame({"loge": np.log10(e_reco), "psi": psi})


@cache
def _fit_monopod_smoothing(qcut=0.25, aeff=3):
    df = load_monopod_angular_errors(aeff=aeff)
    bins = pd.cut(df["loge"], np.linspace(4.55, 7, 11))
    return (
        df["psi"]
        .groupby(bins)
        .apply(smoothing_parameter_from_opening_angles, qcut=qcut)
    )


def get_monopod_smoothing(loge_nodes, qcut=0.25, aeff=3):
    smoothing = _fit_monopod_smoothing(qcut, aeff)
    f = interpolate.interp1d(
        smoothing.index.categories.mid, smoothing, fill_value="extrapolate"
    )
    return f(loge_nodes)


class AngularSmearing(object):
    """
    Smear expectations across HEALpix pixels to account for angular resolution
    """

    def __init__(self, wrapped, angular_smoothing_fwhm):
        self._wrapped = wrapped
        self._angular_smoothing_fwhm = angular_smoothing_fwhm
        self._cached = None

    def __getattr__(self, name: str):
        try:
            return object.__getattr__(self, name)
        except AttributeError:
            return getattr(self._wrapped, name)

    # def __hasattr__(self, name: str):
    #     return object.__hasattr__(self, name) or hasattr(self._wrapped, name)

    def apply_angular_resolution(self, expectations, bin_edges):
        assert isinstance(expectations, np.ndarray)
        smoothed = np.empty(expectations.shape)
        for ei, fwhm in enumerate(
            self._angular_smoothing_fwhm(util.center(np.log10(bin_edges)))
        ):
            # clip smoothed map to avoid small (but deadly)
            # NB: healpy.smoothing mutates input. fun!
            smoothed[:, ei] = np.clip(
                healpy.smoothing(np.copy(expectations[:, ei]), fwhm=fwhm), 0, np.inf
            )
            assert all(smoothed[:, ei] >= 0)
        return smoothed

    def expectations(self, **kwargs):
        return self._expectations(
            **dict(sorted((k, float(v)) for k, v in kwargs.items()))
        )

    @lru_cache(maxsize=32)
    def _expectations(self, *args, **kwargs):
        if self._cached is not None:
            return self._cached

        if hasattr(self._wrapped.expectations, "__call__"):
            return self.apply_angular_resolution(
                self._wrapped.expectations(*args, **kwargs), self._wrapped.bin_edges[1]
            )
        else:
            self._cached = self.apply_angular_resolution(
                self._wrapped.expectations, self._wrapped.bin_edges[1]
            )
            return self._cached


class RingAveraging(AngularSmearing):
    """
    Average expectations over right ascension, mimicking time scrambling
    """

    def apply_angular_resolution(self, expectations, bin_edges):
        avg = np.empty(expectations.shape)
        nside = healpy.npix2nside(avg.shape[0])
        startpix, npix, *_ = healpy.ringinfo(nside, np.arange(1, 4 * nside))
        for i, n in zip(startpix, npix):
            avg[i : i + n, :] = np.mean(
                expectations[i : i + n, :], axis=0, keepdims=True
            )

        return avg
