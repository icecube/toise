import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import logging


logger = logging.getLogger("toise radio muon background")


def get_muon_distribution(
    cosz_edges=np.linspace(-1, 1, 21), energy_edges=np.logspace(6, 12, 61)
):
    """First best guess muon distribution

    Distribution was given as function of zenith and as function of energy
    in Phys. Rev. D 102, 083011 (2020) / DOI: 10.1103/PhysRevD.102.083011
    The zenith dependence is assumed to be independent of energy for this proxy
    """
    ### zenith dependence
    zenith_vs_fraction = np.array(
        [
            [15, 0],
            [20, 0.1201],
            [20.503, 0.1204],
            [22.84, 0.2091],
            [25.043, 0.1651],
            [27.518, 0.2686],
            [29.991, 0.5341],
            [32.607, 0.4607],
            [34.943, 0.6378],
            [37.142, 0.7854],
            [39.888, 1.1688],
            [41.941, 1.832],
            [44.114, 3.4824],
            [46.555, 5.5895],
            [48.727, 7.343],
            [51.342, 7.2991],
            [53.675, 7.6825],
            [55.988, 9.215],
            [58.329, 9.1416],
            [60.97, 7.6392],
            [63.458, 6.9618],
            [65.799, 6.9031],
            [68.43, 5.9458],
            [70.514, 4.7674],
            [73.004, 4.0311],
            [75.77, 3.2506],
            [77.992, 2.0722],
            [80.619, 1.3359],
            [82.966, 0.9384],
            [85.18, 0.2167],
            [87.521, 0.0991],
            [90, 0.0257],
        ]
    )

    # cumulative sum over zenith distribution
    zenith_vs_fraction[:, 1] = np.cumsum(zenith_vs_fraction[:, 1]) / np.sum(
        zenith_vs_fraction[:, 1]
    )

    theta_edges = np.arccos(cosz_edges) * 180 / np.pi
    # interpolate cumulative distribution at the edges
    cosz_distribution = np.interp(
        theta_edges, zenith_vs_fraction[:, 0], zenith_vs_fraction[:, 1]
    )
    cosz_binned = cosz_distribution[:-1] - cosz_distribution[1:]

    ### energy dependence
    energy_bins_paper = np.logspace(6, 10, 20)
    n_events = [
        0.2832,
        0.3132,
        0.3132,
        0.2978,
        0.2978,
        0.2092,
        0.2092,
        0.2092,
        0.06892,
        0.03576,
        0.01856,
        0.007481,
        0.002118,
        0.0005999,
        0.0001536,
        3.379e-5,
        6.075e-6,
        9.389e-7,
        7.529e-8,
        0,
    ]
    n_events_reverse_cumulative = np.flip(np.flip(n_events, 0).cumsum(), 0)

    n_binned_cumulative = np.interp(
        np.log10(energy_edges), np.log10(energy_bins_paper), n_events_reverse_cumulative
    )
    n_muons_binned = n_binned_cumulative[:-1] - n_binned_cumulative[1:]

    muon_distribution = np.outer(cosz_binned, n_muons_binned)

    extended_muon_distribution = muon_distribution[..., None] * np.eye(60) / 100.0
    extended_muon_distribution = np.swapaxes(extended_muon_distribution, 0, 1)
    logger.info(("Total muons per station:", np.sum(extended_muon_distribution)))
    return extended_muon_distribution


def get_tabulated_muon_distribution(pickle_file, cr_cut=True):
    """Get a tabulated muon distribution from a pickle file"""

    import pickle

    with open(pickle_file, "rb") as fin:
        shower_energy_bins, cos_zenith_bins, z_zen, z_zen_crcut = pickle.load(fin)
        # simply check if the shower_energy_bins and cos_zenith_bins match the expected shape or not.
        expected_shower_energy_bins = np.linspace(13.0, 20.0, 71)
        expected_cos_zenith_bins = np.linspace(1.0, 0.0, 11)
        expected_cos_zenith_bins[-1] = 1e-3
        if not np.allclose(shower_energy_bins, expected_shower_energy_bins):
            logger.error(f"energy binning not as expected. Got {shower_energy_bins}, expected {expected_shower_energy_bins}")
            return None
        if not np.allclose(cos_zenith_bins, expected_cos_zenith_bins):
            logger.error(f"cos zenith binning not as expected. Got {cos_zenith_bins}, expected {expected_cos_zenith_bins}")
            return None

        energy_centers = 0.5 * (shower_energy_bins[1:] + shower_energy_bins[:-1])
        z = np.sum(z_zen, axis=0)
        z_crcut = np.sum(z_zen_crcut, axis=0)
        n_tot = np.sum(z)
        n_crcut_tot = np.sum(z_crcut)

        new_cos_zenith_bins = np.linspace(-1, 1, 21)
        new_shower_energy_bins = np.logspace(15 - 9, 21 - 9, 61)

        if cr_cut is True:
            distribution = z_zen_crcut
        else:
            distribution = z_zen

        # atm muon distributions don't contain upgoing region
        upgoing = np.zeros((10, np.shape(distribution)[1]))

        distribution_4pi = np.append(upgoing, np.flip(distribution, axis=0), axis=0)
        distribution_4pi = distribution_4pi[:, 10:]
        extended_muon_distribution = distribution_4pi[..., None] * np.eye(60)
        extended_muon_distribution = np.swapaxes(extended_muon_distribution, 0, 1)

        return (
            new_shower_energy_bins,
            new_cos_zenith_bins,
            new_shower_energy_bins,
        ), extended_muon_distribution
