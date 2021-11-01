import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from icecube.gen2_analysis import plotting, effective_areas, util

# get muon production efficiency
edges, muprod = effective_areas.get_muon_production_efficiency()
# average over neutrinos/antineutrinos, sum over all muon energies
efficiency = muprod.sum(axis=(0, 3)) / 2
# average over neutrinos/antineutrinos
zi = 4
energy_spectrum = muprod[:, :, zi, :].sum(axis=(0)) / 2.0

with plotting.pretty():

    fig = plt.figure(figsize=(8, 4))
    griddy = GridSpec(1, 2)

    ax = plt.subplot(griddy[0])
    for i, zc in itertools.islice(enumerate(util.center(edges[1])), 3, 7, 1):
        ax.loglog(*plotting.stepped_path(edges[0], efficiency[:, i]), label="%.1f" % zc)
    ax.set_ylim(1e-6, 1e-1)
    ax.set_xlim(1e3, 1e10)

    ax.legend(loc="best", title=r"$\cos\theta$")
    ax.set_xlabel(r"$E_{\nu}$ [GeV]")
    ax.set_ylabel("Muon production efficiency")

    ax = plt.subplot(griddy[1])
    for i, ec in itertools.islice(enumerate(edges[2][1:]), 30 - 1, 70 - 1, 10):
        ax.loglog(
            *plotting.stepped_path(edges[2], energy_spectrum[i, :], cumulative=">"),
            label=plotting.format_energy("%d", ec),
            nonposy="clip"
        )

    ax.set_xlim((1e1, 1e7))
    ax.set_xlabel(r"$E_{\mu,\min}$ [GeV]")
    ax.set_ylabel(r"$P(E_{\mu} > E_{\mu,\min})$")

    ax.add_artist(
        AnchoredText(
            r"$\cos\theta=%.1f$" % util.center(edges[1])[zi], loc=1, frameon=False
        )
    )

    ax.legend(loc="lower left", title=r"$E_{\nu}$")

    plt.tight_layout()
    plt.show()
