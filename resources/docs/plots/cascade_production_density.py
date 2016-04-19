
import matplotlib.pyplot as plt
import numpy as np
import itertools
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from icecube.gen2_analysis import plotting, effective_areas, util

# get muon production efficiency
edges, muprod = effective_areas.get_cascade_production_density()
# average over neutrinos/antineutrinos, sum over all muon energies
efficiency = muprod.sum(axis=(0,3))/(2*3)
# average over neutrinos/antineutrinos
zi = 4
energy_spectrum = (muprod[:,:,zi,:].sum(axis=(0))/2.)

with plotting.pretty():
    
    fig = plt.figure(figsize=(8,4))
    griddy = GridSpec(1,2)
    
    ax = plt.subplot(griddy[0])
    for i, zc in itertools.islice(enumerate(util.center(edges[1])), 2, 6, 1):
        ax.loglog(*plotting.stepped_path(edges[0], efficiency[:,i]), label='%.1f' % zc)
    ax.set_ylim((1e-10, 1e-5))
    ax.set_xlim(1e3, 1e9)

    ax.legend(loc='best', title=r'$\cos\theta$')
    ax.set_xlabel(r'$E_{\nu}$ [GeV]')
    ax.set_ylabel('Cascade density in ice [1/m]')
    
    ax = plt.subplot(griddy[1])
    decade = 10
    for i, ec in itertools.islice(enumerate(edges[2][1:]), 2*decade-1, 6*decade-1, decade):
        ax.loglog(*plotting.stepped_path(edges[2], energy_spectrum[i,:], cumulative='>'), label=plotting.format_energy('%d', ec), nonposy='clip')

    ax.set_xlim((1e2, 2e7))
    ax.set_ylim((1e-10, 5e-7))
    ax.set_xlabel(r'$E_{\mu,\min}$ [GeV]')
    ax.set_ylabel(r'$dP(E_{\rm cascade} > E_{\rm cascade,\min})/dx$ [1/m]')
    
    ax.add_artist(AnchoredText(r"$\cos\theta=%.1f$" % util.center(edges[1])[zi], loc=1, frameon=False))

    ax.legend(loc='lower left', title=r'$E_{\nu}$')

    plt.tight_layout()
    plt.show()