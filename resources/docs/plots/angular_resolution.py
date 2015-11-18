
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from icecube.gen2_analysis import plotting, angular_resolution

configs = [
    ('IceCube', None),
    ('EdgeWeighted', 240),
    ('Sunflower', 200),
    ('Sunflower', 240),
    ('Sunflower', 300),
]

emu = np.logspace(3, 8, 21)

from scipy.optimize import bisect
def find_quantile(psf, q, emu, cos_theta=0):
    emu, cos_theta = np.broadcast_arrays(emu, cos_theta)
    quantiles = []
    for e, ct in zip(emu, cos_theta):
        psf_f = lambda p: psf(np.radians(p), e, ct)-q
        qpos = bisect(psf_f, 0, 25)
        quantiles.append(qpos)
    return np.array(quantiles).reshape(emu.shape)

with plotting.pretty():
    
    fig = plt.figure(figsize=(8,8))
    
    griddy = GridSpec(2,2)
    def plot_psf(ax, emu, cos_theta=0):
        for geo, spacing in configs:
            psf = angular_resolution.get_angular_resolution(geo, spacing)

            if spacing is None:
                kwargs = dict(color='k', label=geo)
            else:
                kwargs = dict(label='%s %dm' % (geo, spacing))

            psi = np.linspace(0, 2, 101)
            ax.plot(psi, psf(np.radians(psi), emu, cos_theta), **kwargs)

        ax.set_ylim(0, 1)
        ax.add_artist(AnchoredText(r'$\cos\theta=%.1f$, $E_{\mu}=$%s' % (cos_theta, plotting.format_energy('%d', emu)), loc=2, frameon=False))
        ax.grid(True)
        ax.set_ylabel(r'$P(\Psi < \Psi_{\max})$')
        ax.set_xlabel(r'Angular error $\Psi_{\max}$ [deg]')


    def plot_median_opening_angle(ax, cos_theta=0, legend=False):    
        for geo, spacing in configs:
            psf = angular_resolution.get_angular_resolution(geo, spacing)

            if spacing is None:
                kwargs = dict(color='k', label=geo)
            else:
                kwargs = dict(label='%s %dm' % (geo, spacing))

            ax.semilogx(emu, find_quantile(psf, 0.5, emu, cos_theta), **kwargs)

        ax.add_artist(AnchoredText(r'$\cos\theta=%.1f$' % (cos_theta), loc=3, frameon=False))
        ax.set_ylim(0, 0.8)
        ax.grid(True)
        ax.set_ylabel(r'Median angular error [deg]')
        ax.set_xlabel(r'$E_{\mu}$ [GeV]')

    
    plot_psf(plt.subplot(griddy[0]), 1e5, 0.)
    plot_psf(plt.subplot(griddy[1]), 1e5, -0.3)
    plot_median_opening_angle(plt.subplot(griddy[2]), 0.)
    plot_median_opening_angle(plt.subplot(griddy[3]), -0.3)
    
    fig.axes[0].legend(frameon=True, framealpha=0.8, loc='lower right').get_frame().set_linewidth(0)
    plt.tight_layout()

    plt.show()

