#!/usr/bin/env python

import numpy


def plot_energy_resolution(h2, axis_range=(2.5, 7)):
    """
    Plot the bias and resolution of a muon energy estimator

    :param: a 2D histogram, binned in log10(true muon energy) and log10(reco muon energy)
    :param axis_range: range of the true muon energy axis in log10(GeV)
    """
    import dashi

    dashi.visual()

    import matplotlib.pyplot as plt
    from matplotlib import cm, gridspec
    from matplotlib.colors import LogNorm
    from matplotlib.ticker import NullFormatter

    from toise import plotting

    with plotting.pretty(tex=False):
        fig = plt.figure(figsize=(7, 4))
        sp = dashi.histfuncs.h2profile(h2)
        for i in range(2):
            h2._h_binedges[i][:] = 10 ** h2._h_binedges[i][:]
        norm = LogNorm(1e-3, 0.5)
        colnorm = h2.bincontent.sum(axis=1, keepdims=True)
        h2.bincontent[:] /= colnorm
        h2.squaredweights[:] /= colnorm * colnorm

        outer_grid = gridspec.GridSpec(1, 2)
        left_grid = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=outer_grid[0], height_ratios=[1, 10]
        )
        right_grid = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_grid[1])

        ax = plt.subplot(left_grid[1])
        plt.loglog()
        cmap = cm.get_cmap("viridis")
        img = h2.imshow(norm=norm, cmap=cmap)
        ax.errorbar(
            10**sp.x,
            10**sp.y,
            yerr=(
                10**sp.y - 10 ** (sp.y - sp.yerr),
                10 ** (sp.yerr + sp.y) - 10**sp.y,
            ),
            color="k",
            capsize=2,
            ls=":",
        )
        cax = plt.subplot(left_grid[0])
        cb = plt.colorbar(img, orientation="horizontal", cax=cax)
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
        cb.set_label(r"$P(E_{\rm reco} | E_{\mu})$")
        axis_range = tuple([10**x for x in axis_range])
        ax.set_xlim(axis_range)
        ax.set_ylim(axis_range)
        ax.set_xlabel(r"Muon energy $E_{\mu}$ [GeV]")
        ax.set_ylabel(r"Energy proxy $E_{\rm reco}$ [arb. units]")
        ax.grid()

        ax = plt.subplot(right_grid[0])
        ax.semilogx(10**sp.x, sp.y - sp.x, color=cmap(0))
        ax.set_ylabel(r"$\left< \log_{10}(E_{\rm reco}/E_{\mu}) \right>$")
        ax.xaxis.set_major_formatter(NullFormatter())
        ax.set_xlim(axis_range)
        ax.set_ylim((-1, 0.5))
        ax.grid()

        ax = plt.subplot(right_grid[1])
        ax.semilogx(10**sp.x, sp.yerr, color=cmap(0))
        ax.set_ylabel(r"$\sigma_{\log_{10}E_{\rm reco}}$")
        ax.set_xlabel(r"Muon energy $E_{\mu}$ [GeV]")
        ax.set_xlim(axis_range)
        ax.set_ylim((0.15, 0.5))

        ax.grid()

        plt.tight_layout()
        plt.show()


def save_energy_resolution_profile(h2, fname, smoothing=1e-2):
    """
    Save the bias and resolution of the energy estimator in the format expected
    by energy_resolution.MuonEnergyResolution

    :param: a 2D histogram, binned in log10(true muon energy) and log10(reco muon energy)
    :param fname: name of the npz file to write
    :param smooth: smoothing strength. This will be passed to scipy.interpolate.UnivariateSpline
    """
    sp = dashi.histfuncs.h2profile(h2)

    # extrapolate bias linearly, standard deviation with a constant
    extend = sp.x[-1] - sp.x[0] + sp.x[1:]
    x = numpy.concatenate((sp.x, extend))
    fit = dashi.fitting.leastsq(sp.x[-10:], sp.y[-10:], dashi.fitting.poly(1))
    yextend = extend * fit.params["p1"] + fit.params["p0"]
    mean = numpy.concatenate((sp.y, yextend))
    std = numpy.concatenate(
        (sp.yerr, numpy.ones(sp.yerr.size - 1) * (sp.yerr[-10:].mean()))
    )

    numpy.savez(fname, loge=x, mean=mean, std=std, smoothing=smoothing)


if __name__ == "__main__":

    import dashi
    import tables

    with tables.open_file(
        "/Users/brianclark/Documents/work/Gen2/gen2_optical/gen2-analysis/toise/data/energy_reconstruction/aachen_muon_energy_resolution.hdf5"
    ) as hdf:
        h3 = dashi.histload(hdf, "/muex")

    plot_energy_resolution(h3.project([0, 1]))
