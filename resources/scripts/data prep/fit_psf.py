#!/bin/sh /cvmfs/icecube.opensciencegrid.org/py2-v1/icetray-start
# METAPROJECT /data/user/jvansanten/tarballs/offline-software.openblas.2015-03-06

from icecube.photospline import spglam as glam
from icecube.photospline.utils import pad_knots
from icecube.photospline import splinefitstable
from collections import defaultdict
import numpy as np
import os


def create_psf(
    fname="FinalLevel_IC86.2011_muongun.003333.hdf5",
    muon="MCMuon",
    reco="SplineMPEMuEXDifferential",
    cut=None,
):
    import dashi
    from cubicle.hdfweights import HDFScanner

    hdf = HDFScanner(fname, primary="/" + muon, type="unweighted")
    q = hdf(
        primary="log10({muon}.energy)".format(**locals()),
        ct="cos({muon}.zenith)".format(**locals()),
        alpha="opening_angle({muon}, {reco})/I3Units.degree".format(**locals()),
        where=cut,
    )
    h = q.hist(
        fields=(
            "primary",
            "ct",
            "alpha",
        ),
        bins=(
            np.linspace(3, 8, 51),
            np.linspace(-1, 1, 21),
            np.linspace(0, 25, int(25e2)),
        ),
    )
    del q
    del hdf

    # normalize to total number of events in each energy/zenith slice
    norm = h._h_bincontent.sum(axis=2, keepdims=True)
    # avoid NaNs
    norm[norm == 0] = 1
    h._h_bincontent /= norm
    h._h_squaredweights /= norm ** 2

    return h


def fit_psf(h, smooth=1e-6):
    """
    Fit an energy- and zenith-dependent point spread function with a spline surface

    :param h: a 3D dashi histogram with dimensions log10(muon energy), cos(zenith angle), angular error [deg]
    :param smooth: smoothing strength for spline fit

    :returns: a spline fit to the cumulative angular error
    """

    centers = list(h.bincenters)
    centers[-1] = h.binedges[-1][1:]

    order = [2, 2, 3]
    power = [1.0, 1.0, 2.0]
    nknots = [25, 25, 25]
    knots = [
        pad_knots(np.linspace(e[0] ** (1.0 / p), e[-1] ** (1.0 / p), n) ** p, o)
        for n, o, p, e in zip(nknots, order, power, h.binedges)
    ]

    z = h.bincontent.cumsum(axis=2)
    w = 1.0 / h.squaredweights.cumsum(axis=2)
    w[~np.isfinite(w)] = 0.0

    del h

    penalties = defaultdict(lambda: [0.0] * len(order))
    for i, o in enumerate(order):
        penalties[o][i] = smooth
    penalties[order[0]][0] *= 1e7
    print(penalties)

    return glam.fit(z, w, centers, knots, order, penalties=penalties, monodim=2)

    return z, w, centers, knots


def plot_psf(h, spline, cos_theta=0):
    import matplotlib.pyplot as plt
    import dashi

    dashi.visual()

    ax = plt.gca()
    for logE in np.arange(4, 8):

        ei = h._h_binedges[0].searchsorted(logE) - 1
        zi = h._h_binedges[1].searchsorted(cos_theta) - 1

        sub = h[ei, zi, :]
        sub.rebin(10).scatter(cumulative=True)
        x = sub.bincenters
        y = glam.grideval(
            spline, [[h.bincenters[0][ei - 1]], [h.bincenters[1][zi - 1]], x]
        ).flatten()
        ax.plot(x, y, color=ax.lines[-1].get_color(), label="$10^%.0f$" % logE)
    plt.xlim((0, 5))
    plt.ylim((0, 1))
    plt.legend(title="$E_{\mu}$/GeV")
    plt.title(r"$\cos\theta = %.1f$" % h.bincenters[1][zi - 1])
    plt.show()


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("hdf_file")
    parser.add_argument("fits_file")
    parser.add_argument("--muon", default="MCMuon")
    parser.add_argument("--reco", default="SplineMPEMuEXDifferential")
    parser.add_argument("--smooth", default=1e-3, type=float)
    parser.add_argument("--cut", default=None)
    parser.add_argument(
        "--plot",
        default=False,
        action="store_true",
        help="Plot the result of the fit instead of fitting",
    )
    parser.add_argument(
        "--cos-theta", type=float, default=0.0, help="zenith band to plot"
    )
    opts = parser.parse_args()

    if opts.plot:
        h = create_psf(opts.hdf_file, opts.muon, opts.reco, opts.cut)
        spline = splinefitstable.read(opts.fits_file)
        plot_psf(h, spline, cos_theta=opts.cos_theta)
    else:
        spline = fit_psf(
            create_psf(opts.hdf_file, opts.muon, opts.reco, opts.cut), opts.smooth
        )
        if os.path.exists(opts.fits_file):
            os.unlink(opts.fits_file)

        splinefitstable.write(spline, opts.fits_file)
