#!/usr/bin/env python

import numpy as np
from icecube.toise import surfaces, util


def get_muon_selection_efficiency(
    hdf,
    muon="MCMuon",
    aeff="MuonEffectiveArea",
    fiducial_surface=surfaces.get_fiducial_surface("Sunflower", 200),
    mask=slice(None),
    nfiles=1,
    cos_theta=np.linspace(-1, 1, 21),
    muon_energy=np.logspace(2, 11, 46),
):
    """
    Calculate a muon selection efficiency from MuonGun simulation

    :param hdf: an open tables.File instance
    :param muon: name of the table containing the injected muon
    :param aeff: name of the table containing the muon effective area contribution for each event
    :param fiducial_surface: the surface to which to normalize the effective area
    :param mask: an index representing any cuts to be applied
    :param nfiles: the number of generated files lumped into the HDF5 file
    :param cos_theta: binning in zenith angle
    :param muon_energy: binning in muon energy

    :returns: a tuple (efficiency, error, bin_edges)
    """

    weights = hdf.getNode("/" + aeff).col("value") / nfiles
    muon = hdf.getNode("/" + muon)

    sample = (muon.col("energy")[mask], np.cos(muon.col("zenith")[mask]))
    bins = (muon_energy, cos_theta)
    bincontent, edges = np.histogramdd(sample, weights=weights[mask], bins=bins)
    squaredweights, edges = np.histogramdd(
        sample, weights=weights[mask] ** 2, bins=bins
    )

    # convert GeV m^2 sr to m^2
    bin_volume = 2 * np.pi * np.outer(*map(np.diff, edges))
    # normalize to geometric area
    geometric_area = np.vectorize(fiducial_surface.average_area)(
        cos_theta[:-1], cos_theta[1:]
    )[None, :]

    binerror = np.sqrt(squaredweights)
    for target in bincontent, binerror:
        target /= bin_volume
        target /= geometric_area

    return bincontent, binerror, edges


def fit_muon_selection_efficiency(efficiency, error, binedges, smoothing=1):
    from icecube.photospline import spglam as glam
    from icecube.photospline.utils import pad_knots

    z = efficiency
    w = 1.0 / error**2

    for i in range(z.shape[1]):
        # deweight empty declination bands
        if not (z[:, i] > 0).any():
            w[:, i] = 0
            continue
        # extrapolate efficiency with a constant
        last = np.where(z[:, i] > 0)[0][-1]
        zlast = z[last - 10 : last, i]
        mask = zlast != 0
        w[last:, i] = 1.0 / ((error[last - 10 : last, i][mask] ** 2).mean())
        z[last:, i] = zlast[mask].mean()
        first = np.where(z[:, i] > 0)[0][0]
        w[:first, i] = 1.0 / ((error[first : first + 10, i] ** 2).mean())
    w[~np.isfinite(w) | ~np.isfinite(z)] = 0
    centers = [util.center(np.log10(binedges[0])), util.center(binedges[1])]
    knots = [pad_knots(np.log10(binedges[0]), 2), pad_knots(binedges[1], 2)]
    order = [2, 2]
    penalties = {2: [5 * smoothing, 10 * smoothing]}
    spline = glam.fit(z, w, centers, knots, order, penalties=penalties)

    return spline


if __name__ == "__main__":

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("infile", nargs="*")
    parser.add_argument("outfile")
    parser.add_argument("-g", "--geometry", type=str, default="IceCube")
    parser.add_argument("-s", "--spacing", type=int, default=None)
    parser.add_argument(
        "-n", "--nfiles", type=int, default=1, help="Number of generated files"
    )
    parser.add_argument(
        "--cuts",
        type=str,
        default=None,
        help="e.g. (LineFit.speed < 2*I3Constants.c)&(SplineMPE_recommendedDirectHitsC.dir_track_length > 120)&(SplineMPE_recommendedDirectHitsC.n_dir_doms > 6)&(SplineMPE_recommendedFitParams.rlogl<8.5)&(SplineMPEMuEXDifferential.exists==1)&(SplineMPEMuEXDifferential.SubEvent==0)",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=1.0,
        help="Smoothing strength to apply in spline fit",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        default=False,
        help="Plot efficiencies after fitting",
    )

    opts = parser.parse_args()

    from icecube.photospline import splinefitstable
    import os
    import tables

    with tables.open_file(opts.infile[0]) as hdf:
        if opts.cuts is not None:
            from cubicle.hdfweights import HDFScanner

            q = HDFScanner(hdf, type="unweighted", primary="/MCMuon")(opts.cuts)
            mask = q()
        else:
            mask = slice(None)
        efficiency, error, binedges = get_muon_selection_efficiency(
            hdf,
            fiducial_surface=surfaces.get_fiducial_surface(opts.geometry, opts.spacing),
            nfiles=opts.nfiles,
            mask=mask,
        )

    spline = fit_muon_selection_efficiency(efficiency, error, binedges, opts.smoothing)
    if os.path.exists(opts.outfile):
        os.unlink(opts.outfile)
    splinefitstable.write(spline, opts.outfile)

    if opts.plot:

        import matplotlib.pyplot as plt
        from icecube.toise import plotting
        from icecube.photospline import spglam as glam

        from icecube.photospline import spglam as glam

        with plotting.pretty(tex=False):
            ax = plt.gca()
            x = np.linspace(0, 11, 1001)
            energy = 10 ** (util.center(np.log10(binedges[0])))
            cos_theta = util.center(binedges[1])
            evaluates = glam.grideval(spline, [x, cos_theta])
            for i in range(0, len(binedges[1]) - 1, len(binedges[1]) / 5):
                ct = cos_theta[i]
                label = "%.2f" % ct
                ax.errorbar(
                    energy,
                    efficiency[:, i],
                    yerr=error[:, i],
                    ls="None",
                    marker="o",
                    label=label,
                )
                # ax.semilogx()
                ax.semilogx(10**x, evaluates[:, i], color=ax.lines[-1].get_color())
            ax.legend()
            # ax.set_ylim((0, 1))
            ax.set_xlabel("Muon energy [GeV]")
            ax.set_ylabel("Selection efficiency")
            ax.set_title("%s %sm" % (opts.geometry, opts.spacing))
            plt.tight_layout()
            plt.show()
