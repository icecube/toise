#!/usr/bin/env python

import numpy as np
import operator
from icecube.gen2_analysis import surfaces, util

def get_muon_selection_efficiency(hdf, muon='MCMuon', aeff='MuonEffectiveArea',
    fiducial_surface=surfaces.get_fiducial_surface('Sunflower', 200),
    mask=slice(None),
    nfiles=1, cos_theta=np.linspace(-1, 1, 21), muon_energy=np.logspace(2, 11, 46)):
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
    
    weights = hdf.getNode('/'+aeff).col('value')/nfiles
    muon = hdf.getNode('/'+muon)
    
    sample = (muon.col('energy')[mask], np.cos(muon.col('zenith')[mask]))
    bins = (muon_energy, cos_theta)
    bincontent, edges = np.histogramdd(sample, weights=weights[mask], bins=bins)
    squaredweights, edges = np.histogramdd(sample, weights=weights[mask]**2, bins=bins)
    
    # convert GeV m^2 sr to m^2
    bin_volume = 2*np.pi*np.outer(*map(np.diff, edges))
    # normalize to geometric area
    geometric_area = np.vectorize(fiducial_surface.average_area)(cos_theta[:-1], cos_theta[1:])[None,:]
    
    binerror = np.sqrt(squaredweights)
    for target in bincontent, binerror:
        target /= bin_volume
        target /= geometric_area
    
    return bincontent, binerror, edges

if __name__ == "__main__":

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('infile', nargs='*')
    parser.add_argument('outfile')
    parser.add_argument('-g', '--geometry', type=str, default='IceCube')
    parser.add_argument('-s', '--spacing', type=int, default=None)
    parser.add_argument('-n', '--nfiles', type=int, default=1, help='Number of generated files')

    opts = parser.parse_args()

    import tables
    
    hdf = tables.open_file(opts.infile[0])
    
    efficiency, error, bin_edges = get_muon_selection_efficiency(hdf,
        fiducial_surface=surfaces.get_fiducial_surface(opts.geometry, opts.spacing),
        # NB: average over all angles for demonstration purposes
        cos_theta=[-1, 1])
    
    import matplotlib.pyplot as plt
    from icecube.gen2_analysis import plotting
    
    with plotting.pretty(tex=False):
        centers = 10**(util.center(np.log10(bin_edges[0])))
        plt.errorbar(centers, efficiency[:,0], yerr=error[:,0])
        plt.xlabel('Muon energy [GeV]')
        plt.ylabel('Selection efficiency')
        plt.title('%s %dm' % (opts.geometry, opts.spacing))
        plt.semilogx()
        plt.ylim((0, 1.1))
        plt.tight_layout()
        plt.show()