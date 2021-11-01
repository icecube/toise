import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# from RNOG paper

def get_muon_distribution(cosz_edges = np.linspace(-1,1,21), energy_edges = np.logspace(6,12,61)):
    ### zenith dependence
    zenith_vs_fraction = np.array([[15 , 0],[20 , 0.1201],
     [20.503 , 0.1204],
     [22.84 , 0.2091],
     [25.043 , 0.1651],
     [27.518 , 0.2686],
     [29.991 , 0.5341],
     [32.607 , 0.4607],
     [34.943 , 0.6378],
     [37.142 , 0.7854],
     [39.888 , 1.1688],
     [41.941 , 1.832],
     [44.114 , 3.4824],
     [46.555 , 5.5895],
     [48.727 , 7.343],
     [51.342 , 7.2991],
     [53.675 , 7.6825],
     [55.988 , 9.215],
     [58.329 , 9.1416],
     [60.97 , 7.6392],
     [63.458 , 6.9618],
     [65.799 , 6.9031],
     [68.43 , 5.9458],
     [70.514 , 4.7674],
     [73.004 , 4.0311],
     [75.77 , 3.2506],
     [77.992 , 2.0722],
     [80.619 , 1.3359],
     [82.966 , 0.9384],
     [85.18 , 0.2167],
     [87.521 , 0.0991],
     [90 , 0.0257]])

    #cumulative sum over zenith distribution
    zenith_vs_fraction[:,1] = np.cumsum(zenith_vs_fraction[:,1])/np.sum(zenith_vs_fraction[:,1])

    theta_edges = np.arccos(cosz_edges)*180/np.pi
    # interpolate cumulative distribution at the edges
    cosz_distribution = np.interp(theta_edges,zenith_vs_fraction[:,0],zenith_vs_fraction[:,1])
    cosz_binned = (cosz_distribution[:-1]-cosz_distribution[1:])

    ### energy dependence
    energy_bins_paper = np.logspace(6,10,20)
    n_events = [0.2832,0.3132,0.3132,0.2978,0.2978,0.2092,0.2092,0.2092,0.06892,0.03576,0.01856,0.007481,0.002118,0.0005999,0.0001536,3.379e-5,6.075e-6,9.389e-7,7.529e-8,0]
    n_events_reverse_cumulative = np.flip(np.flip(n_events, 0).cumsum(), 0)

    n_binned_cumulative = np.interp(np.log10(energy_edges),np.log10(energy_bins_paper),n_events_reverse_cumulative)
    n_muons_binned = (n_binned_cumulative[:-1]-n_binned_cumulative[1:])

    muon_distribution = np.outer(cosz_binned, n_muons_binned)


    extended_muon_distribution = muon_distribution[...,None]*np.eye(60)/100.
    extended_muon_distribution = np.swapaxes(extended_muon_distribution,0,1)
    print(("total muons per station:", np.sum(extended_muon_distribution)))
    return extended_muon_distribution
