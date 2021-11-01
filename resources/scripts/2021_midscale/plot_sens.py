import matplotlib.pyplot as plt
import numpy as np
from gen2_analysis import util, grb
import json
import gzip

'''
This code plots the sensitivity or discovery potential (or whatever) for a detector
It along the way also calculates the survey volume in cubic GPc
'''

# some helper functions
def intflux(e, gamma=-2):
	return (e**(1+gamma))/(1+gamma)
def survey_distance(phi, L0=1e45, gamma=-2.):
	"""
	:param phi: E^-2 flux sensitivity in each declination band, in TeV/cm^2 s
	:param L0: standard candle luminosity, in erg/s
	"""
	phi = phi*(intflux(1e5, gamma) - intflux(1e-1, gamma)) # integrate from 100 GeV to 100 PeV
	phi *= (1e3/grb.units.GeV) # TeV / cm^2 s -> erg / cm^2 s
	dl = np.sqrt(L0/(4*np.pi*phi))
	dl *= grb.units.cm/1e3 # cm -> Gpc
	return np.where(np.isfinite(phi), dl, 0)
def survey_volume(sindec, phi, L0=1e45, gamma=-2):
	"""
	:param sindec: sin(declination)
	:param phi: E^-2 flux sensitivity in each declination band, in TeV/cm^2 s
	:param L0: standard candle luminosity, in erg/s
	:returns: the volume in Gpc^3 in which standard candles of luminosity L0 would be detectable
	"""
	dl = survey_distance(phi, L0, gamma)
	return ((sindec.max()-sindec.min())*2*np.pi/3.)*((dl**3).mean())

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("-n",type=str,help="geom name, e.g. hcr", required=True, dest='geom')
parser.add_argument("-g",type=str,help="the flux index", required=True, dest='gamma')
args = parser.parse_args()

the_geom = args.geom
the_index = args.gamma
L0 = 1e45
gamma_num = float(the_index)
if 'Sunflower' not in the_geom:
	the_file = 'sensitivity_Sunflower_' + the_geom + '_' + the_index+'.json'

fig = plt.gcf()
ax = plt.gca()

file = open(the_file, 'r')
dataset = json.load(file)
cos_theta = np.asarray(dataset['data']['cos_zenith'])
xc = -util.center(cos_theta)
for detector in dataset['data']['discovery_potential'].keys():
	yc = np.full(xc.shape, np.nan)
	discovery_potential = []
	for k in dataset['data']['discovery_potential'][detector].keys():
		# items are flux, ns, nb
		the_dp = dataset['data']['discovery_potential'][detector][k]['flux']
		discovery_potential.append(the_dp * 1e-12)
		yc[int(k)] = the_dp
	ax.semilogy(xc, yc*1e-12, label=detector)

	discovery_potential = np.array(discovery_potential)[::-1]
	print("Min discovery potential for {} is {}".format(detector, np.min(discovery_potential/1e-12)))
	volume = survey_volume(xc, discovery_potential, L0=L0, gamma=gamma_num)
	print("the volume for {} is {:.3f} Gpc^3".format(detector, volume))


# and now we should put the IceCube and Gen2 benchmarks on here
benchmark_file = gzip.open('gen2_ic86_1yr_sens_'+the_index+'.json.gz', 'rb')
benchmark_dataset = json.load(benchmark_file)
for detector in benchmark_dataset['data']['discovery_potential'].keys():
	yc = np.full(xc.shape, np.nan)
	discovery_potential = []
	for k in benchmark_dataset['data']['discovery_potential'][detector].keys():
		# items are flux, ns, nb
		the_dp = benchmark_dataset['data']['discovery_potential'][detector][k]['flux']
		discovery_potential.append(the_dp * 1e-12)
		yc[int(k)] = the_dp
	ax.semilogy(xc, yc*1e-12, label=detector)
	
	discovery_potential = np.array(discovery_potential)[::-1]
	# print("Min discovery potential for {} is {}".format(detector, np.min(discovery_potential/1e-12)))
	volume = survey_volume(xc, discovery_potential, L0=L0, gamma=gamma_num)
	# print("the volume for {} is {:.3f} Gpc^3".format(detector, volume))

ax.legend()
ax.set_title('1 yr Discovery Potential, E^{}'.format(the_index))
ax.set_xlabel(r'$\sin \delta$')
ax.set_ylabel(r'$E^2 \Phi_{\nu_x + \overline{\nu_x}}$ $(\rm TeV \,\, cm^{-2} s^{-1})$')
plt.tight_layout(0.1, w_pad=0.5)
# plt.show()
fig.savefig('disc_potential_{}_E{}.png'.format(the_geom, the_index), dpi=300)