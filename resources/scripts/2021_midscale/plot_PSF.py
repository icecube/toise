import matplotlib.pyplot as plt
import numpy as np
from gen2_analysis import plotting, angular_resolution, effective_areas, surfaces, util
from scipy.optimize import fsolve, bisect

import argparse
parser = argparse.ArgumentParser() 
parser.add_argument("-n",type=str,help="geom name, e.g. hcr", required=False, dest='geom', default='Sunflower')
args = parser.parse_args()
spline_name=""
geo_name=""
if 'Sunflower' not in args.geom:
    spline_name = 'Sunflower_{}_240_kingpsf1'.format(args.geom)
    geo_name = 'Sunflower_{}'.format(args.geom)
else:
    spline_name = 'Sunflower_240_kingpsf1'.format(args.geom)
    geo_name = args.geom

@np.vectorize
def median_opening_angle(psf, energy, cos_theta):
    def f(psi): return psf(psi, energy, cos_theta)-0.5
    try:
        return bisect(f, 0, np.radians(5))
    except:
        return np.radians(5)

psf = angular_resolution.SplineKingPointSpreadFunction(spline_name)
psf_ic = angular_resolution.get_angular_resolution('IceCube')
aeff = effective_areas.MuonEffectiveArea(geo_name, 240)

loge = np.arange(3.5, 8.5, 0.5) + 0.25
loge_centers = 10**util.center(loge)

fig = plt.figure(figsize=(7, 2.5))
griddy = plt.GridSpec(1, 2)
ax = plt.subplot(griddy[0])
ct = np.linspace(-1, 1, 101)
for e in 1e4, 1e5, 1e6, 1e7:
    line = ax.plot(ct, aeff(e, ct)/1e6,
                   label=plotting.format_energy('%d', e))[0]

s = surfaces.get_fiducial_surface(geo_name, 240, padding=0)
ax.plot(ct, s.azimuth_averaged_area(ct)/1e6, color='grey', ls=':')
x = 0.7
y = s.azimuth_averaged_area(x)/1e6
ax.annotate(r'$A_{\rm{ geo,IceCube \operatorname{-} Gen2}}$', (x, y,), (-20, 0),
            textcoords='offset points',
            ha='right',
            va='center',
            arrowprops=dict(arrowstyle='-|>',
                            facecolor='k',
                            shrinkA=0.1,
                            connectionstyle='angle3,angleA=0,angleB=90',
                            relpos=(1, 0.5)
                            )
            )

s = surfaces.get_fiducial_surface('IceCube', padding=0)
ax.plot(ct, s.azimuth_averaged_area(ct)/1e6, color='grey', ls='--')
x = ct[np.argmax(s.azimuth_averaged_area(ct))]
y = s.azimuth_averaged_area(x)/1e6
ax.annotate(r'$A_{\rm geo, IceCube}$', (x, y,), (20, -20),
            textcoords='offset points',
            arrowprops=dict(arrowstyle='-|>',
                            facecolor='k',
                            connectionstyle='angle3,angleA=0,angleB=90',
                            relpos=(0, 0.5)
                            )
            )

ax.set_ylabel(r'Muon $A_{\rm eff}$ [km$^2$]')
ax.set_xlabel(r'$\cos\theta_{\rm zenith}$')
ax.set_ylim((0, 8))

ax = plt.subplot(griddy[1])
ctfine = np.linspace(-1, 1, 101)
for e in 1e4, 1e5, 1e6, 1e7:
    line = ax.plot(ctfine, np.degrees(psf.get_quantile(
        0.5, e, ctfine)), label=plotting.format_energy('%d', e))[0]
icmed = np.degrees(median_opening_angle(psf_ic, 1e5, ctfine))
ax.plot([-0.4, 0.4], [np.mean(icmed)]*2, ls='--', color='grey')
ax.annotate('IceCube 100 TeV', (0, np.mean(icmed)), (0, -3),
            textcoords='offset points',
            ha='center',
            va='top',
            fontsize='x-small'
            )

ax.set_ylabel(r'$\Psi_{\rm median}$ [degrees]')
ax.set_xlabel(r'$\cos\theta_{\rm zenith}$')
ax.set_ylim((0, 1.0))
legend = ax.legend(loc='upper center', title='Muon energy at detector border',
                   ncol=2, fontsize='small', handlelength=1)
legend.get_title().set_fontsize('small')

for ax in fig.axes:
    ax.set_xlim(-1, 1)
plt.tight_layout(0.1, w_pad=0.5)

fig.savefig('muon_aeff_psf_{}.png'.format(geo_name), dpi=200)

# quickly, calculate the psf at 100 TeV at the horizon
the_energy = 1e5
the_cos_theta = 0
the_med_angle = np.degrees(psf.get_quantile(0.5, the_energy, the_cos_theta))
the_area = aeff(the_energy, the_cos_theta)/1e6
the_area_downgoing = aeff(the_energy, -0.5)/1e6
print('Median angular resolution at horizon {} is {} deg'.format(plotting.format_energy('%d', the_energy), the_med_angle))
print('Effective area at {} at horizon is {} km^2'.format(plotting.format_energy('%d', the_energy), the_area))
print('Effective area at {} at cos(theta)=-0.5 is {} km^2'.format(plotting.format_energy('%d', the_energy), the_area_downgoing))

