import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
import time
import os
from scipy import interpolate
from gen2_analysis.util import *


class radio_inelasticities:
    def __init__(self):
        self.files = {
            "EM": data_dir + "/inelasticity/2D_inelasticity_EM.npy",
            "EMHAD": data_dir + "/inelasticity/2D_inelasticity_EMHAD.npy",
            "HAD": data_dir + "/inelasticity/2D_inelasticity_HAD.npy",
        }

        self.inelasticities = {}
        for channel in self.files:
            self.inelasticities[channel] = np.cumsum(
                np.load(self.files[channel]), axis=1
            )
        # print(self.inelasticities)

        bins_E = np.arange(16.4, 20, 0.2)[1:]
        bins_logEfinal = np.arange(-5, 0.01, 0.1)[1:]

        self.by_interpolator = {}
        for channel in self.files:
            self.by_interpolator[channel] = interpolate.RectBivariateSpline(
                bins_E, bins_logEfinal, self.inelasticities[channel]
            )

    def get_fraction(self, energy, eout_low, eout_high, channel="HAD"):
        logE = np.log10(energy)

        flow = self.by_interpolator[channel](logE, np.log10(eout_low / energy))
        fhigh = self.by_interpolator[channel](logE, np.log10(eout_high / energy))
        # print(energy, [eout_low, eout_high], fhigh-flow)
        return fhigh - flow

    def smear_energy_slice(self, eslice, bin_edges, channel="HAD"):
        # empty slice of same shape
        eslice_copy = np.zeros_like(eslice)
        for b in range(len(eslice)):
            # unsmeared content
            content = eslice[b]
            # smear the content over all below
            for sb in range(b + 1):  # no need to smear to higher bins #use eV
                eslice_copy[sb] += content * self.get_fraction(
                    bin_edges[b + 1] * 1e9,
                    bin_edges[sb] * 1e9,
                    bin_edges[sb + 1] * 1e9,
                    channel,
                )
        return eslice_copy

    ### for aeff tuple use:
    # aeff_smeared = np.apply_along_axis(smear_energy_slice, 3, aeff.values, aeff.get_bin_edges('reco_energy'))


"""
fin = []

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

y = 0.70

labels = ["HAD", "EM", 'all']
tt = []
ww = []
EE = []
EEnu = []
EEsh = []

for iF, f in enumerate(files):
    fin.append(h5py.File(f, 'r'))
    Enu = np.array(fin[iF]['energies'])
    mask = np.ones_like(Enu, dtype=np.bool)
    Enu = Enu[mask]
    EE.extend(Enu)
    if(iF == 0):
        Esh = Enu * np.array(fin[iF]['inelasticity'])[mask]
    else:
        Esh = Enu * (1 - np.array(fin[iF]['inelasticity'])[mask])
    ws = np.array(fin[iF]['weights'])[mask]
    tt.append(np.log10(Esh / Enu))
    ww.append(ws)
    EEnu.append(Enu)
    EEsh.append(Esh)

tt.append(np.append(tt[0], tt[1]))
ww.append(np.append(ww[0], ww[1]))
EEnu.append(np.append(EEnu[0], EEnu[1]))
EEsh.append(np.append(EEsh[0], EEsh[1]))

fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))
bins = np.arange(16.4, 20, 0.2)
H, xedges, yedges = np.histogram2d(np.log10(EEnu[2]), np.log10(EEsh[2] / EEnu[2]), bins=[bins, np.arange(-5, 0.01, 0.1)])
np.nan_to_num(H)
#         Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
Hmasked = H
H_norm_rows = Hmasked / np.outer(Hmasked.sum(axis=1, keepdims=True), np.ones(H.shape[1]))

im = ax3.imshow(H_norm_rows.T, extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
           cmap=plt.get_cmap("Blues"), aspect='auto', norm=mcolors.LogNorm())
cb = fig.colorbar(im, ax=ax3, orientation='vertical')
cb.set_label("normalized entries per column")
ax3.set_ylim(-2, 0)
ax3.set_xlabel(r'$\log_{10}(E_\nu$ [eV])')
ax3.set_ylabel(r'$\log_{10}(E_\mathrm{sh}/E_\nu)$')
ax3.set_title("EM+HAD showers")
plt.tight_layout()
fig.savefig("figures/2D_inelasticity_EMHAD.png")

# only EM
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))
bins = np.arange(16.4, 20, 0.2)
H, xedges, yedges = np.histogram2d(np.log10(EEnu[1]), np.log10(EEsh[1] / EEnu[1]), bins=[bins, np.arange(-5, 0.01, 0.1)])
np.nan_to_num(H)
#         Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
Hmasked = H
H_norm_rows = Hmasked / np.outer(Hmasked.sum(axis=1, keepdims=True), np.ones(H.shape[1]))

im = ax3.imshow(H_norm_rows.T, extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
           cmap=plt.get_cmap("Blues"), aspect='auto', norm=mcolors.LogNorm())
cb = fig.colorbar(im, ax=ax3, orientation='vertical')
cb.set_label("normalized entries per column")
ax3.set_ylim(-2, 0)
ax3.set_xlabel(r'$\log_{10}(E_\nu$ [eV])')
ax3.set_ylabel(r'$\log_{10}(E_\mathrm{sh}/E_\nu)$')
ax3.set_title("EM showers")
plt.tight_layout()
fig.savefig("figures/2D_inelasticity_EM.png")

# only EM
fig3, ax3 = plt.subplots(1, 1, figsize=(7, 7))
bins = np.arange(16.4, 20, 0.2)
H, xedges, yedges = np.histogram2d(np.log10(EEnu[0]), np.log10(EEsh[0] / EEnu[0]), bins=[bins, np.arange(-5, 0.01, 0.1)])
np.nan_to_num(H)
#         Hmasked = np.ma.masked_where(H==0,H) # Mask pixels with a value of zero
Hmasked = H
H_norm_rows = Hmasked / np.outer(Hmasked.sum(axis=1, keepdims=True), np.ones(H.shape[1]))

im = ax3.imshow(H_norm_rows.T, extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
           cmap=plt.get_cmap("Blues"), aspect='auto', norm=mcolors.LogNorm())
cb = fig.colorbar(im, ax=ax3, orientation='vertical')
cb.set_label("normalized entries per column")
ax3.set_ylim(-2, 0)
ax3.set_xlabel(r'$\log_{10}(E_\nu$ [eV])')
ax3.set_ylabel(r'$\log_{10}(E_\mathrm{sh}/E_\nu)$')
ax3.set_title("HAD showers")
plt.tight_layout()
fig.savefig("figures/2D_inelasticity_HAD.png")
plt.show()

print(H_norm_rows.T)
print(np.shape(H_norm_rows.T))
"""
