# DNN

/data/user/ssclafani/DNNC_effa from @sclafani
https://icecube-spno.slack.com/archives/C046DQM5FEG/p1665668940764479

The dict keys are the minimum declination in the range, and the zenith binning is still every 30 degrees)
https://wiki.icecube.wisc.edu/index.php/Cascade_Neutrino_Source_Dataset/Dataset_Performace

PSF quantiles traced from https://wiki.icecube.wisc.edu/index.php/File:Cascade_Neutrino_Source_Dataset_angres.png

Effective area was made with files for all 3 flavors, but OneWeight normalization is per flavor -> divide by 3 to get average effective area per flavor

# Hans' 6-year cascade

https://icecube-spno.slack.com/files/U02N01EJP/F046H5VUM6G/nue_eff_area_zenith_bins.zip

name of the file is a lie. it actually has columns for each flavor.

# Alina Hans-like selection

https://icecube-spno.slack.com/files/UPR3YESEB/F0489CJBYUS/cascade_selection_nue.npy
https://icecube-spno.slack.com/files/UPR3YESEB/F047URQUK0X/cascade_selection_numu.npy
https://icecube-spno.slack.com/files/UPR3YESEB/F048Z3KB43S/cascade_selection_nutau.npy

The data is a bit sparse in certain places, so there are some zero-entries in this set, unfortunately.
00:47
For the nue and nutau sets, I've binned energy in np.logspace(2.6, 8.0, 13), cos(zenith) in np.linspace(-1.0, 1.0, 4), and depth in [-850.0, -500.0, -250.0, 0.0, 250.0, 500.0, 950.0]
00:47
For the numu set, energy in np.logspace(2.0, 8.0, 13), cos(zenith) in np.linspace(-1.0, 1.0, 4), and depth in [-850.0, -500.0, -250.0, 0.0, 250.0, 500.0, 950.0]

## after some extended discussion...

20:58
Alina Kochocki
 Hello, I am returning with two sets of effective areas. First, I had found there was in fact something I needed to correct in my neutral current simulation, so this change is now represented in the effective areas. This gave a ~30% reduction at these higher energies. This accounts for some of the disagreement (looks like ~260 m^2 around a PeV, so ~13x Hans' result). I will post the new effective areas below. These reflect my original GBDT selection and quality cuts:

# Gen2 cascade reconstruction

monopod distributions from tianlu with de-artifacted splines: https://icecube-spno.slack.com/archives/D2U5YF8R0/p1663790049133469



