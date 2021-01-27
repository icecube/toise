## About
For the midscale proposal, we calculated figures of merit (FOMs) for several "punched out" geometries. These scripts were used to calculate the FOMs.

The geometries considered are:


- the hybrid cluster ring (`Sunflower_hcr`)
- the corner (`Sunflower_cornrer`)
- the standalone (`Sunflower_standalone`)
- the sparse (`Sunflower_sparse`)

In order to make them work in the framework for the geometry keyword, instead of calling the `Sunflower`, you should insteadd call, e.g. the `Sunflower_hcr`.

## Calculating and Plotting Point Spread Functions and Muon Effective Areas
To calculate the PSF and muon selection efficiencies (which are needed for the muon effective areas), we should run `python extract_PSF.py`. This is basically a scripted version of Jakob's gists [here](https://gist.github.com/jvansanten/5eff16a895f6287eeaf9674e60d751a9#file-psf-fitting-ipynb).

The PSF extractor takes two arguments. `-n` to name the geometry. E.g. `-n hcr`. If you want to get the plain Sunflower, pass the keyword `Sunflower`. Otherwise the `hcr` will be handled correctly by the script on its own. There is a flag for "make diagnostic plots", which is `-p True`. Otherwise, no plots are produced.

The outputs of the PSF extractor are 3 fits files. Two represents the PSF (the `*king*.fits` files). These files should be moved into the `gen2-analysis/gen2_analysis/data/psf` directory. And one represents the muon selection efficiency (the `_cut.fits` file). This one should be moved into the `gen2-analysis/gen2_analysis/data/selection_efficiency` folder.

To plot the muon effective area and PSF of a geometry, use the `plot_PSF.py` function. Specifically `python plot_PSF.py -n hcr`

where the convention for the `-n` flag follows the convention from the PSF extraction

## Point Source Sensitivity
Point source senstivity (e.g. discovery potential, survey volumes) are caculated and displayed with the two scripts `calc_sens.py` and `plot_sens.py`. Run `calc` before `plot`. 

Both take two arguments passed through flags.
- the `-n` flag, which specifies a geometry. E.g. `-n hcr`. If you want to get the plain Sunflower, pass the keyword `Sunflower`. Otherwise the `hcr` will be ammended by the code.
- the `-g` flag, which specifies the spectral index of interest. E.g. `-g -2.0`. Please pass the flag as a decimal, e.g. `-2.0` not `-2`.

The plotting script also requires that you have pre-calculated the sensitivity of the IceCube and standard Sunflower. To do this, we can use the standard functions in the framework. So, make sure to also run:

```
gen2-figure-data pointsource.flare.sensitivity -d Gen2-InIce-TracksOnly 1 IceCube-TracksOnly 1 --gamma -2 -o gen2_ic86_1yr_sens_-2.0
```

## Number of Events
To calculate the number of events observed above a given energy proxy threshold (currently 100 TeV) use the `count_events.ipynb` jupyter notebook. (TODO: Make this into a script.) To change the geometry on that file, it must be altered where the geometry is defined, e.g.

```
the_geom = `Sunflower_sparse`
```  

## Cascade Fiducial Volume
Estimates of the cascade fiducial volume were done with the `resources/docs/plots/cascade_volume.py` script. The name of the [evaluated geometry](https://github.com/IceCubeOpenSource/gen2-analysis/blob/fix_fom/resources/docs/plots/cascade_volume.py#L8) was changed in the `configs` variable.


