# gen2_analysis

`gen2_analysis` is a tool for estimating the sensitivity of natural-medium
neutrino detectors such as [IceCube-Gen2](https://www.icecube-gen2.de/) to
sources of high-energy astrophysical neutrinos. It uses parameterizations of a
detector's fiducial area or volume, selection efficiency, energy resolution,
angular resolution, and event classification efficiency to convert (surface)
neutrino fluxes into mean event rates in bins of observable space. These are
then used to estimate statistical quantities of interest, e.g. the median
sensitivity to some flux (i.e. 90% upper limit assuming the true flux is zero)
or the median discovery potential (i.e. the flux level at which the null
hypothesis would be rejected at 5 sigma in 50% of realizations).

## Installation

The way to install the Gen2 analysis package and its depedencies is
with `conda`. If you do not already have `miniconda` on your system, obtain the
installer from https://conda.io/miniconda.html, and install in a location of
your choice, e.g.:

```
     sh ./Miniconda3-latest-Linux-x86_64.sh -p $CONDA_PREFIX
```

replacing `CONDA_PREFIX` with the prefix you chose.

Then, obtain the `gen2-analysis` package:

```
git clone git@github.com:IceCubeOpenSource/gen2-analysis.git
```

Then, from the `gen2_analysis` source directory, create a new environment:
```
  cd gen2-analysis

  ICECUBE_PASSWORD=xxxx $CONDA_PREFIX/bin/conda env create -n gen2-analysis --file environment.yml
```

Replace `xxxx` with the standard IceCube password. This will also download the required data tables.

The above will install the latest available versions of all dependencies. You can also install exactly the versions that were most recently tested with:
```
cd gen2-analysis

PLATFORM_LOCKFILE=$(conda info --json | jq -r '"conda-\(.platform).lock"')
conda create -n gen2-analysis --file $PLATFORM_LOCKFILE
cat $PLATFORM_LOCKFILE | awk '/^# pip/ {print substr($0,7)}' > requirements.txt
conda run -n gen2-analysis pip install -r requirements.txt
ICECUBE_PASSWORD=xxxx conda run -n gen2-analysis pip install -e .
```
This should be much faster, as it does not have to solve for compatible versions of all the dependencies. If you do not have `jq` installed, you can set `PLATFORM_LOCKFILE` by hand to e.g. `conda-osx-64.lock`.

If you have a Jupyter notebook installation from another conda environment, you should now be able to open this notebook in Jupyter and select the "Python [conda env:miniconda3-gen2-analysis]" kernel.

Otherwise, activate the environment with `. $CONDA_PREFIX/bin/activate gen2-analysis`. This should leave you with a prompt that looks like
```
  (gen2-analysis) [jakob@TheInfoSphere3:~]$.
```

If you'd like to install jupyter at this stage, you can do: 
```
  conda install --channel conda-forge notebook
```
