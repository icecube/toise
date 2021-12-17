# toise

`toise` is a tool for estimating the sensitivity of natural-medium
neutrino detectors such as [IceCube-Gen2](https://www.icecube-gen2.de/) to
sources of high-energy astrophysical neutrinos. It uses parameterizations of a
detector's fiducial area or volume, selection efficiency, energy resolution,
angular resolution, and event classification efficiency to convert (surface)
neutrino fluxes into mean event rates in bins of observable space. These are
then used to estimate statistical quantities of interest, e.g. the median
sensitivity to some flux (i.e. 90% upper limit assuming the true flux is zero)
or the median discovery potential (i.e. the flux level at which the null
hypothesis would be rejected at 5 sigma in 50% of realizations).

A [toise](https://en.wikipedia.org/wiki/Toise) is also an archaic French unit of
length, area, or volume, depending on context.

## Installation

The way to install `toise` and its depedencies is
with `conda`. If you do not already have `miniconda` on your system, obtain the
installer from https://conda.io/miniconda.html, and install in a location of
your choice, e.g.:

```
     sh ./Miniconda3-latest-Linux-x86_64.sh -p $CONDA_PREFIX
```

replacing `CONDA_PREFIX` with the prefix you chose.

Then, obtain `toise`:

```
git clone git@github.com:IceCubeOpenSource/toise.git
```

Then, from the `toise` source directory, create a new environment:
```
  cd toise

  ICECUBE_PASSWORD=xxxx $CONDA_PREFIX/bin/conda env create -n toise --file environment.yml
```

Replace `xxxx` with the standard IceCube password. This will also download the required data tables.

The above will install the latest available versions of all dependencies. You can also install exactly the versions that were most recently tested with:
```
cd toise

PLATFORM_LOCKFILE=$(conda info --json | jq -r '"conda-\(.platform).lock"')
conda create -n toise --file $PLATFORM_LOCKFILE
cat $PLATFORM_LOCKFILE | awk '/^# pip/ {print substr($0,7)}' > requirements.txt
conda run -n toise pip install -r requirements.txt
ICECUBE_PASSWORD=xxxx conda run -n toise pip install -e .
```
This should be much faster, as it does not have to solve for compatible versions of all the dependencies. If you do not have `jq` installed, you can set `PLATFORM_LOCKFILE` by hand to e.g. `conda-osx-64.lock`.

If you have a Jupyter notebook installation from another conda environment, you should now be able to open this notebook in Jupyter and select the "Python [conda env:miniconda3-toise]" kernel.

Otherwise, activate the environment with `. $CONDA_PREFIX/bin/activate toise`. This should leave you with a prompt that looks like
```
  (toise) [jakob@TheInfoSphere3:~]$.
```

If you'd like to install jupyter at this stage, you can do: 
```
  conda install --channel conda-forge notebook
```
