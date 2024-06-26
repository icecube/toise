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
length, area, or volume, depending on context. `toise` may also emit projected
sensitivities in different units depending on context.

## Installation

The way to install `toise` and its depedencies is
with [`micromamba`](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)[^1]. If you do not already have `micromamba` on your system,

```console
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

or, os macOS with Homebrew:

```console
brew install micromamba
```

Next, obtain `toise`:

```sh
git clone git@github.com:icecube/toise.git
```

Then, from the `toise` source directory, create a new environment:
```sh
cd toise
micromamba create -n toise --file conda-lock.yml
micromamba run -n toise pip install -e .
```

If you have a Jupyter notebook installation from another conda environment, you should now be able to open this notebook in Jupyter and select the "Python [conda env:miniconda3-toise]" kernel.

Otherwise, activate the environment with `micromamba activate toise`. This should leave you with a prompt that looks like
```
(toise) [jakob@TheInfoSphere3:~]$.
```

If you'd like to install jupyter at this stage, you can do: 
```
micromamba install --channel conda-forge notebook
```

[^1]: micromamba is a fast, small, [mostly] drop-in replacement for the [conda](https://docs.conda.io/projects/conda/en/stable/) package manager. 

## Adding your detector to toise

`toise` uses parameterized response functions to calculate expected
sensitivities to scenarios of astrophysical neutrino production. It comes with
parameterizations for a fictive under-ice optical detector as well as a radio
detector. To use `toise` with your own detector design, you will have to provide
parameterizations its performance.

### Optical detector

For an optical detector, you will need to provide:

1. The instrumented volume of the detector.
2. For muons that enter the instrumented volume from the outside:
    1. Selection efficiency, as a function of muon energy at the boundary of the
       instrumented volume and zenith angle.
    2. Energy resolution, in the form of a transfer matrix from muon energy at
       the boundary of the instrumented volume to energy observable.
    3. Angular resolution, in the form of a cumulative distribution of opening
       angle between the true and reconstructed muon direction, as a function of
       muon energy at the boundary of the instrumented volume and zenith angle.
3. For neutrino interactions inside the instrumented volume:
    1. Selection efficiency, as a function of deposited energy and zenith angle.
    2. Energy resolution, in the form of a transfer matrix from deposited energy
       to energy observable.
    3. Angular resolution, in the form of a cumulative distribution of opening
       angle between the true and reconstructed neutrino direction, as a
       function of deposited energy and zenith angle.
    4. Flavor signature classification efficiency, in terms of an
       energy-dependent transfer matrix from interaction type to flavor
       signature (cascade, double cascade, starting track).

### Radio detector

For a radio detector, you will need to provide:

1. The (zenith and neutrino energy dependent) effective volumes for neutrinos at trigger level
2. (Optionally) The PDF of triggered shower energies for given neutrino energy.
   Otherwise the default transfer matrices of provided within `toise` will be used.

Default parametrization functions for analysis efficiency, energy resolution and angular point spread are
available within the framework. The provided function parameters can be steered by the configuration `.yaml` files.
To find appropriate parameters describing the detector to be studied, producing the following distributions is sufficient:

3. The efficiency with which triggered events can be reconstructed and analysed as a function of shower energy.
4. The (1D) energy resolution in terms of `log10(E_rec/E_shower)`.
5. The angular resolution point spread function. If using the default double-Gaussian implementation,
   the fraction of well reconstrucable events and the fraction of poorly reconstructable events with corresponding sigma.
   These angular resolution quantities may be provided energy dependently.
