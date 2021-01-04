## Installation

The easiest way to install the Gen2 analysis package and its depedencies is
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

If you have a Jupyter notebook installation from another conda environment, you should now be able to open this notebook in Jupyter and select the "Python [conda env:miniconda3-gen2-analysis]" kernel.

Otherwise, activate the environment with `. $CONDA_PREFIX/bin/activate gen2-analysis`. This should leave you with a prompt that looks like
```
  (gen2-analysis) [jakob@TheInfoSphere3:~]$.
```

If you'd like to install jupyter at this stage, you can do: 
```
  conda install --channel conda-forge notebook
```
