![nuFATE Logo](/resources/nufate.png)

nuFATE is a code that rapidly calculates the attenuated neutrino flux as a function of energy, neutrino flavor and zenith angle, due to the earth's opacity. The software as implemented employs a user-specified power-law isotropic neutrino flux, the STW105 reference earth model, and neutrino-nucleon cross sections computed with the CT10nlo PDF distribution. The attenuation rates can be used to calculate the upgoing nu and nubar fluxes for high-energy neutrino observatories such as IceCube or ANTARES. Full description is available here: https://arxiv.org/abs/1706.09895

Prerequisites
-------------

The following packages are required to use the library, and
can probably be obtained from your favorite package manager:

* numpy: http://www.numpy.org/
* scipy: http://www.scipy.org/
* tables: https://pypi.python.org/pypi/tables

Recommended:
* ipython: http://ipython.org/
* jupyter: http://jupyter.readthedocs.io/
* matplotlib: http://matplotlib.org/

For the C++ version, you will need:

* hdf5 with c bindings: http://www.hdfgroup.org/HDF5/
* gsl (>= 1.15): http://www.gnu.org/software/gsl/
* C++ compiler with C++11 support


Compiling
---------

The Python interface can be installed by simply running:

  python setup.py install

Without write permissions, you can install it using:

  python setup.py install --user

The library can be compiled by running:

	make

An example program demonstrating usage and functionality
can be compiled with the command:

	make examples

The resulting example executables can then be found in the
subdirectory of `examples`

Finally the library can be installed using:

	make install

Compile pybinding (nuFATEpy)
-----------------------------

To compile pybinding, cvmfs service (for icecube) is required.

Also, the main nuFATE project must be installed with 'make install' command in advance.

Load icecube cvmfs environment before typing make command.

        eval `/cvmfs/icecube.opensciencegrid.org/py2-v3/setup.sh`
        cd nuFATE/src/pybinding
        make 

Then, add nuFATE/src/pybinding to your PYTHONPATH to use nuFATEpy.

Example script for using nuFATEpy is nuFATE/src/pybinding/example.py.


Example
-------

The one thing to remember is that the solution provided by nuFATE is E^2 * phi

The example script is called example.py. To run it do

python example.py

There is also an iPython notebook (notebook.ipynb) which shows some examples, including plots. This requires the "recommended" packages, above. 

To run the C++ example:

./example.exe


Cautions and some restrictions of NuFATE
-----------------------------------------------

NuFATE provides two modes to calculate attenuated flux. 
The first mode(Mode 1) calculates attenuation of initial neutrino flux which particle type (mentioned as "flavor" in program code) is defined via flavor_id.
The second mode(Mode 2) includes contribution of NuTau's regeneration effect on NuE or NuMu flux, as well as NuTau to NuTau regeneration effect. 
Note that the Mode 2 can be used only when we assume initial flux of NuE:NuEBar:NuMu:NuMuBar:NuTau:NuTauBar as 1:1:1:1:1:1, and if the condition is fulfilled, the authors always recommend using Mode 2 to include NuTau regeneration effect.

The flavor_id must be one of the integer numbers listed below.

* 1 for NuE
* -1 for NuEBar
* 2 for NuMu
* -2 for NuMuBar
* 3 for NuTau
* -3 for NuTauBar

NuEBar, NuTau and NuTauBar may generate neutrinos with other particle types due to Glashow Resonance or Tau regeneration process. Read notes listed below for each case.

### NuEBar

NuEBar may have Glashow Resonance interaction and the outgoing neutrinos from W- decay could be any flavor.

1) NuEBar + e- -> (W-) -> e- + NuEBar
2) NuEBar + e- -> (W-) -> mu- + NuMuBar
3) NuEBar + e- -> (W-) -> tau- + NuTauBar
4) NuEBar + e- -> (W-) -> hadrons

NuFATE takes into account of the first case (NuEBar + e- -> (W-) -> e- + NuEBar) only. In other words, there is no function to get the arrival flux from NuEBar to NuMuBar or NuTauBar. To estimate these contributions, use nuSQuIDS.

Cross sections of Glashow Resonance is hard coded in programs.


### NuTau and NuTauBar

NuTau and NuTauBar may generate tau via CC interaction, and the tau may decay into any flavor:

1) NuTau -> tau -> e + NuEBar + NuTau (18%)
2) NuTau -> tau -> mu + NuMuBar + NuTau (18%)
3) NuTau -> tau -> hadron + NuTau (64%)

With nuFATE Mode 1, the result attenuation ratio includes regeneration from NuTau to NuTau only.

To calculate attenuation ratio of NuE(NuEBar) or NuMu(NuMuBar) with contribution from NuTau's regeneration effect, use cascade_sec.py for python mode or activate "include_secondaries" option in constructor of c++ version. Allowed flavor_ids are -1, 1, -2, and 2.  As mentioned above, with this mode (Mode 2) initial flux is assumed to be 1:1:1:1:1:1 for all particle types.

**Example: Calculate NuEBar's attenuation ratio with NuTau's regeneration effect (Mode 2):**

Set flavor_id as -1, and give initial flux (which is same for all flavors).  
The obtained attenuation ratio has a length of 2N, where N is size of energy_nodes. The first half of the attenuation ratio is for the NuEBar with respect to initial NuEBar flux.  
The second half of attenuation ratio gives NuTauBar's attenuation, and that is same as the attenuation ratio obtained with Mode 1 simulation.


Format of cross sections
------------------------

### Format of NuFATECrossSections.h5

Total cross sections must be charged-current(CC) + neutral-current(NC) cross sections as a function of neutrino energy.
Unit for energy is GeV and for cross sections is cm^2.

Here is the details of each component. () represents shape of matrices, N represents isoscalar particle(0.5(p+n)).

- **total_cross_sections.nuexs(200,)** : NuE-N total cross section (CC + NC)
- **total_cross_sections.nuebarxs(200,)** : NuEBar-N total cross section (CC + NC)
- **total_cross_sections.numuxs(200,)** : NuMu-N total cross section (CC + NC)
- **total_cross_sections.numubarxs(200,)** : NuMuBar-N total cross section (CC + NC)
- **total_cross_sections.nutauxs(200,)** : NuTau-N total cross section (CC + NC)
- **total_cross_sections.nutaubarxs(200,)** : NuTauBar-N total cross section (CC + NC)


- **diffferential_cross_sections.dxsnu (200, 200)** : Nu-N NC differential cross section, the first axis is for primary nu energy and the second axis is for scattered nu energy. Same cross section is used for all flavors.
- **diffferential_cross_sections.dxsnubar (200, 200)** : NuBar-N NC differential cross section, the first axis for primary nubar energy and the second axis for scattered nubar energy. Same cross section is used for all flavors.


- **tau_decay_spectrum.tfull (200,200)** : Differential cross section for NuTau regeneration NuTau -> tau -> NuTau + something. The first axis for primary NuTau energy and the second axis is for regenerated NuTau energy.
- **tau_decay_spectrum.tbarfull (200,200)** : Differential cross section for NuTauBar regeneration NuTauBar -> TauBar -> NuTauBar + something. The first axis is for primary NuTauBar energy and the second axis is for regenerated NuTauBar energy.

** TODO : the following notes may be wrong, confirm it **

- **tau_decay_spectrum_secfull (200,200)** : Differential cross section for NuTau -> tau -> e+NuEBar or mu+NuMuBar. The first axis is for primary NuTau energy and the second axisis for NuEBar or NuMuBar energy.
- **tau_decay_spectrum_secbarfull (200,200)** : Differential cross section for NuTauBar -> taubar -> e+ + NuE or mu+ + NuMu. The first axis is for primary NuTauBar energy and the second axis is for NuE or NuMu energy.

For tau decay component, Appendix A and Table 1 of https://arxiv.org/abs/hep-ph/0005310 is used.


### Format of text-based cross sections
NuFATE accepts text-based cross section files in format of nuSQuIDS cross section.
Currently NuTau regeneration is not supported yet for text-based cross sections. 

Format of total cross section file must have 7 column and N rows where N = number of energy bins.  

- Energy  NuE_Xsec  NuEBar_Xsec  NuMu_Xsec  NuMuBar_Xsec  NuTau_Xsec  NuTauBar_Xsec

Energies must be in GeV and cross sections(Xsec) must be in cm^2.  
Cross section files are separated for CC interaction and NC interaction.


NuFATE uses dsigma/dE differential cross section for NC interaction.  
The differential cross section file must have 8 column and N rows where N = number of energy bins.  

- Energy_in  Energy_out  NuE_Xsec  NuEBar_Xsec  NuMu_Xsec  NuMuBar_Xsec  NuTau_Xsec  NuTauBar_Xsec

Energies must be in GeV and cross sections(Xsec) must be in cm^2.




Citation
--------

If you want cite this work, or want to look at further description
please refer to

High-energy neutrino attenuation in the Earth and its associated uncertainties

Aaron C. Vincent, Carlos A. Arguelles, and A. Kheirandish

arXiv:1706.09895

Contributors
------------

- Aaron C. Vincent
- Carlos A. Arguelles
- Ali Kheirandish
- Ibrahim Safa
- Kotoyo Hoshina

