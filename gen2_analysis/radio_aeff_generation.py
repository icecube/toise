##
from __future__ import print_function
##
import dashi
import tables
from scipy import interpolate
import numpy as np

import logging
logger = logging.getLogger("aeff calculation")


import os
import numpy
import itertools
import healpy
import warnings
import copy

from surfaces import get_fiducial_surface
from energy_resolution import get_energy_resolution
from angular_resolution import get_angular_resolution
from inelasticity import radio_inelasticities
from classification_efficiency import get_classification_efficiency
from util import *


def _load_rno_veff(filename= data_dir + "/aeff/run_input_500km2_01_surface_4LPDA_1dipole_RNOG_1.50km_config_Alv2009_nonoise_100ns_D01surface_4LPDA_1dipole_250MHz_dipoles_RNOG_200m_3.00km_D02single_dipole_250MHz_e.json", trigger="dipole_2.5sigma"):
    """
    :returns: a tuple (edges, veff). veff has units of m^3
    """
    import pandas as pd
    import json

    if not filename.startswith('/'):
        filename = os.path.join(data_dir, 'aeff', filename)
    with open(filename) as f:
        dats = json.load(f)
    index = []
    arrays = {'veff': []}
    for zenith, values in dats.items():
        for selection, items in values.items():
            if selection == trigger: 
                #print(selection, items)
                for energy, veff in zip(items['energies'], items['Veff']):
                    index.append((selection, energy, np.cos(float(zenith))))
                    arrays['veff'].append(veff)
    veff = pd.DataFrame(arrays, index=pd.MultiIndex.from_tuples(
        index, names=['selection', 'energy', 'cos_zenith']))
    #print(veff, veff.veff)
    veff.sort_index(level=[0, 1, 2], inplace=True)
    # add right-hand bin edges
    energy = veff.index.levels[1].values/1e9
    energy = np.concatenate([energy, [energy[-1]**2/energy[-2]]])
    # left-hand bin edges were specified in zenith, so add in reverse
    cos_zenith = veff.index.levels[2].values
    cos_zenith = np.concatenate(
        ([2*cos_zenith[0] - cos_zenith[1]], cos_zenith))
    omega = 2*np.pi*np.diff(cos_zenith)
    print("WARNING: aeff is being normalised to per sr")
    print(veff['veff'].unstack(level=-1).values.reshape((energy.size-1, cos_zenith.size-1)))
    return (energy, cos_zenith), veff['veff'].unstack(level=-1).values.reshape((energy.size-1, cos_zenith.size-1)) / omega[None, :]


def _load_radio_review_veff(filename = data_dir + "/aeff/review_array_dict_e.pkl", trigger=None):
    """
    :returns: a tuple (edges, veff). veff has units of m^3
    """
    import pandas as pd
    import numpy as np
    import json

    if not filename.startswith('/'):
        filename = os.path.join(data_dir, 'aeff', filename)

    # read filename and convert to pandas
    data = open(filename, 'r')
    jsonfile = json.load(data)
    dataframe = pd.DataFrame(jsonfile)
    #print(dataframe)
    def _list_of_triggers(df):
        triggerlist = [list(df.veff[i]) for i in range(len(df))]
        triggerlist_flat = np.concatenate(triggerlist).ravel().tolist()
        triggerlist_set = list(set(triggerlist_flat))
        return triggerlist_set
    if trigger==None:
        #simply take the first
        trigger = _list_of_triggers(dataframe)[0]

    def _extract_veff(df, triggername):
        veff = np.array([df.veff[i][triggername][0] for i in range(len(df))])
        return veff

    dataframe['veff_values'] = _extract_veff(dataframe, trigger)
    dataframe['cos_theta'] = (np.cos(dataframe.thetamin) + np.cos(dataframe.thetamax)) / 2

    #get bin edges in energy and cos(theta)    
    def _energy_bin_edges(energy):
        bin_centers = list(set(list(energy)))
        bin_centers = list(np.sort(bin_centers))
        # padding left right
        bin_centers.append(bin_centers[-1]**2/bin_centers[-2])
        bin_centers.append(bin_centers[0]**2/bin_centers[1])
        bin_centers = list(np.sort(bin_centers))
        # calculate centers
        logedges = [0.5*(np.log10(bin_centers[i])+np.log10(bin_centers[i+1])) for i in range(len(bin_centers)-1)]
        edges = 10**np.array(logedges)
        return edges
    bin_edges_energy_gev = _energy_bin_edges(dataframe.energy)/1e9

    bin_edges_costheta = list(set(list(np.round(np.cos(dataframe.thetamin),5)) + list(np.round(np.cos(dataframe.thetamax)))))
    bin_edges_costheta = np.sort(bin_edges_costheta)
        
    # keep only necessary columns
    veff = dataframe.filter(['energy','cos_theta','domega', 'veff_values'], axis=1)    
    veff.sort_index(level=[0, 1, 2, 3], inplace=True)
    
    veff_values = np.array(veff.veff_values)
    return (bin_edges_energy_gev, bin_edges_costheta), veff_values.reshape(bin_edges_energy_gev.size-1, bin_edges_costheta.size-1)




def _interpolate_rno_veff(energy_edges, ct_edges=None, filename="jaml.file"):
    from scipy import interpolate
    #print("interpolating effective area")
    edges, veff = _load_radio_review_veff(filename)
    #print("shape", np.shape(veff))
    # NB: occasionally there are NaN effective volumes. intepolate through them
    def interp_masked(arr, x, xp):
        valid = ~np.ma.masked_invalid(arr).mask

        if np.sum(valid)==0:
            return -np.inf*np.ones_like(x)
        else:
            interpolator = interpolate.interp1d(xp, arr, fill_value='extrapolate')
            interpolation_result = interpolator(x)



            ####interpolation_result =  np.interp(x, xp[valid], arr[valid], left=-np.inf)
            #interpolation_result = np.maximum(np.nan_to_num(interpolation_result, 0.),-10)
            #print("interp res", interpolation_result)
            return interpolation_result
    veff = 10**(np.apply_along_axis(
        interp_masked,
        0,
        np.log10(np.clip(veff,1e-10,np.inf)),
        center(np.log10(energy_edges)),
        center(np.log10(edges[0]))
        )
        )
    #print("INTERPOLATION RESULT", np.shape(veff), veff)
    #print("Interpolated shape", np.shape(veff))
    if ct_edges is None:
        return (energy_edges, edges[1]), veff
    else:
        interp = interpolate.interp1d(
            center(edges[1]),
            veff,
            'nearest',
            axis=1,
            bounds_error=False,
            fill_value=0
        )
        center_ct = np.clip(center(ct_edges),min(center(edges[1]))+1e-3,max(center(edges[1]))-1e-3)
        
        return (energy_edges, ct_edges), interp(center_ct)



def efficiency_sigmoid(x, eff_low, eff_high, loge_turn, loge_halfmax):
    # sigmoid function in logE for efficiency between 0 and 1
    logx = np.log10(x)
    # choose factors conveniently
    # loge_halfmax should correspond to units in logE from turnover, where 0.25/0.75 of max are reached
    # = number of orders of magnitude in x between 0.25..0.75*(max-min) range
    b = np.log(3)/loge_halfmax

    eff = ((eff_low-eff_high) / (1 + (np.exp(b*(logx-loge_turn))))) + eff_high
    # do not allow below 0
    eff = np.maximum(0, eff)
    #for fit, upper range max 1 prevents fit from converging if called without start values
    #val = np.minimum(1,val)
    return  eff

def bound_efficiency_sigmoid(x, eff_low, eff_high, e_turn, e_fwhm):
    # sigmoid function in logE for efficiency between 0 and 1
    # hard limits between 0 and 1
    eff = efficiency_sigmoid(x, eff_low, eff_high, e_turn, e_fwhm)
    #limit to range between 0 and 1
    eff = np.maximum(0, eff)
    eff = np.minimum(1, eff)
    return  eff


def rno_analysis_efficiency(E, minval, maxval, log_turnon_gev, log_turnon_width):
    #return bound_efficiency_sigmoid(E, -0.16480194,  0.76853897,  8.46903659,  1.03517252)
    return bound_efficiency_sigmoid(E, minval, maxval, log_turnon_gev, log_turnon_width)




class StationOverlap:
    ''' overlap is parametrised for different values of station spacing as function of energy'''
    overlap_fit_data = {
        0:     [1.16915194, 0.76565583],
        100:   [14.71768875,  0.59238456],
        250:   [16.15636026,  0.50626117],
        500:   [16.92610908,  0.520047  ],
        750:   [17.49885381,  0.45712464],
        1000:  [17.90166045,  0.48601167],
        1250:  [18.26189115,  0.5322479 ],
        1500:  [18.6003936,   0.55918667],
        2000:  [19.1751211,   0.58951958],
        2500:  [19.64801491,  0.58333683],
        3000:  [20.06492668,  0.56412106]}
    
    def __init__(self, spacing):
        self.find_spacing(spacing)
        
    def find_spacing(self, spacing):
        print("requested spacing %f" %spacing)
        spacings = np.array(self.overlap_fit_data.keys())
        best = spacings[np.abs(spacings - spacing).argmin()]
        print("using nearest available:", best)
        self.spacing = best
        
    def overlap_sigmoid(self, x, loge_turn, loge_halfmax):
        # sigmoid function in logE for overlap between 0 and 1
        logx = np.log10(x)
        # choose factors conveniently
        # loge_halfmax should correspond to units in logE from turnover, where 0.25/0.75 of max are reached
        # = number of orders of magnitude in x between 0.25..0.75*(max-min) range
        b = np.log(3)/loge_halfmax
    
        y_low=0 # in the limit of zero energy there will be no overlap
        y_high=1 # in the limit of infinite energy 100% overlap can be expected

        y = ((y_low-y_high) / (1 + (np.exp(b*(logx-loge_turn))))) + y_high
        return  y

    def overlap_sigmoid_fit(self, energies, the_overlap):
        #print(loge, the_eff)
        res, vals = optimize.curve_fit(self.overlap_sigmoid, energies, the_overlap)
        return res
    
    def get_overlap(self, e):
        turn  = self.overlap_fit_data[self.spacing][0]
        width = self.overlap_fit_data[self.spacing][1]
        return self.overlap_sigmoid(e, turn, width)

        '''
        Idea, apply downscaling to account for station overlap, like:
        Caveat!: This needs modification if applied!!!

        if 'spacing_m' in configuration['detector_setup']:
            spacing = configuration['detector_setup']['spacing_m']
            print('requested station spacing: %f' %spacing)
            overlap = StationOverlap(spacing)

            scale_factor = 1.-overlap.get_overlap(e_nu[:-1]*1e9)
            print('scale factor:', e_nu, scale_factor)
            aeff = np.apply_along_axis(np.multiply, 1, aeff, scale_factor)
        '''


#from radio_response import StationOverlap
from radio_response import radio_analysis_efficiency
from radio_response import RadioPointSpreadFunction
from radio_response import RadioEnergyResolution
from effective_areas import calculate_cascade_production_density
from effective_areas import effective_area

class radio_aeff:
    def __init__(self, config=os.path.realpath(os.path.join(os.path.dirname(__file__), 'radio_config.yaml')), psi_bins=None):
        self.configfile = config
        # set parameters provided in config file
        self.restore_config()
        self.logger = logger
        self.logger.info(self.configuration)
        self.set_psi_bins(psi_bins=psi_bins)

    def switch_smearing_inelasticity(self, on=True):
        if type(on)==bool:
            self.configuration['apply_smearing_inelasticity'] = on
        else:
            self.logger.warning('no boolean provided for switch')

    def switch_energy_resolution(self, on=True):
        if type(on)==bool:
            self.configuration['apply_energy_resolution'] = on
        else:
            self.logger.warning('no boolean provided for switch')

    def switch_analysis_efficiency(self, on=True):
        if type(on)==bool:
            self.configuration['apply_analysis_efficiency'] = on
        else:
            self.logger.warning('no boolean provided for switch')

    def restore_config(self):
        import yaml
        config = self.configfile
        with open(config) as file:
            self.configuration = yaml.full_load(file)
    
    def scale_veff(self, factor):
        if not 'nstations' in self.configuration['detector_setup']:
            self.configuration['detector_setup']['nstations'] = 1.
        self.configuration['detector_setup']['nstations'] *= factor

    def get_parameter(self, group, name):
        return self.configuration[group][name]

    def set_parameter(self, group, name, value):
        self.logger.debug("changing parameter {}/{}".format(group, name))
        self.logger.debug(" old value: %f" %self.get_parameter(group, name))
        self.configuration[group][name] = value
        self.logger.info("New parameter value: %f" %self.get_parameter(group, name))

    def set_psi_bins(self, psi_bins):
        self.psi_bins = psi_bins

    def create_muon_background(self,
            energy_resolution=RadioEnergyResolution(),
            cos_theta=np.linspace(-1, 1, 21), neutrino_energy=np.logspace(6, 12, 61)):
        from radio_muon_background import get_muon_distribution
        psi_bins = self.psi_bins
        configuration = self.configuration

        aeff = get_muon_distribution(cos_theta, neutrino_energy)
        edges = [neutrino_energy, cos_theta, neutrino_energy]
        #print(cos_theta)
        self.logger.warning("Energy/direction resolution smearing not applied for atm. muons for now!")
        
        # apply smearing for shower energy resolution
        configuration = self.configuration
        if configuration['apply_energy_resolution'] == True:
            self.logger.info("applying energy resolution smearing")
            energy_resolution.set_params(configuration['energy_resolution'])
            response = energy_resolution.get_response_matrix(neutrino_energy, neutrino_energy)
            self.logger.debug('... energy response shape: {}'.format(np.shape(response)))
            aeff = np.apply_along_axis(np.inner, 2, aeff, response)
        else:
            self.logger.warning("requested to skip accounting for energy resolution")
        
        return effective_area(edges, aeff, 'cos_theta')

    def create(self,
            energy_resolution=RadioEnergyResolution(),
            psf=RadioPointSpreadFunction(),
            cos_theta=np.linspace(-1, 1, 21), neutrino_energy=np.logspace(6, 12, 61)):

        psi_bins = self.psi_bins
        configuration = self.configuration
        """
        Create an effective area for radio
        """
        nside = None
        if isinstance(cos_theta, int):
            nside = cos_theta

        # Step 1: Density of final states per metre
        self.logger.debug('STEP 1: Density of final states per metre')

        (e_nu, cos_theta, e_showering), aeff = calculate_cascade_production_density(
            cos_theta, neutrino_energy)

        # quick fix to ignore Earth absorption... this is already included in the effective volumes
        for i in range((len(cos_theta)-1)//2):
            aeff[:,:,i,:] = aeff[:,:,-i-1,:]

        self.logger.debug('Shape of aeff tuple: {}'.format(np.shape(aeff)))

        # Step 2: Trigger effective volume in terms of shower energy
        self.logger.debug('STEP 2: Trigger effective volume per station in terms of shower energy')

        veff_filename = configuration['effective_volume']
        edges_e, veff_e = _interpolate_rno_veff(e_showering, cos_theta, filename=veff_filename['e'])
        edges_mu, veff_mu = _interpolate_rno_veff(e_showering, cos_theta, filename=veff_filename['mu'])
        edges_tau, veff_tau = _interpolate_rno_veff(e_showering, cos_theta, filename=veff_filename['tau'])

        if 'simulated_stations' not in veff_filename:
            veff_filename['simulated_stations'] = 1
            self.logger.info('will not rescale effective volumes to per-station')

        # Step 3: Scale trigger effective volume to desired number of stations
        self.logger.debug('STEP 3: Scaling trigger effective volume for number of stations and applying to density of final states')
        if 'nstations' in configuration['detector_setup']:
            nstations = configuration['detector_setup']['nstations']
            self.logger.info("using number of stations from config")
        else:
            nstations = 1
            self.logger.warning("Number of stations not passed in config. Using single station proxy.")
        veff_scale = nstations/float(veff_filename['simulated_stations'])
        self.logger.debug('Veff scaling factor:{}'.format(veff_scale))

        ### applying triggered veff in terms of cosz / e_shower
        ###aeff[0:2,...] *= (veff_e.T)[None,None,...]*veff_scale # electron neutrino
        ###aeff[2:4,...] *= (veff_mu.T)[None,None,...]*veff_scale # muon neutrino
        ###aeff[4:6,...] *= (veff_tau.T)[None,None,...]*veff_scale # tau neutrino

        ### applying triggered veff in terms of e_neutrino / cosz
        aeff[0:2,...] *= (veff_e)[None,...,None]*veff_scale # electron neutrino
        aeff[2:4,...] *= (veff_mu)[None,...,None]*veff_scale # muon neutrino
        aeff[4:6,...] *= (veff_tau)[None,...,None]*veff_scale # tau neutrino

        # Step 4: Account for additional downsmear in energy for triggered shower energy (i.e. inelasticity smearing)
        # this is slow and should be changed to something more reasonable. Enu -> Eshower already taken care of in step 1. Accurate enough for our purpose?
        if configuration['apply_smearing_inelasticity'] == True:
            self.logger.info("smearing neutrino energy to deposited shower energy")
            ### smear for inelasticity
            triggered_inelasticities = radio_inelasticities()
            self.logger.info("... EM for nue")
            aeff[0:2,...] = np.apply_along_axis(triggered_inelasticities.smear_energy_slice, 3, aeff[0:2,...], e_showering, "EM")
            self.logger.info("... HAD for numu, nutau")
            aeff[2:4,...] = np.apply_along_axis(triggered_inelasticities.smear_energy_slice, 3, aeff[2:4,...], e_showering, "HAD")
            aeff[4:6,...] = np.apply_along_axis(triggered_inelasticities.smear_energy_slice, 3, aeff[4:6,...], e_showering, "HAD")
        else:
            self.logger.warning("requested to skip inelasticity smearing")

        # Step 5: Scale down effective volume to account for analysis efficiency
        if configuration['apply_analysis_efficiency'] == True:
            self.logger.info("applying analysis efficiency as function of neutrino energy")

            ana_efficiency = rno_analysis_efficiency(e_nu[:-1],
                    configuration['analysis_efficiency']['minval'],
                    configuration['analysis_efficiency']['maxval'],
                    configuration['analysis_efficiency']['log_turnon_gev'],
                    configuration['analysis_efficiency']['log_turnon_width'])
            aeff = np.apply_along_axis(np.multiply, 1, aeff, ana_efficiency)
        else:
            self.logger.warning("requested to skip accounting for analysis efficiency")            

        # Step 6: apply smearing for energy resolution
        if configuration['apply_energy_resolution'] == True:
            self.logger.info("applying energy resolution smearing")
            energy_resolution.set_params(configuration['energy_resolution'])
            response = energy_resolution.get_response_matrix(e_showering, e_showering)
            self.logger.debug('... energy response shape: {}'.format(np.shape(response)))
            aeff = np.apply_along_axis(np.inner, 3, aeff, response)
        else:
            self.logger.warning("requested to skip accounting for energy resolution")


        # Step 7: apply smearing for angular resolution
        # should at some point account for fraction of events with large offsets in zenith "banana-arcs"...
        self.logger.debug('STEP 7: angular smearing ... 1D for now')
        # psi bins given in radians

        # Add an overflow bin if none present
        if np.isfinite(psi_bins[-1]):
            psi_bins = np.concatenate((psi_bins, [np.inf]))

        psf.set_params(configuration['angular_resolution'])
        cdf = psf.CDF(np.degrees(psi_bins[:-1]))
        # set overflow bin to 1
        cdf = np.concatenate((cdf, [1]))
        self.logger.debug('angular resolution cdf: {}'.format(np.diff(cdf)))
        
        #print('cdf shape... angular res:', np.shape(cdf))
        # an empty array
        total_aeff = np.zeros_like(np.empty(aeff.shape + (psi_bins.size-1,)))
        # expand differential contributions along the opening-angle axis
        total_aeff[...,:] = aeff[..., None]*np.diff(cdf)[None, ...]

        edges = (e_nu, cos_theta, e_showering, psi_bins)
        self.logger.info('generated aeff of shape {}'.format(np.shape(total_aeff)))
        return effective_area(edges, total_aeff, 'cos_theta' if nside is None else 'healpix')
