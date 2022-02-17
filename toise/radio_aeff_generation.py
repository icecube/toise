import os
from copy import copy
import json

import logging
import numpy as np
import pandas as pd
from scipy import interpolate

from toise.externals import nuFATE

from .radio_response import radio_analysis_efficiency
from .radio_response import RadioPointSpreadFunction
from .radio_response import RadioEnergyResolution
from .effective_areas import calculate_cascade_production_density
from .effective_areas import effective_area
from .radio_muon_background import get_muon_distribution
from .radio_muon_background import get_tabulated_muon_distribution

from .pointsource import is_zenith_weight
from .util import *

logger = logging.getLogger("toise aeff calculation for radio")


def _load_rno_veff(
    filename=data_dir
    + "/aeff/run_input_500km2_01_surface_4LPDA_1dipole_RNOG_1.50km_config_Alv2009_nonoise_100ns_D01surface_4LPDA_1dipole_250MHz_dipoles_RNOG_200m_3.00km_D02single_dipole_250MHz_e.json",
    trigger="dipole_2.5sigma",
):
    """
    Loads the effective volume for an RNO-G simulations
    :param filename: name of the json export of the Veff calculated using NuRadioMC
    :param trigger: name of the trigger to be used

    :returns: a tuple (edges, veff). veff has units of m^3
    """

    if not filename.startswith("/"):
        filename = os.path.join(data_dir, "aeff", filename)
    with open(filename) as jsonfile:
        dats = json.load(jsonfile)
    index = []
    arrays = {"veff": []}
    for zenith, values in list(dats.items()):
        for selection, items in list(values.items()):
            if selection == trigger:
                # print(selection, items)
                for energy, veff in zip(items["energies"], items["Veff"]):
                    index.append((selection, energy, np.cos(float(zenith))))
                    arrays["veff"].append(veff)
    veff = pd.DataFrame(
        arrays,
        index=pd.MultiIndex.from_tuples(
            index, names=["selection", "energy", "cos_zenith"]
        ),
    )
    # print(veff, veff.veff)
    veff.sort_index(level=[0, 1, 2], inplace=True)
    # add right-hand bin edges
    energy = veff.index.levels[1].values / 1e9
    energy = np.concatenate([energy, [energy[-1] ** 2 / energy[-2]]])
    # left-hand bin edges were specified in zenith, so add in reverse
    cos_zenith = veff.index.levels[2].values
    cos_zenith = np.concatenate(([2 * cos_zenith[0] - cos_zenith[1]], cos_zenith))
    omega = 2 * np.pi * np.diff(cos_zenith)
    logger.info("Veff is being normalised to per sr")
    print(
        veff["veff"]
        .unstack(level=-1)
        .values.reshape((energy.size - 1, cos_zenith.size - 1))
    )
    return (energy, cos_zenith), veff["veff"].unstack(level=-1).values.reshape(
        (energy.size - 1, cos_zenith.size - 1)
    ) / omega[None, :]


def _load_radio_veff_json(
    filename=data_dir + "/aeff/fictive_radio_dict_e.json", trigger=None
):
    """
    Loads the effective volume for an exported via NuRadioMC

    :param filename: name of the json export of the Veff calculated using NuRadioMC
    :param trigger: name of the trigger to be used
    :returns: a tuple (edges, veff). veff has units of m^3
    """

    if not os.path.isfile(filename):
        # look in the data directory if provided file does not exist
        filename = os.path.join(data_dir, "aeff", filename)

    # read filename and convert to pandas
    with open(filename, "r") as data:
        jsonfile = json.load(data)
    dataframe = pd.DataFrame(jsonfile)
    # sort input data by energy and cos(zenith)
    dataframe.sort_values(
        by=["energy", "thetamin"], ascending=[True, False], inplace=True
    )
    dataframe.reset_index(inplace=True)

    def _list_of_triggers(df):
        triggerlist = [list(df.veff[i]) for i in range(len(df))]
        triggerlist_flat = np.concatenate(triggerlist).ravel().tolist()
        triggerlist_set = list(set(triggerlist_flat))
        logger.debug(f"found triggers: {triggerlist_set}")
        return triggerlist_set

    if trigger is None:
        # simply take the first
        found_triggers = _list_of_triggers(dataframe)
        trigger = found_triggers[0]
        logger.warning(
            f"No trigger name requested, simply picking the first ('{trigger}') from available {found_triggers}."
        )

    logger.info(f"Using trigger: {trigger}")

    def _extract_veff(df, triggername):
        veff = np.array([df.veff[i][triggername][0] for i in range(len(df))])
        return veff

    dataframe["veff_values"] = _extract_veff(dataframe, trigger)
    dataframe["cos_theta"] = (
        np.cos(dataframe.thetamin) + np.cos(dataframe.thetamax)
    ) / 2

    # get bin edges in energy and cos(theta)
    def _energy_bin_edges(energy):
        bin_centers = list(set(list(energy)))
        bin_centers = list(np.sort(bin_centers))
        # padding left right
        bin_centers.append(bin_centers[-1] ** 2 / bin_centers[-2])
        bin_centers.append(bin_centers[0] ** 2 / bin_centers[1])
        bin_centers = list(np.sort(bin_centers))
        # calculate centers
        logedges = [
            0.5 * (np.log10(bin_centers[i]) + np.log10(bin_centers[i + 1]))
            for i in range(len(bin_centers) - 1)
        ]
        edges = 10 ** np.array(logedges)
        return edges

    bin_edges_energy_gev = _energy_bin_edges(dataframe.energy) / 1e9

    bin_edges_costheta = list(
        set(
            list(np.round(np.cos(dataframe.thetamin), 5))
            + list(np.round(np.cos(dataframe.thetamax), 5))
        )
    )
    bin_edges_costheta = np.sort(bin_edges_costheta)

    # keep only necessary columns
    veff = dataframe.filter(["energy", "cos_theta", "domega", "veff_values"], axis=1)
    veff.sort_index(level=[0, 1, 2, 3], inplace=True)

    veff_values = np.array(veff.veff_values)
    return (bin_edges_energy_gev, bin_edges_costheta), veff_values.reshape(
        bin_edges_energy_gev.size - 1, bin_edges_costheta.size - 1
    )


def _interpolate_radio_veff(
    energy_edges, ct_edges=None, filename="json.file", trigger=None
):
    """
            Loads a NuRadioMC effective volume and interpolates for the requested energy / cos theta binning

        :param energy_edges: final energy binning to interpolate Veff to
        :param ct_edges: final cos theta binning to interpolate Veff to, if None, keep original binning
        :param filename: name of the json export of the Veff calculated using NuRadioMC
        :param trigger: name of the trigger to be used

    :returns: a tuple (edges, veff). veff has units of m^3
    """
    logger.debug("interpolating effective area")
    edges, veff = _load_radio_veff_json(filename, trigger)
    logger.debug(f"Veff shape before interpolation: {np.shape(veff)}")

    def interp_masked(arr, x, xp):
        # NB: occasionally there may be NaN effective volumes. interpolate through them
        valid = ~np.ma.masked_invalid(arr).mask

        if np.sum(valid) == 0:
            # a safeguard against not having any valid Veffs in the slice
            return -10 * np.ones_like(x)
        else:
            interpolator = interpolate.interp1d(
                xp, arr, bounds_error=False, fill_value=-10
            )
            interpolation_result = interpolator(x)
            return interpolation_result

    # interpolate in log-log space, where the curve is ~smooth and slowly varying
    veff = 10 ** (
        np.apply_along_axis(
            interp_masked,
            0,
            np.log10(np.clip(veff, 1e-10, np.inf)),
            center(np.log10(energy_edges)),
            center(np.log10(edges[0])),
        )
    )
    logger.debug(f"Veff shape after interpolation: {np.shape(veff)}")

    if ct_edges is None:
        # do return original cos theta binning
        return (energy_edges, edges[1]), veff
    else:
        # use 'nearest' neighbour binning to return finer Veff ins cos theta
        interp = interpolate.interp1d(
            center(edges[1]), veff, "nearest", axis=1, bounds_error=False, fill_value=0
        )
        center_ct = np.clip(
            center(ct_edges), min(center(edges[1])) + 1e-3, max(center(edges[1])) - 1e-3
        )

        return (energy_edges, ct_edges), interp(center_ct)

class radio_aeff:
    """ convenience class to generate neutrino aeffs and background aeffs from config file settings """
    def __init__(self, config="radio_config.yaml", psi_bins=None):
        if not os.path.isfile(config):
            config = os.path.realpath(os.path.join(os.path.dirname(__file__), config))
        self.configfile = config

        # set parameters provided in config file
        self.restore_config()
        self.logger = logger
        self.logger.info(self.configuration)
        self.set_psi_bins(psi_bins=psi_bins)

    def switch_energy_resolution(self, on=True):
        """ (De)activate energy smearing """
        if type(on) == bool:
            self.configuration["apply_energy_resolution"] = on
        else:
            self.logger.warning("no boolean provided for switch")

    def switch_analysis_efficiency(self, on=True):
        """ (De)activate reduction to Aeff due to analysis efficiency """
        if type(on) == bool:
            self.configuration["apply_analysis_efficiency"] = on
        else:
            self.logger.warning("no boolean provided for switch")

    def restore_config(self):
        """ Revert all changed settings back to what is in the provided config file """
        import yaml

        config = self.configfile
        with open(config) as file:
            self.configuration = yaml.full_load(file)

    def scale_veff(self, factor):
        """ Apply up/down scaling by a factor *factor* """
        if not "nstations" in self.configuration["detector_setup"]:
            self.configuration["detector_setup"]["nstations"] = 1.0
        self.configuration["detector_setup"]["nstations"] *= factor

    def get_parameter(self, group, name):
        return self.configuration[group][name]

    def set_parameter(self, group, name, value):
        self.logger.debug(f"Old value for parameter {group}/{name}: {self.get_parameter(group, name)}")
        self.configuration[group][name] = value
        self.logger.info(f"New value for parameter {group}/{name}: {self.get_parameter(group, name)}")

    def set_psi_bins(self, psi_bins):
        self.psi_bins = psi_bins

    def create_muon_background(
        self,
        energy_resolution=RadioEnergyResolution(),
        cos_theta=np.linspace(-1, 1, 21),
        neutrino_energy=np.logspace(6, 12, 61),
    ):
        from .radio_muon_background import get_muon_distribution

        psi_bins = self.psi_bins
        configuration = self.configuration

        if "nstations" in configuration["detector_setup"]:
            nstations = configuration["detector_setup"]["nstations"]
            self.logger.info("using number of stations from config")
        else:
            nstations = 1
            self.logger.warning(
                "Number of stations not passed in config. Using single station proxy."
            )
        veff_scale = nstations  # simulation was done for ~100 stations

        aeff = get_muon_distribution(cos_theta, neutrino_energy) * veff_scale
        edges = (neutrino_energy, cos_theta, neutrino_energy)
        # print(cos_theta)
        self.logger.warning(
            "Direction resolution smearing not applied for atm. muons for now!"
        )

        # apply smearing for shower energy resolution
        configuration = self.configuration
        if configuration["apply_energy_resolution"] == True:
            self.logger.info("applying energy resolution smearing")
            energy_resolution.set_params(configuration["energy_resolution"])
            response = energy_resolution.get_response_matrix(
                neutrino_energy, neutrino_energy
            )
            self.logger.debug(
                "... energy response shape: {}".format(np.shape(response))
            )
            aeff = np.apply_along_axis(np.inner, 2, aeff, response)
        else:
            self.logger.warning("requested to skip accounting for energy resolution")

        return effective_area(edges, aeff, "cos_theta")

    def create_muon_background_from_tabulated(
        self,
        energy_resolution=RadioEnergyResolution(),
        cos_theta=np.linspace(-1, 1, 21),
        neutrino_energy=np.logspace(6, 12, 61),
    ):
        from .radio_muon_background import get_tabulated_muon_distribution

        psi_bins = self.psi_bins
        configuration = self.configuration

        if ("nstations" in configuration["detector_setup"]) and ("simulated_stations" in configuration["effective_volume"]):
            veff_scale = configuration["detector_setup"]["nstations"]/configuration["effective_volume"]["simulated_stations"]
            self.logger.info(f"Using number of stations from config. Rescaling atm. muon count by {veff_scale}.")
        else:
            veff_scale = 1
            self.logger.warning(
                "Number of stations not passed in config. Will not rescale muon background."
            )

        cr_cut = False
        if "cr_cut" in configuration["muon_background"]:
            cr_cut = configuration["muon_background"]["cr_cut"]
        (e_cr_shower, cos_t, e_shower), muon_distro = get_tabulated_muon_distribution(configuration["muon_background"]["table"], cr_cut)#cos_theta, neutrino_energy)
        aeff = muon_distro * veff_scale
        edges = (neutrino_energy, cos_theta, neutrino_energy)
        # print(cos_theta)
        self.logger.warning(
            "Direction resolution smearing not applied for atm. muons for now!"
        )

        # apply smearing for shower energy resolution
        configuration = self.configuration
        if configuration["apply_energy_resolution"] == True:
            self.logger.info("applying energy resolution smearing")
            energy_resolution.set_params(configuration["energy_resolution"])
            response = energy_resolution.get_response_matrix(
                neutrino_energy, neutrino_energy
            )
            self.logger.debug(
                "... energy response shape: {}".format(np.shape(response))
            )
            aeff = np.apply_along_axis(np.inner, 2, aeff, response)
        else:
            self.logger.warning("requested to skip accounting for energy resolution")

        return effective_area(edges, aeff, "cos_theta", source="atm_muon")

    def create(
        self,
        energy_resolution=RadioEnergyResolution(),
        psf=RadioPointSpreadFunction(),
        cos_theta=np.linspace(-1, 1, 21),
        neutrino_energy=np.logspace(6, 12, 61),
    ):

        """
        Create an effective area for radio
        """
        psi_bins = self.psi_bins
        configuration = self.configuration        

        nside = None
        if isinstance(cos_theta, int):
            nside = cos_theta

        # Step 1: Density of final states per metre
        self.logger.debug("STEP 1: Density of final states per metre")

        use_default_transfer = True
        if "transfer_matrix" in configuration:
            # use a transfer matrix taking account for the inelasticity distribution of triggered events
            self.logger.warning("using transfer matrices from external file")
            def neutrino_interaction_length_ice(flavor, energy_edges):
                ice_density=.917 * 1e-3 / (1e-2)**3 # kg/m**3
                from scipy import constants
                m_n = constants.m_p  # in kg  * units.kg  # nucleon mass, assuming proton mass
                cc = nuFATE.NeutrinoCascade(10**(0.5*(np.log10(energy_edges[:-1]) + np.log10(energy_edges[1:]))))
                L_int = m_n / (cc.total_cross_section(0) * 1e-4) / ice_density    
                return L_int

            matrix_data = np.load(configuration["transfer_matrix"]["table"], allow_pickle=True)
            if (not np.allclose(neutrino_energy,  matrix_data["bin_edges"][0]*1e-9)) or (not np.allclose(cos_theta,  matrix_data["bin_edges"][1])) or (not np.allclose(neutrino_energy,  matrix_data["bin_edges"][2]*1e-9)):
                self.logger.error("shapes of requested veff and transfer matrices do not match. Will fall back to default transfer matrix!")
                print("got:\n", matrix_data["bin_edges"], "\nexpected:\n", neutrino_energy, cos_theta)
                use_default_transfer = True

            else:
                aeffs_flavor = []
                for fi, flavor in enumerate(["e", "e", "mu", "mu", "tau", "tau"]):
                    data = matrix_data[f"transfer_matrix_{flavor}"]
                    prod_dens = 1./neutrino_interaction_length_ice(fi, neutrino_energy)
                    data[:,:,:] *= prod_dens[:,np.newaxis,np.newaxis]
                    aeffs_flavor.append(data)
                aeff = np.array(aeffs_flavor)
                e_nu = neutrino_energy
                e_showering = neutrino_energy
                use_default_transfer = False

        if use_default_transfer:
            # use the default
            (e_nu, cos_theta, e_showering), aeff = calculate_cascade_production_density(
                cos_theta, neutrino_energy
            )
            self.logger.warning("using default transfer matrices designed for optical as proxy. Using downgoing region also for upgoing to avoid double accounting for Earth absorption")
            # quick fix to ignore Earth absorption... this is already included in the effective volumes
            for i in range((len(cos_theta) - 1) // 2):
                aeff[:, :, i, :] = aeff[:, :, -i - 1, :]

        self.logger.debug("Shape of aeff tuple: {}".format(np.shape(aeff)))

        # Step 2: Trigger effective volume in terms of neutrino energy
        self.logger.debug(
            "... Trigger effective volume per station in terms of neutrino energy"
        )

        veff_filename = configuration["effective_volume"]
        if not "trigger_name" in veff_filename:
            veff_filename["trigger_name"] = None
        edges_e, veff_e = _interpolate_radio_veff(
            e_showering,
            cos_theta,
            filename=veff_filename["e"],
            trigger=veff_filename["trigger_name"],
        )
        edges_mu, veff_mu = _interpolate_radio_veff(
            e_showering,
            cos_theta,
            filename=veff_filename["mu"],
            trigger=veff_filename["trigger_name"],
        )
        edges_tau, veff_tau = _interpolate_radio_veff(
            e_showering,
            cos_theta,
            filename=veff_filename["tau"],
            trigger=veff_filename["trigger_name"],
        )

        if "simulated_stations" not in veff_filename:
            veff_filename["simulated_stations"] = 1
            self.logger.info("will not rescale effective volumes to per-station")

        # Step 3: Scale trigger effective volume to desired number of stations
        self.logger.debug(
            "Scaling trigger effective volume for number of stations and applying to density of final states"
        )
        if "nstations" in configuration["detector_setup"]:
            nstations = configuration["detector_setup"]["nstations"]
            self.logger.info("using number of stations from config")
        else:
            nstations = 1
            self.logger.warning(
                "Number of stations not passed in config. Using single station proxy."
            )
        veff_scale = nstations / float(veff_filename["simulated_stations"])
        self.logger.debug("Veff scaling factor:{}".format(veff_scale))

        ### applying triggered veff in terms of cosz / e_shower
        ###aeff[0:2,...] *= (veff_e.T)[None,None,...]*veff_scale # electron neutrino
        ###aeff[2:4,...] *= (veff_mu.T)[None,None,...]*veff_scale # muon neutrino
        ###aeff[4:6,...] *= (veff_tau.T)[None,None,...]*veff_scale # tau neutrino

        ### applying triggered veff in terms of e_neutrino / cosz
        aeff[0:2, ...] *= (veff_e)[None, ..., None] * veff_scale  # electron neutrino
        aeff[2:4, ...] *= (veff_mu)[None, ..., None] * veff_scale  # muon neutrino
        aeff[4:6, ...] *= (veff_tau)[None, ..., None] * veff_scale  # tau neutrino


        # Step 4: Scale down effective volume to account for analysis efficiency
        if configuration["apply_analysis_efficiency"] == True:
            self.logger.info(
                "applying analysis efficiency as function of neutrino energy"
            )

            ana_efficiency = radio_analysis_efficiency(
                e_nu[:-1],
                configuration["analysis_efficiency"]["minval"],
                configuration["analysis_efficiency"]["maxval"],
                configuration["analysis_efficiency"]["log_turnon_gev"],
                configuration["analysis_efficiency"]["log_turnon_width"],
            )
            aeff = np.apply_along_axis(np.multiply, 3, aeff, ana_efficiency) #3 if shower E, 1 if neutrino E
        else:
            self.logger.warning("requested to skip accounting for analysis efficiency")

        # Step 5: apply smearing for energy resolution
        if configuration["apply_energy_resolution"] == True:
            self.logger.info("applying energy resolution smearing")
            energy_resolution.set_params(configuration["energy_resolution"])
            response = energy_resolution.get_response_matrix(e_showering, e_showering)
            self.logger.debug(
                "... energy response shape: {}".format(np.shape(response))
            )
            aeff = np.apply_along_axis(np.inner, 3, aeff, response)
        else:
            self.logger.warning("requested to skip accounting for energy resolution")

        # Step 6: apply smearing for angular resolution
        # should at some point account for fraction of events with large offsets in zenith "banana-arcs"...
        self.logger.debug("applying angular smearing")
        # psi bins given in radians

        # Add an overflow bin if none present
        if np.isfinite(psi_bins[-1]):
            psi_bins = np.concatenate((psi_bins, [np.inf]))

        # check if angular resolution is array or integers
        # if it is not an array, simply apply it on the last dimension
        one_dimensional = True
        # check if any of the keys is higher dimension
        for key in configuration["angular_resolution"]:
            if not np.isscalar(configuration["angular_resolution"][key]):
                psf_shape = np.shape(configuration["angular_resolution"][key])
                one_dimensional = False
        if one_dimensional:
            psf.set_params(configuration["angular_resolution"])
            cdf = psf.CDF(np.degrees(psi_bins[:-1]))
            # set overflow bin to 1
            cdf = np.concatenate((cdf, [1]))
            self.logger.debug("angular resolution cdf: {}".format(np.diff(cdf)))

            # an empty array
            total_aeff = np.zeros_like(np.empty(aeff.shape + (psi_bins.size - 1,)))
            # expand differential contributions along the opening-angle axis
            #print(np.shape(np.diff(cdf)))
            total_aeff[..., :] = aeff[..., None] * np.diff(cdf)[None, ...]
        else:
            for key in configuration["angular_resolution"]:
                if np.isscalar(configuration["angular_resolution"][key]):
                    configuration["angular_resolution"][key] = np.full(psf_shape, configuration["angular_resolution"][key])
            psf_params = pd.DataFrame(configuration["angular_resolution"]).to_dict(orient="records")
            psf_array = []
            for psf_param in psf_params:
                psf.set_params(psf_param)
                cdf = psf.CDF(np.degrees(psi_bins[:-1]))
                cdf = np.concatenate((cdf, [1]))
                psf_array.append(np.diff(cdf))
            psf_array = np.array(psf_array)
            # an empty array
            total_aeff = np.zeros_like(np.empty(aeff.shape + (psi_bins.size - 1,)))
            # expand differential contributions along the opening-angle axis
            total_aeff[..., :] = aeff[..., None] * psf_array[None, ...]

        edges = (e_nu, cos_theta, e_showering, psi_bins)
        self.logger.info("generated aeff of shape {}".format(np.shape(total_aeff)))
        return effective_area(
            edges, total_aeff, "cos_theta" if nside is None else "healpix"
        )


def combine_aeffs(
    aeff1,
    aeff2,
    overlap_E=10 ** np.array([7, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11.0]),
    overlap_values=np.array(
        [0.004, 0.007, 0.019, 0.074, 0.153, 0.245, 0.323, 0.368, 0.393]
    ),
):
    """
    Combine two effective area tuples while removing the overlap of events seen in both
    (as done in the Feb. review array sims, where deep + shallow arrays had been simulated separately)
  
    :param aeff1: first aeff tuple
    :param aeff2: second aeff tuple
    :param overlap_E: energies for overlap to be subtracted
    :param overlap_values: values for overlap to be subtracted

    :returns: combined aeff
    """
    if not aeff1.compatible_with(aeff2):
        logger.error("provided incompatible aeffs to combine function")
    else:
        aeff = copy.deepcopy(aeff1)

        def overlap(E, logE, ovl):
            interpolator = interpolate.interp1d(loge, ovl, fill_value="extrapolate")
            interpolation_result = np.maximum(interpolator(np.log10(E)), 0)
            return (interpolation_result + 1.0) ** -1

        logger.info("Subtracting overlap in aeff combination.")
        logger.info(f"E={aeff.get_bin_centers('true_energy')}")
        logger.info(f"overlap={overlap(aeff.get_bin_centers('true_energy'))}")

        # apply to neutrino energy axis
        aeff.values = (aeff1.values + aeff2.values) * overlap(
            aeff.get_bin_centers("true_energy"), np.log10(overlap_E), overlap_values
        )[None, :, None, None, None]
        return aeff


class MuonBackground(object):
    """add the muon background assuming the Aeff is in reality just the number of events per zenith/energy bin"""

    def __init__(self, effective_area, livetime=1.0):
        self._aeff = effective_area

        emu, cos_theta = effective_area.bin_edges[:2]
        # FIXME: account for healpix binning
        self._solid_angle = 2 * np.pi * np.diff(self._aeff.bin_edges[1])


        total = (self._aeff.values).sum(axis=0) * (
            livetime
        )
        self._livetime = livetime

        # up to now we've assumed that everything is azimuthally symmetric and
        # dealt with zenith bins/healpix rings. repeat the values in each ring
        # to broadcast onto a full healpix map.
        if effective_area.is_healpix:
            total = total.repeat(effective_area.ring_repeat_pattern, axis=0)

        self.seed = 1.0
        self.uncertainty = None

        self.expectations = total

        self.bin_edges = effective_area.bin_edges

    def point_source_background(
        self, zenith_index, psi_bins, livetime=None, n_sources=None, with_energy=True
    ):
        """
        Convert flux to a form suitable for calculating point source backgrounds.
        The predictions in **expectations** will be differential in the opening-angle
        bins `psi` instead of being integrated over them.

        :param zenith_index: index of the sky bin to use. May be either an integer
                             (for single point source searches) or a slice (for
                             stacking searches)
        :param livetime: if not None, the actual livetime to integrate over in seconds
        :param n_sources: number of search windows in each zenith band
        :param with_energy: if False, integrate over reconstructed energy. Otherwise,
                            provide a differential prediction in reconstructed energy.
        """
        assert (
            not self._aeff.is_healpix
        ), "Don't know how to make PS backgrounds from HEALpix maps yet"

        background = copy(self)
        bin_areas = (np.pi * np.diff(psi_bins ** 2))[None, ...]
        # observation time shorter for triggered transient searches
        if livetime is not None:
            bin_areas *= livetime / self._livetime / constants.annum
        if is_zenith_weight(zenith_index, self._aeff):
            omega = self._solid_angle[:, None]
        elif isinstance(zenith_index, slice):
            omega = self._solid_angle[zenith_index, None]
            bin_areas = bin_areas[None, ...]
        else:
            omega = self._solid_angle[zenith_index]
        # scale the area in each bin by the number of search windows
        if n_sources is not None:
            expand = [None] * bin_areas.ndim
            expand[0] = slice(None)
            bin_areas = bin_areas * n_sources[expand]

        # dimensions of the keys in expectations are now energy, radial bin
        if is_zenith_weight(zenith_index, self._aeff):
            background.expectations = (
                np.nansum(
                    (self.expectations * zenith_index[:, None]) / omega, axis=0
                )[..., None]
                * bin_areas
            )
        else:
            background.expectations = (self.expectations[zenith_index, :] / omega)[
                ..., None
            ] * bin_areas
        if not with_energy:
            # just radial bins
            background.expectations = background.expectations.sum(axis=0)
        return background
