import numbers
import os
import pickle as pickle
from functools import partial

import numpy as np
from scipy import interpolate

from . import (
    angular_resolution,
    classification_efficiency,
    effective_areas,
    multillh,
    surface_veto,
    surfaces,
)
from .util import data_dir, defer


def make_key(opts, kwargs):
    key = dict(opts.__dict__)
    key.update(kwargs)
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray):
            key[k] = (v[0], v[-1], len(v))
        else:
            key[k] = v

    return tuple(key.items())


def create_aeff(opts, **kwargs):

    cache_file = os.path.join(data_dir, "cache", "throughgoing_aeff")
    # try:
    # 	cache = pickle.load(open(cache_file))
    # except IOError:
    # 	cache = dict()
    # key = make_key(opts, kwargs)
    # if key in cache:
    # 	return cache[key]

    if opts.veto_area > 0:
        kwargs["veto_coverage"] = surface_veto.GeometricVetoCoverage(
            opts.geometry, opts.spacing, opts.veto_area
        )

    seleff_kwargs = dict()
    if opts.energy_threshold is not None:
        seleff_kwargs["energy_threshold"] = opts.energy_threshold
    if opts.psf_class is not None:
        # assume that all PSF classes have equal effective area
        seleff_kwargs["scale"] = 1.0 / opts.psf_class[1]
    if opts.efficiency_scale != 1:
        if seleff_kwargs.get("scale", 1) != 1:
            raise ValueError("psf_class and efficiency_scale are mutually exclusive")
        seleff_kwargs["scale"] = opts.efficiency_scale
    seleff = effective_areas.get_muon_selection_efficiency(
        opts.geometry, opts.spacing, **seleff_kwargs
    )

    if opts.no_cuts:

        def selection_efficiency(emu, cos_theta):
            return seleff(emu, cos_theta=0)

        # selection_efficiency = lambda emu, cos_theta: np.ones(emu.shape)
        # selection_efficiency = effective_areas.get_muon_selection_efficiency("IceCube", None)
    else:
        selection_efficiency = seleff

    for k in "psi_bins", "cos_theta":
        if k in kwargs:
            v = kwargs[k]
        elif hasattr(opts, k):
            v = getattr(opts, k)
        else:
            continue
        try:
            len(v)
            kwargs[k] = np.asarray(v)
        except TypeError:
            kwargs[k] = v

    if hasattr(opts, "psf"):
        kwargs["psf"] = getattr(opts, "psf")
    else:
        kwargs["psf"] = angular_resolution.get_angular_resolution(
            opts.geometry, opts.spacing, opts.angular_resolution_scale, opts.psf_class
        )

    # hack -- always use the standard sunflower for energy resolutions
    resolution_geometry = opts.geometry
    fiducial_geometry = opts.geometry
    if "Sunflower" in opts.geometry:
        if opts.geometry != "Sunflower":
            resolution_geometry = "Sunflower"
            print(
                (
                    "Warning! Hack! For energy resolution, overriding the requested geometry ({}) with the standard ({})".format(
                        opts.geometry, resolution_geometry
                    )
                )
            )

    neutrino_aeff = effective_areas.create_throughgoing_aeff(
        energy_resolution=effective_areas.get_energy_resolution(
            resolution_geometry, opts.spacing
        ),
        selection_efficiency=selection_efficiency,
        surface=effective_areas.get_fiducial_surface(fiducial_geometry, opts.spacing),
        **kwargs
    )

    bundle_aeff = effective_areas.create_bundle_aeff(
        energy_resolution=effective_areas.get_energy_resolution(
            resolution_geometry, opts.spacing
        ),
        selection_efficiency=selection_efficiency,
        surface=effective_areas.get_fiducial_surface(fiducial_geometry, opts.spacing),
        veto_efficiency=(
            effective_areas.StepFunction(opts.veto_threshold, 90)
            if isinstance(opts.veto_threshold, numbers.Number)
            else opts.veto_threshold
        ),
        **kwargs
    )

    # cache[key] = aeff
    # pickle.dump(cache, open(cache_file, 'w'), 2)
    return neutrino_aeff, bundle_aeff


def create_cascade_aeff(opts, **kwargs):

    # cache_file = os.path.join(data_dir, 'cache', 'cascade_aeff')
    # try:
    # 	cache = pickle.load(open(cache_file))
    # except IOError:
    # 	cache = dict()
    # key = make_key(opts, kwargs)
    # if key in cache:
    # 	return cache[key]

    for k in "psi_bins", "cos_theta":
        if k in kwargs:
            kwargs[k] = np.asarray(kwargs[k])
        elif hasattr(opts, k):
            kwargs[k] = np.asarray(getattr(opts, k))

    aeff = effective_areas.create_cascade_aeff(
        energy_resolution=effective_areas.get_energy_resolution(
            opts.geometry, opts.spacing, channel="cascade"
        ),
        selection_efficiency=effective_areas.HECascadeSelectionEfficiency(
            opts.geometry, opts.spacing, opts.cascade_energy_threshold
        ),
        surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
        # energy_threshold=effective_areas.StepFunction(opts.veto_threshold, 90),
        psf=angular_resolution.get_angular_resolution(
            opts.geometry,
            opts.spacing,
            opts.angular_resolution_scale,
            channel="cascade",
        ),
        **kwargs
    )

    # cache[key] = aeff
    # pickle.dump(cache, open(cache_file, 'w'), 2)
    return aeff


def create_starting_aeff(opts, **kwargs):

    for k in "psi_bins", "cos_theta":
        if k in kwargs:
            kwargs[k] = np.asarray(kwargs[k])
        elif hasattr(opts, k):
            kwargs[k] = np.asarray(getattr(opts, k))

    return effective_areas.create_starting_aeff(
        energy_resolution=effective_areas.get_energy_resolution(
            opts.geometry, opts.spacing, channel="cascade"
        ),
        selection_efficiency=effective_areas.HECascadeSelectionEfficiency(
            opts.geometry, opts.spacing, opts.cascade_energy_threshold
        ),
        classification_efficiency=classification_efficiency.get_classification_efficiency(
            opts.geometry, opts.spacing
        ),
        surface=effective_areas.get_fiducial_surface(opts.geometry, opts.spacing),
        psf=angular_resolution.get_angular_resolution(
            opts.geometry,
            opts.spacing,
            opts.angular_resolution_scale,
            channel="cascade",
        ),
        **kwargs
    )


class aeff_factory(object):
    """
    Create effective areas, lazily
    """

    def __init__(self):
        # arguments for creating effective areas
        self._recipes = dict()
        # cached effective areas
        self._aeffs = dict()

    def set_kwargs(self, **kwargs):
        self._aeffs.clear()
        for opts, kw in self._recipes.values():
            kw.update(**kwargs)

    def add(self, name, opts, **kwargs):
        self._recipes[name] = (opts, kwargs)
        if name in self._aeffs:
            del self._aeffs[name]

    def _create(self, opts, **kwargs):
        aeffs = {}

        if hasattr(opts, "__call__"):
            return opts(**kwargs)

        if "custom_radio" in opts:
            if opts.custom_radio == True:
                aeffs["radio_events"] = opts.aeffs
                return aeffs

        if opts.geometry in ("ARA", "Radio"):
            psi_bins = kwargs.pop("psi_bins")
            for k in "cos_theta", "neutrino_energy":
                if k in kwargs:
                    kwargs[k] = np.asarray(kwargs[k])
                elif hasattr(opts, k):
                    kwargs[k] = np.asarray(getattr(opts, k))
            kwargs["psi_bins"] = psi_bins["radio"]
            if hasattr(opts, "config_file"):
                from . import radio_aeff_generation

                radio = radio_aeff_generation.radio_aeff(
                    psi_bins=psi_bins["radio"], config=opts.config_file
                )
                aeffs["radio_events"] = (
                    radio.create(cos_theta=kwargs["cos_theta"]),
                    radio.create_muon_background_from_tabulated(
                        cos_theta=kwargs["cos_theta"]
                    ),
                )
            else:
                kwargs["nstations"] = opts.nstations
                if hasattr(opts, "veff_filename"):
                    kwargs["veff_filename"] = opts.veff_filename
                if opts.geometry == "ARA":
                    kwargs["depth"] = opts.depth
                    aeffs["radio_events"] = (
                        effective_areas.create_ara_aeff(**kwargs),
                        None,
                    )
                else:
                    aeffs["radio_events"] = (
                        effective_areas.create_radio_aeff(**kwargs),
                        None,
                    )
        elif opts.geometry == "Sunflower_240_EHE":
            psi_bins = kwargs.pop("psi_bins")
            aeffs["EHE_events"] = (
                effective_areas.create_gen2_ehe_aeff(**kwargs),
                None,
            )
        else:
            psi_bins = kwargs.pop("psi_bins")
            nu, mu = create_aeff(opts, psi_bins=psi_bins["tracks"], **kwargs)
            aeffs["shadowed_tracks"] = (nu[0], mu[0])
            aeffs["unshadowed_tracks"] = (nu[1], mu[1])
            if opts.cascade_energy_threshold is not None:
                for channel, aeff in create_starting_aeff(
                    opts, psi_bins=psi_bins["cascades"], **kwargs
                ).items():
                    aeffs[channel] = (aeff, None)
        return aeffs

    def __call__(self, name):
        if not name in self._recipes:
            raise KeyError("Unknown configuration '{}'".format(name))
        if not name in self._aeffs:
            opts, kwargs = self._recipes[name]
            self._aeffs[name] = self._create(opts, **kwargs)
        return self._aeffs[name]

    def get_kwargs(self, name):
        opts, kwargs = self._recipes[name]
        return kwargs

    def get_opts(self, name):
        opts, kwargs = self._recipes[name]
        return opts

    @classmethod
    def get(cls):
        if not hasattr(cls, "instance"):
            cls.instance = cls()
        return cls.instance


class component_bundle(object):
    def __init__(self, livetimes, component_factory, **kwargs):
        self.components = dict()
        self.livetimes = dict(livetimes)
        self.detectors = dict()
        for detector, livetime in livetimes.items():
            for channel, aeff in aeff_factory.get()(detector).items():
                key = detector + "_" + channel
                if "restricted" in kwargs:
                    emin = kwargs["restricted"][0]
                    emax = kwargs["restricted"][1]
                    truncate = False
                    if truncate:
                        aeff_c = aeff[0].truncate_energy_range(emin, emax)
                        try:
                            aeff_mu = aeff[1].truncate_energy_range(emin, emax)
                        except:
                            aeff_mu = aeff[1]
                            self.components[key] = component_factory(
                                (aeff_c, aeff_mu), **kwargs
                            )
                    else:
                        self.components[key] = component_factory(
                            aeff, emin=emin, emax=emax, **kwargs
                        )
                else:
                    self.components[key] = component_factory(aeff, **kwargs)
                self.detectors[key] = detector

    def get_component(self, key, livetimes=None):
        if livetimes is None:
            livetimes = self.livetimes
        return multillh.Combination(
            {
                k: (self.components[k][key], livetimes[self.detectors[k]])
                for k in self.components
                if key in self.components[k]
            }
        )

    def get_components(self, livetimes=None):
        keys = set()
        for v in self.components.values():
            keys.update(v.keys())
        return {key: self.get_component(key, livetimes) for key in keys}


def gen2_throughgoing_muon_efficiency_correction(energy, scale):
    x, y = [1.0, 1.5, 2.2, 3.0, 5.0], [-0.0, 1138.0, 1985.0, 2307.0, 2397.0]
    b = interpolate.interp1d(x, y, 2, fill_value="extrapolate")(scale)
    return 1 + b / energy


def gen2_throughgoing_muon_angular_resolution_correction(
    energy, scale, ssmpe=True, mdom=True
):
    """
    :param scale: cherenkov effective area per sensor, relative to PDOM
    :param sspmpe: approximate resolution improvement from Segmented SplineMPE
    :param mdom: approxmate resolution improvement from mDOM
    """

    def b(x):
        return -0.82 + 14.54 / x

    def med(emu, b):
        return 0.11 + b / np.sqrt(emu)

    scale = med(energy, b(scale)) / med(energy, b(1))
    if ssmpe:
        # see: https://github.com/fbradascio/IceCube/blob/b8556b7b3d3c53a1cfab4bf53737bebff1264707/SensitivityStudy_SSMPE_vs_MPE.ipynb
        # https://doi.org/10.1051/epjconf/201920705002
        # NB: this improvement was evaluated at 2x PDOM sensitivty
        scale *= np.minimum(
            1,
            np.polyval(
                [0.01266943, -0.1901559, 0.80568256, 0.04948373],
                np.log10(energy / 2),
            ),
        )
    if mdom:
        # see: https://events.icecube.wisc.edu/contributionDisplay.py?contribId=148&sessionId=1&confId=100
        scale *= 1 - 0.2
    return scale


def make_options(**kwargs):
    import argparse

    defaults = dict(
        geometry="Sunflower",
        spacing=240,
        veto_area=75.0,
        angular_resolution_scale=1.0,
        efficiency_scale=1.0,
        cascade_energy_threshold=None,
        veto_threshold=1e5,
        energy_threshold=0,
        no_cuts=False,
        psf_class=(0, 1),
        livetime=1.0,
    )
    # icecube has no veto...yet
    if kwargs.get("geometry") == "IceCube":
        defaults["veto_area"] = 0.0
        defaults["veto_threshold"] = np.inf
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def add_configuration(name, opts, **kwargs):
    """
    Add an effective area calculation to the cache
    """
    aeff_factory.get().add(name, opts, **kwargs)


def add_aeffs(name, aeffs):
    """
    Add a calculated effective area to the cache
    """
    add_configuration(
        name, make_options(**dict(aeffs=aeffs, geometry=name, custom_radio=True))
    )


set_kwargs = aeff_factory.get().set_kwargs


def scale_gen2_sensors(
    scale=1.0, ssmpe=True, mdom=True, with_cascades=True, veto_area=5.8
):
    """
    Approximate a Gen2 instrumented with sensors `scale` times the photon
    effective area of a PDOM
    """
    return dict(
        geometry="Sunflower",
        spacing=240,
        cascade_energy_threshold=2e5 / scale if with_cascades else None,
        veto_area=veto_area,
        veto_threshold=defer(surface_veto.UDelSurfaceVeto),
        angular_resolution_scale=partial(
            gen2_throughgoing_muon_angular_resolution_correction,
            scale=scale,
            ssmpe=ssmpe,
            mdom=mdom,
        ),
        efficiency_scale=partial(
            gen2_throughgoing_muon_efficiency_correction, scale=scale
        ),
    )


default_configs = {
    "IceCube": dict(
        geometry="IceCube",
        spacing=125,
        cascade_energy_threshold=6e4,
        veto_area=1.0,
        veto_threshold=1e5,
    ),
    "IceCube-TracksOnly": dict(
        geometry="IceCube", spacing=125, veto_area=1.0, veto_threshold=1e5
    ),
    "Fictive-Optical": dict(
        geometry="Fictive",
        cascade_energy_threshold=1e5,
        veto_area=5.0,
        veto_threshold=1e5,
    ),
    "Fictive-Optical-TracksOnly": dict(
        geometry="Fictive",
        veto_area=5.0,
        veto_threshold=1e5,
    ),
    "Gen2-InIce": scale_gen2_sensors(3.0),
    "Gen2-InIce-TracksOnly": (
        scale_gen2_sensors(3.0, with_cascades=False)
        | {"veto_threshold": defer(surface_veto.UDelSurfaceVeto)}
    ),
    "Gen2-InIce-NoVeto": scale_gen2_sensors(3.0, veto_area=0.0),
    "Gen2-InIce-TracksOnly-NoVeto": (
        scale_gen2_sensors(3.0, with_cascades=False, veto_area=0.0)
        | {"veto_threshold": defer(surface_veto.UDelSurfaceVeto)}
    ),
    "Gen2-Radio": dict(
        geometry="Radio",
        config_file="data/gen2-radio/baseline.yaml",
    ),
    "Fictive-Radio": dict(geometry="Radio", nstations=30),
    "Sunflower_240": dict(
        geometry="Sunflower",
        spacing=240,
        cascade_energy_threshold=2e5,
        veto_area=75.0,
        veto_threshold=1e5,
    ),
    "Sunflower_240_EHE": dict(
        geometry="Sunflower_240_EHE",
    ),
    "ARA_37": dict(geometry="ARA", nstations=37, depth=200),
    "ARA_200": dict(geometry="ARA", nstations=200, depth=200),
    "ARA_300": dict(geometry="ARA", nstations=300, depth=200),
    "IceCube_NoCasc": dict(
        geometry="IceCube", spacing=125, veto_area=1.0, veto_threshold=1e5
    ),
    "Sunflower_240_NoCasc": dict(
        geometry="Sunflower", spacing=240, veto_area=75.0, veto_threshold=1e5
    ),
    "KM3NeT": dict(
        geometry="IceCube",
        spacing=125,
        veto_area=0.0,
        veto_threshold=None,
        angular_resolution_scale=0.2,
    ),
    "Gen2-Phase2-Radio": dict(
        geometry="Radio",
        config_file=os.path.join(os.path.dirname(__file__), "radio_config_16.yaml"),
    ),
}


# Add midscale geometry candidates
# FIXME: add corner22, scan22 geometries
for midscale in "corner", "sparse", "hcr":
    surface = defer(
        surfaces.get_fiducial_surface, "Sunflower_" + midscale, spacing=240, padding=0
    )
    # artificially fix veto area to the footprint of the geometry
    area = defer(lambda: surface.azimuth_averaged_area(1) / 1e6)
    default_configs["Gen2-Phase2-" + midscale] = dict(
        geometry="Sunflower_" + midscale,
        spacing=240,
        veto_area=area,
        veto_threshold=1e5,
        cascade_energy_threshold=2e5,
    )
    default_configs["Gen2-Phase2-" + midscale + "-TracksOnly"] = dict(
        geometry="Sunflower_" + midscale,
        spacing=240,
        veto_area=area,
        veto_threshold=1e5,
    )


default_psi_bins = {
    "tracks": np.linspace(0, np.radians(1.5) ** 2, 150) ** (1.0 / 2),
    "cascades": np.linspace(0, np.radians(60) ** 2, 50) ** (1.0 / 2),
    "radio": np.linspace(0, np.radians(15) ** 2, 50) ** (1.0 / 2),
}

# default_cos_theta_bins = np.linspace(-1, 1, 21)
default_cos_theta_bins = [
    -1.0,
    -0.95,
    -0.85,
    -0.75,
    -0.65,
    -0.55,
    -0.45,
    -0.35,
    -0.25,
    -0.15,
    -0.05,
    0.05,
    0.15,
    0.25,
    0.35,
    0.45,
    0.55,
    0.65,
    0.75,
    0.85,
    0.95,
    1.0,
]

for k, config in default_configs.items():
    psi_bins = dict(default_psi_bins)
    psi_bins.update(config.pop("psi_bins", {}))
    kwargs = {
        "cos_theta": config.pop("cos_theta", default_cos_theta_bins),
        "psi_bins": psi_bins,
    }
    add_configuration(k, make_options(**config), **kwargs)


def get(configuration):
    return aeff_factory.instance(configuration)
