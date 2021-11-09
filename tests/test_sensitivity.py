from typing import Optional

import numpy as np
import pytest
from gen2_analysis import diffuse, factory, figures_of_merit, multillh, surface_veto
from gen2_analysis.effective_areas import effective_area
from gen2_analysis.figures import figure
from scipy import stats
from scipy.optimize import bisect


@pytest.fixture(scope="session")
def dummy_configuration():
    # FIXME: make this independent of IceCube data tables
    factory.add_configuration(
        "__dummy_config__",
        factory.make_options(
            **dict(geometry="IceCube", spacing=125, veto_area=1.0, veto_threshold=1e5),
        ),
        psi_bins={k: [0, np.pi] for k in ("cascades", "tracks", "radio")},
    )


@pytest.fixture
def components(dummy_configuration):
    def make_components(aeffs: tuple[effective_area, Optional[effective_area]]):
        aeff, muon_aeff = aeffs

        energy_threshold = np.inf
        atmo = diffuse.AtmosphericNu.conventional(
            aeff, 1, hard_veto_threshold=energy_threshold
        )
        atmo.prior = lambda v, **kwargs: -((v - 1) ** 2) / (2 * 0.1 ** 2)
        prompt = diffuse.AtmosphericNu.prompt(
            aeff, 1, hard_veto_threshold=energy_threshold
        )
        prompt.min = 0.5
        prompt.max = 3.0
        astro = diffuse.DiffuseAstro(aeff, 1)
        astro.seed = 2.0

        if muon_aeff is None:
            muon = diffuse.NullComponent(aeff)
        else:
            muon = surface_veto.MuonBundleBackground(muon_aeff, 1)

        return dict(atmo=atmo, prompt=prompt, astro=astro, muon=muon)

    bundle = factory.component_bundle({"__dummy_config__": 1.0}, make_components)
    components = bundle.get_components()
    return components


@pytest.fixture
def asimov_llh(components):
    return multillh.asimov_llh(components, astro=0)


def test_nullhypo(asimov_llh: multillh.LLHEval, snapshot):
    constrained_fit = asimov_llh.fit(astro=0)
    assert constrained_fit.pop("astro") == 0
    assert constrained_fit == pytest.approx({k: 1 for k in constrained_fit.keys()})
    assert (
        asimov_llh.llh(**{"astro": 0, **{k: 1 for k in constrained_fit.keys()}})
        == snapshot
    )


def test_likelihood_ratio(asimov_llh: multillh.LLHEval, snapshot):
    # test statistic between astro = f and astro = 0
    ts = lambda f: -2 * (
        asimov_llh.llh(**asimov_llh.fit(astro=f))
        - asimov_llh.llh(**asimov_llh.fit(astro=0))
    )
    # fit for \Delta TS = 2.705 (90% CL for 1 degree of freedom)
    critical_ts = stats.chi2(1).ppf(0.9)
    limit = bisect(lambda f: ts(f) - critical_ts, 0, 1)
    assert round(limit, 3) == snapshot


def test_component_expectations(
    components: dict[str, multillh.Combination], snapshot, round_to_scale
):
    assert (
        round_to_scale(
            {k: v.expectations() for k, v in components.items()}, 5
        )
        == snapshot
    )


def test_expectations(
    asimov_llh: multillh.LLHEval, snapshot, round_to_scale
):
    assert (
        round_to_scale(
            asimov_llh.expectations(**{k: 1 for k in asimov_llh.components.keys()}), 5
        )
        == snapshot
    )


@pytest.mark.parametrize(
    "metric,kwargs",
    [
        (figures_of_merit.TOT.ul, {}),
        (figures_of_merit.DIFF.ul, {"decades": 1}),
        (figures_of_merit.DIFF.dp, {"decades": 1}),
    ],
)
def test_gzk(dummy_configuration, metric, kwargs, snapshot):
    fom = figures_of_merit.GZK({"__dummy_config__": 15})
    assert tuple(np.round(v, 3) for v in fom.benchmark(metric, **kwargs)) == snapshot
