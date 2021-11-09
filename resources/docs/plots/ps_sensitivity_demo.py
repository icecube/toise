import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
from icecube.gen2_analysis import (
    plotting,
    effective_areas,
    diffuse,
    pointsource,
    multillh,
)
from icecube.gen2_analysis import angular_resolution, energy_resolution


def create_aeff(
    geometry="IceCube",
    spacing=200,
    veto_threshold=1e4,
    veto_area=0.0,
    energy_threshold=None,
    no_cuts=False,
    psf_spacing=None,
    angular_resolution_scale=1.0,
    **kwargs
):

    if psf_spacing is None:
        psf_spacing = spacing

    if veto_area > 0:
        kwargs["veto_coverage"] = surface_veto.GeometricVetoCoverage(
            geometry, spacing, veto_area
        )

    if energy_threshold is not None:
        seleff = effective_areas.get_muon_selection_efficiency(
            geometry, spacing, energy_threshold=energy_threshold
        )
    else:
        seleff = effective_areas.get_muon_selection_efficiency(geometry, spacing)
    if no_cuts:
        selection_efficiency = lambda emu, cos_theta: seleff(emu, cos_theta=0)
        # selection_efficiency = lambda emu, cos_theta: np.ones(emu.shape)
        # selection_efficiency = effective_areas.get_muon_selection_efficiency("IceCube", None)
    else:
        selection_efficiency = seleff

    return effective_areas.create_throughgoing_aeff(
        energy_resolution=energy_resolution.get_energy_resolution(geometry, spacing),
        selection_efficiency=selection_efficiency,
        surface=effective_areas.get_fiducial_surface(geometry, spacing),
        energy_threshold=effective_areas.StepFunction(veto_threshold, 90),
        psf=angular_resolution.get_angular_resolution(
            geometry, psf_spacing, scale=angular_resolution_scale
        ),
        psi_bins=np.sqrt(np.linspace(0, np.radians(2) ** 2, 100)),
        # psi_bins=(np.linspace(0, np.radians(2), 800)),
        **kwargs
    )


def ps_components(aeff, livetime=1.0, sindec=0):
    # find the index of the appropriate zenith band
    zi = aeff.bin_edges[1].searchsorted(-sindec) - 1

    # create background components
    atmo = diffuse.AtmosphericNu.conventional(aeff, livetime).point_source_background(
        zi
    )
    astro = diffuse.DiffuseAstro(aeff, livetime).point_source_background(zi)
    astro.seed = 2
    gamma = multillh.NuisanceParam(-2.3, 0.5, min=-2.7, max=-1.7)

    diffuse_components = dict(atmo=atmo, astro=astro, gamma=gamma)
    # fix background estimate for simplicity & speed
    fixed = dict(atmo=1, astro=2, gamma=-2.3)

    ps = pointsource.SteadyPointSource(aeff, livetime, zenith_bin=zi)
    return ps, diffuse_components, fixed


def demo_plot(
    geo="Sunflower", spacing=240, sindec=0.1, livetime=1.0, angular_resolution_scale=1.0
):
    from matplotlib import cm

    aeff = create_aeff(geo, spacing, angular_resolution_scale=angular_resolution_scale)
    ps, background, fixed = ps_components(aeff, livetime=livetime, sindec=sindec)

    dp = pointsource.discovery_potential(ps, background, **fixed)

    with plotting.pretty():
        fig = plt.figure(figsize=(8, 4))
        griddy = GridSpec(1, 2, width_ratios=(3, 4))

        ax = plt.subplot(griddy[0])
        psi = np.degrees(aeff.bin_edges[-1][:-1])
        bin_area = np.pi * np.diff(psi ** 2)[0]
        ax.plot(
            *plotting.stepped_path(psi, dp * ps.expectations()["tracks"].sum(axis=0)),
            label=r"$%.2f \times 10^{-12} \rm \,\, TeV \, cm^{-2} \, s^{-1}$" % dp
        )
        ax.plot(
            *plotting.stepped_path(
                psi, background["atmo"].expectations["tracks"].sum(axis=0)
            ),
            label="Atmos. background"
        )
        ax.plot(
            *plotting.stepped_path(
                psi,
                background["astro"]
                .expectations(gamma=fixed["gamma"])["tracks"]
                .sum(axis=0),
            ),
            label="Astro. background"
        )

        bkg = background["atmo"].expectations["tracks"].sum(axis=0) + background[
            "astro"
        ].expectations(gamma=fixed["gamma"])["tracks"].sum(axis=0)
        sig = dp * ps.expectations()["tracks"].sum(axis=0)
        ax.plot(*plotting.stepped_path(psi, bkg + sig), color="grey")

        ax.set_xlabel("Angular distance to source $\Psi$ [deg]")
        ax.set_ylabel("Mean events/livetime (%.1f bins/sqdeg)" % (1.0 / bin_area))
        ax.legend(title=r"Rates at $\sin\delta=%.1f$" % sindec)
        ax.set_title("%.0f years of %s %sm" % (livetime, geo, spacing))

        ax = plt.subplot(griddy[1])

        components = dict(background)
        components["ps"] = ps
        allh = multillh.asimov_llh(components, ps=dp, **fixed)

        ts = 2 * (
            allh.llh_contributions(ps=dp, **fixed)["tracks"]
            - allh.llh_contributions(ps=0, **fixed)["tracks"]
        )
        mappy = ax.pcolor(psi, aeff.bin_edges[2], ts, cmap=cm.get_cmap("viridis"))
        ax.semilogy()
        ax.set_ylabel("Reconstructed energy [MuEx GeV]")
        ax.set_xlabel("Angular distance to source $\Psi$ [deg]")

        ax.set_xlim((0, 1))
        ax.set_ylim((1e2, 1e6))
        ax.grid(color="w")
        plt.colorbar(mappy).set_label("Test statistic contribution")

        plt.tight_layout()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "--geometry",
        choices=("IceCube", "Sunflower", "EdgeWeighted"),
        default="Sunflower",
    )
    parser.add_argument("--spacing", type=int, default=240)
    parser.add_argument("--veto-area", type=float, default=25)
    parser.add_argument("--veto-threshold", type=float, default=1e4)
    parser.add_argument("--no-cuts", default=False, action="store_true")
    parser.add_argument("--livetime", type=float, default=10.0)
    parser.add_argument("--energy-threshold", type=float, default=None)
    parser.add_argument("--sin-dec", type=float, default=0.0)
    parser.add_argument("--angular-resolution-scale", type=float, default=1.0)
    opts = parser.parse_args()

    import pylab

    demo_plot(
        geo=opts.geometry,
        spacing=opts.spacing,
        sindec=opts.sin_dec,
        livetime=opts.livetime,
        angular_resolution_scale=opts.angular_resolution_scale,
    )
    pylab.show()
