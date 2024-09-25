from toise.figures import figure


@figure
def performance(tabulated_psf=False):
    """
    :param tabulated_psf: also show tabulated version of the King PSF
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import bisect, fsolve

    from toise import (
        angular_resolution,
        effective_areas,
        plotting,
        surfaces,
        util,
    )

    @np.vectorize
    def median_opening_angle(psf, energy, cos_theta):
        def f(psi):
            return psf(psi, energy, cos_theta) - 0.5

        try:
            return bisect(f, 0, np.radians(5))
        except:
            return np.radians(5)

    psf = angular_resolution.SplineKingPointSpreadFunction()
    if tabulated_psf:
        psf_tab = angular_resolution.get_angular_resolution(
            "Sunflower", 240, psf_class=(0, 1)
        )
    psf_ic = angular_resolution.get_angular_resolution("IceCube")
    aeff = effective_areas.MuonEffectiveArea("Sunflower", 240)

    loge = np.arange(3.5, 8.5, 0.5) + 0.25
    loge_centers = 10 ** util.center(loge)

    fig = plt.figure(figsize=(7, 2.5))
    griddy = plt.GridSpec(1, 2)
    ax = plt.subplot(griddy[0])
    ct = np.linspace(-1, 1, 101)
    for e in 1e4, 1e5, 1e6, 1e7:
        line = ax.plot(ct, aeff(e, ct) / 1e6, label=plotting.format_energy("%d", e))[0]

    s = surfaces.get_fiducial_surface("Sunflower", 240, padding=0)
    ax.plot(ct, s.azimuth_averaged_area(ct) / 1e6, color="grey", ls=":")
    x = 0.7
    y = s.azimuth_averaged_area(x) / 1e6
    ax.annotate(
        r"$A_{\rm{ geo,IceCube \operatorname{-} Gen2}}$",
        (
            x,
            y,
        ),
        (-20, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            facecolor="k",
            shrinkA=0.1,
            connectionstyle="angle3,angleA=0,angleB=90",
            relpos=(1, 0.5),
        ),
    )

    s = surfaces.get_fiducial_surface("IceCube", padding=0)
    ax.plot(ct, s.azimuth_averaged_area(ct) / 1e6, color="grey", ls="--")
    x = ct[np.argmax(s.azimuth_averaged_area(ct))]
    y = s.azimuth_averaged_area(x) / 1e6
    ax.annotate(
        r"$A_{\rm geo, IceCube}$",
        (
            x,
            y,
        ),
        (20, -20),
        textcoords="offset points",
        arrowprops=dict(
            arrowstyle="-|>",
            facecolor="k",
            connectionstyle="angle3,angleA=0,angleB=90",
            relpos=(0, 0.5),
        ),
    )

    ax.set_ylabel(r"Muon $A_{\rm eff}$ [km$^2$]")
    ax.set_xlabel(r"$\cos\theta_{\rm zenith}$")
    ax.set_ylim((0, 8))

    ax = plt.subplot(griddy[1])
    ctfine = np.linspace(-1, 1, 101)
    for e in 1e4, 1e5, 1e6, 1e7:
        line = ax.plot(
            ctfine,
            np.degrees(psf.get_quantile(0.5, e, ctfine)),
            label=plotting.format_energy("%d", e),
        )[0]
        if tabulated_psf:
            ax.plot(
                ctfine,
                np.degrees(psf_tab.get_quantile(0.5, e, ctfine)),
                color=line.get_color(),
                ls=":",
            )
    icmed = np.degrees(median_opening_angle(psf_ic, 1e4, ctfine))
    ax.plot([-0.4, 0.4], [np.mean(icmed)] * 2, ls="--", color="grey")
    ax.annotate(
        "IceCube 10 TeV",
        (0, np.mean(icmed)),
        (0, -3),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize="x-small",
    )

    ax.set_ylabel(r"$\Psi_{\rm median}$ [degrees]")
    ax.set_xlabel(r"$\cos\theta_{\rm zenith}$")
    ax.set_ylim((0, 1.0))
    legend = ax.legend(
        loc="upper center",
        title="Muon energy at detector border",
        ncol=2,
        fontsize="small",
        handlelength=1,
    )
    legend.get_title().set_fontsize("small")

    for ax in fig.axes:
        ax.set_xlim(-1, 1)
    plt.tight_layout(pad=0.1, w_pad=0.5)
    return fig
