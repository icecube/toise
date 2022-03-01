import numpy as n
import pylab as p


def xerr(bins, log=False):
    binc = get_binc(bins, log)
    l = binc - bins[:-1]
    h = bins[1:] - binc
    return (l, h)


def create_figure(format=None):
    figsize = (5.5, 3.5) if format == "wide" else (4.5, 3.5)
    fig = p.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r"$E\,\,[\mathrm{GeV}]$")
    ax.set_ylabel(
        r"$E^{2}\times\Phi\,\,[\mathrm{GeV}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{cm}^{-2}]$"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")

    return fig, ax


def get_auger():
    # Values from EPJC 2021 https://arxiv.org/pdf/2109.13400.pdf
    e = n.logspace(8.05, 11.15, 32)
    f = 10 ** n.array(
        [
            -5.5966,
            -5.6949,
            -5.8010,
            -5.9155,
            -6.0365,
            -6.1585,
            -6.2787,
            -6.4077,
            -6.5316,
            -6.6504,
            -6.7818,
            -6.9201,
            -7.0709,
            -7.1787,
            -7.3135,
            -7.4426,
            -7.5659,
            -7.6629,
            -7.7103,
            -7.7650,
            -7.8202,
            -7.8828,
            -7.9818,
            -8.0745,
            -8.2026,
            -8.3019,
            -8.4214,
            -8.7088,
            -8.8903,
            -9.3435,
            -9.6584,
            -9.7364,
        ]
    )
    fd = 10 ** n.array(
        [
            -8.2227,
            -8.1988,
            -8.2207,
            -8.2077,
            -8.2186,
            -8.2227,
            -8.2446,
            -8.2998,
            -8.3217,
            -8.2978,
            -8.3196,
            -8.3537,
            -8.3554,
            -8.3674,
            -9.7537,
            -9.7756,
            -9.7838,
            -9.7947,
            -9.7574,
            -9.6957,
            -9.6998,
            -9.7217,
            -9.6978,
            -9.7988,
            -9.6957,
            -9.6998,
            -9.6937,
            -9.7684,
            -9.8446,
            -9.9967,
            -10.1435,
            -10.1196,
        ]
    )
    fu = 10 ** n.array(
        [
            -8.2227,
            -8.1988,
            -8.2207,
            -8.2077,
            -8.2186,
            -8.2227,
            -8.2446,
            -8.2998,
            -8.3217,
            -8.2978,
            -8.3196,
            -8.3537,
            -8.3653,
            -8.3516,
            -9.7537,
            -9.7756,
            -9.7838,
            -9.5838,
            -9.7574,
            -9.6957,
            -9.6998,
            -9.7217,
            -9.6978,
            -9.7988,
            -9.6957,
            -9.6998,
            -9.6937,
            -9.7684,
            -9.7988,
            -9.8998,
            -10.0009,
            -9.9684,
        ]
    )
    ul = n.zeros_like(e, dtype=int)
    return dict(x=e, y=f, yerr=[fd, fu], uplims=ul)


def plot_auger(ax, lh, ll, label=None, **kwargs):

    vals = get_auger()
    plot_kwargs = dict(
        color="k", ecolor="k", marker="s", markersize=4, linestyle="None", capsize=2
    )
    plot_kwargs.update(kwargs)
    art = ax.errorbar(
        vals["x"], vals["y"], yerr=vals["yerr"], uplims=vals["uplims"], **plot_kwargs
    )
    if label is not None:
        lh.append(art)
        ll.append(label)


def get_ta():
    # Values from ICRC2019https://doi.org/10.22323/1.358.0298
    e = n.logspace(18.25, 20.25, 21)
    f = n.array(
        [
            1.087e-07,
            7.959e-08,
            5.850e-08,
            4.338e-08,
            3.374e-08,
            2.649e-08,
            2.285e-08,
            1.921e-08,
            1.730e-08,
            1.500e-08,
            1.317e-08,
            1.075e-08,
            7.930e-09,
            7.131e-09,
            5.770e-09,
            5.733e-09,
            3.907e-09,
            1.935e-09,
            1.737e-09,
            1.983e-10,
            4.889e-10,
        ])

    fd = n.array([
            1.961e-09,
            1.253e-09,
            9.055e-10,
            7.176e-10,
            5.986e-10,
            5.534e-10,
            5.262e-10,
            5.163e-10,
            5.665e-10,
            6.086e-10,
            5.755e-10,
            5.714e-10,
            5.536e-10,
            5.892e-10,
            6.125e-10,
            6.862e-10,
            5.964e-10,
            4.440e-10,
            5.046e-10,
            1.251e-10,
            3.080e-10,
        ])

    fu = n.array([
            1.961e-09,
            1.203e-09,
            9.055e-10,
            7.176e-10,
            6.185e-10,
            5.502e-10,
            5.262e-10,
            5.163e-10,
            5.539e-10,
            6.066e-10,
            5.755e-10,
            5.764e-10,
            5.536e-10,
            5.892e-10,
            6.125e-10,
            6.862e-10,
            5.964e-10,
            5.814e-10,
            6.055e-10,
            3.472e-10,
            5.499e-10,
        ])
    ul = n.zeros_like(e)
    e *= 1e-9
    return dict(x=e, y=f, yerr=[fd, fu], uplims=ul)


def plot_ta(ax, lh, ll, label=None, **kwargs):
    vals = get_ta()
    plot_kwargs = dict(
        color="k", ecolor="k", marker="s", markersize=4, linestyle="None", capsize=2
    )
    plot_kwargs.update(kwargs)
    art = ax.errorbar(vals["x"], vals["y"], yerr=vals["yerr"], uplims=vals["uplims"], **plot_kwargs)
    if label is not None:
        lh.append(art)
        ll.append(label)


def get_fermi_igrb_2014():
    ebins = n.logspace(-1, 2.9125, 27)
    e = 10 ** (0.5 * (n.log10(ebins[1:]) + n.log10(ebins[:-1])))
    e_down = e - ebins[:-1]
    e_up = ebins[1:] - e
    flux = n.array(
        [
            9.4852e-7,
            8.3118e-7,
            7.3310e-7,
            6.5520e-7,
            6.2516e-7,
            6.4592e-7,
            5.4767e-7,
            4.2894e-7,
            3.4529e-7,
            3.2094e-7,
            2.7038e-7,
            2.3997e-7,
            2.2308e-7,
            2.5266e-7,
            1.9158e-7,
            1.7807e-7,
            1.4804e-7,
            1.4320e-7,
            1.2303e-7,
            9.8937e-8,
            6.0016e-8,
            5.4006e-8,
            3.6609e-8,
            3.4962e-8,
            1.1135e-8,
            4.7647e-9,
        ]
    )
    flux_up = n.array(
        [
            1.1324e-6,
            1.0393e-6,
            9.4078e-7,
            8.5215e-7,
            7.6656e-7,
            7.2692e-7,
            6.0432e-7,
            4.8297e-7,
            3.9361e-7,
            3.5653e-7,
            3.0630e-7,
            2.8097e-7,
            2.6124e-7,
            2.8820e-7,
            2.1703e-7,
            2.0047e-7,
            1.6668e-7,
            1.6121e-7,
            1.3852e-7,
            1.0853e-7,
            7.1681e-8,
            6.4886e-8,
            4.7313e-8,
            4.6671e-8,
            1.9719e-8,
            4.7647e-9,
        ]
    )
    flux_down = n.array(
        [
            7.5851e-7,
            6.1847e-7,
            5.1746e-7,
            4.5949e-7,
            4.8715e-7,
            5.7000e-7,
            4.8657e-7,
            3.7629e-7,
            2.9871e-7,
            2.8890e-7,
            2.3552e-7,
            2.0231e-7,
            1.9059e-7,
            2.2169e-7,
            1.6688e-7,
            1.5717e-7,
            1.3068e-7,
            1.2641e-7,
            1.0722e-7,
            8.5645e-8,
            4.8961e-8,
            4.2901e-8,
            2.6368e-8,
            2.3876e-8,
            3.3003e-9,
            0.5 * 4.7647e-9,
        ]
    )
    lims = n.zeros_like(flux)
    lims[-1] = 1
    return dict(x=e, y=flux, yerr=[flux - flux_down, flux_up - flux], uplims=lims)


def plot_fermi_igrb_2014(ax, lh, ll, label=r"$Fermi\,IGRB$", **kwargs):
    ebins = n.logspace(-1, 2.9125, 27)
    e = 10 ** (0.5 * (n.log10(ebins[1:]) + n.log10(ebins[:-1])))
    e_down = e - ebins[:-1]
    e_up = ebins[1:] - e
    flux = n.array(
        [
            9.4852e-7,
            8.3118e-7,
            7.3310e-7,
            6.5520e-7,
            6.2516e-7,
            6.4592e-7,
            5.4767e-7,
            4.2894e-7,
            3.4529e-7,
            3.2094e-7,
            2.7038e-7,
            2.3997e-7,
            2.2308e-7,
            2.5266e-7,
            1.9158e-7,
            1.7807e-7,
            1.4804e-7,
            1.4320e-7,
            1.2303e-7,
            9.8937e-8,
            6.0016e-8,
            5.4006e-8,
            3.6609e-8,
            3.4962e-8,
            1.1135e-8,
            4.7647e-9,
        ]
    )
    flux_up = n.array(
        [
            1.1324e-6,
            1.0393e-6,
            9.4078e-7,
            8.5215e-7,
            7.6656e-7,
            7.2692e-7,
            6.0432e-7,
            4.8297e-7,
            3.9361e-7,
            3.5653e-7,
            3.0630e-7,
            2.8097e-7,
            2.6124e-7,
            2.8820e-7,
            2.1703e-7,
            2.0047e-7,
            1.6668e-7,
            1.6121e-7,
            1.3852e-7,
            1.0853e-7,
            7.1681e-8,
            6.4886e-8,
            4.7313e-8,
            4.6671e-8,
            1.9719e-8,
            4.7647e-9,
        ]
    )
    flux_down = n.array(
        [
            7.5851e-7,
            6.1847e-7,
            5.1746e-7,
            4.5949e-7,
            4.8715e-7,
            5.7000e-7,
            4.8657e-7,
            3.7629e-7,
            2.9871e-7,
            2.8890e-7,
            2.3552e-7,
            2.0231e-7,
            1.9059e-7,
            2.2169e-7,
            1.6688e-7,
            1.5717e-7,
            1.3068e-7,
            1.2641e-7,
            1.0722e-7,
            8.5645e-8,
            4.8961e-8,
            4.2901e-8,
            2.6368e-8,
            2.3876e-8,
            3.3003e-9,
            0.5 * 4.7647e-9,
        ]
    )
    lims = n.zeros_like(flux)
    lims[-1] = 1

    plot_kwargs = dict(color="k", marker="o", markersize=4, linestyle="None", capsize=2)
    plot_kwargs.update(kwargs)
    art = ax.errorbar(
        e, flux, yerr=[flux - flux_down, flux_up - flux], uplims=lims, **plot_kwargs
    )
    lh.append(art)
    ll.append(label)


def plot_unfolding(
    ax, lh, ll, e, best, low, high, label, rescale=True, plotlimits=False, **kwargs
):
    plot_kwargs = dict(
        color="k",
        marker="o",
        mfc="steelblue",
        markersize=5,
        linestyle="None",
        capsize=2,
        linewidth=1.5,
        capthick=1.5,
    )
    plot_kwargs.update(kwargs)
    if rescale:
        best, low, high = 1e-8 * best, 1e-8 * low, 1e-8 * high
    ul = n.zeros_like(e, dtype="int")
    if plotlimits:
        m = best < 1e-10
        best[m] = high[m]
        low[m] = 0.5 * high[m]
        ul[m] = 1
    art = ax.errorbar(
        e, best, xerr=None, yerr=[best - low, high - best], uplims=ul, **plot_kwargs
    )
    lh.append(art)
    ll.append(label)


def plot_legend(ax, lh, ll, **kwargs):
    plot_kwargs = dict(loc="upper right", ncol=1, frameon=False, fontsize=10)
    plot_kwargs.update(kwargs)
    return ax.legend(lh, ll, **plot_kwargs)


def set_limits(ax, erange, yrange, ax2=None):
    ax.set_xlim(erange)
    ax.set_ylim(yrange)
    if ax2 is not None:
        ax2.set_xlim(n.log10(erange[0]), n.log10(erange[1]))
        ax2.set_ylim(n.log10(yrange[0]), n.log10(yrange[1]))


if __name__ == "__main__":

    p.interactive(0)

    p.matplotlib.rcParams["figure.facecolor"] = "white"
    p.matplotlib.rcParams["text.usetex"] = True
    p.matplotlib.rcParams["font.size"] = 10
    p.matplotlib.rcParams["font.sans-serif"] = "computer modern"
    p.matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{upgreek}"]

    ic_flux = n.loadtxt("icecube_flux.txt")

    f, ax = create_figure(loc=[0.15, 0.14, 0.84, 0.83])
    lh, ll = [], []
    plot_fermi_igrb_2014(
        ax,
        lh,
        ll,
        label=r"Diffuse $\upgamma$ (Fermi LAT)",
        marker="D",
        color="darkorange",
        ecolor="k",
    )
    plot_unfolding(
        ax,
        lh,
        ll,
        *ic_flux,
        label=r"Cosmic $\upnu$ (IceCube, this work)",
        plotlimits=True,
        linewidth=1,
        capthick=0.5
    )
    plot_auger(ax, lh, ll, label="Cosmic rays (Auger)", marker="o", color="tomato")
    plot_ta(ax, lh, ll, label="Cosmic rays (TA)", color="mediumseagreen")
    l = plot_legend(
        ax,
        lh,
        ll,
        loc="upper center",
        ncol=2,
        numpoints=1,
        fontsize=9,
        columnspacing=0.4,
        handletextpad=0.1,
    )
    set_limits(ax, (1, 5e11), (2e-11, 5e-5))

    f.savefig("flux_result_compare.pdf")
