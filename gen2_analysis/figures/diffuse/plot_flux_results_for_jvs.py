import numpy as n
import pylab as p




def xerr(bins, log=False):
    binc = get_binc(bins, log)
    l = binc - bins[:-1]
    h = bins[1:] - binc
    return (l, h)



def create_figure(format=None):
    figsize = (5.5, 3.5) if format == 'wide' else (4.5, 3.5)
    fig = p.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.set_xlabel(r'$E\,\,[\mathrm{GeV}]$')
    ax.set_ylabel(r'$E^{2}\times\Phi\,\,[\mathrm{GeV}\,\mathrm{s}^{-1}\,\mathrm{sr}^{-1}\,\mathrm{cm}^{-2}]$')
    ax.set_xscale('log')
    ax.set_yscale('log')

    return fig, ax

def get_auger():
    unit = 1e-10/31536000.
    e = n.logspace(17.55, 20.45, 30)
    f = 10**n.array([37.8535,37.8426,37.8097,37.7914,37.7622,
                     37.7146,37.7476,37.6928,37.6599,37.6709,
                     37.6380,37.6124,37.6052,37.6455,37.6767,
                     37.7152,37.7483,37.7502,37.7667,37.7484,
                     37.7522,37.5874,37.4556,37.3787,36.8824,
                     36.6792,36.3972,36.9229,37.1226,37.3241])
    fd = -n.inf*n.ones_like(f)
    fd[21:27] = 10**n.array([37.5544,37.4043,37.3055,36.4337,36.1810,35.]) # last point unknown
    fu = n.inf*n.ones_like(f)
    fu[21:27] = 10**n.array([37.6204,37.5069,37.4520,37.0418,36.9429,36.8661])
    ul = n.zeros_like(e, dtype=int)
    ul[-3:] = 1
    fd[-3:] = 0.5*f[-3:]
    fu[-3:] = f[-3:]
    f *= unit/e*1e-9
    fd *= unit/e*1e-9
    fu *= unit/e*1e-9
    e *= 1e-9
    return dict(x=e, y=f, yerr=[f-fd, fu-f], uplims=ul)

def plot_auger(ax, lh, ll, label=None, **kwargs):
    unit = 1e-10/31536000.
    e = n.logspace(17.55, 20.45, 30)
    f = 10**n.array([37.8535,37.8426,37.8097,37.7914,37.7622,
                     37.7146,37.7476,37.6928,37.6599,37.6709,
                     37.6380,37.6124,37.6052,37.6455,37.6767,
                     37.7152,37.7483,37.7502,37.7667,37.7484,
                     37.7522,37.5874,37.4556,37.3787,36.8824,
                     36.6792,36.3972,36.9229,37.1226,37.3241])
    fd = -n.inf*n.ones_like(f)
    fd[21:27] = 10**n.array([37.5544,37.4043,37.3055,36.4337,36.1810,35.]) # last point unknown
    fu = n.inf*n.ones_like(f)
    fu[21:27] = 10**n.array([37.6204,37.5069,37.4520,37.0418,36.9429,36.8661])
    ul = n.zeros_like(e, dtype='int')
    ul[-3:] = 1
    fd[-3:] = 0.5*f[-3:]
    fu[-3:] = f[-3:]
    f *= unit/e*1e-9
    fd *= unit/e*1e-9
    fu *= unit/e*1e-9
    e *= 1e-9
    plot_kwargs = dict(color='k', ecolor='k', marker='s', markersize=4, linestyle='None', capsize=2)
    plot_kwargs.update(kwargs)
    art = ax.errorbar(e, f, yerr=[f-fd, fu-f], uplims=ul, **plot_kwargs)
    if label is not None:
        lh.append(art)
        ll.append(label)

def get_ta():
    unit = 1e24*1e-4
    e = n.logspace(18.25, 20.35, 22)
    f = 10**n.array([0.28849,0.25662,0.21420,0.18496,0.15571,
                     0.16611,0.21745,0.23048,0.28447,0.28428,
                     0.33695,0.38962,0.41850,0.44737,0.39173,
                     0.45760,0.34513,0.06622,-0.03165,-0.11905,
                     0.06571,0.37729])
    fd = -n.inf*n.ones_like(f)
    fd[8:21] = 10**n.array([0.25538,0.24730,0.29468,0.34470,0.36300,
                            0.37868,0.29925,0.34928,0.19322,-0.17156,
                            -0.47029,-0.56030,-0.37289])
    fu = n.inf*n.ones_like(f)
    fu[8:21] = 10**n.array([0.30823,0.31599,0.37393,0.42924,0.46341,
                            0.50551,0.46569,0.54480,0.47724,0.30137,
                            0.29855,0.32480,0.50956])
    ul = n.zeros_like(e, dtype=int)
    ul[-1] = 1
    fd[-1] = 0.5*f[-1]
    fu[-1] = f[-1]
    f *= unit/e*1e-9
    fd *= unit/e*1e-9
    fu *= unit/e*1e-9
    e *= 1e-9
    return dict(x=e, y=f, yerr=[f-fd, fu-f], uplims=ul)

def plot_ta(ax, lh, ll, label=None, **kwargs):
    unit = 1e24*1e-4
    e = n.logspace(18.25, 20.35, 22)
    f = 10**n.array([0.28849,0.25662,0.21420,0.18496,0.15571,
                     0.16611,0.21745,0.23048,0.28447,0.28428,
                     0.33695,0.38962,0.41850,0.44737,0.39173,
                     0.45760,0.34513,0.06622,-0.03165,-0.11905,
                     0.06571,0.37729])
    fd = -n.inf*n.ones_like(f)
    fd[8:21] = 10**n.array([0.25538,0.24730,0.29468,0.34470,0.36300,
                            0.37868,0.29925,0.34928,0.19322,-0.17156,
                            -0.47029,-0.56030,-0.37289])
    fu = n.inf*n.ones_like(f)
    fu[8:21] = 10**n.array([0.30823,0.31599,0.37393,0.42924,0.46341,
                            0.50551,0.46569,0.54480,0.47724,0.30137,
                            0.29855,0.32480,0.50956])
    ul = n.zeros_like(e, dtype='int')
    ul[-1] = 1
    fd[-1] = 0.5*f[-1]
    fu[-1] = f[-1]
    f *= unit/e*1e-9
    fd *= unit/e*1e-9
    fu *= unit/e*1e-9
    e *= 1e-9
    plot_kwargs = dict(color='k', ecolor='k', marker='s', markersize=4, linestyle='None', capsize=2)
    plot_kwargs.update(kwargs)
    art = ax.errorbar(e, f, yerr=[f-fd, fu-f], uplims=ul, **plot_kwargs)
    if label is not None:
        lh.append(art)
        ll.append(label)

def get_fermi_igrb_2014():
    ebins = n.logspace(-1, 2.9125, 27)
    e = 10**(0.5*(n.log10(ebins[1:])+n.log10(ebins[:-1])))
    e_down = e - ebins[:-1]
    e_up = ebins[1:] - e
    flux = n.array([9.4852e-7, 8.3118e-7, 7.3310e-7, 6.5520e-7, 6.2516e-7, 6.4592e-7, 5.4767e-7, 4.2894e-7,
                    3.4529e-7, 3.2094e-7, 2.7038e-7, 2.3997e-7, 2.2308e-7, 2.5266e-7, 1.9158e-7, 1.7807e-7,
                    1.4804e-7, 1.4320e-7, 1.2303e-7, 9.8937e-8, 6.0016e-8, 5.4006e-8, 3.6609e-8, 3.4962e-8,
                    1.1135e-8, 4.7647e-9])
    flux_up = n.array([1.1324e-6, 1.0393e-6, 9.4078e-7, 8.5215e-7, 7.6656e-7, 7.2692e-7, 6.0432e-7, 4.8297e-7, 
                       3.9361e-7, 3.5653e-7, 3.0630e-7, 2.8097e-7, 2.6124e-7, 2.8820e-7, 2.1703e-7, 2.0047e-7, 
                       1.6668e-7, 1.6121e-7, 1.3852e-7, 1.0853e-7, 7.1681e-8, 6.4886e-8, 4.7313e-8, 4.6671e-8, 
                       1.9719e-8, 4.7647e-9])
    flux_down = n.array([7.5851e-7, 6.1847e-7, 5.1746e-7, 4.5949e-7, 4.8715e-7, 5.7000e-7, 4.8657e-7, 3.7629e-7, 
                         2.9871e-7, 2.8890e-7, 2.3552e-7, 2.0231e-7, 1.9059e-7, 2.2169e-7, 1.6688e-7, 1.5717e-7, 
                         1.3068e-7, 1.2641e-7, 1.0722e-7, 8.5645e-8, 4.8961e-8, 4.2901e-8, 2.6368e-8, 2.3876e-8, 
                         3.3003e-9, 0.5*4.7647e-9])
    lims = n.zeros_like(flux)
    lims[-1] = 1
    return dict(x=e, y=flux, yerr=[flux-flux_down, flux_up-flux], uplims=lims)

def plot_fermi_igrb_2014(ax, lh, ll, label=r'$Fermi\,IGRB$', **kwargs):
    ebins = n.logspace(-1, 2.9125, 27)
    e = 10**(0.5*(n.log10(ebins[1:])+n.log10(ebins[:-1])))
    e_down = e - ebins[:-1]
    e_up = ebins[1:] - e
    flux = n.array([9.4852e-7, 8.3118e-7, 7.3310e-7, 6.5520e-7, 6.2516e-7, 6.4592e-7, 5.4767e-7, 4.2894e-7,
                    3.4529e-7, 3.2094e-7, 2.7038e-7, 2.3997e-7, 2.2308e-7, 2.5266e-7, 1.9158e-7, 1.7807e-7,
                    1.4804e-7, 1.4320e-7, 1.2303e-7, 9.8937e-8, 6.0016e-8, 5.4006e-8, 3.6609e-8, 3.4962e-8,
                    1.1135e-8, 4.7647e-9])
    flux_up = n.array([1.1324e-6, 1.0393e-6, 9.4078e-7, 8.5215e-7, 7.6656e-7, 7.2692e-7, 6.0432e-7, 4.8297e-7, 
                       3.9361e-7, 3.5653e-7, 3.0630e-7, 2.8097e-7, 2.6124e-7, 2.8820e-7, 2.1703e-7, 2.0047e-7, 
                       1.6668e-7, 1.6121e-7, 1.3852e-7, 1.0853e-7, 7.1681e-8, 6.4886e-8, 4.7313e-8, 4.6671e-8, 
                       1.9719e-8, 4.7647e-9])
    flux_down = n.array([7.5851e-7, 6.1847e-7, 5.1746e-7, 4.5949e-7, 4.8715e-7, 5.7000e-7, 4.8657e-7, 3.7629e-7, 
                         2.9871e-7, 2.8890e-7, 2.3552e-7, 2.0231e-7, 1.9059e-7, 2.2169e-7, 1.6688e-7, 1.5717e-7, 
                         1.3068e-7, 1.2641e-7, 1.0722e-7, 8.5645e-8, 4.8961e-8, 4.2901e-8, 2.6368e-8, 2.3876e-8, 
                         3.3003e-9, 0.5*4.7647e-9])
    lims = n.zeros_like(flux)
    lims[-1] = 1
    
    plot_kwargs = dict(color='k', marker='o', markersize=4, linestyle='None', capsize=2)
    plot_kwargs.update(kwargs)
    art = ax.errorbar(e, flux, yerr=[flux-flux_down, flux_up-flux], uplims=lims, **plot_kwargs)
    lh.append(art)
    ll.append(label)


def plot_unfolding(ax, lh, ll, e, best, low, high, label, rescale=True, plotlimits=False, **kwargs):
    plot_kwargs = dict(color='k', marker='o', mfc='steelblue', markersize=5, linestyle='None', capsize=2, linewidth=1.5, capthick=1.5)
    plot_kwargs.update(kwargs)
    if rescale:
        best, low, high = 1e-8*best, 1e-8*low, 1e-8*high
    ul = n.zeros_like(e, dtype='int')
    if plotlimits:
        m = best<1e-10
        best[m] = high[m]
        low[m] = 0.5*high[m]
        ul[m] = 1
    art = ax.errorbar(e, best, xerr=None, yerr=[best-low, high-best], uplims=ul, **plot_kwargs)
    lh.append(art)
    ll.append(label)


def plot_legend(ax, lh, ll, **kwargs):
    plot_kwargs = dict(loc='upper right', ncol=1, frameon=False, fontsize=10)
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

	p.matplotlib.rcParams['figure.facecolor'] = 'white'
	p.matplotlib.rcParams['text.usetex'] = True
	p.matplotlib.rcParams['font.size'] = 10
	p.matplotlib.rcParams['font.sans-serif'] = 'computer modern'
	p.matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{upgreek}']

	ic_flux = n.loadtxt('icecube_flux.txt')

	f,ax = create_figure(loc=[0.15, 0.14, 0.84, 0.83])
	lh, ll = [], []
	plot_fermi_igrb_2014(ax, lh, ll, label='Diffuse $\upgamma$ (Fermi LAT)', marker='D', color='darkorange', ecolor='k')
	plot_unfolding(ax, lh, ll, *ic_flux, label='Cosmic $\upnu$ (IceCube, this work)', plotlimits=True, linewidth=1, capthick=0.5)
	plot_auger(ax, lh, ll, label='Cosmic rays (Auger)', marker='o', color='tomato')
	plot_ta(ax, lh, ll, label='Cosmic rays (TA)', color='mediumseagreen')
	l = plot_legend(ax, lh, ll, loc='upper center', ncol=2, numpoints=1, fontsize=9, columnspacing=0.4, handletextpad=0.1)
	set_limits(ax, (1, 5e11), (2e-11, 5e-5))

	f.savefig('flux_result_compare.pdf')
