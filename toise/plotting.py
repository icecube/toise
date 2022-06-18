from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np


def stepped_path(edges, bins, cumulative=False):
    """
    Create a stepped path suitable for histogramming

    :param edges: bin edges
    :param bins: bin contents
    """
    if len(edges) != len(bins) + 1:
        raise ValueError("edges must be 1 element longer than bins")

    x = np.zeros((2 * len(edges)))
    y = np.zeros((2 * len(edges)))

    if cumulative is not False:
        if cumulative == "<":
            bins = bins.cumsum()
        elif cumulative == ">":
            bins = bins[::-1].cumsum()[::-1]

    x[0::2], x[1::2] = edges, edges
    y[1:-1:2], y[2::2] = bins, bins

    return x, y


def format_energy(fmt, energy):
    places = int(np.log10(energy) / 3) * 3
    if places == 0:
        unit = "GeV"
    elif places == 3:
        unit = "TeV"
    elif places == 6:
        unit = "PeV"
    elif places == 9:
        unit = "EeV"
    elif places == 12:
        unit = "ZeV"
    elif places == 15:
        unit = "YeV"
    return (fmt % (energy / 10 ** (places))) + " " + unit


def plot_profile2d(profile, x, y, levels=[68, 90, 99], colors="k", **kwargs):
    from scipy.stats import chi2
    import matplotlib.pyplot as plt

    xv = np.unique(profile[x])
    yv = np.unique(profile[y])
    shape = (xv.size, yv.size)

    ts = 2 * (np.nanmax(profile["LLH"]) - profile["LLH"]).reshape(shape)
    pvalue = chi2.cdf(ts.T, 2) * 100

    ax = plt.gca()
    cs = ax.contour(xv, yv, pvalue, levels=levels, colors=colors, **kwargs)
    if ax.get_xlabel() == "":
        ax.set_xlabel(x)
    if ax.get_ylabel() == "":
        ax.set_ylabel(y)
    return cs


def pretty_style(tex=True):
    import matplotlib

    style = {
        "figure.figsize": (3.375, 3.375),
        "legend.frameon": False,
        "legend.fontsize": "small",
        "lines.linewidth": 1.5,
        "grid.linewidth": 0.1,
        "grid.linestyle": "-",
        "axes.titlesize": "medium",
        "image.cmap": "viridis",
    }
    if tex:
        style["font.family"] = "serif"
        style["font.serif"] = "Computer Modern"
        style["text.usetex"] = True
    return style


def pretty(*args, **kwargs):
    import matplotlib.pyplot as plt

    return plt.rc_context(pretty_style(*args, **kwargs))


def save_all(fname_base):
    import matplotlib.pyplot as plt

    plt.savefig(fname_base + ".pdf", transparent=True)
    plt.savefig(fname_base + ".hires.png", dpi=300)
    plt.savefig(fname_base + ".png")


def label_curve(ax, line, x=None, y=None, orientation="parallel", offset=0, **kwargs):
    """
    Place a label on an existing line. The anchor point may be specified
    either as an x or y coordinate. Extra keyword arguments will be passed
    to ax.text().

    :param ax: an Axes to draw the text in
    :param line: a Line2D to label
    :param x: x-position of label anchor
    :param y: y-position of label anchor
    :param orientation: if parallel, rotate label so that it matches
                        the local slope of the target line. Since rotations
                        must be specified in screen coordinates, this should
                        only be called once all layout (axis boundaries,
                        aspect ratios, etc.) is finished.
    :param offset: if nonzero, then shift the label from the anchor point along
                   the local vector this number of points

    Example
    -------

    >>> x = linspace(-1, 1, 101)
    >>> line = plot(x, x**2, label='foo!')[0]
    >>> label_curve(gca(), line, x=0.5, va='bottom',)
    >>> label_curve(gca(), line, x=-0.5, va='top', label='something\ndifferent!')

    """

    # extract points from line
    xd = line.get_xdata()
    yd = line.get_ydata()
    # sort if necessary
    if (np.diff(xd) < 0).any():
        order = xd.argsort()
        xd = xd[order]
        yd = yd[order]
    # de-step if necessary
    ux = np.unique(xd)
    if 2 * ux.size == xd.size and (ux == xd[::2]).all():
        xd = (xd[::2] + xd[1::2]) / 2
        yd = yd[1::2]

    # interpolate for x if y is supplied
    if x is None:
        x = np.interp([y], yd, xd)[0]
    # get points on either side of the anchor point
    i = np.searchsorted(xd, x)
    if i > xd.size - 2:
        i = xd.size - 2
    xb, yb = xd[i : i + 2], yd[i : i + 2]
    # interpolate for y
    y = yb[0] + (yb[1] - yb[0]) * (x - xb[0]) / (xb[1] - xb[0])

    text = kwargs.pop("label", line.get_label())
    kw = {
        "color": line.get_color(),
        "rotation_mode": "anchor",
        "ha": "center",
    }

    # get local slope in *screen coordinates*
    p1 = ax.transData.transform_point([xb[0], yb[0]])
    p2 = ax.transData.transform_point([xb[1], yb[1]])
    if orientation == "parallel":
        kw["rotation"] = np.degrees(np.arctan2(p2[1] - p1[1], p2[0] - p1[0]))

    kw.update(**kwargs)

    text = ax.text(x, y, text, **kw)
    # calculate normal in *screen coordinates*
    if offset != 0:
        xy = ax.transData.transform_point(text.get_position())
        norm = np.array([p2[1] - p1[1], p2[0] - p1[0]])
        norm = norm / (np.hypot(norm[0], norm[1]) / offset)
        xy = ax.transData.inverted().transform_point((xy[0] - norm[0], xy[1] + norm[1]))
        text.set_position(xy)
    return text
