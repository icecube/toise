"""
Custom axes for ternary projection, roughly adapted and updated from
ternary_project.py by Kevin L. Davies 
"""

import matplotlib.axis as maxis
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import matplotlib.spines as mspines
import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Polygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.ticker import MultipleLocator
from matplotlib.transforms import (
    Affine2D,
    BboxTransformFrom,
    BboxTransformTo,
    ScaledTranslation,
    TransformedBbox,
)


class TernaryAxes(Axes):
    name = "ternary1"
    angle = 0.0

    def __init__(self, *args, **kwargs):
        Axes.__init__(self, *args, **kwargs)
        self.cla()
        # C is for center.
        self.set_aspect(aspect="equal", adjustable="box", anchor="C")

    def cla(self):
        super(TernaryAxes, self).cla()

        self.set_xlim(0, 1)
        self.set_ylim(0, 1)
        self.yaxis.set_visible(False)
        self.xaxis.set_ticks_position("bottom")
        self.xaxis.set_ticks(np.linspace(0, 1, 11))

        self.xaxis.set_label_coords(0.5, 0, transform=self._xlabel_transform)

    @classmethod
    def create(cls, fig=None, subplotspec=None):
        """
        Top-level factory method. Use this to create new axes.
        """
        if fig is None:
            import pylab

            fig = pylab.gcf()
        if subplotspec is None:
            subplotspec = fig.add_gridspec()[0]
        try:
            self = fig.add_subplot(subplotspec, projection="ternary1")
        except TypeError:
            self = fig.add_axes(subplotspec, projection="ternary1")

        self.ab = self
        self.ab.patch.set_visible(False)
        self.bc = self.figure.add_axes(
            self.get_position(True),
            sharex=self.ab,
            projection="ternary2",
            frameon=True,
            zorder=-1,
        )
        self.bc.patch.set_visible(False)
        self.ca = self.figure.add_axes(
            self.get_position(True),
            sharex=self.ab,
            projection="ternary3",
            frameon=True,
            zorder=-1,
        )
        return self

    def grid(self, *args, **kwargs):
        if not hasattr(self, "ab"):
            super(TernaryAxes, self).grid(*args, **kwargs)
        else:
            for k in "ab", "bc", "ca":
                ax = getattr(self, k)
                ax.xaxis.grid(*args, **kwargs)

    def _rotate(self, trans):
        pass

    def _set_lim_and_transforms(self):
        """
        This is called once when the plot is created to set up all the
        transforms for the data, text and grids.
        """
        # Transform from the lower triangular half of the unit square
        # to an equilateral triangle
        h = np.sqrt(3) / 2
        self.transProjection = Affine2D().scale(1, h).skew_deg(30, 0)

        # Shift and and rotate the axis into place
        self.transAffine = Affine2D().translate(0.0, 1 - h)
        self._rotate(self.transAffine)

        # 3) This is the transformation from axes space to display
        # space.
        self.transAxes = BboxTransformTo(self.bbox)

        # An affine transformation on the data, generally to limit the
        # range of the axes
        self.transLimits = BboxTransformFrom(
            TransformedBbox(self.viewLim, self.transScale)
        )

        # Put all the transforms together
        self.transData = (
            self.transProjection
            + self.transAffine
            + (self.transLimits + self.transAxes)
        )

        self._xaxis_transform = self.transData
        self._yaxis_transform = self.transData

        # Set up a special skew-less transform for the axis labels
        self._xlabel_transform = Affine2D()
        self._rotate(self._xlabel_transform)
        self._xlabel_transform += self.transAxes

    def get_xaxis_transform(self, which="grid"):
        """
        Override this method to provide a transformation for the
        x-axis grid and ticks.
        """
        assert which in ["tick1", "tick2", "grid"]
        return self._xaxis_transform

    def get_xaxis_text1_transform(self, pad_points):
        """
        Override this method to provide a transformation for the
        x-axis tick labels.

        Returns a tuple of the form (transform, valign, halign)
        """
        pad = pad_points / 72.0
        va, ha = "center", "center"
        angle = self.angle
        return (
            self.transData
            + ScaledTranslation(
                pad * (np.sin(angle) + 0.5 * np.sin(angle - np.pi / 3)),
                pad * (-np.cos(angle) - 0.5 * np.cos(angle - np.pi / 3)),
                self.figure.dpi_scale_trans,
            ),
            va,
            ha,
        )

    def get_yaxis_transform(self, which="grid"):
        """
        Override this method to provide a transformation for the
        y-axis grid and ticks.
        """
        assert which in ["tick1", "tick2", "grid"]
        return self._yaxis_transform

    def _gen_axes_patch(self):
        h = np.sqrt(3) / 2.0
        return Polygon([(0, 1 - h), (1, 1 - h), (0.5, 1)], closed=True)

    def _gen_axes_spines(self):
        path = Path([(0.0, 0.0), (1.0, 0.0)])
        return dict(bottom=mspines.Spine(self, "bottom", path))

    def _init_axis(self):
        self.xaxis = maxis.XAxis(self)
        self.spines["bottom"].register_axis(self.xaxis)
        self.yaxis = maxis.YAxis(self)
        self._update_transScale()


class TernaryAxes2(TernaryAxes):
    name = "ternary2"
    angle = 2 * np.pi / 3

    def _rotate(self, trans):
        h = np.sqrt(3) / 2
        trans.rotate_around(1, 1 - h, self.angle).translate(-0.5, h)

    def cla(self):
        super(TernaryAxes2, self).cla()

        self.patch.set_visible(False)


class TernaryAxes3(TernaryAxes2):
    name = "ternary3"
    angle = 4 * np.pi / 3

    def _rotate(self, trans):
        h = np.sqrt(3) / 2
        trans.rotate_around(1, 1 - h, self.angle).translate(-1, 0)


def flavor_triangle(fig=None, subplotspec=None, grid=False):
    ax = TernaryAxes.create(fig, subplotspec)
    ax.grid(grid, "minor", zorder=0, linewidth=0.5)
    ax.ab.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.ab.xaxis.set_minor_locator(MultipleLocator(0.1))
    rot = 60
    sgn = 1
    for subax in (ax.ab, ax.bc, ax.ca):
        # rotate tick labels
        subax.tick_params(axis="x", which="major", pad=10)
        plt.setp(
            subax.xaxis.get_majorticklabels(), rotation=rot, va="center", ha="center"
        )

        # NB: Since axis ticks are drawn as a single-point line with a special
        # marker (tickup/tickdown for x axes), and markers have their own
        # transform, the axis transformation that skews the grid lines has no
        # effect on the tick marks. The least bad way to make them parallel to
        # the grid lines is to construct a custom marker with an appropriate
        # slant and set that as the style for the tick line.
        t = sgn * np.radians(90 - rot)
        x = np.sin(t)
        y = np.cos(t)
        p = mpath.Path(np.array([[0, 0], [-x, -y]]))
        for xt in subax.xaxis.get_major_ticks():
            l = xt.tick1line
            l.set_marker(p)
            l.set_markersize(10)
        for xt in subax.xaxis.get_minor_ticks():
            l = xt.tick1line
            l.set_marker(p)
            l.set_markersize(3)

        rot -= 60
        sgn *= -1
    fontdict = dict(size="xx-large")
    ax.ab.set_xlabel(r"$f_{e}$", fontdict=fontdict).set_position((0.5, 0))
    ax.bc.set_xlabel(r"$f_{\mu}$", fontdict=fontdict).set_position((0.5, -0.07))
    ax.ca.set_xlabel(r"$f_{\tau}$", fontdict=fontdict).set_position((0.5, -0.1))

    # Flavor ratios at Earth for various production scenarios, assuming inverted hierarchy
    # See: http://arxiv.org/pdf/1507.03991v2.pdf
    # http://arxiv.org/pdf/1506.02645v4.pdf
    ax.ab.scatter(
        [0.93 / 3],
        [1.05 / 3],
        marker="o",
        facecolor="k",
        edgecolor="w",
        lw=0.5,
        s=20,
        label="1:2:0",
    ).set_zorder(100)
    ax.ab.scatter(
        [0.19],
        [0.43],
        marker="s",
        facecolor="k",
        edgecolor="w",
        lw=0.5,
        s=20,
        label="0:1:0",
    ).set_zorder(100)
    ax.ab.scatter(
        [0.55],
        [0.19],
        marker="^",
        facecolor="k",
        edgecolor="w",
        lw=0.5,
        s=20,
        label="1:0:0",
    ).set_zorder(100)

    return ax


register_projection(TernaryAxes)
register_projection(TernaryAxes2)
register_projection(TernaryAxes3)
