import os

import numpy

from .util import data_dir


def get_gcd(geometry="Sunflower", spacing=200):
    if geometry == "EdgeWeighted":
        gcd = "IceCubeHEX_{geometry}_spacing{spacing}m_ExtendedDepthRange.GCD.i3.bz2".format(
            **locals()
        )
    elif "Sunflower" in geometry:
        gcd = (
            "IceCubeHEX_{geometry}_{spacing}m_v3_ExtendedDepthRange.GCD.i3.bz2".format(
                **locals()
            )
        )
    elif geometry == "Banana":
        gcd = "IceCubeHEX_bananaLayout_v2.0_ExtendedDepthRange.GCD.i3.bz2"
    elif geometry == "IceCube":
        gcd = "GeoCalibDetectorStatus_IC86_Merged.i3.bz2"
    elif geometry == "Phase1":
        gcd = "GeoCalibDetectorStatus_pingu_V47_Jokinen_7_s33_d1_pDOM.i3.bz2"
    else:
        raise ValueError("Unknown geometry %s" % geometry)
    return os.path.join(data_dir, "geometries", gcd)


def get_geometry_file(geometry="Sunflower", spacing=200):
    if geometry == "EdgeWeighted":
        gcd = "IceCubeHEX_{geometry}_spacing{spacing}m_ExtendedDepthRange.GCD.txt.gz".format(
            **locals()
        )
    elif "Sunflower" in geometry:
        gcd = (
            "IceCubeHEX_{geometry}_{spacing}m_v3_ExtendedDepthRange.GCD.txt.gz".format(
                **locals()
            )
        )
    elif geometry == "Banana":
        gcd = "IceCubeHEX_bananaLayout_v2.0_ExtendedDepthRange.GCD.txt.gz"
    elif geometry == "IceCube":
        gcd = "GeoCalibDetectorStatus_IC86_Merged.txt.gz"
    else:
        raise ValueError("Unknown geometry %s" % geometry)
    return os.path.join(data_dir, "geometries", gcd)


def get_radio_geometry_file(geometry="Gen2_baseline_array"):

    if geometry == "Gen2_baseline_array":
        path = os.path.join(data_dir, "geometries", "{}.json.bz2".format(geometry))
        # the center of the arrays in the json files are at 0,0
        # but they really need to be displaced into the IceCube coordinate system
        center_x = -11760.249614931281
        center_y = 3354.665759087001
    else:
        raise ValueError("Uknown radio geometry {}".format(geometry))

    return path, center_x, center_y


def get_fiducial_surface(geometry="Sunflower", spacing=200, padding=60):
    if geometry == "Fictive":
        return Cylinder(500, 700)
    elif geometry == "IceCube":
        return Cylinder()
    else:
        gcd = get_geometry_file(geometry, spacing)

    return ExtrudedPolygon.from_file(gcd, padding=padding)


def get_inner_volume(geometry="Sunflower", spacing=200):
    """Get HESE-like inner volume"""
    if geometry == "Fictive":
        return Cylinder(400, 700)
    else:
        side_padding = spacing / 2.0
        return ExtrudedPolygon.from_file(
            get_geometry_file(geometry, spacing), padding=-side_padding
        )


# -----------------------------------------------------------------------------
# Original copyright notice from eventinjector
# -----------------------------------------------------------------------------
# Copyright (c) 2014
# Claudio Kopper <claudio.kopper@icecube.wisc.edu>
# and the IceCube Collaboration <http://www.icecube.wisc.edu>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION
# OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN
# CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# -----------------------------------------------------------------------------
def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.

    Lifted from http://code.icecube.wisc.edu/svn/sandbox/ckopper/eventinjector/python/util/__init__.py
    """

    # convert to a list of tuples
    points = [(p[0], p[1]) for p in points]

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    hull = lower[:-1] + upper[:-1]

    # convert into numpy array
    return numpy.array(hull)


def hull_to_normals(points):
    # append first point at the end to close the hull
    points = numpy.append(points, [points[0]], axis=0)

    vecs = points[1:] - points[:-1]
    magn = numpy.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)

    normals = numpy.array(
        [vecs[:, 1] / magn, -vecs[:, 0] / magn, numpy.zeros(magn.shape)]
    ).T

    return normals


def hull_to_lengths(points):
    # append first point at the end to close the hull
    points = numpy.append(points, [points[0]], axis=0)

    vecs = points[1:] - points[:-1]

    return numpy.sqrt(vecs[:, 0] ** 2 + vecs[:, 1] ** 2)


def signed_area(points):
    """Returns the signed area of a given simple (i.e. non-intersecting) polygon.
    Positive if points are sorted counter-clockwise.
    """

    # append first point at the end to close the hull
    points = numpy.append(points, [points[0]], axis=0)

    return (
        numpy.sum(
            points[:, 0][:-1] * points[:, 1][1:] - points[:, 0][1:] * points[:, 1][:-1]
        )
        / 2.0
    )


class UprightSurface(object):
    """
    A closed volume consisting entirely of vertical and horizontal surfaces.
    """

    def get_cap_area(self):
        """
        Get the area of the vertical surfaces
        """
        raise NotImplementedError

    def get_side_area(self):
        """
        Get azimuth-averaged area of the vertical surfaces
        """
        raise NotImplementedError

    def azimuth_averaged_area(self, cos_theta):
        """
        Return projected area at the given zenith angle, averaged over all
        azimuth angles.

        :param cos_theta: cosine of the zenith angle
        """
        return self.get_cap_area() * abs(cos_theta) + self.get_side_area() * numpy.sqrt(
            1 - cos_theta**2
        )

    def get_maximum_area(self):
        ct_max = numpy.cos(numpy.arctan(self.get_side_area() / self.get_cap_area()))
        return self.azimuth_averaged_area(ct_max)

    def sample_direction(self, cos_min=-1, cos_max=1, size=1):
        max_area = self.get_maximum_area()
        blocksize = min((size * 4, 65535))
        accepted = 0
        directions = numpy.empty((size, 3))
        while accepted < size:
            ct = numpy.random.uniform(cos_min, cos_max, blocksize)
            st = numpy.sqrt((1 - ct) * (1 + ct))
            azi = numpy.random.uniform(0, 2 * numpy.pi, blocksize)
            candidates = numpy.vstack((numpy.cos(azi) * st, numpy.sin(azi) * st, ct)).T
            candidates = candidates[
                numpy.random.uniform(0, max_area, blocksize)
                <= self.projected_area(candidates),
                :,
            ]
            if accepted + len(candidates) > size:
                candidates = candidates[: size - accepted]
            directions[accepted : accepted + len(candidates)] = candidates
            accepted += len(candidates)

        return directions

    @staticmethod
    def _integrate_area(a, b, cap, sides):
        return (
            cap * (b**2 - a**2)
            + sides
            * (
                numpy.arccos(a)
                - numpy.arccos(b)
                - numpy.sqrt(1 - a**2) * a
                + numpy.sqrt(1 - b**2) * b
            )
        ) / 2.0

    def etendue(self, cosMin=-1.0, cosMax=1.0):
        r"""
        Integrate A * d\Omega over the given range of zenith angles

        :param cosMin: cosine of the maximum zenith angle
        :param cosMax: cosine of the minimum zenith angle
        :returns: a product of area and solid angle. Divide by
                  2*pi*(cosMax-cosMin) to obtain the average projected area in
                  this zenith angle range
        """

        sides = self.get_side_area()
        cap = self.get_cap_area()

        if cosMin >= 0 and cosMax >= 0:
            area = self._integrate_area(cosMin, cosMax, cap, sides)
        elif cosMin < 0 and cosMax <= 0:
            area = self._integrate_area(-cosMax, -cosMin, cap, sides)
        elif cosMin < 0 and cosMax > 0:
            area = self._integrate_area(0, -cosMin, cap, sides) + self._integrate_area(
                0, cosMax, cap, sides
            )
        else:
            area = numpy.nan
            raise ValueError(
                "Can't deal with zenith range [%.1e, %.1e]" % (cosMin, cosMax)
            )
        return 2 * numpy.pi * area

    def average_area(self, cosMin=-1, cosMax=1):
        """
        Projected area of the surface, averaged between the given zenith angles
        and over all azimuth angles.

        :param cosMin: cosine of the maximum zenith angle
        :param cosMax: cosine of the minimum zenith angle
        :returns: the average projected area in the zenith angle range
        """
        return self.etendue(cosMin, cosMax) / (2 * numpy.pi * (cosMax - cosMin))

    def volume(self):
        return self.get_cap_area() * self.length


class ExtrudedPolygon(UprightSurface):
    """
    A *convex* polygon in the x-y plane, extruded in the z direction
    """

    def __init__(self, xy_points, z_range):
        assert len(xy_points) >= 3
        hull = convex_hull(xy_points)
        # hull points, in counterclockwise order
        self._x = hull
        # next neighbor in the hull
        self._nx = numpy.roll(hull, -1, axis=0)
        # vector connecting each pair of points in the hull
        self._dx = self._nx - self._x

        self._z_range = z_range
        self.length = z_range[1] - z_range[0]

        side_normals = hull_to_normals(hull)
        self._side_lengths = hull_to_lengths(hull)
        side_areas = self._side_lengths * self.length
        cap_area = [signed_area(hull)] * 2
        cap_normals = numpy.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])

        self._areas = numpy.concatenate((side_areas, cap_area))
        self._normals = numpy.concatenate((side_normals, cap_normals))
        assert self._areas.size == self._normals.shape[0]

    def get_z_range(self):
        return self._z_range

    def get_cap_area(self):
        return self._areas[-1]

    def get_side_area(self):
        # the projected area of a plane, averaged over a 2\pi rotation that
        # passes through the normal, is
        # A*\int_0^\pi \Theta(\sin\alpha)\sin\alpha d\alpha / 2\pi = A/\pi
        return self._side_lengths.sum() * self.length / numpy.pi

    def projected_area(self, direction):
        inner = numpy.dot(direction, self._normals.T)
        areas = numpy.where(inner < 0, -inner * self._areas, 0)
        return areas.sum(axis=areas.ndim - 1)

    def _sample_on_caps(self, directions, bottom, offset, scale):
        """
        :param directions: direction unit vectors
        :param top: boolean mask; False if top, True if bottom
        """
        size = len(directions)
        accepted = 0
        blocksize = min((size * 4, 65535))
        positions = numpy.empty((len(directions), 3))
        while accepted < size:
            cpos = numpy.random.uniform(size=(blocksize, 2)) * scale + offset
            mask = numpy.array(list(map(self._point_in_hull, cpos)))
            cpos = cpos[mask]
            if len(cpos) + accepted > size:
                cpos = cpos[: size - accepted]
            positions[accepted : accepted + len(cpos), :-1] = cpos
            accepted += len(cpos)

        positions[bottom, -1] = self._z_range[0]
        positions[~bottom, -1] = self._z_range[1]

        return positions

    def sample_impact_ray(self, cos_min=-1, cos_max=1, size=1):
        directions = numpy.empty((size, 3))
        positions = numpy.empty((size, 3))
        accepted = 0
        blocksize = min((size * 4, 65535))

        bbox_offset = self._x.min(axis=0)
        bbox_scale = self._x.max(axis=0) - bbox_offset

        while accepted < size:
            block = min((blocksize, size - accepted))
            cdir = self.sample_direction(cos_min, cos_max, block)
            directions[accepted : accepted + block] = cdir
            inner = numpy.dot(cdir, self._normals.T)
            areas = numpy.where(inner < 0, -inner * self._areas, 0)
            prob = areas.cumsum(axis=1)
            prob /= prob[:, -1:]
            p = numpy.random.uniform(size=block)
            target = numpy.array([prob[i, :].searchsorted(p[i]) for i in range(block)])

            # first, handle sides
            sides = target < len(self._areas) - 2
            side_target = target[sides]
            nsides = sides.sum()
            xy = (
                self._x[side_target]
                + numpy.random.uniform(size=nsides)[:, None] * self._dx[side_target]
            )
            xyz = numpy.concatenate(
                (
                    xy,
                    (
                        self._z_range[0]
                        + numpy.random.uniform(size=nsides) * self.length
                    )[:, None],
                ),
                axis=1,
            )
            positions[accepted : accepted + block][sides] = xyz

            # now, the caps
            caps = ~sides
            cap_target = target[caps]
            positions[accepted : accepted + block][caps] = self._sample_on_caps(
                cdir[caps], cap_target == self._areas.size - 1, bbox_offset, bbox_scale
            )

            accepted += block

        return directions, positions

    def expand(self, padding):
        """
        Expand the x-y footprint by moving each edge out by a distance *padding*.
        """
        # A convex polygon can be offset by moving each vertex parallel to the
        # edges by a distance that is inversely proportional to the sine of the
        # counterclockwise angle between the edges that meet at each vertex.
        # This breaks down for edges that are [anti]parallel or, but neither
        # case should occur for maximally simplified polygons.

        # normalized vector connecting each vertex to the next one
        d = self._dx / self._side_lengths[:, None]
        # and the one connecting the previous vertex
        prev_d = numpy.roll(d, 1, axis=0)
        # sine of the inner angle of each vertex
        det = prev_d[:, 0] * d[:, 1] - prev_d[:, 1] * d[:, 0]
        assert (det != 0.0).all(), "Edges can't be [anti]parallel"
        points = self._x + (padding / det[:, None]) * (prev_d - d)

        z_range = [self._z_range[0] - padding, self._z_range[1] + padding]

        return type(self)(points, z_range)

    @classmethod
    def from_I3Geometry(cls, i3geo, padding=0):
        """
        Create from an I3Geometry object file

        :param i3geo: an I3Geometry
        :param padding: distance, in meters, to expand the surface in all directions
        """
        from collections import defaultdict

        strings = defaultdict(list)
        for omkey, omgeo in i3geo.omgeo:
            if omgeo.omtype != omgeo.IceTop:
                strings[omkey.string].append(list(omgeo.position))
        mean_xy = [
            numpy.mean(positions, axis=0)[0:2] for positions in list(strings.values())
        ]
        zmax = max(max(p[2] for p in positions) for positions in list(strings.values()))
        zmin = min(min(p[2] for p in positions) for positions in list(strings.values()))

        self = cls(mean_xy, [zmin, zmax])
        if padding != 0:
            return self.expand(padding)
        else:
            return self

    @classmethod
    def from_file(cls, fname, padding=0):
        dats = numpy.loadtxt(fname)
        zmax = dats[:, -1].max()
        zmin = dats[:, -1].min()
        mean_xy = [
            dats[dats[:, 0] == i].mean(axis=0)[2:4] for i in numpy.unique(dats[:, 0])
        ]

        self = cls(mean_xy, [zmin, zmax])
        if padding != 0:
            return self.expand(padding)
        else:
            return self

    @classmethod
    def from_i3file(cls, fname, padding=0):
        """
        Create from a GCD file

        :param fname: path to an I3 file containing at least a G frame
        :param padding: distance, in meters, to expand the surface in all directions
        """
        from icecube import dataclasses, dataio, icetray

        f = dataio.I3File(fname)
        fr = f.pop_frame(icetray.I3Frame.Geometry)
        f.close()
        return cls.from_I3Geometry(fr["I3Geometry"], padding)

    def _direction_to_vec(self, cos_zenith, azimuth):
        ct, azi = numpy.broadcast_arrays(cos_zenith, azimuth)
        st = numpy.sqrt(1.0 - ct**2)
        cp = numpy.cos(azi)
        sp = numpy.sin(azi)
        return -numpy.array([st * cp, st * sp, ct])

    def area(self, cos_zenith, azimuth):
        """
        Return projected area as seen from cos_zenith,azimuth (IceCube convention)
        """
        vec = self._direction_to_vec(cos_zenith, azimuth)
        # inner product with component normals
        inner = numpy.dot(self._normals, vec)
        # only surfaces that face the requested direction count towards the area
        mask = inner < 0
        return -(inner * self._areas[:, None] * mask).sum(axis=0)

    def _point_in_hull(self, point):
        """
        Test whether point is inside the 2D hull by ray casting
        """
        x, y = point[0:2]
        # Find segments whose y range spans the current point
        mask = ((self._x[:, 1] > y) & (self._nx[:, 1] <= y)) | (
            (self._x[:, 1] <= y) & (self._nx[:, 1] > y)
        )
        # Count crossings to the right of the current point
        xc = self._x[:, 0] + (y - self._x[:, 1]) * self._dx[:, 0] / self._dx[:, 1]
        crossings = (x < xc[mask]).sum()
        inside = (crossings % 2) == 1

        return inside

    def _distance_to_hull(self, point, vec):
        """
        Calculate the most extreme displacements from x,y along dx,dy to points
        on the 2D hull
        """
        # calculate the distance along the ray to each line segment
        x, y = (self._x - point[:2]).T
        dx, dy = self._dx.T
        dirx, diry = vec[0:2]

        assert dirx + diry != 0, "Direction vector may not have zero length"

        # proportional distance along edge to intersection point
        # NB: if diry/dirx == dy/dx, the ray is parallel to the line segment
        nonparallel = diry * dx != dirx * dy
        alpha = numpy.where(
            nonparallel, (dirx * y - diry * x) / (diry * dx - dirx * dy), numpy.nan
        )
        # check whether the intersection is actually in the segment
        mask = (alpha >= 0) & (alpha < 1)

        # distance along ray to intersection point
        if dirx != 0:
            beta = ((x + alpha * dx) / dirx)[mask]
        else:
            beta = ((y + alpha * dy) / diry)[mask]

        if beta.size == 0:
            return (numpy.nan,) * 2
        else:
            return (numpy.nanmin(beta), numpy.nanmax(beta))

    def _distance_to_cap(self, point, dir, cap_z):
        d = (point[2] - cap_z) / dir[2]
        if self._point_in_hull(point + d * dir):
            return d
        else:
            return numpy.nan

    def _distance_to_caps(self, point, dir):
        return sorted(
            (self._distance_to_cap(point, dir, cap_z) for cap_z in self._z_range)
        )

    def point_in_footprint(self, point):
        return self._point_in_hull(point)

    def intersections(self, x, y, z, cos_zenith, azimuth):
        point = numpy.array((x, y, z))
        vec = self._direction_to_vec(cos_zenith, azimuth)

        # perfectly vertical track: only check intersections with caps
        if abs(cos_zenith) == 1.0:
            return self._distance_to_caps(point, vec)
        # perfectly horizontal track: only check intersections with sides
        elif cos_zenith == 0.0:
            return self._distance_to_hull(point, vec)
        # general case: both rho and z components nonzero
        else:
            sin_zenith = numpy.sqrt(1.0 - cos_zenith**2)
            sides = numpy.array(self._distance_to_hull(point, vec)) / sin_zenith
            caps = self._distance_to_caps(point, vec)

            return numpy.nanmax((sides[0], caps[0])), numpy.nanmin((sides[1], caps[1]))


class Cylinder(UprightSurface):
    def __init__(self, length=1000, radius=587):
        self.length = length
        self.radius = radius

    def expand(self, margin):
        return Cylinder(self.length + 2 * margin, self.radius + margin)

    def point_in_footprint(self, point):
        return numpy.hypot(point[0], point[1]) < self.radius

    def get_z_range(self):
        return (-self.length / 2.0, self.length / 2)

    def get_cap_area(self):
        return numpy.pi * self.radius**2

    def get_side_area(self):
        return 2 * self.radius * self.length

    def area(self, cos_zenith, azimuth=numpy.nan):
        return self.azimuth_averaged_area(cos_zenith)

    def projected_area(self, direction):
        ct = direction[..., -1]
        st = numpy.sqrt((1 + ct) * (1 - ct))
        return self.get_cap_area() * abs(ct) + self.get_side_area() * st

    def sample_impact_ray(self, cos_min=-1, cos_max=1, size=1):
        directions = numpy.empty((size, 3))
        positions = numpy.empty((size, 3))
        accepted = 0
        blocksize = min((size * 4, 65535))

        while accepted < size:
            block = min((blocksize, size - accepted))
            cdir = self.sample_direction(cos_min, cos_max, block)
            directions[accepted : accepted + block] = cdir
            ct = -cdir[:, -1]
            st = numpy.sqrt((1 + ct) * (1 - ct))

            areas = numpy.array(
                [
                    self.get_side_area() * st,
                    self.get_cap_area() * numpy.where(ct > 0, ct, 0),
                    self.get_cap_area() * numpy.where(ct < 0, abs(ct), 0),
                ]
            ).T

            prob = areas.cumsum(axis=1)
            prob /= prob[:, -1:]
            p = numpy.random.uniform(size=block)
            target = numpy.array([prob[i, :].searchsorted(p[i]) for i in range(block)])

            # first, handle sides
            sides = target == 0
            nsides = sides.sum()
            beta = numpy.arcsin(
                numpy.random.uniform(-1, 1, size=nsides)
            ) + numpy.arctan2(-cdir[sides, 1], -cdir[sides, 0])
            positions[accepted : accepted + block][sides] = numpy.stack(
                (
                    self.radius * numpy.cos(beta) * st[sides],
                    self.radius * numpy.sin(beta) * st[sides],
                    numpy.random.uniform(
                        -self.length / 2, self.length / 2, size=nsides
                    ),
                )
            ).T

            # now, the caps
            caps = ~sides
            ncaps = caps.sum()
            cap_target = target[caps]
            beta = numpy.random.uniform(0, 2 * numpy.pi, size=ncaps)
            r = numpy.sqrt(numpy.random.uniform(size=ncaps)) * self.radius
            positions[accepted : accepted + block][caps] = numpy.stack(
                (
                    r * numpy.cos(beta),
                    r * numpy.sin(beta),
                    numpy.where(cap_target == 1, self.length / 2, -self.length / 2),
                )
            ).T
            accepted += block

        return directions, positions
