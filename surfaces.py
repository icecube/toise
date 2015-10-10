
import numpy
import os

data_dir = os.path.join(os.path.dirname(__file__), 'data')

def get_gcd(geometry="Sunflower", spacing=200):
    if geometry == "EdgeWeighted":
        gcd='IceCubeHEX_{geometry}_spacing{spacing}m_ExtendedDepthRange.GCD.i3.bz2'.format(**locals())
    elif geometry == "Sunflower":
        gcd='IceCubeHEX_{geometry}_{spacing}m_v3_ExtendedDepthRange.GCD.i3.bz2'.format(**locals())
    else:
        raise ValueError("Unknown geometry %s" % geometry)
    return os.path.join(data_dir, 'geometries', gcd)

def get_fiducial_surface(geometry="Sunflower", spacing=200, padding=60):
    if geometry == "IceCube":
        return Cylinder()
    else:
        gcd = get_gcd(geometry, spacing)
    
    return ExtrudedPolygon.from_file(gcd, padding=padding)

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.
 
    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    
    Lifted from http://code.icecube.wisc.edu/svn/sandbox/ckopper/eventinjector/python/util/__init__.py
    """
 
    # convert to a list of tuples
    points = [(p[0],p[1]) for p in points]
    
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
    points = numpy.append(points, [points[0]], axis=0 )
    
    vecs = points[1:]-points[:-1]
    magn = numpy.sqrt(vecs[:,0]**2 + vecs[:,1]**2)
    
    normals = numpy.array([vecs[:,1]/magn, -vecs[:,0]/magn, numpy.zeros(magn.shape)]).T
    
    return normals

def hull_to_lengths(points):
    # append first point at the end to close the hull
    points = numpy.append(points, [points[0]], axis=0 )
    
    vecs = points[1:]-points[:-1]
    
    return numpy.sqrt(vecs[:,0]**2 + vecs[:,1]**2)

def signed_area(points):
    """Returns the signed area of a given simple (i.e. non-intersecting) polygon.
    Positive if points are sorted counter-clockwise.
    """

    # append first point at the end to close the hull
    points = numpy.append(points, [points[0]], axis=0 )
    
    return numpy.sum(points[:,0][:-1]*points[:,1][1:] - points[:,0][1:]*points[:,1][:-1])/2.

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
        return self.get_cap_area()*abs(cos_theta) + self.get_side_area()*numpy.sqrt(1-cos_theta**2)
    
    @staticmethod
    def _integrate_area(a, b, cap, sides):
        return (cap*(b**2-a**2) + sides*(numpy.arccos(a) - numpy.arccos(b) - numpy.sqrt(1-a**2)*a + numpy.sqrt(1-b**2)*b))/2.
    
    def entendue(self, cosMin=-1., cosMax=1.):
        """
        Integrate A * d\Omega over the given range of zenith angles
        
        :param cosMin: cosine of the maximum zenith angle
        :param cosMax: cosine of the minimum zenith angle
        :returns: a product of area and solid angle. Divide by
                  2*pi*(cosMax-cosMin) to obtain the average projected area in
                  this zenith angle range
        """
        
        sides = self.get_side_area()
        cap = self.get_cap_area()
        
        if (cosMin >= 0 and cosMax >= 0):
            area = self._integrate_area(cosMin, cosMax, cap, sides)
        elif (cosMin < 0 and cosMax <= 0):
            area = self._integrate_area(-cosMax, -cosMin, cap, sides)
        elif (cosMin < 0 and cosMax > 0):
            area = self._integrate_area(0, -cosMin, cap, sides) \
                + self._integrate_area(0, cosMax, cap, sides)
        else:
            area = numpy.nan
            raise ValueError("Can't deal with zenith range [%.1e, %.1e]" % (cosMin, cosMax))
        return 2*numpy.pi*area
    
    def average_area(self, cosMin=-1, cosMax=1):
        """
        Projected area of the surface, averaged between the given zenith angles
        and over all azimuth angles.
        
        :param cosMin: cosine of the maximum zenith angle
        :param cosMax: cosine of the minimum zenith angle
        :returns: the average projected area in the zenith angle range
        """
        return self.entendue(cosMin, cosMax)/(2*numpy.pi*(cosMax-cosMin))
    
    def volume(self):
        return self.get_cap_area()*self.length

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
        side_areas = self._side_lengths*self.length
        cap_area = [signed_area(hull)]*2
        cap_normals = numpy.array([[0., 0., 1.], [0., 0., -1.]])
        
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
        return self._side_lengths.sum()*self.length/numpy.pi
    
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
        d = self._dx/self._side_lengths[:,None]
        # and the one connecting the previous vertex
        prev_d = numpy.roll(d, 1, axis=0)
        # sine of the inner angle of each vertex
        det = prev_d[:,0]*d[:,1] - prev_d[:,1]*d[:,0]
        assert (det != 0.).all(), "Edges can't be [anti]parallel"
        points = self._x + (padding/det[:,None])*(prev_d - d)
        
        z_range = [self._z_range[0]-padding, self._z_range[1]+padding]
        
        return type(self)(points, z_range)
    
    @classmethod
    def from_I3Geometry(cls, i3geo, padding=0):
        from collections import defaultdict
        strings = defaultdict(list)
        for omkey, omgeo in i3geo.omgeo:
            if omgeo.omtype != omgeo.IceTop:
                strings[omkey.string].append(list(omgeo.position))
        mean_xy = [numpy.mean(positions, axis=0)[0:2] for positions in strings.values()]
        zmax = max(max(p[2] for p in positions) for positions in strings.values())
        zmin = min(min(p[2] for p in positions) for positions in strings.values())
        
        self = cls(mean_xy, [zmin, zmax])
        if padding != 0:
            return self.expand(padding)
        else:
            return self
    
    @classmethod
    def from_file(cls, fname, padding=0):
        from icecube import icetray, dataio, dataclasses
        f = dataio.I3File(fname)
        fr = f.pop_frame(icetray.I3Frame.Geometry)
        f.close()
        return cls.from_I3Geometry(fr['I3Geometry'], padding)
    
    def _direction_to_vec(self, cos_zenith, azimuth):
        ct, azi = numpy.broadcast_arrays(cos_zenith, azimuth)
        st = numpy.sqrt(1.-ct**2)
        cp = numpy.cos(azi)
        sp = numpy.sin(azi)
        return -numpy.array([st*cp, st*sp, ct])
    
    def area(self, cos_zenith, azimuth):
        """
        Return projected area as seen from cos_zenith,azimuth (IceCube convention)
        """
        vec = self._direction_to_vec(cos_zenith, azimuth)
        # inner product with component normals
        inner = numpy.dot(self._normals, vec)
        # only surfaces that face the requested direction count towards the area
        mask = inner < 0
        return -(inner*self._areas[:,None]*mask).sum(axis=0)
    
    def _point_in_hull(self, point):
        """
        Test whether point is inside the 2D hull by ray casting
        """
        x, y = point[0:2]
        # Find segments whose y range spans the current point
        mask = ((self._x[:,1] > y)&(self._nx[:,1] <= y))|((self._x[:,1] <= y)&(self._nx[:,1] > y))
        # Count crossings to the right of the current point
        xc = self._x[:,0] + (y-self._x[:,1])*self._dx[:,0]/self._dx[:,1]
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
        
        assert dirx+diry != 0, "Direction vector may not have zero length"
        
        # proportional distance along edge to intersection point
        # NB: if diry/dirx == dy/dx, the ray is parallel to the line segment
        nonparallel = diry*dx != dirx*dy
        alpha = numpy.where(nonparallel, (dirx*y - diry*x)/(diry*dx - dirx*dy), numpy.nan) 
        # check whether the intersection is actually in the segment
        mask = (alpha >= 0)&(alpha < 1)
        
        # distance along ray to intersection point
        if dirx != 0:
            beta = ((x + alpha*dx)/dirx)[mask]
        else:
            beta = ((y + alpha*dy)/diry)[mask]
        
        if beta.size == 0:
            return (numpy.nan,)*2
        else:
            return (numpy.nanmin(beta), numpy.nanmax(beta))
    
    def _distance_to_cap(self, point, dir, cap_z):
        d = (point[2]-cap_z)/dir[2]
        if self._point_in_hull(point + d*dir):
            return d
        else:
            return numpy.nan
    
    def _distance_to_caps(self, point, dir):
        return sorted((self._distance_to_cap(point, dir, cap_z) for cap_z in self._z_range))
    
    def point_in_footprint(self, point):
        return self._point_in_hull(point)
    
    def intersections(self, x, y, z, cos_zenith, azimuth):
        point = numpy.array((x, y, z))
        vec = self._direction_to_vec(cos_zenith, azimuth)
        
        # perfectly vertical track: only check intersections with caps
        if abs(cos_zenith) == 1.:
            return self._distance_to_caps(point, vec)
        # perfectly horizontal track: only check intersections with sides
        elif cos_zenith == 0.:
            return self._distance_to_hull(point, vec)
        # general case: both rho and z components nonzero
        else:
            sin_zenith = numpy.sqrt(1.-cos_zenith**2)
            sides = numpy.array(self._distance_to_hull(point, vec))/sin_zenith
            caps = self._distance_to_caps(point, vec)
            intersections = numpy.concatenate((sides, caps))
            return [numpy.nanmin(intersections), numpy.nanmax(intersections)] 

class Cylinder(UprightSurface):
    def __init__(self, length=1000, radius=587):
        self.length = length
        self.radius = radius

    def expand(self, margin):
        return Cylinder(self.length+2*margin, self.radius+margin)
    
    def point_in_footprint(self, point):
        return numpy.hypot(point[0], point[1]) < self.radius
    
    def get_z_range(self):
        return (-self.length/2., self.length/2)
    
    def get_cap_area(self):
        return numpy.pi*self.radius**2
    
    def get_side_area(self):
        return 2*self.radius*self.length
    
    def area(self, cos_zenith, azimuth=numpy.nan):
        return self.azimuth_averaged_area(cos_zenith)

