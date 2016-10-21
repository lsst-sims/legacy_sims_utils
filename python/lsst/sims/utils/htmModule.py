import numpy as np
from lsst.sims.utils import cartesianFromSpherical, sphericalFromCartesian

__all__ = ["Trixel", "HalfSpace", "findHtmId", "trixelFromLabel",
           "basic_trixels"]

class HalfSpace(object):

    def __init__(self, vector, length):
        self._v = vector/np.sqrt(np.power(vector, 2).sum())
        self._d = length
        self._phi = np.arccos(self._d)  # half angular extent of the half space

    def __eq__(self, other):
        tol = 1.0e-10
        if np.abs(self.dd-other.dd)>tol:
            return False
        if np.abs(np.dot(self.vector, other.vector)-1.0)>tol:
            return False
        return True

    @property
    def vector(self):
        """
        The unit vector from the origin to the center of the Half Space.
        """
        return self._v

    @property
    def dd(self):
        """
        The distance along the Half Space's vector that defines the
        extent of the Half Space.
        """
        return self._d

    @property
    def phi(self):
        """
        The angular radius of the Half Space on the surface of the sphere
        in radians.
        """
        return self._phi

    def contains_pt(self, pt):
        """
        Cartesian point
        """
        norm_pt = pt/np.sqrt(np.power(pt, 2).sum())

        dot_product = np.dot(pt, self._v)

        if self._d >= 0.0:
            if dot_product > self._d:
                return True
        else:
            if dot_product < np.abs(self._d):
                return True

        return False

    def _intersects_edge(self, pt1, pt2):
        """
        pt1 and pt2 are two unit vectors; the edge goes from pt1 to pt2

        see equation 4.8 of Szalay et al 2005
        https://www.microsoft.com/en-us/research/wp-content/uploads/2005/09/tr-2005-123.pdf
        """
        costheta = np.dot(pt1, pt2)
        u=np.sqrt((1-costheta)/(1+costheta))  # using trig identity for tan(theta/2)
        gamma1 = np.dot(self._v, pt1)
        gamma2 = np.dot(self._v, pt2)
        b = gamma1*(u*u-1.0) + gamma2*(u*u+1)
        a = -u*u*(gamma1+self._d)
        c = gamma1 - self._d

        det = b*b - 4*a*c
        if det<0.0:
            return False

        sqrt_det = np.sqrt(det)
        pos = (-b + sqrt_det)/(2.0*a)

        if pos >= 0.0 and pos <= 1.0:
            return True

        neg = (-b - sqrt_det)/(2.0*a)
        if neg >= 0.0 and neg <= 1.0:
            return True

        return False

    def contains_trixel(self, tx):

        n_corners_contained = 0
        for corner in tx.corners:
            if self.contains_pt(corner):
                n_corners_contained += 1

        if n_corners_contained == 3:
            return "full"
        elif n_corners_contained > 0:
            return "partial"

        # check if the trixel's bounding circle intersects
        # the halfspace
        theta = np.arccos(np.dot(tx.bounding_circle[0], self._v))
        if theta > self._phi + tx.bounding_circle[2]:
            return "outside"

        # need to test that the bounding circle intersect the halfspace
        # boundary

        intersection = False
        for edge in ((tx.corners[0], tx.corners[1]),
                     (tx.corners[1], tx.corners[2]),
                     (tx.corners[2], tx.corners[0])):

            if self._intersects_edge(edge[0], edge[1]):
                intersection = True
                break

        if intersection:
            return "partial"

        if tx.contains_pt(self._v):
            return "partial"

        return "outside"



class Trixel(object):

    def __init__(self, present_label, present_corners):
        """
        Corners in ccw order from lower left hand corner (v0)
        """
        self._corners = present_corners
        self._label = present_label
        self._level = (len('{0:b}'.format(self._label))/2)-1
        self._cross01 = None
        self._cross12 = None
        self._cross20 = None
        self._w_arr = None
        self._bounding_circle = None


    def contains(self, ra, dec):
        """
        In degrees
        """
        xyz = cartesianFromSpherical(np.radians(ra), np.radians(dec))
        return self.contains_pt(xyz)

    def contains_pt(self, pt):
        """
        Cartesian point
        """

        if self._cross01 is None:
            self._cross01 = np.cross(self._corners[0], self._corners[1])

        if np.dot(self._cross01,pt)>0.0:
            if self._cross12 is None:
                self._cross12 = np.cross(self._corners[1], self._corners[2])

            if np.dot(self._cross12, pt)>0.0:
                if self._cross20 is None:
                    self._cross20 = np.cross(self._corners[2], self._corners[0])
                if np.dot(self._cross20, pt)>0.0:
                    return True

        return False

    def _create_w(self):

        w0 = self._corners[1]+self._corners[2]
        w0 = w0/np.sqrt(np.power(w0, 2).sum())
        w1 = self._corners[0]+self._corners[2]
        w1 = w1/np.sqrt(np.power(w1, 2).sum())
        w2 = self._corners[0]+self._corners[1]
        w2 = w2/np.sqrt(np.power(w2, 2).sum())

        self._w_arr = [w0, w1, w2]

    def get_children(self):

        if self._w_arr is None:
            self._create_w()

        base_child = self._label << 2

        t0 = Trixel(base_child, [self._corners[0], self._w_arr[2], self._w_arr[1]])
        t1 = Trixel(base_child+1, [self._corners[1], self._w_arr[0],self._w_arr[2]])
        t2 = Trixel(base_child+2, [self._corners[2], self._w_arr[1],self._w_arr[0]])
        t3 = Trixel(base_child+3, [self._w_arr[0], self._w_arr[1], self._w_arr[2]])

        return [t0, t1, t2, t3]

    def get_child(self, dex):
        children = self.get_children()
        return children[dex]

    def get_center(self):
        xyz = self._corners[0] + self._corners[1] + self._corners[2]
        xyz = xyz/np.sqrt(np.power(xyz, 2).sum())
        ra, dec = sphericalFromCartesian(xyz)
        return np.degrees(ra), np.degrees(dec)

    @property
    def level(self):
        return self._level

    @property
    def label(self):
        return self._label

    @property
    def corners(self):
        return self._corners

    @property
    def bounding_circle(self):
        """
        Returns a tuple.
        Zeroth element is the 'z-axis' vector of the bounding circle.
        First element is the d of the bounding circle.
        Second element is the half angular extent of the bounding circle.
        """
        if self._bounding_circle is None:
            vb = np.cross((self._corners[1]-self._corners[0]), (self._corners[2]-self._corners[1]))
            vb = vb/np.sqrt(np.power(vb, 2).sum())
            dd = np.dot(self._corners[0], vb)
            if np.abs(dd)>1.0:
                raise RuntimeError("Bounding circle has dd %e (should be between -1 and 1)" % dd)
            self._bounding_circle = (vb, dd, np.arccos(dd))

        return self._bounding_circle


_N0_trixel = Trixel(12, [np.array([1.0, 0.0, 0.0]),
                         np.array([0.0, 0.0, 1.0]),
                         np.array([0.0, -1.0, 0.0])])

_N1_trixel = Trixel(13,[np.array([0.0, -1.0, 0.0]),
                        np.array([0.0, 0.0, 1.0]),
                        np.array([-1.0, 0.0, 0.0])])

_N2_trixel = Trixel(14, [np.array([-1.0, 0.0, 0.0]),
                         np.array([0.0, 0.0, 1.0]),
                         np.array([0.0, 1.0, 0.0])])

_N3_trixel = Trixel(15, [np.array([0.0, 1.0, 0.0]),
                         np.array([0.0, 0.0, 1.0]),
                         np.array([1.0, 0.0, 0.0])])

_S0_trixel = Trixel(8, [np.array([1.0, 0.0, 0.0]),
                        np.array([0.0, 0.0, -1.0]),
                        np.array([0.0, 1.0, 0.0])])

_S1_trixel = Trixel(9, [np.array([0.0, 1.0, 0.0]),
                        np.array([0.0, 0.0, -1.0]),
                        np.array([-1.0, 0.0, 0.0])])

_S2_trixel = Trixel(10, [np.array([-1.0, 0.0, 0.0]),
                         np.array([0.0, 0.0, -1.0]),
                         np.array([0.0, -1.0, 0.0])])

_S3_trixel = Trixel(11, [np.array([0.0, -1.0, 0.0]),
                         np.array([0.0, 0.0, -1.0]),
                         np.array([1.0, 0.0, 0.0])])

basic_trixels = {'N0': _N0_trixel,
                 'N1': _N1_trixel,
                 'N2': _N2_trixel,
                 'N3': _N3_trixel,
                 'S0': _S0_trixel,
                 'S1': _S1_trixel,
                 'S2': _S2_trixel,
                 'S3': _S3_trixel}


def trixelFromLabel(label):
    label_0 = label
    tree = []
    reduced_label = label
    while reduced_label > 0:
        reduced_label = label >> 2
        d_label = label - (reduced_label << 2)
        tree.append(d_label)
        label = reduced_label

    tree.reverse()

    ans = None

    if tree[0] == 3:
        if tree[1] == 0:
            ans = _N0_trixel
        elif tree[1] == 1:
            ans = _N1_trixel
        elif tree[1] == 2:
            ans = _N2_trixel
        elif tree[1] == 3:
            ans = _N3_trixel
    elif tree[0] == 2:
        if tree[1] == 0:
            ans = _S0_trixel
        elif tree[1] == 1:
            ans = _S1_trixel
        elif tree[1] == 2:
            ans = _S2_trixel
        elif tree[1] == 3:
            ans = _S3_trixel

    if ans is None:
        raise RuntimeError("Unable to find trixel for id %d\n %s"
                           % (label_0, str(tree)))

    for ix in range(2, len(tree)):
        ans = ans.get_child(tree[ix])

    return ans


def _iterateTrixelFinder(pt, parent, max_level):
    children = parent.get_children()
    for child in children:
        if child.contains_pt(pt):
            if child.level == max_level:
                return child.label
            else:
                return _iterateTrixelFinder(pt, child, max_level)

def findHtmId(ra, dec, max_level):

    raRad = np.radians(ra)
    decRad = np.radians(dec)
    pt = cartesianFromSpherical(raRad, decRad)

    if _S0_trixel.contains_pt(pt):
        parent = _S0_trixel
    elif _S1_trixel.contains_pt(pt):
        parent = _S1_trixel
    elif _S2_trixel.contains_pt(pt):
        parent = _S2_trixel
    elif _S3_trixel.contains_pt(pt):
        parent = _S3_trixel
    elif _N0_trixel.contains_pt(pt):
        parent = _N0_trixel
    elif _N1_trixel.contains_pt(pt):
        parent = _N1_trixel
    elif _N2_trixel.contains_pt(pt):
        parent = _N2_trixel
    elif _N3_trixel.contains_pt(pt):
        parent = _N3_trixel
    else:
        raise RuntimeError("could not find parent Trixel")

    return _iterateTrixelFinder(pt, parent, max_level)
