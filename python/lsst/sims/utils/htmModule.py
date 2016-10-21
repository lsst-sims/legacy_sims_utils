import numpy as np
from lsst.sims.utils import cartesianFromSpherical, sphericalFromCartesian

__all__ = ["Trixel", "HalfSpace", "Convex", "findHtmId", "trixelFromLabel",
           "basic_trixels"]

_CONVEX_SIGN_POS=1
_CONVEX_SIGN_NEG=-1
_CONVEX_SIGN_ZERO=0
_CONVEX_SIGN_MIXED=2

class HalfSpace(object):

    def __init__(self, vector, length):
        self._v = vector/np.sqrt(np.power(vector, 2).sum())
        self._d = length
        if np.abs(self._d)<1.0:
            self._phi = np.arccos(self._d)  # half angular extent of the half space
        else:
            if self._d > 0.0:
                self._phi = np.pi
            else:
                self._phi = 0.0

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

        if dot_product > self._d:
            return True

        return False

    def intersects_edge(self, pt1, pt2):
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

    def intersects_bounding_circle(self, tx):

        dotproduct = np.dot(tx.bounding_circle[0], self._v)
        if np.abs(dotproduct) < 1.0:
            theta = np.arccos(np.dot(tx.bounding_circle[0], self._v))
        elif dotproduct<1.000000001:
            theta = 0.0
        elif dotproduct>-1.000000001:
            theta = np.pi
        else:
            raise RuntimeError("Dot product between unit vectors is %e" % dotproduct)

        if theta > self._phi + tx.bounding_circle[2]:
            return False

        return True

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
        if not self.intersects_bounding_circle(tx):
            return "outside"

        # need to test that the bounding circle intersect the halfspace
        # boundary

        intersection = False
        for edge in ((tx.corners[0], tx.corners[1]),
                     (tx.corners[1], tx.corners[2]),
                     (tx.corners[2], tx.corners[0])):

            if self.intersects_edge(edge[0], edge[1]):
                intersection = True
                break

        if intersection:
            return "partial"

        if tx.contains_pt(self._v):
            return "partial"

        return "outside"


class Convex(object):

    def __init__(self, half_space_list):
        """
        half_space_list is a list of HalfSpaces
        """
        self._half_space_list = half_space_list
        self._is_null = False
        self._is_whole_sphere = False
        self._trim_half_space_list()
        self._roots = []
        if (not self._is_null and not self._is_whole_sphere
            and len(self._half_space_list) > 1):

            self._find_roots()

        n_pos = 0
        n_neg = 0
        tol = 1.0e-10
        for hs in self._half_space_list:
            if hs.dd < -1.0*tol:
                n_neg += 1
            elif hs.dd > tol:
                n_pos += 1

        if n_neg > 0 and n_pos == 0:
            self._sign = _CONVEX_SIGN_NEG
        elif n_pos > 0 and n_neg == 0:
            self._sign = _CONVEX_SIGN_POS
        elif n_pos > 0 and n_neg > 0:
            self._sign = _CONVEX_SIGN_MIXED
        else:
            self._sign = _CONVEX_SIGN_ZERO

        # sort half spaces in order of size
        self._half_space_list = np.array(self._half_space_list)
        phi_arr = []
        for hs in self._half_space_list:
            phi_arr.append(hs.phi)
        phi_arr = np.array(phi_arr)
        sorted_dex = np.argsort(phi_arr)
        self._half_space_list = self._half_space_list[sorted_dex]


    def _trim_half_space_list(self):

        # check that none of the half spaces exclude the whole sphere
        for hs in self._half_space_list:
            if hs.dd > 1.0:
                self._is_null = True
                return

        # remove half spaces that are identical
        redundant_half_spaces = []
        for ix in range(len(self._half_space_list)):
            for iy in range(ix+1, len(self._half_space_list)):
                if self._half_space_list[ix]==self._half_space_list[iy]:
                    if iy not in redundant_half_spaces:
                        redundant_half_spaces.append(iy)

        redundant_half_spaces.sort(reverse=True)
        for ix in redundant_half_spaces:
            self._half_space_list.pop(ix)

        # check for half spaces that are the whole sphere
        redundant_half_spaces = []
        for ix in range(len(self._half_space_list)):
            if self._half_space_list[ix].dd < -1.0:
                if len(self._half_space_list) > 1:
                    redundant_half_spaces.append(ix)
                else:
                    self._is_whole_sphere = True
                    return

        redundant_half_spaces.sort(reverse=True)
        for ix in redundant_half_spaces:
            self._half_space_list.pop(ix)

        # check to see if two half spaces conflict with each other,
        # causing the whole convex to be null
        for ix in range(len(self._half_space_list)):
            hs1 = self._half_space_list[ix]
            for iy in range(ix+1, len(self._half_space_list)):
                hs2 = self._half_space_list[iy]
                angle_between = np.arccos(np.dot(hs1.vector, hs2.vector))
                if angle_between >= hs1.phi + hs2.phi:
                    self._is_null = True
                    return

        # look for cases where two half spaces are complements of each other
        tol = 1.0e-10
        for ix in range(len(self._half_space_list)):
            hs1 = self._half_space_list[ix]
            for iy in range(ix+1, len(self._half_space_list)):
                hs2 = self._half_space_list[iy]
                delta_dd = hs1.dd+hs2.dd
                if np.abs(delta_dd) < tol:
                    if np.abs(np.dot(hs1.vector, hs2.vector)-1.0) < tol:
                        self._is_null = True
                        return

        # look for half spaces that completely contain another
        redundant_half_spaces = []
        for ix in range(len(self._half_space_list)):
            hs1 = self._half_space_list[ix]
            for iy in range(ix+1, len(self._half_space_list)):
                hs2 = self._half_space_list[iy]
                if hs1.phi > hs2.phi:
                    bigger = hs1
                    bigger_dex = ix
                    smaller = hs2
                    smaller_dex = iy
                else:
                    bigger = hs2
                    bigger_dex = iy
                    smaller = hs1
                    smaller_dex = ix

                angle_between = np.arccos(np.dot(bigger.vector, smaller.vector))
                if bigger.phi - smaller.phi >= angle_between:
                    redundant_half_spaces.append(smaller_dex)

        redundant_half_spaces.sort(reverse=True)
        for ix in redundant_half_spaces:
            self._half_space_list.pop(ix)

    def _find_roots(self):
        """
        See section 3.5 of Szalay et al. 2005
        """
        tol = 1.0e-10
        for ix in range(len(self._half_space_list)):
            hs1 = self._half_space_list[ix]
            for iy in range(ix+1, len(self._half_space_list)):
                hs2 = self._half_space_list[iy]
                gamma = np.dot(hs1.vector, hs2.vector)
                if np.abs(gamma-1.0)<tol or np.abs(gamma+1.0)<tol:
                    break

                denom = 1.0 - gamma*gamma
                if denom <= 0.0:
                    break

                num = hs1.dd*hs1.dd + hs2.dd*hs2.dd - 2.0*gamma*hs1.dd*hs2.dd

                if denom < num:
                    break

                w_plus = np.sqrt((1.0-num/denom)/denom)
                w_minus = -1.0*w_plus

                uu = (hs1.dd - gamma*hs2.dd)/denom
                vv = (hs2.dd - gamma*hs1.dd)/denom

                cross = np.cross(hs1.vector, hs2.vector)

                vv_plus = uu*hs1.vector + vv*hs2.vector + w_plus*cross
                vv_plus_valid = True
                for iz in range(len(self._half_space_list)):
                    if iz != ix and iz != iy:
                        if not self._half_space_list[iz].contains_pt(vv_plus):
                            vv_plus_valid = False
                            break
                if vv_plus_valid:
                    self._roots.append(vv_plus)

                vv_minus = uu*hs1.vector + vv*hs2.vector + w_minus*cross
                vv_minus_valid = True
                for iz in range(len(self._half_space_list)):
                    if iz != ix and iz != iy:
                        if not self._half_space_list[iz].contains_pt(vv_minus):
                            vv_minus_valid = False
                            break
                if vv_minus_valid:
                    self._roots.append(vv_minus)

    def contains_pt(self, pt):
        """
        Test that convex contains a Cartesian pt
        """
        for hs in self._half_space_list:
            if not hs.contains_pt(pt):
                return False
        return True


    def _contains_trixel_pos(self, tx):
        """
        Test that convex contains trixel when convex is positive
        """
        if self._sign != _CONVEX_SIGN_POS and self._sign != _CONVEX_SIGN_ZERO:
            raise RuntimeError("Calling _contains_trixel_pos when sign is %d" % self._sign)

        corner_contained_in = []
        for corner in tx.corners:
            n_in = 0
            if self.contains_pt(corner):
                n_in += 1
            corner_contained_in.append(n_in)

        all_corners_in = True
        for ix in corner_contained_in:
            if ix != len(self._half_space_list):
                all_corners_in = False
                break

        if all_corners_in:
            return "full"

        for ix in corner_contained_in:
            if ix == len(self._half_space_list):
                return "partial"

        bounding_circle_intersects_all = True
        for hs in self._half_space_list:
            if not hs.intersects_bounding_circle(tx):
                bounding_circle_intersects_all = False
                break

        if not bounding_circle_intersects_all:
            return "outside"

        smallest_intersects = False
        edge_tuple = ((tx.corners[0], tx.corners[1]),
                      (tx.corners[1], tx.corners[2]),
                      (tx.corners[2], tx.corners[0]))

        for edge in edge_tuple:
            if self._half_space_list[0].intersects_edge(edge[0], edge[1]):
                smallest_intersects = True
                break

        if smallest_intersects:
            another_does_not_intersect = False
            for ix in range(1, len(self._half_space_list)):
                intersects_an_edge = False
                for edge in edge_tuple:
                    if self._half_space_list[ix].intersects_edge(edge[0], edge[1]):
                        intersects_an_edge = True
                        break
                if not intersects_an_edge:
                    another_does_not_intersect = True
                    break

            if not another_does_not_intersect:
                return "partial"

        constraint_inside = True
        for root in self._roots:
            if not tx.contains_pt(root):
                constraint_inside = False
                break

        if constraint_inside:
            return "partial"

        return "outside"

    def contains_trixel(self, tx):
        if self._is_null:
            return "outside"
        if self._is_whole_sphere:
            return "full"

        if self._sign==_CONVEX_SIGN_POS or self._sign==_CONVEX_SIGN_ZERO:
            return self._contains_trixel_pos(tx)
        elif self._sign==_CONVEX_SIGN_NEG:
            return self._contains_trixel_neg(tx)
        elif self._sign==_CONVEX_SIGN_MIXED:
            return self._contains_trixel_mixed(tx)


    @property
    def half_space_list(self):
        return self._half_space_list

    @property
    def is_null(self):
        return self._is_null

    @property
    def is_whole_sphere(self):
        return self._is_whole_sphere

    @property
    def roots(self):
        return self._roots

    @property
    def sign(self):
        return self._sign

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
