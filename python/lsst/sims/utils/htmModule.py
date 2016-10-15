import numpy as np
from lsst.sims.utils import cartesianFromSpherical, sphericalFromCartesian

__all__ = ["Trixel", "findHtmId", "trixelFromLabel"]

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


    def contains(self, ra, dec):
        """
        In degrees
        """
        xyz = cartesianFromSpherical(np.radians(ra), np.radians(dec))
        return self._contains(xyz)

    def _contains(self, pt):
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
        if child._contains(pt):
            if child.level == max_level:
                return child.label
            else:
                return _iterateTrixelFinder(pt, child, max_level)

def findHtmId(ra, dec, max_level):

    raRad = np.radians(ra)
    decRad = np.radians(dec)
    pt = cartesianFromSpherical(raRad, decRad)

    if _S0_trixel._contains(pt):
        parent = _S0_trixel
    elif _S1_trixel._contains(pt):
        parent = _S1_trixel
    elif _S2_trixel._contains(pt):
        parent = _S2_trixel
    elif _S3_trixel._contains(pt):
        parent = _S3_trixel
    elif _N0_trixel._contains(pt):
        parent = _N0_trixel
    elif _N1_trixel._contains(pt):
        parent = _N1_trixel
    elif _N2_trixel._contains(pt):
        parent = _N2_trixel
    elif _N3_trixel._contains(pt):
        parent = _N3_trixel
    else:
        raise RuntimeError("could not find parent Trixel")

    return _iterateTrixelFinder(pt, parent, max_level)
