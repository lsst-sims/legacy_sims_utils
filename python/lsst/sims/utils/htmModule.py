from lsst.sims.utils import cartesianFromSpherical
import numpy as np

__all__ = ["Trixel", "findHtmId"]

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

    def contains(self, pt):

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

    def get_children(self):

        base_child = self._label << 2

        w0 = self._corners[1]+self._corners[2]
        w0 = w0/np.sqrt(np.power(w0, 2).sum())
        w1 = self._corners[0]+self._corners[2]
        w1 = w1/np.sqrt(np.power(w1, 2).sum())
        w2 = self._corners[0]+self._corners[1]
        w2 = w2/np.sqrt(np.power(w2, 2).sum())

        t0 = Trixel(base_child, [self._corners[0], w2, w1])
        t1 = Trixel(base_child+1, [self._corners[1], w0, w2])
        t2 = Trixel(base_child+2, [self._corners[2], w1, w0])
        t3 = Trixel(base_child+3, [w0, w1, w2])

        return [t0, t1, t2, t3]

    @property
    def level(self):
        return self._level

    @property
    def label(self):
        return self._label

    @property
    def corners(self):
        return self._corners


def _iterateTrixelFinder(pt, parent, max_level):
    children = parent.get_children()
    for child in children:
        if child.contains(pt):
            if child.level == max_level:
                return child.label
            else:
                return _iterateTrixelFinder(pt, child, max_level)

def findHtmId(ra, dec, max_level):
    v0 = np.array([0.0, 0.0, 1.0])
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])
    v3 = np.array([-1.0, 0.0, 0.0])
    v4 = np.array([0.0, -1.0, 0.0])
    v5 = np.array([0.0, 0.0, -1.0])

    raRad = np.radians(ra)
    decRad = np.radians(dec)
    pt = cartesianFromSpherical(raRad, decRad)

    S0 = Trixel(8, [v1, v5, v2])
    if S0.contains(pt):
        parent = S0
    else:
        S1 = Trixel(9, [v2, v5, v3])
        if S1.contains(pt):
            parent = S1
        else:
           S2 = Trixel(10, [v3, v5, v4])
           if S2.contains(pt):
              parent = S2
           else:
               S3 = Trixel(11, [v4, v5, v1])
               if S3.contains(pt):
                   parent = S3
               else:
                   N0 = Trixel(12, [v1, v0, v4])
                   if N0.contains(pt):
                       parent = N0
                   else:
                       N1 = Trixel(13,[v4, v0, v3])
                       if N1.contains(pt):
                           parent = N1
                       else:
                           N2 = Trixel(14, [v3, v0, v2])
                           if N2.contains(pt):
                               parent = N2
                           else:
                               N3 = Trixel(15, [v2, v0, v1])
                               if N3.contains(pt):
                                   parent = N3
                               else:
                                   raise RuntimeError("could not find parent Trixel")

    return _iterateTrixelFinder(pt, parent, max_level)
