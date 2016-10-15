from lsst.sims.utils import cartesianFromSpherical
import numpy as np

class TrixelFinder(object):

    def _trixelContains(self, corners, pt):
        """
        Corners in ccw order from lower left hand corner (v0)
        """
        if np.dot(np.cross(corners[0], corners[1]), pt)>0.0:
            if np.dot(np.cross(corners[1], corners[2]), pt)>0.0:
                if np.dot(np.cross(corners[2], corners[0]), pt)>0.0:
                    return True

        return False


    def _iterateTrixel(self):
        self.current_level += 1
        self.label = self.label << 2

        self.w0 = self.corners[1]+self.corners[2]
        self.w0 = self.w0/np.sqrt(np.power(self.w0, 2).sum())
        self.w1 = self.corners[0]+self.corners[2]
        self.w1 = self.w1/np.sqrt(np.power(self.w1, 2).sum())
        self.w2 = self.corners[0]+self.corners[1]
        self.w2 = self.w2/np.sqrt(np.power(self.w2, 2).sum())

        if self._trixelContains([self.corners[0], self.w2, self.w1], self.pt):
            self.corners = [self.corners[0], self.w2, self.w1]
        elif self._trixelContains([self.corners[1], self.w0, self.w2], self.pt):
            self.label += 1
            self.corners = [self.corners[1], self.w0, self.w2]
        elif self._trixelContains([self.corners[2], self.w1, self.w0], self.pt):
            self.label += 2
            self.corners= [self.corners[2], self.w1, self.w0]
        elif self._trixelContains([self.w0, self.w1, self.w2], self.pt):
            self.label += 3
            self.corners = [self.w0, self.w1, self.w2]

        if self.current_level==self.max_level:
            return self.label

        return self._iterateTrixel()


    def findHtmId(self, ra, dec, level):
        v0 = np.array([0.0, 0.0, 1.0])
        v1 = np.array([1.0, 0.0, 0.0])
        v2 = np.array([0.0, 1.0, 0.0])
        v3 = np.array([-1.0, 0.0, 0.0])
        v4 = np.array([0.0, -1.0, 0.0])
        v5 = np.array([0.0, 0.0, -1.0])

        raRad = np.radians(ra)
        decRad = np.radians(dec)
        self.pt = cartesianFromSpherical(raRad, decRad)

        if self._trixelContains([v1, v5, v2], self.pt):
            self.label = 8 # S0
            self.corners = [v1, v5, v2]
        elif self._trixelContains([v2, v5, v3], self.pt):
            self.label = 9 # S1
            self.corners = [v2, v5, v3]
        elif self._trixelContains([v3, v5, v4], self.pt):
            self.label = 10 # S2
            self.corners = [v3, v5, v4]
        elif self._trixelContains([v4, v5, v1], self.pt):
            self.label = 11 # S3
            self.corners = [v4, v5, v1]
        elif self._trixelContains([v1, v0, v4], self.pt):
            self.label = 12 # N0
            self.corners = [v1, v0, v4]
        elif self._trixelContains([v4, v0, v3], self.pt):
            self.label = 13 # N1
            self.corners = [v4, v0, v3]
        elif self._trixelContains([v3, v0, v2], self.pt):
            self.label = 14 # N2
            self.corners = [v3, v0, v2]
        elif self._trixelContains([v2, v0, v1], self.pt):
            self.label = 15 # N3
            self.corners = [v2, v0, v1]

        self.current_level = 1
        self.max_level = level
        return self._iterateTrixel()
