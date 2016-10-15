import unittest
from lsst.sims.utils.htmModule import TrixelFinder
import time
import numpy as np

from lsst.sims.utils import sphericalFromCartesian

class TrixelFinderTest(unittest.TestCase):

    longMessage = True

    def setUp(self):
        self.finder = TrixelFinder()

    def check_pt(self, pt, answer):
        ra, dec = sphericalFromCartesian(pt)
        ii = self.finder.findHtmId(np.degrees(ra), np.degrees(dec), 3)
        binary = '{0:b}'.format(ii)
        self.assertEqual(binary, answer)


    def test_trixel_finding(self):
        epsilon = 1.0e-6
        dx = np.array([epsilon, 0.0, 0.0])
        dy = np.array([0.0, epsilon, 0.0])
        dz = np.array([0.0, 0.0, epsilon])

        xx = np.array([1.0, 0.0, 0.0])
        yy = np.array([0.0, 1.0, 0.0])
        zz = np.array([0.0, 0.0, 1.0])

        pt = xx + dy + dz
        # N320
        self.check_pt(pt, '11111000')

        pt = xx - dy + dz
        # N000
        self.check_pt(pt, '11000000')

        pt = xx - dy - dz
        # S320
        self.check_pt(pt, '10111000')

        pt = yy + dx + dz
        # N300
        self.check_pt(pt, '11110000')

        pt = yy - dx + dz
        # N220
        self.check_pt(pt, '11101000')

        pt = yy - dx - dz
        # S100
        self.check_pt(pt, '10010000')

        pt = zz + dy + dx
        # N310
        self.check_pt(pt, '11110100')

        pt = zz - dy + dx
        # N010
        self.check_pt(pt, '11000100')

        pt = zz - dy - dx
        # N110
        self.check_pt(pt, '11010100')

        pt = -xx + dz + dy
        #N200
        self.check_pt(pt, '11100000')

        pt = -xx -dz + dy
        # S120
        self.check_pt(pt, '10011000')

        pt = -xx - dz - dy
        #S200
        self.check_pt(pt, '10100000')

        pt = -yy + dx + dz
        #N020
        self.check_pt(pt, '11001000')

        pt = -yy - dx + dz
        # N100
        self.check_pt(pt, '11010000')

        pt = -yy - dx - dz
        # S220
        self.check_pt(pt, '10101000')

        pt = -zz + dx + dy
        #S010
        self.check_pt(pt, '10000100')

        pt = -zz -dx +dy
        #S110
        self.check_pt(pt, '10010100')

        pt = -zz -dx -dy
        # S210
        self.check_pt(pt, '10100100')

        pt = xx + yy + zz
        # N333
        self.check_pt(pt, '11111111')


if __name__ == "__main__":
    unittest.main()
