import unittest
from lsst.sims.utils import findHtmId, trixelFromLabel
from lsst.sims.utils import HalfSpace, basic_trixels
import time
import numpy as np

from lsst.sims.utils import sphericalFromCartesian, cartesianFromSpherical
from lsst.sims.utils import rotAboutY, rotAboutX, rotAboutZ

class HalfSpaceTest(unittest.TestCase):

    longMessage = True

    def test_half_space_contains_pt(self):
        hs = HalfSpace(np.array([0.0, 0.0, 1.0]), 0.1)
        nhs = HalfSpace(np.array([0.0, 0.0, 1.0]), -0.1)
        theta = np.arcsin(0.1)
        rng = np.random.RandomState(88)
        n_tests = 200
        ra_list = rng.random_sample(n_tests)*2.0*np.pi
        dec_list = rng.random_sample(n_tests)*(0.5*np.pi-theta)+theta
        for ra, dec, in zip(ra_list, dec_list):
            xyz = cartesianFromSpherical(ra, dec)
            self.assertTrue(hs.contains_pt(xyz))
            self.assertFalse(nhs.contains_pt(xyz))

        ra_list = rng.random_sample(n_tests)*2.0*np.pi
        dec_list = theta - rng.random_sample(n_tests)*(0.5*np.pi+theta)
        for ra, dec, in zip(ra_list, dec_list):
            xyz = cartesianFromSpherical(ra, dec)
            self.assertFalse(hs.contains_pt(xyz))
            self.assertTrue(nhs.contains_pt(xyz))

        hs = HalfSpace(np.array([1.0, 0.0, 0.0]), 0.2)
        nhs = HalfSpace(np.array([1.0, 0.0, 0.0]), -0.2)
        theta = np.arcsin(0.2)
        ra_list = rng.random_sample(n_tests)*2.0*np.pi
        dec_list = rng.random_sample(n_tests)*(0.5*np.pi-theta)+theta
        for ra, dec in zip(ra_list, dec_list):
            xyz_rot = cartesianFromSpherical(ra, dec)
            xyz = rotAboutY(xyz_rot, 0.5*np.pi)
            self.assertTrue(hs.contains_pt(xyz))
            self.assertFalse(nhs.contains_pt(xyz))

        ra_list = rng.random_sample(n_tests)*2.0*np.pi
        dec_list = theta - rng.random_sample(n_tests)*(0.5*np.pi+theta)
        for ra, dec, in zip(ra_list, dec_list):
            xyz_rot = cartesianFromSpherical(ra, dec)
            xyz = rotAboutY(xyz_rot, 0.5*np.pi)
            self.assertFalse(hs.contains_pt(xyz))
            self.assertTrue(nhs.contains_pt(xyz))

        vv = np.array([0.5*np.sqrt(2), -0.5*np.sqrt(2), 0.0])
        hs = HalfSpace(vv, 0.3)
        nhs = HalfSpace(vv, -0.3)
        theta = np.arcsin(0.3)
        ra_list = rng.random_sample(n_tests)*2.0*np.pi
        dec_list = rng.random_sample(n_tests)*(0.5*np.pi-theta)+theta

        for ra, dec in zip(ra_list, dec_list):
            xyz_rot = cartesianFromSpherical(ra, dec)
            xyz_rot = rotAboutX(xyz_rot, 0.5*np.pi)
            xyz = rotAboutZ(xyz_rot, 0.25*np.pi)
            self.assertTrue(hs.contains_pt(xyz))
            self.assertFalse(nhs.contains_pt(xyz))

        ra_list = rng.random_sample(n_tests)*2.0*np.pi
        dec_list = theta - rng.random_sample(n_tests)*(0.5*np.pi+theta)
        for ra, dec, in zip(ra_list, dec_list):
            xyz_rot = cartesianFromSpherical(ra, dec)
            xyz_rot = rotAboutX(xyz_rot, 0.5*np.pi)
            xyz = rotAboutZ(xyz_rot, 0.25*np.pi)
            self.assertFalse(hs.contains_pt(xyz))
            self.assertTrue(nhs.contains_pt(xyz))


    def test_halfspace_contains_trixel(self):

        # test half space that is on the equator wher N3 and S0 meet
        hs = HalfSpace(np.array([1.0, 1.0, 0.0]), 0.8)
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = 'Failed on %s' % tx
            if tx not in ('S0', 'N3'):
                self.assertEqual(status, 'outside', msg=msg)
            else:
                self.assertEqual(status, 'partial', msg=msg)

        # test halfspace that is centered on vertex where S0, S3, N0, N3 meet
        hs = HalfSpace(np.array([1.0, 0.0, 0.0]), 0.8)
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = 'Failed on %s' % tx
            if tx not in ('S0', 'S3', 'N0', 'N3'):
                self.assertEqual(status, 'outside', msg=msg)
            else:
                self.assertEqual(status, 'partial', msg=msg)

        # test halfspace fully contained in N3
        hs = HalfSpace(np.array([1.0, 1.0, 1.0]), 0.9)
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = 'Failed on %s' % tx
            if tx != 'N3':
                self.assertEqual(status, 'outside', msg=msg)
            else:
                self.assertEqual(status, 'partial', msg=msg)

        # test halfspace that totally contains N3
        ra, dec = basic_trixels['N3'].get_center()
        xyz = cartesianFromSpherical(np.radians(ra), np.radians(dec))
        hs = HalfSpace(np.array([1.0, 1.0, 1.0]), np.cos(0.31*np.pi))
        for tx in basic_trixels:
            status = hs.contains_trixel(basic_trixels[tx])
            msg = 'Failed on %s' % tx
            if tx == 'N3':
                self.assertEqual(status, 'full', msg=msg)
            elif tx in ('N1', 'N2', 'N0', 'S0', 'S1', 'S3'):
                self.assertEqual(status, 'partial', msg=msg)
            else:
                self.assertEqual(status, 'outside', msg=msg)

    def test_half_space_eq(self):
        """
        Test that __eq__() works for HalfSpace
        """
        vv = np.array([1.0, 0.9, 2.4])
        hs1 = HalfSpace(vv, 0.1)
        hs2 = HalfSpace(2.0*vv, 0.1)
        self.assertEqual(hs1, hs2)
        hs2 = HalfSpace(vv, 0.09)
        self.assertNotEqual(hs1, hs2)
        hs2 = HalfSpace(vv-1.0e-6*np.array([1.0, 0.0, 0.0]), 0.1)
        self.assertNotEqual(hs1, hs2)


class TrixelFinderTest(unittest.TestCase):

    longMessage = True

    def check_pt(self, pt, answer):
        ra, dec = sphericalFromCartesian(pt)
        ii = findHtmId(np.degrees(ra), np.degrees(dec), 3)
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


    def test_trixel_from_label(self):
        rng = np.random.RandomState(88)
        n_tests = 100
        for i_test in range(n_tests):
            pt = rng.normal(0.0, 1.0, 3)
            ra, dec = sphericalFromCartesian(pt)
            ra = np.degrees(ra)
            dec = np.degrees(dec)
            ii = findHtmId(ra, dec, 5)
            tt = trixelFromLabel(ii)
            self.assertTrue(tt.contains(ra, dec))
            tt1 = trixelFromLabel(ii-1)
            self.assertFalse(tt1.contains(ra, dec))
            tt2 = trixelFromLabel(ii+1)
            self.assertFalse(tt2.contains(ra, dec))

if __name__ == "__main__":
    unittest.main()
