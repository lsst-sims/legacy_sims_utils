import unittest
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.sims.utils import findHtmid, trixelFromHtmid
from lsst.sims.utils import HalfSpace, Convex, basic_trixels
from lsst.sims.utils import halfSpaceFromRaDec, levelFromHtmid
import time
import numpy as np
import os

from lsst.sims.utils import sphericalFromCartesian, cartesianFromSpherical
from lsst.sims.utils import rotAboutY, rotAboutX, rotAboutZ
from lsst.sims.utils import angularSeparation


def setup_module(module):
    lsst.utils.tests.init()


class HalfSpaceTest(unittest.TestCase):

    longMessage = True

    def test_half_space_contains_pt(self):
        hs = HalfSpace(np.array([0.0, 0.0, 1.0]), 0.1)
        nhs = HalfSpace(np.array([0.0, 0.0, -1.0]), -0.1)
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
        nhs = HalfSpace(np.array([-1.0, 0.0, 0.0]), -0.2)
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
        nhs = HalfSpace(-1.0*vv, -0.3)
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

    def test_findAllTrixels(self):
        """
        Test the method that attempts to find all of the trixels
        inside a given half space
        """
        level = 5

        # approximate the linear angular scale (in degrees)
        # of a trixel grid using the fact that there are
        # 8*4**(level-1) trixels in the grid as per equation 2.5 of
        #
        # https://www.microsoft.com/en-us/research/wp-content/uploads/2005/09/tr-2005-123.pdf
        angular_scale= np.sqrt(4.0*np.pi*(180.0/np.pi)**2/(8.0*4.0**(level-1)))

        ra = 43.0
        dec = 22.0
        radius = 20.0
        half_space = halfSpaceFromRaDec(ra, dec, radius)
        trixel_list = half_space.findAllTrixels(level)
        self.assertGreater(len(trixel_list), 2)

        # first, check that all of the returned trixels are
        # inside the HalfSpace
        good_htmid_list = []
        for i_limit, limits in enumerate(trixel_list):

            # verify that the tuples have been sorted by
            # htmid_min
            if i_limit>0:
                self.assertGreater(limits[0], trixel_list[i_limit-1][1])

            for htmid in range(limits[0], limits[1]+1):
                test_trixel = trixelFromHtmid(htmid)
                ra_trix, dec_trix = test_trixel.get_center()
                good_htmid_list.append(htmid)
                self.assertNotEqual(half_space.contains_trixel(test_trixel),
                                    'outside')

                # check that the returned trixels are within
                # radius+angular_scale of the center of the HalfSpace
                self.assertLess(angularSeparation(ra, dec, ra_trix, dec_trix),
                                radius+angular_scale)

        # next, verify that all of the possible trixels that
        # were not returned are outside the HalfSpace
        for base_htmid in range(8,16):
            htmid_0 = base_htmid<<2*(level-1)
            self.assertEqual(levelFromHtmid(htmid_0), level)
            for ii in range(2**(2*level-2)):
                htmid = htmid_0 + ii
                self.assertEqual(levelFromHtmid(htmid), level)
                if htmid not in good_htmid_list:
                    test_trixel = trixelFromHtmid(htmid)
                    self.assertEqual(half_space.contains_trixel(test_trixel), 'outside')
                    ra_trix, dec_trix = test_trixel.get_center()
                    self.assertGreater(angularSeparation(ra, dec, ra_trix, dec_trix),
                                       radius)

class ConvexTestCase(unittest.TestCase):

    longMessage = True

    def test_trimming_of_half_spaces(self):
        """
        Test that convex correctly pairs down list of HalfSpaces
        """

        # test when one of the HalfSpaces excludes the entire sphere
        hs_list = [HalfSpace(np.array([1.0, 1.3, 2.1]), 0.1),
                   HalfSpace(np.array([1.1, 2.1, 0.0]), 0.7),
                   HalfSpace(np.array([1.0, 0.0, 0.0]), 1.1),
                   HalfSpace(np.array([0.0, 1.0, 0.0]), -1.1)]

        conv = Convex(hs_list)
        self.assertTrue(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)

        # test case when a convex and its complement are in the convex
        hs_list = [HalfSpace(np.array([1.9, 0.1, 0.3]), 0.2),
                   HalfSpace(np.array([1.9, 0.1, 0.3]), -0.2),
                   HalfSpace(np.array([1.2, 3.1, 0.0]), 0.2)]

        conv = Convex(hs_list)
        self.assertTrue(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)

        # test case where two half spaces serve to cut off the whole
        # sphere
        hs_list = [HalfSpace(np.array([1.1, 2.1, 3.1]), 0.9),
                   HalfSpace(np.array([-1.1, -2.1, -3.1]), 0.9),
                   HalfSpace(np.array([0.0, 0.0, 0.1]), 0.01)]
        conv = Convex(hs_list)
        self.assertTrue(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)

        # test case where convex is just one half space that contains
        # the entire sphere
        hs_list = [HalfSpace(np.array([1.1, 2.3, 0.0]), -1.1)]
        conv = Convex(hs_list)
        self.assertFalse(conv.is_null)
        self.assertTrue(conv.is_whole_sphere)

        # test case where one half space is contained within another
        hs1 = HalfSpace(np.array([1.1, 0.2, 0.3]), 0.5)
        hs2 = HalfSpace(np.array([2.2, 0.4, 0.61]), 0.9) # hs2 should be in hs1
        hs3 = HalfSpace(np.array([0.0, 0.0, 1.0]), 0.01)
        hs_list = [hs1, hs2, hs3]
        conv = Convex(hs_list)
        self.assertFalse(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)
        self.assertEqual(len(conv.half_space_list), 2)
        self.assertIn(hs1, conv.half_space_list)
        self.assertIn(hs3, conv.half_space_list)

        hs_list = [hs2, hs1, hs3]
        conv = Convex(hs_list)
        self.assertFalse(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)
        self.assertEqual(len(conv.half_space_list), 2)
        self.assertIn(hs1, conv.half_space_list)
        self.assertIn(hs3, conv.half_space_list)

        # test case when two convexes are identical
        hs_list = [hs1, hs3, hs1]
        conv = Convex(hs_list)
        self.assertFalse(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)
        self.assertEqual(len(conv.half_space_list), 2)
        self.assertIn(hs1, conv.half_space_list)
        self.assertIn(hs3, conv.half_space_list)

        # test case when one convex is whole sphere
        hs4 = HalfSpace(np.array([1.1, 3.1, 0.9]), -1.1)
        hs_list = [hs1, hs3, hs4]
        conv = Convex(hs_list)
        self.assertFalse(conv.is_null)
        self.assertFalse(conv.is_whole_sphere)
        self.assertEqual(len(conv.half_space_list), 2)
        self.assertIn(hs1, conv.half_space_list)
        self.assertIn(hs3, conv.half_space_list)

        # test case of a hole in a half space
        axis = np.array([1.2, 0.9, -0.3])
        hs = HalfSpace(axis, 0.1)
        whole = HalfSpace(-1.0*axis, -0.9)
        conv = Convex([hs, whole])
        self.assertEqual(len(conv.half_space_list), 2)


    def test_roots(self):
        """
        Test that Convex finds the correct roots
        """

        v1 = np.array([0.0, 1.0, 1.0])
        hs1 = HalfSpace(v1, 0.0)
        v2 = np.array([0.0, -1.0, 1.0])
        hs2 = HalfSpace(v2, 0.0)
        conv = Convex([hs1, hs2])
        self.assertEqual(len(conv.roots), 2)
        for root in conv.roots:
            try:
                np.testing.assert_array_almost_equal(root,
                                                     np.array([1.0, 0.0, 0.0]),
                                                     10)
            except:
                np.testing.assert_array_almost_equal(root,
                                                     np.array([-1.0, 0.0, 0.0]),
                                                     10)

        self.assertAlmostEqual(np.dot(conv.roots[0], conv.roots[1]), -1.0, 10)

        xaxis = np.array([1.0, 0.0, 0.0])
        hs3 =  HalfSpace(xaxis, 0.5)
        conv = Convex([hs1, hs2, hs3])
        self.assertEqual(len(conv.roots), 3)
        np.testing.assert_array_almost_equal(conv.roots[0], xaxis, 10)
        self.assertAlmostEqual(np.dot(conv.roots[1], xaxis), np.cos(np.pi/3.0), 10)
        # take advantage of the fact that we know the order in which HalfSpaces are checked
        # for roots to make sure that the roots are at the correct angle with respect to
        # the normal vectors of the half spaces
        self.assertAlmostEqual(conv.roots[1][0], 0.5, 10)
        self.assertAlmostEqual(conv.roots[2][0], 0.5, 10)
        self.assertAlmostEqual(np.dot(conv.roots[1], v1), 0.0, 10)
        self.assertAlmostEqual(np.dot(conv.roots[2], v2), 0.0, 10)

        hs4 = HalfSpace(-1.0*xaxis, -0.9)
        conv = Convex([hs1, hs4, hs2, hs3])
        self.assertEqual(len(conv.roots), 4)

        # again, take advantage of the fact that we know the order in which
        # the roots will be found
        self.assertAlmostEqual(np.dot(conv.roots[0], xaxis), 0.9, 10)
        self.assertAlmostEqual(np.dot(conv.roots[1], xaxis), 0.5, 10)
        self.assertAlmostEqual(np.dot(conv.roots[2], xaxis), 0.9, 10)
        self.assertAlmostEqual(np.dot(conv.roots[3], xaxis), 0.5, 10)
        self.assertAlmostEqual(np.dot(conv.roots[0], v1), 0.0, 10)
        self.assertAlmostEqual(np.dot(conv.roots[1], v1), 0.0, 10)
        self.assertAlmostEqual(np.dot(conv.roots[2], v2), 0.0, 10)
        self.assertAlmostEqual(np.dot(conv.roots[3], v2), 0.0, 10)

    def test_positive_convex_contains_trixel(self):
        """
        Test the contains_trixel() method for positive/zero sign convexes
        """

        hs1 = HalfSpace(np.array([0.0, 0.0, 1.0]), 0.001)
        hs2 = HalfSpace(np.array([0.0, 1.0, 0.0]), 0.001)
        hs3 = HalfSpace(np.array([1.0, 0.0, 0.0]), 0.001)
        conv = Convex([hs1, hs2, hs3])  # only accept the positive octant
        self.assertEqual(conv.sign, 1)

        self.assertEqual(conv.contains_trixel(basic_trixels['N2']), 'outside')
        self.assertEqual(conv.contains_trixel(basic_trixels['N0']), 'outside')
        self.assertEqual(conv.contains_trixel(basic_trixels['N3']), 'partial')


class TrixelFinderTest(unittest.TestCase):

    longMessage = True

    def check_pt(self, pt, answer):
        ra, dec = sphericalFromCartesian(pt)
        ii = findHtmid(np.degrees(ra), np.degrees(dec), 3)
        binary = '{0:b}'.format(ii)
        self.assertEqual(binary, answer)

    def test_against_fatboy(self):
        """
        Test findHtmid against a random selection of stars from fatboy
        """
        dtype = np.dtype([('htmid', int), ('ra', float), ('dec', float)])
        data = np.genfromtxt(os.path.join(getPackageDir('sims_utils'), 'tests',
                                          'testData', 'htmid_test_data.txt'),
                             dtype=dtype)
        self.assertGreater(len(data), 20)
        for i_pt in range(len(data)):
            htmid_test = findHtmid(data['ra'][i_pt], data['dec'][i_pt], 21)
            self.assertEqual(htmid_test, data['htmid'][i_pt])
            level_test = levelFromHtmid(htmid_test)
            self.assertEqual(level_test, 21)

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
            ii = findHtmid(ra, dec, 5)
            tt = trixelFromHtmid(ii)
            self.assertTrue(tt.contains(ra, dec))
            tt1 = trixelFromHtmid(ii-1)
            self.assertFalse(tt1.contains(ra, dec))
            tt2 = trixelFromHtmid(ii+1)
            self.assertFalse(tt2.contains(ra, dec))


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
