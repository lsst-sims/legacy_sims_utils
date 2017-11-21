import unittest
import lsst.utils.tests
from lsst.utils import getPackageDir
from lsst.sims.utils import findHtmid, trixelFromHtmid
from lsst.sims.utils import HalfSpace, Convex, basic_trixels
from lsst.sims.utils import halfSpaceFromRaDec, levelFromHtmid
from lsst.sims.utils import getAllTrixels
from lsst.sims.utils import arcsecFromRadians
import time
import numpy as np
import os

from lsst.sims.utils import sphericalFromCartesian, cartesianFromSpherical
from lsst.sims.utils import rotAboutY, rotAboutX, rotAboutZ
from lsst.sims.utils import angularSeparation, _angularSeparation


def setup_module(module):
    lsst.utils.tests.init()


def trixel_intersects_half_space(trix, hspace):
    """
    This is a brute force method to determine whether a trixel
    is inside, or at least intersects, a halfspace.
    """
    if hspace.phi > 0.25*np.pi:
        raise RuntimeError("trixel_intersects_half_space is not safe for "
                           "large HalfSpaces")

    # if any of the trixel's corners are within the
    # HalfSpace, return True
    raRad, decRad = sphericalFromCartesian(hspace.vector)
    for corner in trix.corners:
        raRad1, decRad1 = sphericalFromCartesian(corner)
        if _angularSeparation(raRad, decRad, raRad1, decRad1) < hspace.phi:
            return True

    # if the trixel contains the HalfSpace's center,
    # return True
    if trix.contains_pt(hspace.vector):
        return True

    sinphi = np.abs(np.sin(hspace.phi))

    # Iterate over each pair of corners (c1, c2).  For each pair,
    # construct a coordinate basis in which +z is in the
    # direction of c3, and +x is along the
    # unit vector defining c_i such that the angle
    # phi of c_j in the x,y plane is positive.  This coordinate
    # system is such that the trixel edge defined by c1, c2 is
    # now along the equator of the unit sphere.  Find the point
    # of closest approach of the HalfSpace's center to the equator.
    # If that point is between c1 and c2, return True.
    for i_c_1 in range(3):
        c1 = trix.corners[i_c_1]
        for i_c_2 in range(3):
            if i_c_2 <= i_c_1:
                continue
            c2 = trix.corners[i_c_2]
            i_c_3 = 3 - (i_c_1+i_c_2)
            c3 = trix.corners[i_c_3]
            assert i_c_3 != i_c_2
            assert i_c_3 != i_c_1
            assert i_c_1 != i_c_2

            z_axis = np.array([c1[1]*c2[2]-c1[2]*c2[1],
                               c2[0]*c1[2]-c1[0]*c2[2],
                               c1[0]*c2[1]-c2[0]*c1[1]])
            z_axis = z_axis/np.sqrt((z_axis**2).sum())

            if np.dot(z_axis,c3)<0.0:
                z_axis *= -1.0

            assert np.abs(1.0-np.dot(z_axis,z_axis))<1.0e-10
            assert np.abs(1.0-np.dot(c1,c1))<1.0e-10
            assert np.abs(1.0-np.dot(c2,c2))<1.0e-10
            assert np.abs(np.dot(z_axis, c1))<1.0e-10
            assert np.abs(np.dot(z_axis, c2))<1.0e-10

            # if the dot product of the center of the HalfSpace
            # with the z axis of the new coordinate system is
            # greater than the sine of the radius of the
            # halfspace, then there is no way that the halfspace
            # intersects the equator of the unit sphere in this
            # coordinate system
            if np.abs(np.dot(z_axis, hspace.vector)) > sinphi:
                continue

            x_axis = c1
            y_axis = -1.0*np.array([x_axis[1]*z_axis[2]-x_axis[2]*z_axis[1],
                                    z_axis[0]*x_axis[2]-x_axis[0]*z_axis[2],
                                    x_axis[0]*z_axis[1]-z_axis[0]*x_axis[1]])

            cos_a = np.dot(x_axis, c2)
            sin_a = np.dot(y_axis, c2)

            if sin_a < 0.0:
                x_axis = c2
                y_axis = -1.0*np.array([x_axis[1]*z_axis[2]-x_axis[2]*z_axis[1],
                                        z_axis[0]*x_axis[2]-x_axis[0]*z_axis[2],
                                        x_axis[0]*z_axis[1]-z_axis[0]*x_axis[1]])

                cos_a = np.dot(x_axis, c1)
                sin_a = np.dot(y_axis, c1)

            assert cos_a >= 0.0
            assert sin_a >= 0.0
            assert np.abs(1.0-cos_a**2-sin_a**2)<1.0e-10
            assert np.abs(np.dot(x_axis, z_axis))<1.0e-10
            assert np.abs(np.dot(x_axis, y_axis))<1.0e-10
            assert np.abs(np.dot(y_axis, z_axis))<1.0e-10

            x_center = np.dot(x_axis, hspace.vector)

            # if the x-coordinate of the HalfSpace's center is
            # negative, the HalfSpace is on the opposite side
            # of the unit sphere; ignore this pair c1, c2
            if x_center<0.0:
                continue

            y_center = np.dot(y_axis, hspace.vector)

            # tan_a is the tangent of the angle between
            # the x_axis and the other trixel corner in
            # the x, y plane
            tan_a = sin_a/cos_a

            # tan_extreme is the tangent of the angle in
            # the x, y plane defining the point of closest
            # approach of the HalfSpace's center to the
            # equator.  If this point is between c1, c2,
            # return True.
            tan_extreme = y_center/x_center
            if tan_extreme > 0.0 and tan_extreme < tan_a:
                return True

    return False

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
        hs2 = HalfSpace(vv-1.0e-4*np.array([1.0, 0.0, 0.0]), 0.1)
        self.assertNotEqual(hs1, hs2)

    def test_findAllTrixels_radius(self):
        """
        Test the method that attempts to find all of the trixels
        inside a given half space by approximating the angular
        scale of the trixels and verifying that all returned
        trixels are within radius+angular scale of the center
        of the half space.
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

    def test_findAllTrixels_brute(self):
        """
        Use the method trixel_intersects_half_space defined at the
        top of this script to verify that HalfSpace.findAllTrixels works
        """
        level = 7
        trixel_dict = getAllTrixels(level)
        all_htmid = []
        for htmid in trixel_dict.keys():
            if levelFromHtmid(htmid) == level:
                all_htmid.append(htmid)

        hspace = halfSpaceFromRaDec(36.0, 22.1, 2.0)

        # make sure that the two methods of determining if
        # a HalfSpace contains a trixel (HalfSpace.contains_trixel
        # and trixel_interects_half_space) agree
        for htmid in all_htmid:
            trix = trixel_dict[htmid]
            msg = 'offending htmid %d' % htmid
            if trixel_intersects_half_space(trix, hspace):
                self.assertNotEqual(hspace.contains_trixel(trix), 'outside',
                                    msg=msg)
            else:
                self.assertEqual(hspace.contains_trixel(trix), 'outside',
                                 msg=msg)

        trixel_limits = hspace.findAllTrixels(level)
        intersecting_htmid = set()

        # check that all of the trixels included in the limits
        # do, in fact, intersect or exist in the HalfSpace
        for lim in trixel_limits:
            for htmid in range(lim[0], lim[1]+1):
                trix = trixel_dict[htmid]
                self.assertTrue(trixel_intersects_half_space(trix, hspace))
                intersecting_htmid.add(htmid)

        # check that all of the trixels not included in the limits
        # are, in fact, outside of the HalfSpace
        self.assertLess(len(intersecting_htmid), len(all_htmid))
        self.assertGreater(len(intersecting_htmid), 0)
        for htmid in all_htmid:
            if htmid in intersecting_htmid:
                continue
            trix = trixel_dict[htmid]
            self.assertFalse(trixel_intersects_half_space(trix, hspace))


        #print(trixel_intersects_half_space(trix, hspace))
        #print(hspace.contains_trixel(trix))


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


    def test_trixel_from_htmid(self):
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

    def test_trixel_eq_ne(self):
        """
        Test thatthe __eq__ and __ne__ operators on the Trixel class work
        """
        t1 = trixelFromHtmid(8*16+1)
        t2 = trixelFromHtmid(8*16+1)
        self.assertEqual(t1, t2)
        t3 = trixelFromHtmid(8*16+3)
        self.assertNotEqual(t1, t3)
        self.assertTrue(t1==t2)
        self.assertFalse(t1==t3)
        self.assertTrue(t1!=t3)
        self.assertFalse(t2==t3)
        self.assertTrue(t2!=t3)

    def test_get_all_trixels(self):
        """
        Test method to get all trixels up to a certain level
        """
        max_level = 5
        n_trixel_per_level = {}
        n_trixel_per_level[0] = 0
        for level in range(1,max_level+1):
            n_trixel_per_level[level] = 8*(4**(level-1))

        trixel_dict = getAllTrixels(max_level)
        n_found = {}
        for level in range(max_level+1):
            n_found[level] = 0

        for htmid in trixel_dict:
            level = levelFromHtmid(htmid)
            n_found[level] += 1

        # verify that the correct number of trixels were
        # found per level
        for level in n_found:
            msg = 'failed on level %d' % level
            self.assertEqual(n_found[level], n_trixel_per_level[level],
                             msg=msg)

        # make sure no trixels were duplicated
        self.assertEqual(len(np.unique(list(trixel_dict.keys()))),
                         len(trixel_dict))

        for htmid in trixel_dict.keys():
            level = levelFromHtmid(htmid)
            self.assertLessEqual(level, max_level)
            self.assertGreaterEqual(level, 1)
            t0 = trixelFromHtmid(htmid)
            self.assertEqual(t0, trixel_dict[htmid])

    def test_trixel_bounding_circle(self):
        """
        Verify that the trixel's bounding_circle method returns
        a circle that contains all of the corners of the
        trixel
        """
        rng = np.random.RandomState(142)
        n_test_cases = 5
        for i_test in range(n_test_cases):
            htmid = (13<<6)+rng.randint(1,2**6-1)
            trixel = trixelFromHtmid(htmid)
            bounding_circle = trixel.bounding_circle
            ra_0, dec_0 = sphericalFromCartesian(bounding_circle[0])
            ra_list = []
            dec_list = []
            for cc in trixel.corners:
                ra, dec = sphericalFromCartesian(cc)
                ra_list.append(ra)
                dec_list.append(dec)
            ra_list = np.array(ra_list)
            dec_list = np.array(dec_list)
            distance = _angularSeparation(ra_0, dec_0,
                                          ra_list, dec_list)
            distance = arcsecFromRadians(distance)
            radius = arcsecFromRadians(bounding_circle[2])
            self.assertLessEqual(distance.max()-radius,1.0e-8)
            self.assertLess(np.abs(distance.max()-radius), 1.0e-8)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
