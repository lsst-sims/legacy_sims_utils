from __future__ import division
from builtins import zip
from builtins import range
from past.utils import old_div
import numpy as np
import unittest
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils


def controlEquationOfEquinoxes(mjd):
    """
    Taken from http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] mjd is Terrestrial Time as a Modified Julian Date

    @param [out] the equation of equinoxes in radians
    """

    JD = mjd + 2400000.5
    D = JD - 2451545.0
    omegaDegrees = 125.04 - 0.052954*D
    Ldegrees = 280.47 + 0.98565*D
    deltaPsiHours = -0.000319*np.sin(np.radians(omegaDegrees)) \
                    - 0.000024 * np.sin(2.0*np.radians(Ldegrees))
    epsilonDegrees = 23.4393 - 0.0000004*D
    return (old_div(deltaPsiHours,24.0))*2.0*np.pi*np.cos(np.radians(epsilonDegrees))


def controlCalcGmstGast(mjd):
    # From http://aa.usno.navy.mil/faq/docs/GAST.php Nov. 9 2013
    mjdConv = 2400000.5
    jd2000 = 2451545.0
    mjd_o = np.floor(mjd)
    jd = mjd + mjdConv
    jd_o = mjd_o + mjdConv
    h = 24.*(jd-jd_o)
    d = jd - jd2000
    d_o = jd_o - jd2000
    t = old_div(d,36525.)
    gmst = 6.697374558 + 0.06570982441908*d_o + 1.00273790935*h + 0.000026*t**2
    gast = gmst + 24.0*utils.equationOfEquinoxes(mjd)/(2.0*np.pi)
    gmst %= 24.
    gast %= 24.
    return gmst, gast


class testCoordinateTransformations(unittest.TestCase):

    def setUp(self):
        np.random.seed(32)
        ntests = 100
        self.mjd = 57087.0 - 1000.0*(np.random.sample(ntests)-0.5)
        self.tolerance = 1.0e-5

    def testExceptions(self):
        """
        Test to make sure that methods complain when incorrect data types are passed.
        """
        mjdFloat = 52000.0
        mjd2 = np.array([52000.0, 53000.0])
        mjd3 = np.array([53000.0, 53000.0, 54000.0])

        longFloat = 1.2
        longArr = np.array([1.2, 1.4])

        self.assertRaises(RuntimeError, utils.calcLmstLast, mjdFloat, longArr)
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjd3, longArr)
        self.assertRaises(RuntimeError, utils.calcLmstLast, list(mjd2), longArr)
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjd2, list(longArr))
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjdFloat, longArr)
        utils.calcLmstLast(mjd2, longFloat)
        utils.calcLmstLast(mjdFloat, longFloat)
        utils.calcLmstLast(int(mjdFloat), longFloat)
        utils.calcLmstLast(mjdFloat, int(longFloat))
        utils.calcLmstLast(int(mjdFloat), int(longFloat))
        utils.calcLmstLast(mjd2, longArr)

    def testEquationOfEquinoxes(self):
        """
        Test equation of equninoxes calculation
        """

        # test vectorized version
        control = controlEquationOfEquinoxes(self.mjd)
        test = utils.equationOfEquinoxes(self.mjd)
        self.assertTrue(np.abs(test-control).max() < self.tolerance)

        # test non-vectorized version
        for mm in self.mjd:
            control = controlEquationOfEquinoxes(mm)
            test = utils.equationOfEquinoxes(mm)
            self.assertTrue(np.abs(test-control) < self.tolerance)

    def testGmstGast(self):
        """
        Test calculation of Greenwich mean and apparent sidereal times
        """

        controlGmst, controlGast = controlCalcGmstGast(self.mjd)
        testGmst, testGast = utils.calcGmstGast(self.mjd)
        self.assertTrue(np.abs(testGmst - controlGmst).max() < self.tolerance)
        self.assertTrue(np.abs(testGast - controlGast).max() < self.tolerance)

        # test non-vectorized version
        for mm in self.mjd:
            controlGmst, controlGast = controlCalcGmstGast(mm)
            testGmst, testGast = utils.calcGmstGast(mm)
            self.assertTrue(np.abs(testGmst - controlGmst) < self.tolerance)
            self.assertTrue(np.abs(testGast - controlGast) < self.tolerance)

    def testLmstLast(self):
        """
        Test calculation of local mean and apparent sidereal time
        """

        gmst, gast = utils.calcGmstGast(self.mjd)
        ll = [1.2, 2.2]

        # test passing a float for longitude and a numpy array for mjd
        for longitude in ll:
            hours = old_div(np.degrees(longitude),15.0)
            if hours > 24.0:
                hours -= 24.0
            controlLmst = gmst + hours
            controlLast = gast + hours
            controlLmst %= 24.0
            controlLast %= 24.0
            testLmst, testLast = utils.calcLmstLast(self.mjd, longitude)
            self.assertTrue(np.abs(testLmst - controlLmst).max() < self.tolerance)
            self.assertTrue(np.abs(testLast - controlLast).max() < self.tolerance)
            self.assertIsInstance(testLmst, np.ndarray)
            self.assertIsInstance(testLast, np.ndarray)

        # test passing two floats
        for longitude in ll:
            for mm in self.mjd:
                gmst, gast = utils.calcGmstGast(mm)
                hours = old_div(np.degrees(longitude),15.0)
                if hours > 24.0:
                    hours -= 24.0
                controlLmst = gmst + hours
                controlLast = gast + hours
                controlLmst %= 24.0
                controlLast %= 24.0
                testLmst, testLast = utils.calcLmstLast(mm, longitude)
                self.assertTrue(np.abs(testLmst - controlLmst) < self.tolerance)
                self.assertTrue(np.abs(testLast - controlLast) < self.tolerance)
                self.assertIsInstance(testLmst, np.float)
                self.assertIsInstance(testLast, np.float)

        # test passing two numpy arrays
        ll = np.random.random_sample(len(self.mjd))*2.0*np.pi
        testLmst, testLast = utils.calcLmstLast(self.mjd, ll)
        self.assertIsInstance(testLmst, np.ndarray)
        self.assertIsInstance(testLast, np.ndarray)
        for ix, (longitude, mm) in enumerate(zip(ll, self.mjd)):
            controlLmst, controlLast = utils.calcLmstLast(mm, longitude)
            self.assertAlmostEqual(controlLmst, testLmst[ix], 10)
            self.assertAlmostEqual(controlLast, testLast[ix], 10)

    def test_galacticFromEquatorial(self):

        ra = np.zeros((3), dtype=float)
        dec = np.zeros((3), dtype=float)

        ra[0] = 2.549091039839124218e+00
        dec[0] = 5.198752733024248895e-01
        ra[1] = 8.693375673649429425e-01
        dec[1] = 1.038086165642298164e+00
        ra[2] = 7.740864769302191473e-01
        dec[2] = 2.758053025017753179e-01

        glon, glat = utils._galacticFromEquatorial(ra, dec)

        self.assertIsInstance(glon, np.ndarray)
        self.assertIsInstance(glat, np.ndarray)

        self.assertAlmostEqual(glon[0], 3.452036693523627964e+00, 6)
        self.assertAlmostEqual(glat[0], 8.559512505657201897e-01, 6)
        self.assertAlmostEqual(glon[1], 2.455968474619387720e+00, 6)
        self.assertAlmostEqual(glat[1], 3.158563770667878468e-02, 6)
        self.assertAlmostEqual(glon[2], 2.829585540991265358e+00, 6)
        self.assertAlmostEqual(glat[2], -6.510790587552289788e-01, 6)

        # test passing in floats as args
        for ix, (rr, dd) in enumerate(zip(ra, dec)):
            gl, gb = utils._galacticFromEquatorial(rr, dd)
            self.assertIsInstance(rr, np.float)
            self.assertIsInstance(dd, np.float)
            self.assertIsInstance(gl, np.float)
            self.assertIsInstance(gb, np.float)
            self.assertAlmostEqual(gl, glon[ix], 10)
            self.assertAlmostEqual(gb, glat[ix], 10)

    def test_equatorialFromGalactic(self):

        lon = np.zeros((3), dtype=float)
        lat = np.zeros((3), dtype=float)

        lon[0] = 3.452036693523627964e+00
        lat[0] = 8.559512505657201897e-01
        lon[1] = 2.455968474619387720e+00
        lat[1] = 3.158563770667878468e-02
        lon[2] = 2.829585540991265358e+00
        lat[2] = -6.510790587552289788e-01

        ra, dec = utils._equatorialFromGalactic(lon, lat)

        self.assertIsInstance(ra, np.ndarray)
        self.assertIsInstance(dec, np.ndarray)

        self.assertAlmostEqual(ra[0], 2.549091039839124218e+00, 6)
        self.assertAlmostEqual(dec[0], 5.198752733024248895e-01, 6)
        self.assertAlmostEqual(ra[1], 8.693375673649429425e-01, 6)
        self.assertAlmostEqual(dec[1], 1.038086165642298164e+00, 6)
        self.assertAlmostEqual(ra[2], 7.740864769302191473e-01, 6)
        self.assertAlmostEqual(dec[2], 2.758053025017753179e-01, 6)

        # test passing in floats as args
        for ix, (ll, bb) in enumerate(zip(lon, lat)):
            rr, dd = utils._equatorialFromGalactic(ll, bb)
            self.assertIsInstance(ll, np.float)
            self.assertIsInstance(bb, np.float)
            self.assertIsInstance(rr, np.float)
            self.assertIsInstance(dd, np.float)
            self.assertAlmostEqual(rr, ra[ix], 10)
            self.assertAlmostEqual(dd, dec[ix], 10)

    def testSphericalFromCartesian(self):
        """
        Note that xyz[i][j] is the ith component of the jth vector

        Each column of xyz is a vector
        """
        np.random.seed(42)
        nsamples = 10
        radius = np.random.random_sample(nsamples)*10.0
        theta = np.random.random_sample(nsamples)*np.pi-0.5*np.pi
        phi = np.random.random_sample(nsamples)*2.0*np.pi

        points = []
        for ix in range(nsamples):
            vv = [radius[ix]*np.cos(theta[ix])*np.cos(phi[ix]),
                  radius[ix]*np.cos(theta[ix])*np.sin(phi[ix]),
                  radius[ix]*np.sin(theta[ix])]

            points.append(vv)

        points = np.array(points)
        lon, lat = utils.sphericalFromCartesian(points)
        for ix in range(nsamples):
            self.assertAlmostEqual(np.cos(lon[ix]), np.cos(phi[ix]), 5)
            self.assertAlmostEqual(np.sin(lon[ix]), np.sin(phi[ix]), 5)
            self.assertAlmostEqual(np.cos(lat[ix]), np.cos(theta[ix]), 5)
            self.assertAlmostEqual(np.sin(lat[ix]), np.sin(theta[ix]), 5)

        # test passing in the points one at a time
        for pp, th, ph in zip(points, theta, phi):
            lon, lat = utils.sphericalFromCartesian(pp)
            self.assertAlmostEqual(np.cos(lon), np.cos(ph), 5)
            self.assertAlmostEqual(np.sin(lon), np.sin(ph), 5)
            self.assertAlmostEqual(np.cos(lat), np.cos(th), 5)
            self.assertAlmostEqual(np.sin(lat), np.sin(th), 5)

    def testCartesianFromSpherical(self):
        np.random.seed(42)
        nsamples = 10
        theta = np.random.random_sample(nsamples)*np.pi-0.5*np.pi
        phi = np.random.random_sample(nsamples)*2.0*np.pi

        points = []
        for ix in range(nsamples):
            vv = [np.cos(theta[ix])*np.cos(phi[ix]),
                  np.cos(theta[ix])*np.sin(phi[ix]),
                  np.sin(theta[ix])]

            points.append(vv)

        points = np.array(points)
        lon, lat = utils.sphericalFromCartesian(points)
        outPoints = utils.cartesianFromSpherical(lon, lat)

        for pp, oo in zip(points, outPoints):
            np.testing.assert_array_almost_equal(pp, oo, decimal=6)

        # test passing in arguments as floats
        for ix, (ll, bb) in enumerate(zip(lon, lat)):
            xyz = utils.cartesianFromSpherical(ll, bb)
            self.assertIsInstance(xyz[0], np.float)
            self.assertIsInstance(xyz[1], np.float)
            self.assertIsInstance(xyz[2], np.float)
            self.assertAlmostEqual(xyz[0], outPoints[ix][0], 12)
            self.assertAlmostEqual(xyz[1], outPoints[ix][1], 12)
            self.assertAlmostEqual(xyz[2], outPoints[ix][2], 12)

    def testHaversine(self):
        arg1 = 7.853981633974482790e-01
        arg2 = 3.769911184307751517e-01
        arg3 = 5.026548245743668986e+00
        arg4 = -6.283185307179586232e-01

        output = utils.haversine(arg1, arg2, arg3, arg4)

        self.assertAlmostEqual(output, 2.162615946398791955e+00, 10)

    def testRotationMatrixFromVectors(self):
        v1 = np.zeros((3), dtype=float)
        v2 = np.zeros((3), dtype=float)
        v3 = np.zeros((3), dtype=float)

        v1[0] = -3.044619987218469825e-01
        v2[0] = 5.982190522311925385e-01
        v1[1] = -5.473550908956383854e-01
        v2[1] = -5.573565912346714057e-01
        v1[2] = 7.795545496018386755e-01
        v2[2] = -5.757495946632366079e-01

        output = utils.rotationMatrixFromVectors(v1, v2)

        for i in range(3):
            for j in range(3):
                v3[i] += output[i][j]*v1[j]

        for i in range(3):
            self.assertAlmostEqual(v3[i], v2[i], 7)

        v1 = np.array([1.0, 1.0, 1.0])
        self.assertRaises(RuntimeError, utils.rotationMatrixFromVectors, v1, v2)
        self.assertRaises(RuntimeError, utils.rotationMatrixFromVectors, v2, v1)


def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(testCoordinateTransformations)

    return unittest.TestSuite(suites)


def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
