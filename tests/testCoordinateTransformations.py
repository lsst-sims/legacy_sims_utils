import os
import numpy
import unittest
import palpy as palpy
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils
from lsst.sims.utils import Site


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
    deltaPsiHours = -0.000319*numpy.sin(numpy.radians(omegaDegrees)) \
                    - 0.000024 * numpy.sin(2.0*numpy.radians(Ldegrees))
    epsilonDegrees = 23.4393 - 0.0000004*D
    return (deltaPsiHours/24.0)*2.0*numpy.pi*numpy.cos(numpy.radians(epsilonDegrees))

def controlCalcGmstGast(mjd):
    #From http://aa.usno.navy.mil/faq/docs/GAST.php Nov. 9 2013
    mjdConv = 2400000.5
    jd2000 = 2451545.0
    mjd_o = numpy.floor(mjd)
    jd = mjd + mjdConv
    jd_o = mjd_o + mjdConv
    h = 24.*(jd-jd_o)
    d = jd - jd2000
    d_o = jd_o - jd2000
    t = d/36525.
    gmst = 6.697374558 + 0.06570982441908*d_o + 1.00273790935*h + 0.000026*t**2
    gast = gmst + 24.0*utils.equationOfEquinoxes(mjd)/(2.0*numpy.pi)
    gmst %= 24.
    gast %= 24.
    return gmst, gast

class testCoordinateTransformations(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(32)
        ntests = 100
        self.mjd = 57087.0 - 1000.0*(numpy.random.sample(ntests)-0.5)
        self.tolerance = 1.0e-5

    def testExceptions(self):
        """
        Test to make sure that methods complain when incorrect data types are passed.
        """
        mjdFloat = 52000.0
        mjd2 = numpy.array([52000.0, 53000.0])
        mjd3 = numpy.array([53000.0, 53000.0, 54000.0])

        longFloat = 1.2
        longArr = numpy.array([1.2, 1.4])

        self.assertRaises(RuntimeError, utils.calcLmstLast, mjdFloat, longArr)
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjd3, longArr)
        self.assertRaises(RuntimeError, utils.calcLmstLast, list(mjd2), longArr)
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjd2, list(longArr))
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjdFloat, longArr)
        ans = utils.calcLmstLast(mjd2, longFloat)
        ans = utils.calcLmstLast(mjdFloat, longFloat)
        ans = utils.calcLmstLast(mjd2, longArr)


    def testEquationOfEquinoxes(self):
        """
        Test equation of equninoxes calculation
        """

        #test vectorized version
        control = controlEquationOfEquinoxes(self.mjd)
        test = utils.equationOfEquinoxes(self.mjd)
        self.assertTrue(numpy.abs(test-control).max() < self.tolerance)

        #test non-vectorized version
        for mm in self.mjd:
            control = controlEquationOfEquinoxes(mm)
            test = utils.equationOfEquinoxes(mm)
            self.assertTrue(numpy.abs(test-control) < self.tolerance)

    def testGmstGast(self):
        """
        Test calculation of Greenwich mean and apparent sidereal times
        """

        controlGmst, controlGast = controlCalcGmstGast(self.mjd)
        testGmst, testGast = utils.calcGmstGast(self.mjd)
        self.assertTrue(numpy.abs(testGmst - controlGmst).max() < self.tolerance)
        self.assertTrue(numpy.abs(testGast - controlGast).max() < self.tolerance)

        #test non-vectorized version
        for mm in self.mjd:
            controlGmst, controlGast = controlCalcGmstGast(mm)
            testGmst, testGast = utils.calcGmstGast(mm)
            self.assertTrue(numpy.abs(testGmst - controlGmst) < self.tolerance)
            self.assertTrue(numpy.abs(testGast - controlGast) < self.tolerance)

    def testLmstLast(self):
        """
        Test calculation of local mean and apparent sidereal time
        """

        gmst, gast = utils.calcGmstGast(self.mjd)
        ll = [1.2, 2.2]

        # test passing a float for longitude and a numpy array for mjd
        for longitude in ll:
            hours = numpy.degrees(longitude)/15.0
            if hours > 24.0:
                hours -= 24.0
            controlLmst = gmst + hours
            controlLast = gast + hours
            controlLmst %= 24.0
            controlLast %= 24.0
            testLmst, testLast = utils.calcLmstLast(self.mjd, longitude)
            self.assertTrue(numpy.abs(testLmst - controlLmst).max() < self.tolerance)
            self.assertTrue(numpy.abs(testLast - controlLast).max() < self.tolerance)
            self.assertIsInstance(testLmst, numpy.ndarray)
            self.assertIsInstance(testLast, numpy.ndarray)

        # test passing two floats
        for longitude in ll:
            for mm in self.mjd:
                gmst, gast = utils.calcGmstGast(mm)
                hours = numpy.degrees(longitude)/15.0
                if hours > 24.0:
                    hours -= 24.0
                controlLmst = gmst + hours
                controlLast = gast + hours
                controlLmst %= 24.0
                controlLast %= 24.0
                testLmst, testLast = utils.calcLmstLast(mm, longitude)
                self.assertTrue(numpy.abs(testLmst - controlLmst) < self.tolerance)
                self.assertTrue(numpy.abs(testLast - controlLast) < self.tolerance)
                self.assertIsInstance(testLmst, numpy.float)
                self.assertIsInstance(testLast, numpy.float)

        # test passing two numpy arrays
        ll = numpy.random.random_sample(len(self.mjd))*2.0*numpy.pi
        testLmst, testLast = utils.calcLmstLast(self.mjd, ll)
        self.assertIsInstance(testLmst, numpy.ndarray)
        self.assertIsInstance(testLast, numpy.ndarray)
        for ix, (longitude, mm) in enumerate(zip(ll, self.mjd)):
            controlLmst, controlLast = utils.calcLmstLast(mm, longitude)
            self.assertAlmostEqual(controlLmst, testLmst[ix], 10)
            self.assertAlmostEqual(controlLast, testLast[ix], 10)


    def test_galacticFromEquatorial(self):

        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)

        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        ra[1]=8.693375673649429425e-01
        dec[1]=1.038086165642298164e+00
        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01

        output=utils._galacticFromEquatorial(ra,dec)

        self.assertAlmostEqual(output[0][0],3.452036693523627964e+00,6)
        self.assertAlmostEqual(output[1][0],8.559512505657201897e-01,6)
        self.assertAlmostEqual(output[0][1],2.455968474619387720e+00,6)
        self.assertAlmostEqual(output[1][1],3.158563770667878468e-02,6)
        self.assertAlmostEqual(output[0][2],2.829585540991265358e+00,6)
        self.assertAlmostEqual(output[1][2],-6.510790587552289788e-01,6)

    def test_equatorialFromGalactic(self):

        lon=numpy.zeros((3),dtype=float)
        lat=numpy.zeros((3),dtype=float)

        lon[0]=3.452036693523627964e+00
        lat[0]=8.559512505657201897e-01
        lon[1]=2.455968474619387720e+00
        lat[1]=3.158563770667878468e-02
        lon[2]=2.829585540991265358e+00
        lat[2]=-6.510790587552289788e-01

        output=utils._equatorialFromGalactic(lon,lat)

        self.assertAlmostEqual(output[0][0],2.549091039839124218e+00,6)
        self.assertAlmostEqual(output[1][0],5.198752733024248895e-01,6)
        self.assertAlmostEqual(output[0][1],8.693375673649429425e-01,6)
        self.assertAlmostEqual(output[1][1],1.038086165642298164e+00,6)
        self.assertAlmostEqual(output[0][2],7.740864769302191473e-01,6)
        self.assertAlmostEqual(output[1][2],2.758053025017753179e-01,6)


    def testCartesianFromSpherical(self):
        arg1=2.19911485751
        arg2=5.96902604182
        output=utils.cartesianFromSpherical(arg1,arg2)

        vv=numpy.zeros((3),dtype=float)
        vv[0]=numpy.cos(arg2)*numpy.cos(arg1)
        vv[1]=numpy.cos(arg2)*numpy.sin(arg1)
        vv[2]=numpy.sin(arg2)

        self.assertAlmostEqual(output[0],vv[0],7)
        self.assertAlmostEqual(output[1],vv[1],7)
        self.assertAlmostEqual(output[2],vv[2],7)


    def testSphericalFromCartesian(self):
        """
        Note that xyz[i][j] is the ith component of the jth vector

        Each column of xyz is a vector
        """
        numpy.random.seed(42)
        nsamples = 10
        radius = numpy.random.random_sample(nsamples)*10.0
        theta = numpy.random.random_sample(nsamples)*numpy.pi-0.5*numpy.pi
        phi = numpy.random.random_sample(nsamples)*2.0*numpy.pi

        points = []
        for ix in range(nsamples):
            vv = [radius[ix]*numpy.cos(theta[ix])*numpy.cos(phi[ix]),
                  radius[ix]*numpy.cos(theta[ix])*numpy.sin(phi[ix]),
                  radius[ix]*numpy.sin(theta[ix])]

            points.append(vv)

        points = numpy.array(points)
        lon, lat = utils.sphericalFromCartesian(points)
        for ix in range(nsamples):
            self.assertAlmostEqual(numpy.cos(lon[ix]), numpy.cos(phi[ix]), 5)
            self.assertAlmostEqual(numpy.sin(lon[ix]), numpy.sin(phi[ix]), 5)
            self.assertAlmostEqual(numpy.cos(lat[ix]), numpy.cos(theta[ix]), 5)
            self.assertAlmostEqual(numpy.sin(lat[ix]), numpy.sin(theta[ix]), 5)

        for pp, th, ph in zip(points, theta, phi):
            lon, lat = utils.sphericalFromCartesian(pp)
            self.assertAlmostEqual(numpy.cos(lon), numpy.cos(ph), 5)
            self.assertAlmostEqual(numpy.sin(lon), numpy.sin(ph), 5)
            self.assertAlmostEqual(numpy.cos(lat), numpy.cos(th), 5)
            self.assertAlmostEqual(numpy.sin(lat), numpy.sin(th), 5)


    def testCartesianFromSpherical(self):
        numpy.random.seed(42)
        nsamples = 10
        theta = numpy.random.random_sample(nsamples)*numpy.pi-0.5*numpy.pi
        phi = numpy.random.random_sample(nsamples)*2.0*numpy.pi

        points = []
        for ix in range(nsamples):
            vv = [numpy.cos(theta[ix])*numpy.cos(phi[ix]),
                  numpy.cos(theta[ix])*numpy.sin(phi[ix]),
                  numpy.sin(theta[ix])]

            points.append(vv)


        points = numpy.array(points)
        lon, lat = utils.sphericalFromCartesian(points)
        outPoints = utils.cartesianFromSpherical(numpy.array(lon), numpy.array(lat))

        for pp, oo in zip(points, outPoints):
            numpy.testing.assert_array_almost_equal(pp, oo, decimal=6)

    def testHaversine(self):
        arg1 = 7.853981633974482790e-01
        arg2 = 3.769911184307751517e-01
        arg3 = 5.026548245743668986e+00
        arg4 = -6.283185307179586232e-01

        output=utils.haversine(arg1,arg2,arg3,arg4)

        self.assertAlmostEqual(output,2.162615946398791955e+00,10)



    def testRotationMatrixFromVectors(self):
        v1=numpy.zeros((3),dtype=float)
        v2=numpy.zeros((3),dtype=float)
        v3=numpy.zeros((3),dtype=float)

        v1[0]=-3.044619987218469825e-01
        v2[0]=5.982190522311925385e-01
        v1[1]=-5.473550908956383854e-01
        v2[1]=-5.573565912346714057e-01
        v1[2]=7.795545496018386755e-01
        v2[2]=-5.757495946632366079e-01

        output=utils.rotationMatrixFromVectors(v1,v2)

        for i in range(3):
            for j in range(3):
                v3[i]+=output[i][j]*v1[j]

        for i in range(3):
            self.assertAlmostEqual(v3[i],v2[i],7)

        v1 = numpy.array([1.0, 1.0, 1.0])
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
