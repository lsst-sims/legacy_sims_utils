import os
import numpy
import unittest
import palpy as palpy
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils

def controlRaDecToAltAz(raRad, decRad, longRad, latRad, mjd):
    """
    Converts RA and Dec to altitude and azimuth

    @param [in] raRad is the RA in radians

    @param [in] decRad is the Dec in radians

    @param [in] longRad is the longitude of the observer in radians
    (positive east of the prime meridian)

    @param [in[ latRad is the latitude of the observer in radians
    (positive north of the equator)

    @param [in] mjd is the universal time expressed as an MJD

    @param [out] altitude in radians

    @param [out[ azimuth in radians

    see: http://www.stargazing.net/kepler/altaz.html#twig04
    """
    lst = utils.calcLmstLast(mjd, longRad)
    last = lst[1]
    haRad = numpy.radians(last*15.) - raRad
    sinDec = numpy.sin(decRad)
    cosLat = numpy.cos(latRad)
    sinLat = numpy.sin(latRad)
    sinAlt = sinDec*sinLat+numpy.cos(decRad)*cosLat*numpy.cos(haRad)
    altRad = numpy.arcsin(sinAlt)
    azRad = numpy.arccos((sinDec - sinAlt*sinLat)/(numpy.cos(altRad)*cosLat))
    azRadOut = numpy.where(numpy.sin(haRad)>=0.0, 2.0*numpy.pi-azRad, azRad)
    return altRad, azRadOut

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
        longList = numpy.array([1.2, 1.4])
        latFloat = 0.5
        latList = numpy.array([0.5, 0.6])

        raFloat = 1.1
        raList = numpy.array([0.2, 0.3])

        decFloat = 1.1
        decList = numpy.array([0.2, 0.3])

        self.assertRaises(RuntimeError, utils.calcLmstLast, mjdFloat, longList)
        self.assertRaises(RuntimeError, utils.calcLmstLast, mjd3, longList)
        ans = utils.calcLmstLast(mjdFloat, longFloat)
        ans = utils.calcLmstLast(mjd2, longList)

        self.assertRaises(RuntimeError, utils.raDecToAltAzPa, raList, decList, longList, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils.raDecToAltAzPa, raList, decList, longFloat, latList, mjd2)
        self.assertRaises(RuntimeError, utils.raDecToAltAzPa, raList, decFloat, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils.raDecToAltAzPa, raFloat, decList, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils.raDecToAltAzPa, raFloat, decFloat, longFloat, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils.raDecToAltAzPa, raList, decList, longFloat, latFloat, mjd3)
        ans = utils.raDecToAltAzPa(raFloat, decFloat, longFloat, latFloat, mjdFloat)
        ans = utils.raDecToAltAzPa(raList, decList, longFloat, latFloat, mjdFloat)
        ans = utils.raDecToAltAzPa(raList, decList, longFloat, latFloat, mjd2)

        self.assertRaises(RuntimeError, utils.altAzToRaDec, raList, decList, longList, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils.altAzToRaDec, raList, decList, longFloat, latList, mjd2)
        self.assertRaises(RuntimeError, utils.altAzToRaDec, raList, decFloat, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils.altAzToRaDec, raFloat, decList, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils.altAzToRaDec, raFloat, decFloat, longFloat, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils.altAzToRaDec, raList, decList, longFloat, latFloat, mjd3)
        ans = utils.altAzToRaDec(raFloat, decFloat, longFloat, latFloat, mjdFloat)
        ans = utils.altAzToRaDec(raList, decList, longFloat, latFloat, mjdFloat)
        ans = utils.altAzToRaDec(raList, decList, longFloat, latFloat, mjd2)


    def testEquationOfEquinoxes(self):

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

        gmst, gast = utils.calcGmstGast(self.mjd)
        ll = [1.2, 2.2]
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

        #test non-vectorized version
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


    def testRaDecToAltAz(self):

        numpy.random.seed(32)
        ra = numpy.random.sample(len(self.mjd))*2.0*numpy.pi
        dec = (numpy.random.sample(len(self.mjd))-0.5)*numpy.pi
        longitude = 1.467
        latitude = -0.234
        controlAlt, controlAz = controlRaDecToAltAz(ra, dec, \
                                                    longitude, latitude, \
                                                    self. mjd)

        testAlt, testAz, testPa = utils.raDecToAltAzPa(ra, dec, \
                                                       longitude, latitude, \
                                                       self.mjd)
        self.assertTrue(numpy.abs(testAz - controlAz).max() < self.tolerance)
        self.assertTrue(numpy.abs(testAlt - controlAlt).max() < self.tolerance)

        #test non-vectorized version
        for r,d in zip(ra, dec):
            controlAlt, controlAz = controlRaDecToAltAz(r, d, longitude, latitude, self.mjd[0])
            testAlt, testAz, testPa = utils.raDecToAltAzPa(r, d, longitude, latitude, self.mjd[0])
            self.assertTrue(numpy.abs(testAz - controlAz) < self.tolerance)
            self.assertTrue(numpy.abs(testAlt - controlAlt) < self.tolerance)

    def testAltAzToRaDec(self):
        numpy.random.seed(32)
        raIn = numpy.random.sample(len(self.mjd))*2.0*numpy.pi
        decIn = (numpy.random.sample(len(self.mjd))-0.5)*numpy.pi
        longitude = 1.467
        latitude = -0.234
        lst, last = utils.calcLmstLast(self.mjd, longitude)
        hourAngle = numpy.radians(last*15.0) - raIn
        controlAz, azd, azdd, \
        controlAlt, eld, eldd, \
        pa, pad, padd = palpy.altazVector(hourAngle, decIn, latitude)

        raOut, decOut, = utils.altAzToRaDec(controlAlt, controlAz,
                                           longitude, latitude, self.mjd)

        self.assertTrue(numpy.abs(numpy.cos(raOut) - numpy.cos(raIn)).max() < self.tolerance)
        self.assertTrue(numpy.abs(numpy.sin(raOut) - numpy.sin(raIn)).max() < self.tolerance)
        self.assertTrue(numpy.abs(numpy.cos(decOut) - numpy.cos(decIn)).max() < self.tolerance)
        self.assertTrue(numpy.abs(numpy.sin(decOut) - numpy.sin(decIn)).max() < self.tolerance)

        #test non-vectorized version
        for alt, az, r, d, m in zip(controlAlt, controlAz, raIn, decIn, self.mjd):
            raOut, decOut = utils.altAzToRaDec(alt, az, longitude, latitude, m)
            self.assertTrue(numpy.abs(numpy.cos(raOut) - numpy.cos(r)) < self.tolerance)
            self.assertTrue(numpy.abs(numpy.sin(raOut) - numpy.sin(r)) < self.tolerance)
            self.assertTrue(numpy.abs(numpy.cos(decOut) - numpy.cos(d)) < self.tolerance)
            self.assertTrue(numpy.abs(numpy.sin(decOut) - numpy.sin(d)) < self.tolerance)


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
