import unittest
import numpy as np
import palpy
import lsst.utils.tests as utilsTests
import lsst.sims.utils.CompoundCoordinateTransformations as utils
from lsst.sims.utils import arcsecFromRadians, haversine

def controlAltAzFromRaDec(raRad, decRad, longRad, latRad, mjd):
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
    haRad = np.radians(last*15.) - raRad
    sinDec = np.sin(decRad)
    cosLat = np.cos(latRad)
    sinLat = np.sin(latRad)
    sinAlt = sinDec*sinLat+np.cos(decRad)*cosLat*np.cos(haRad)
    altRad = np.arcsin(sinAlt)
    azRad = np.arccos((sinDec - sinAlt*sinLat)/(np.cos(altRad)*cosLat))
    azRadOut = np.where(np.sin(haRad)>=0.0, 2.0*np.pi-azRad, azRad)
    return altRad, azRadOut


class CompoundCoordinateTransformationsTests(unittest.TestCase):

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
        longList = np.array([1.2, 1.4])
        latFloat = 0.5
        latList = np.array([0.5, 0.6])

        raFloat = 1.1
        raList = np.array([0.2, 0.3])

        decFloat = 1.1
        decList = np.array([0.2, 0.3])

        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raList, decList, longList, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raList, decList, longFloat, latList, mjd2)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raList, decFloat, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raFloat, decList, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raFloat, decFloat, longFloat, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raList, decList, longFloat, latFloat, mjd3)
        ans = utils._altAzPaFromRaDec(raFloat, decFloat, longFloat, latFloat, mjdFloat)
        ans = utils._altAzPaFromRaDec(raList, decList, longFloat, latFloat, mjdFloat)
        ans = utils._altAzPaFromRaDec(raList, decList, longFloat, latFloat, mjd2)

        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raList, decList, longList, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raList, decList, longFloat, latList, mjd2)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raList, decFloat, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raFloat, decList, longFloat, latFloat, mjdFloat)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raFloat, decFloat, longFloat, latFloat, mjd2)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raList, decList, longFloat, latFloat, mjd3)
        ans = utils._raDecFromAltAz(raFloat, decFloat, longFloat, latFloat, mjdFloat)
        ans = utils._raDecFromAltAz(raList, decList, longFloat, latFloat, mjdFloat)
        ans = utils._raDecFromAltAz(raList, decList, longFloat, latFloat, mjd2)


    def test_raDecFromAltAz(self):
        """
        Test conversion of Alt, Az to Ra, Dec
        """
        np.random.seed(32)
        raIn = np.random.sample(len(self.mjd))*2.0*np.pi
        decIn = (np.random.sample(len(self.mjd))-0.5)*np.pi
        longitude = 1.467
        latitude = -0.234
        lst, last = utils.calcLmstLast(self.mjd, longitude)
        hourAngle = np.radians(last*15.0) - raIn
        controlAz, azd, azdd, \
        controlAlt, eld, eldd, \
        pa, pad, padd = palpy.altazVector(hourAngle, decIn, latitude)

        raOut, decOut, = utils._raDecFromAltAz(controlAlt, controlAz,
                                           longitude, latitude, self.mjd)

        self.assertTrue(np.abs(np.cos(raOut) - np.cos(raIn)).max() < self.tolerance)
        self.assertTrue(np.abs(np.sin(raOut) - np.sin(raIn)).max() < self.tolerance)
        self.assertTrue(np.abs(np.cos(decOut) - np.cos(decIn)).max() < self.tolerance)
        self.assertTrue(np.abs(np.sin(decOut) - np.sin(decIn)).max() < self.tolerance)

        #test non-vectorized version
        for alt, az, r, d, m in zip(controlAlt, controlAz, raIn, decIn, self.mjd):
            raOut, decOut = utils._raDecFromAltAz(alt, az, longitude, latitude, m)
            self.assertTrue(np.abs(np.cos(raOut) - np.cos(r)) < self.tolerance)
            self.assertTrue(np.abs(np.sin(raOut) - np.sin(r)) < self.tolerance)
            self.assertTrue(np.abs(np.cos(decOut) - np.cos(d)) < self.tolerance)
            self.assertTrue(np.abs(np.sin(decOut) - np.sin(d)) < self.tolerance)


    def testAltAzRADecRoundTrip(self):
        """
        Test that altAzPaFromRaDec and raDecFromAltAz really invert each other
        """

        np.random.seed(42)
        nSamples = 1000
        mjd = 58350.0

        alt_in = []
        az_in = []
        for alt in np.arange(0.0, 90.0, 10.0):
            for az in np.arange(0.0, 360.0, 10.0):
                alt_in.append(alt)
                az_in.append(az)

        alt_in = np.array(alt_in)
        az_in = np.array(az_in)

        for lon in (0.0, 90.0, 135.0):
            for lat in (60.0, 30.0, -60.0, -30.0):

                ra_in, dec_in = utils.raDecFromAltAz(alt_in, az_in, lon, lat, mjd)

                self.assertFalse(np.isnan(ra_in).any())
                self.assertFalse(np.isnan(dec_in).any())

                alt_out, az_out, pa_out = utils.altAzPaFromRaDec(ra_in, dec_in, lon, lat, mjd)

                self.assertFalse(np.isnan(pa_out).any())

                for alt_c, az_c, alt_t, az_t in \
                    zip(np.radians(alt_in), np.radians(az_in), np.radians(alt_out), np.radians(az_out)):

                    distance = arcsecFromRadians(haversine(az_c, alt_c, az_t, alt_t))
                    if az_c<0.01 or az_c>3.14159:
                        self.assertAlmostEqual(distance, 0.0, 2)
                    else:
                        self.assertAlmostEqual(distance, 0.0, 8)


    def testAltAzFromRaDec(self):
        """
        Test conversion from RA, Dec to Alt, Az
        """

        np.random.seed(32)
        ra = np.random.sample(len(self.mjd))*2.0*np.pi
        dec = (np.random.sample(len(self.mjd))-0.5)*np.pi
        longitude = 1.467
        latitude = -0.234
        controlAlt, controlAz = controlAltAzFromRaDec(ra, dec, \
                                                    longitude, latitude, \
                                                    self. mjd)

        #verify parallactic angle against an expression from
        #http://www.astro.washington.edu/groups/APO/Mirror.Motions/Feb.2000.Image.Jumps/report.html#Image%20motion%20directions
        #
        lmst, last = utils.calcLmstLast(self.mjd, longitude)
        hourAngle = np.radians(last*15.0) - ra
        controlSinPa = np.sin(hourAngle)*np.cos(latitude)/np.cos(controlAlt)

        testAlt, testAz, testPa = utils._altAzPaFromRaDec(ra, dec, \
                                                       longitude, latitude, \
                                                       self.mjd)

        self.assertTrue(np.abs(testAz - controlAz).max() < self.tolerance)
        self.assertTrue(np.abs(testAlt - controlAlt).max() < self.tolerance)
        self.assertTrue(np.abs(np.sin(testPa) - controlSinPa).max() < self.tolerance)

        #test non-vectorized version
        for r,d,m in zip(ra, dec, self.mjd):
            controlAlt, controlAz = controlAltAzFromRaDec(r, d, longitude, latitude, m)
            testAlt, testAz, testPa = utils._altAzPaFromRaDec(r, d, longitude, latitude, m)
            lmst, last = utils.calcLmstLast(m, longitude)
            hourAngle = np.radians(last*15.0) - r
            controlSinPa = np.sin(hourAngle)*np.cos(latitude)/np.cos(controlAlt)
            self.assertTrue(np.abs(testAz - controlAz) < self.tolerance)
            self.assertTrue(np.abs(testAlt - controlAlt) < self.tolerance)
            self.assertTrue(np.abs(np.sin(testPa) - controlSinPa) < self.tolerance)




def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(CompoundCoordinateTransformationsTests)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
