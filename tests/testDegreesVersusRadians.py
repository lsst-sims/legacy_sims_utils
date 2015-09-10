import unittest
import numpy
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils

class testDegrees(unittest.TestCase):
    """
    Test that all the pairs of methods that deal in degrees versus
    radians agree with each other.
    """

    def setUp(self):
        numpy.random.seed(87334)
        self.raList = numpy.random.random_sample(100)*2.0*numpy.pi
        self.decList = (numpy.random.random_sample(100)-0.5)*numpy.pi
        self.lon = numpy.random.random_sample(1)[0]*2.0*numpy.pi
        self.lat = (numpy.random.random_sample(1)[0]-0.5)*numpy.pi


    def testUnitConversion(self):
        """
        Test that arcsecFromRadians, arcsecFromDegrees,
        radiansFromArcsec, and degreesFromArcsec are all
        self-consistent
        """

        radList = numpy.random.random_sample(100)*2.0*numpy.pi
        degList = numpy.degrees(radList)

        arcsecRadList = utils.arcsecFromRadians(radList)
        arcsecDegList = utils.arcsecFromDegrees(degList)

        numpy.testing.assert_array_equal(arcsecRadList, arcsecDegList)

        arcsecList = numpy.random.random_sample(100)*1.0
        radList = utils.radiansFromArcsec(arcsecList)
        degList = utils.degreesFromArcsec(arcsecList)
        numpy.testing.assert_array_equal(numpy.radians(degList), radList)


    def testGalacticFromEquatorial(self):
        raList = self.raList
        decList = self.decList

        lonRad, latRad = utils._galacticFromEquatorial(raList, decList)
        lonDeg, latDeg = utils.galacticFromEquatorial(numpy.degrees(raList),
                                                     numpy.degrees(decList))

        numpy.testing.assert_array_almost_equal(lonRad, numpy.radians(lonDeg), 10)
        numpy.testing.assert_array_almost_equal(latRad, numpy.radians(latDeg), 10)

        for ra, dec in zip(raList, decList):
            lonRad, latRad = utils._galacticFromEquatorial(ra, dec)
            lonDeg, latDeg = utils.galacticFromEquatorial(numpy.degrees(ra), numpy.degrees(dec))
            self.assertAlmostEqual(lonRad, numpy.radians(lonDeg), 10)
            self.assertAlmostEqual(latRad, numpy.radians(latDeg), 10)


    def testEquaorialFromGalactic(self):
        lonList = self.raList
        latList = self.decList

        raRad, decRad = utils._equatorialFromGalactic(lonList, latList)
        raDeg, decDeg = utils.equatorialFromGalactic(numpy.degrees(lonList),
                                                     numpy.degrees(latList))

        numpy.testing.assert_array_almost_equal(raRad, numpy.radians(raDeg), 10)
        numpy.testing.assert_array_almost_equal(decRad, numpy.radians(decDeg), 10)

        for lon, lat in zip(lonList, latList):
            raRad, decRad = utils._equatorialFromGalactic(lon, lat)
            raDeg, decDeg = utils.equatorialFromGalactic(numpy.degrees(lon), numpy.degrees(lat))
            self.assertAlmostEqual(raRad, numpy.radians(raDeg), 10)
            self.assertAlmostEqual(decRad, numpy.radians(decDeg), 10)



    def testAltAzPaFromRaDec(self):
        mjdList = numpy.random.random_sample(len(self.raList))*5000.0 + 52000.0

        altRad, azRad, paRad = utils._altAzPaFromRaDec(self.raList, self.decList,
                                                       self.lon, self.lat, mjdList)

        altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(numpy.degrees(self.raList),
                                                      numpy.degrees(self.decList),
                                                      numpy.degrees(self.lon),
                                                      numpy.degrees(self.lat),
                                                      mjdList)


        numpy.testing.assert_array_almost_equal(altRad, numpy.radians(altDeg), 10)
        numpy.testing.assert_array_almost_equal(azRad, numpy.radians(azDeg), 10)
        numpy.testing.assert_array_almost_equal(paRad, numpy.radians(paDeg), 10)


        altRad, azRad, paRad = utils._altAzPaFromRaDec(self.raList, self.decList,
                                                       self.lon, self.lat, mjdList[0])

        altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(numpy.degrees(self.raList),
                                                      numpy.degrees(self.decList),
                                                      numpy.degrees(self.lon),
                                                      numpy.degrees(self.lat),
                                                      mjdList[0])


        numpy.testing.assert_array_almost_equal(altRad, numpy.radians(altDeg), 10)
        numpy.testing.assert_array_almost_equal(azRad, numpy.radians(azDeg), 10)
        numpy.testing.assert_array_almost_equal(paRad, numpy.radians(paDeg), 10)


        for ra, dec, mjd in zip(self.raList, self.decList, mjdList):
            altRad, azRad, paRad = utils._altAzPaFromRaDec(ra, dec, self.lon, self.lat, mjd)
            altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(numpy.degrees(ra),
                                                          numpy.degrees(dec),
                                                          numpy.degrees(self.lon),
                                                          numpy.degrees(self.lat),
                                                          mjd)

            self.assertAlmostEqual(altRad, numpy.radians(altDeg), 10)
            self.assertAlmostEqual(azRad, numpy.radians(azDeg), 10)
            self.assertAlmostEqual(paRad, numpy.radians(paDeg), 10)


    def raDecFromAltAz(self):
        azList = self.raList
        altList = self.decList
        mjdList = numpy.random.random_sample(len(self.raList))*5000.0 + 52000.0

        raRad, decRad = utils._raDecFromAltAz(altList, azList,
                                              self.lon, self.lat, mjdList)

        raDeg, decDeg = utils.raDecFromAltAz(numpy.degrees(altList),
                                             numpy.degrees(azList),
                                             numpy.degrees(self.lon),
                                             numpy.degrees(self.lat),
                                             mjdList)


        numpy.testing.assert_array_almost_equal(raRad, numpy.radians(raDeg), 10)
        numpy.testing.assert_array_almost_equal(decRad, numpy.radians(decDeg), 10)

        raRad, decRad = utils._raDecFromAltAz(altList, azList,
                                              lon, lat, mjdList[0])

        raDeg, decDeg = utils.raDecFromAltAz(numpy.degrees(altList),
                                             numpy.degrees(azList),
                                             numpy.degrees(self.lon),
                                             numpy.degrees(self.lat),
                                             mjdList[0])


        numpy.testing.assert_array_almost_equal(raRad, numpy.radians(raDeg), 10)
        numpy.testing.assert_array_almost_equal(decRad, numpy.radians(decDeg), 10)


        for alt, az, mjd in zip(altList, azList, mjdList):
            raRad, decRad = utils._raDecFromAltAz(alt, az, lon, lat, mjd)
            raDeg, decDeg = utils.raDecFromAltAz(numpy.degrees(alt),
                                                 numpy.degrees(az),
                                                 numpy.degrees(self.lon),
                                                 numpy.degrees(self.lat),
                                                 mjd)

            self.assertAlmostEqual(raRad, numpy.radians(raDeg), 10)
            self.assertAlmostEqual(decRad, numpy.radians(decDeg), 10)


    def testGetRotSkyPos(self):
        rotTelList = numpy.random.random_sample(len(self.raList))*2.0*numpy.pi
        mjdList = numpy.random.random_sample(len(self.raList))*5000.0+52000.0

        rotSkyRad = utils._getRotSkyPos(self.raList, self.decList,
                                        self.lon, self.lat,
                                        mjdList, rotTelList)

        rotSkyDeg = utils.getRotSkyPos(numpy.degrees(self.raList),
                                       numpy.degrees(self.decList),
                                       numpy.degrees(self.lon),
                                       numpy.degrees(self.lat),
                                       mjdList, numpy.degrees(rotTelList))

        numpy.testing.assert_array_almost_equal(rotSkyRad, numpy.radians(rotSkyDeg), 10)

        rotSkyRad = utils._getRotSkyPos(self.raList, self.decList,
                                        self.lon, self.lat,
                                        mjdList[0], rotTelList[0])

        rotSkyDeg = utils.getRotSkyPos(numpy.degrees(self.raList),
                                       numpy.degrees(self.decList),
                                       numpy.degrees(self.lon),
                                       numpy.degrees(self.lat),
                                       mjdList[0], numpy.degrees(rotTelList[0]))

        numpy.testing.assert_array_almost_equal(rotSkyRad, numpy.radians(rotSkyDeg), 10)


        for ra, dec, mjd, rotTel in \
        zip(self.raList, self.decList, mjdList, rotTelList):

            rotSkyRad = utils._getRotSkyPos(ra, dec, self.lon, self.lat, mjd, rotTel)

            rotSkyDeg = utils.getRotSkyPos(numpy.degrees(ra), numpy.degrees(dec),
                                           numpy.degrees(self.lon), numpy.degrees(self.lat),
                                           mjd, numpy.degrees(rotTel))

            self.assertAlmostEqual(rotSkyRad, numpy.radians(rotSkyDeg), 10)


def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(testDegrees)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
