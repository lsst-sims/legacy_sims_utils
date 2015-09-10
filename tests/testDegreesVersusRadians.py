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
        lon = numpy.random.random_sample(1)[0]*2.0*numpy.pi
        lat = (numpy.random.random_sample(1)[0]-0.5)*0.5*numpy.pi
        mjdList = numpy.random.random_sample(len(self.raList))*5000.0 + 52000.0

        altRad, azRad, paRad = utils._altAzPaFromRaDec(self.raList, self.decList,
                                                       lon, lat, mjdList)

        altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(numpy.degrees(self.raList),
                                                      numpy.degrees(self.decList),
                                                      numpy.degrees(lon), numpy.degrees(lat),
                                                      mjdList)


        numpy.testing.assert_array_almost_equal(altRad, numpy.radians(altDeg), 10)
        numpy.testing.assert_array_almost_equal(azRad, numpy.radians(azDeg), 10)
        numpy.testing.assert_array_almost_equal(paRad, numpy.radians(paDeg), 10)


        altRad, azRad, paRad = utils._altAzPaFromRaDec(self.raList, self.decList,
                                                       lon, lat, mjdList[0])

        altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(numpy.degrees(self.raList),
                                                      numpy.degrees(self.decList),
                                                      numpy.degrees(lon), numpy.degrees(lat),
                                                      mjdList[0])


        numpy.testing.assert_array_almost_equal(altRad, numpy.radians(altDeg), 10)
        numpy.testing.assert_array_almost_equal(azRad, numpy.radians(azDeg), 10)
        numpy.testing.assert_array_almost_equal(paRad, numpy.radians(paDeg), 10)


        for ra, dec, mjd in zip(self.raList, self.decList, mjdList):
            altRad, azRad, paRad = utils._altAzPaFromRaDec(ra, dec, lon, lat, mjd)
            altDeg, azDeg, paDeg = utils.altAzPaFromRaDec(numpy.degrees(ra),
                                                          numpy.degrees(dec),
                                                          numpy.degrees(lon),
                                                          numpy.degrees(lat),
                                                          mjd)

            self.assertAlmostEqual(altRad, numpy.radians(altDeg), 10)
            self.assertAlmostEqual(azRad, numpy.radians(azDeg), 10)
            self.assertAlmostEqual(paRad, numpy.radians(paDeg), 10)


    def raDecFromAltAz(self):
        azList = self.raList
        altList = self.decList
        lon = numpy.random.random_sample(1)[0]*2.0*numpy.pi
        lat = (numpy.random.random_sample(1)[0]-0.5)*0.5*numpy.pi
        mjdList = numpy.random.random_sample(len(self.raList))*5000.0 + 52000.0

        raRad, decRad = utils._raDecFromAltAz(altList, azList,
                                              lon, lat, mjdList)

        raDeg, decDeg = utils.raDecFromAltAz(numpy.degrees(altList),
                                             numpy.degrees(azList),
                                             numpy.degrees(lon), numpy.degrees(lat),
                                             mjdList)


        numpy.testing.assert_array_almost_equal(raRad, numpy.radians(raDeg), 10)
        numpy.testing.assert_array_almost_equal(decRad, numpy.radians(decDeg), 10)

        raRad, decRad = utils._raDecFromAltAz(altList, azList,
                                              lon, lat, mjdList[0])

        raDeg, decDeg = utils.raDecFromAltAz(numpy.degrees(altList),
                                             numpy.degrees(azList),
                                             numpy.degrees(lon), numpy.degrees(lat),
                                             mjdList[0])


        numpy.testing.assert_array_almost_equal(raRad, numpy.radians(raDeg), 10)
        numpy.testing.assert_array_almost_equal(decRad, numpy.radians(decDeg), 10)


        for alt, az, mjd in zip(altList, azList, mjdList):
            raRad, decRad = utils._raDecFromAltAz(alt, az, lon, lat, mjd)
            raDeg, decDeg = utils.raDecFromAltAz(numpy.degrees(alt),
                                                 numpy.degrees(az),
                                                 numpy.degrees(lon),
                                                 numpy.degrees(lat),
                                                 mjd)

            self.assertAlmostEqual(raRad, numpy.radians(raDeg), 10)
            self.assertAlmostEqual(decRad, numpy.radians(decDeg), 10)





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
