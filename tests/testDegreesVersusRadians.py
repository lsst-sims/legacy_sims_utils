import unittest
import numpy
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils
from lsst.sims.utils import ObservationMetaData

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


    def testGetRotTelPos(self):
        rotSkyList = numpy.random.random_sample(len(self.raList))*2.0*numpy.pi
        mjdList = numpy.random.random_sample(len(self.raList))*5000.0+52000.0

        rotTelRad = utils._getRotTelPos(self.raList, self.decList,
                                        self.lon, self.lat,
                                        mjdList, rotSkyList)

        rotTelDeg = utils.getRotTelPos(numpy.degrees(self.raList),
                                       numpy.degrees(self.decList),
                                       numpy.degrees(self.lon),
                                       numpy.degrees(self.lat),
                                       mjdList, numpy.degrees(rotSkyList))

        numpy.testing.assert_array_almost_equal(rotTelRad, numpy.radians(rotTelDeg), 10)

        rotTelRad = utils._getRotTelPos(self.raList, self.decList,
                                        self.lon, self.lat,
                                        mjdList[0], rotSkyList[0])

        rotTelDeg = utils.getRotTelPos(numpy.degrees(self.raList),
                                       numpy.degrees(self.decList),
                                       numpy.degrees(self.lon),
                                       numpy.degrees(self.lat),
                                       mjdList[0], numpy.degrees(rotSkyList[0]))

        numpy.testing.assert_array_almost_equal(rotTelRad, numpy.radians(rotTelDeg), 10)


        for ra, dec, mjd, rotSky in \
        zip(self.raList, self.decList, mjdList, rotSkyList):

            rotTelRad = utils._getRotTelPos(ra, dec, self.lon, self.lat, mjd, rotSky)

            rotTelDeg = utils.getRotTelPos(numpy.degrees(ra), numpy.degrees(dec),
                                           numpy.degrees(self.lon), numpy.degrees(self.lat),
                                           mjd, numpy.degrees(rotSky))

            self.assertAlmostEqual(rotTelRad, numpy.radians(rotTelDeg), 10)

class AstrometryDegreesTest(unittest.TestCase):

    def setUp(self):
        self.nStars = 10
        numpy.random.seed(8273)
        self.raList = numpy.random.random_sample(self.nStars)*2.0*numpy.pi
        self.decList = (numpy.random.random_sample(self.nStars)-0.5)*numpy.pi
        self.mjdList = numpy.random.random_sample(10)*5000.0 + 52000.0
        self.pm_raList = utils.radiansFromArcsec(numpy.random.random_sample(self.nStars)*10.0 - 5.0)
        self.pm_decList = utils.radiansFromArcsec(numpy.random.random_sample(self.nStars)*10.0 - 5.0)
        self.pxList = utils.radiansFromArcsec(numpy.random.random_sample(self.nStars)*2.0)
        self.v_radList = numpy.random.random_sample(self.nStars)*500.0 - 250.0


    def testApplyPrecession(self):
        for mjd in self.mjdList:
            raRad, decRad = utils._applyPrecession(self.raList,
                                                        self.decList,
                                                        mjd=mjd)

            raDeg, decDeg = utils.applyPrecession(numpy.degrees(self.raList),
                                                       numpy.degrees(self.decList),
                                                       mjd=mjd)

            dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


    def testApplyProperMotion(self):
        for mjd in self.mjdList:
            raRad, decRad = utils._applyProperMotion(self.raList, self.decList,
                                                          self.pm_raList, self.pm_decList,
                                                          self.pxList, self.v_radList, mjd=mjd)

            raDeg, decDeg = utils.applyProperMotion(numpy.degrees(self.raList),
                                                         numpy.degrees(self.decList),
                                                         utils.arcsecFromRadians(self.pm_raList),
                                                         utils.arcsecFromRadians(self.pm_decList),
                                                         utils.arcsecFromRadians(self.pxList),
                                                         self.v_radList, mjd=mjd)

            dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


        for ra, dec, pm_ra, pm_dec, px, v_rad in \
        zip(self.raList, self.decList, self.pm_raList, self.pm_decList, \
        self.pxList, self.v_radList):

            raRad, decRad = utils._applyProperMotion(ra, dec, pm_ra, pm_dec, px, v_rad,
                                                          mjd=self.mjdList[0])

            raDeg, decDeg = utils.applyProperMotion(numpy.degrees(ra), numpy.degrees(dec),
                                                         utils.arcsecFromRadians(pm_ra), utils.arcsecFromRadians(pm_dec),
                                                         utils.arcsecFromRadians(px), v_rad, mjd=self.mjdList[0])

            self.assertAlmostEqual(utils.arcsecFromRadians(raRad-numpy.radians(raDeg)), 0.0, 9)
            self.assertAlmostEqual(utils.arcsecFromRadians(decRad-numpy.radians(decDeg)), 0.0, 9)


    def testAppGeoFromICRS(self):
        mjd = 42350.0
        for pmRaList in [self.pm_raList, None]:
            for pmDecList in [self.pm_decList, None]:
                for pxList in [self.pxList, None]:
                    for vRadList in [self.v_radList, None]:
                        raRad, decRad = utils._appGeoFromICRS(self.raList, self.decList,
                                                                   pmRaList, pmDecList,
                                                                   pxList, vRadList, mjd=mjd)

                        raDeg, decDeg = utils.appGeoFromICRS(numpy.degrees(self.raList),
                                                                 numpy.degrees(self.decList),
                                                                 utils.arcsecFromRadians(pmRaList),
                                                                 utils.arcsecFromRadians(pmDecList),
                                                                 utils.arcsecFromRadians(pxList),
                                                                 vRadList, mjd=mjd)

                        dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
                        numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

                        dDec = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
                        numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)



    def testObservedFromAppGeo(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0,
                                  mjd=43572.0)

        for includeRefraction in [True, False]:
            raRad, decRad = utils._observedFromAppGeo(self.raList, self.decList,
                                                           includeRefraction=includeRefraction,
                                                           altAzHr=False, obs_metadata=obs)

            raDeg, decDeg = utils.observedFromAppGeo(numpy.degrees(self.raList),
                                                          numpy.degrees(self.decList),
                                                          includeRefraction=includeRefraction,
                                                          altAzHr=False, obs_metadata=obs)

            dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


            raDec, altAz = utils._observedFromAppGeo(self.raList, self.decList,
                                                          includeRefraction=includeRefraction,
                                                          altAzHr=True, obs_metadata=obs)

            raRad = raDec[0]
            decRad = raDec[1]
            altRad = altAz[0]
            azRad = altAz[1]

            raDec, altAz = utils.observedFromAppGeo(numpy.degrees(self.raList),
                                                         numpy.degrees(self.decList),
                                                         includeRefraction=includeRefraction,
                                                         altAzHr=True, obs_metadata=obs)

            raDeg = raDec[0]
            decDeg = raDec[1]
            altDeg = altAz[0]
            azDeg = altAz[1]

            dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)

            dAz = utils.arcsecFromRadians(azRad-numpy.radians(azDeg))
            numpy.testing.assert_array_almost_equal(dAz, numpy.zeros(self.nStars), 9)

            dAlt = utils.arcsecFromRadians(altRad-numpy.radians(altDeg))
            numpy.testing.assert_array_almost_equal(dAlt, numpy.zeros(self.nStars), 9)


    def testAppGeoFromObserved(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0,
                                  mjd=43572.0)

        for includeRefraction in (True, False):
            for wavelength in (0.5, 0.2, 0.3):

                raRad, decRad = utils._appGeoFromObserved(self.raList, self.decList,
                                                               includeRefraction=includeRefraction,
                                                               wavelength=wavelength,
                                                               obs_metadata=obs)


                raDeg, decDeg = utils.appGeoFromObserved(numpy.degrees(self.raList), numpy.degrees(self.decList),
                                                              includeRefraction=includeRefraction,
                                                              wavelength=wavelength,
                                                              obs_metadata=obs)

                dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
                numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(len(dRa)), 9)

                dDec = utils.arcsecFromRadians(decRad-numpy.radians(decDeg))
                numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(len(dDec)), 9)


    def testIcrsFromAppGeo(self):

        for mjd in (53525.0, 54316.3, 58463.7):
            for epoch in( 2000.0, 1950.0, 2010.0):

                raRad, decRad = utils._icrsFromAppGeo(self.raList, self.decList,
                                                           epoch=epoch, mjd=mjd)

                raDeg, decDeg = utils.icrsFromAppGeo(numpy.degrees(self.raList),
                                                          numpy.degrees(self.decList),
                                                          epoch=epoch, mjd=mjd)

                dRa = utils.arcsecFromRadians(numpy.abs(raRad-numpy.radians(raDeg)))
                self.assertLess(dRa.max(), 1.0e-9)

                dDec = utils.arcsecFromRadians(numpy.abs(decRad-numpy.radians(decDeg)))
                self.assertLess(dDec.max(), 1.0e-9)


    def testObservedFromICRS(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0,
                                  mjd=43572.0)
        for pmRaList in [self.pm_raList, None]:
            for pmDecList in [self.pm_decList, None]:
                for pxList in [self.pxList, None]:
                    for vRadList in [self.v_radList, None]:
                        for includeRefraction in [True, False]:


                            raRad, decRad = utils._observedFromICRS(self.raList, self.decList,
                                                                         pm_ra=pmRaList, pm_dec=pmDecList,
                                                                         parallax=pxList, v_rad=vRadList,
                                                                         obs_metadata=obs, epoch=2000.0,
                                                                         includeRefraction=includeRefraction)

                            raDeg, decDeg = utils.observedFromICRS(numpy.degrees(self.raList), numpy.degrees(self.decList),
                                                                         pm_ra=utils.arcsecFromRadians(pmRaList),
                                                                         pm_dec=utils.arcsecFromRadians(pmDecList),
                                                                         parallax=utils.arcsecFromRadians(pxList),
                                                                         v_rad=vRadList,
                                                                         obs_metadata=obs, epoch=2000.0,
                                                                     includeRefraction=includeRefraction)


                            dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
                            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

                            dDec = utils.arcsecFromRadians(decRad-numpy.radians(decDeg))
                            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)


    def testIcrsFromObserved(self):
        obs = ObservationMetaData(pointingRA=35.0, pointingDec=-45.0,
                                  mjd=43572.0)

        for includeRefraction in [True, False]:

            raRad, decRad = utils._icrsFromObserved(self.raList, self.decList,
                                                         obs_metadata=obs, epoch=2000.0,
                                                         includeRefraction=includeRefraction)

            raDeg, decDeg = utils.icrsFromObserved(numpy.degrees(self.raList), numpy.degrees(self.decList),
                                                        obs_metadata=obs, epoch=2000.0,
                                                        includeRefraction=includeRefraction)

            dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
            numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(self.nStars), 9)

            dDec = utils.arcsecFromRadians(decRad-numpy.radians(decDeg))
            numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(self.nStars), 9)



    def testraDecFromPupilCoords(self):
        obs = ObservationMetaData(pointingRA=23.5, pointingDec=-115.0, mjd=42351.0, rotSkyPos=127.0)

        xpList = numpy.random.random_sample(100)*0.25*numpy.pi
        ypList = numpy.random.random_sample(100)*0.25*numpy.pi

        raRad, decRad = utils._raDecFromPupilCoords(xpList, ypList, obs_metadata=obs, epoch=2000.0)
        raDeg, decDeg = utils.raDecFromPupilCoords(xpList, ypList, obs_metadata=obs, epoch=2000.0)

        dRa = utils.arcsecFromRadians(raRad-numpy.radians(raDeg))
        numpy.testing.assert_array_almost_equal(dRa, numpy.zeros(len(xpList)), 9)

        dDec = utils.arcsecFromRadians(decRad-numpy.radians(decDeg))
        numpy.testing.assert_array_almost_equal(dDec, numpy.zeros(len(xpList)), 9)



    def testpupilCoordsFromRaDec(self):
        obs = ObservationMetaData(pointingRA=23.5, pointingDec=-115.0, mjd=42351.0, rotSkyPos=127.0)

        # need to make sure the test points are tightly distributed around the bore site, or
        # PALPY will throw an error
        raList = numpy.random.random_sample(self.nStars)*numpy.radians(1.0) + numpy.radians(23.5)
        decList = numpy.random.random_sample(self.nStars)*numpy.radians(1.0) + numpy.radians(-115.0)

        xpControl, ypControl = utils._pupilCoordsFromRaDec(raList, decList,
                                                                     obs_metadata=obs, epoch=2000.0)

        xpTest, ypTest = utils.pupilCoordsFromRaDec(numpy.degrees(raList), numpy.degrees(decList),
                                                              obs_metadata=obs, epoch=2000.0)

        dx = utils.arcsecFromRadians(xpControl-xpTest)
        numpy.testing.assert_array_almost_equal(dx, numpy.zeros(self.nStars), 9)

        dy = utils.arcsecFromRadians(ypControl-ypTest)
        numpy.testing.assert_array_almost_equal(dy, numpy.zeros(self.nStars), 9)



def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(testDegrees)
    suites += unittest.makeSuite(AstrometryDegreesTest)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
