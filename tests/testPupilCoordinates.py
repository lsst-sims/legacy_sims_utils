import numpy as np
import unittest
import lsst.utils.tests as utilsTests

from lsst.sims.utils import ObservationMetaData, _nativeLonLatFromRaDec
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.utils import _raDecFromPupilCoords
from lsst.sims.utils import _observedFromICRS, _icrsFromObserved
from lsst.sims.utils import haversine, arcsecFromRadians, solarRaDec, ModifiedJulianDate, distanceToSun

class PupilCoordinateUnitTest(unittest.TestCase):


    def testExceptions(self):
        """
        Test that exceptions are raised when they ought to be
        """
        obs_metadata = ObservationMetaData(pointingRA=25.0,
                                           pointingDec=25.0,
                                           rotSkyPos=25.0,
                                           mjd=52000.0)

        np.random.seed(42)
        ra = np.random.random_sample(10)*np.radians(1.0) + np.radians(obs_metadata.pointingRA)
        dec = np.random.random_sample(10)*np.radians(1.0) + np.radians(obs_metadata.pointingDec)
        raShort = np.array([1.0])
        decShort = np.array([1.0])

        #test without obs_metadata
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0)

        #test without pointingRA
        dummy = ObservationMetaData(pointingDec=obs_metadata.pointingDec,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        #test without pointingDec
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        #test without rotSkyPos
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    pointingDec=obs_metadata.pointingDec,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        #test without mjd
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    pointingDec=obs_metadata.pointingDec,
                                    rotSkyPos=obs_metadata.rotSkyPos)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)


        #test for mismatches
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    pointingDec=obs_metadata.pointingDec,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)

        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, decShort, epoch=2000.0,
                          obs_metadata=dummy)

        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, raShort, dec, epoch=2000.0,
                          obs_metadata=dummy)

        # test that it actually runs (and that passing in either numpy arrays or floats gives
        # the same results)
        xx_arr, yy_arr = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata)
        self.assertIsInstance(xx_arr, np.ndarray)
        self.assertIsInstance(yy_arr, np.ndarray)

        for ix in range(len(ra)):
            xx_f, yy_f = _pupilCoordsFromRaDec(ra[ix], dec[ix], obs_metadata=obs_metadata)
            self.assertIsInstance(xx_f, np.float)
            self.assertIsInstance(yy_f, np.float)
            self.assertAlmostEqual(xx_arr[ix], xx_f, 12)
            self.assertAlmostEqual(yy_arr[ix], yy_f, 12)
            self.assertFalse(np.isnan(xx_f))
            self.assertFalse(np.isnan(yy_f))


    def testCardinalDirections(self):
        """
        This unit test verifies that the following conventions hold:

        if rotSkyPos = 0, then north is +y the camera and east is -x

        if rotSkyPos = -90, then north is +x on the camera and east is +y

        if rotSkyPos = 90, then north is -x on the camera and east is -y

        if rotSkyPos = 180, then north is -y on the camera and east is +x

        This is consistent with rotSkyPos = rotTelPos - parallacticAngle

        parallacticAngle is negative when the pointing is east of the meridian.
        http://www.petermeadows.com/html/parallactic.html

        rotTelPos is the angle between up on the telescope and up on
        the camera, where positive rotTelPos goes from north to west
        (from an email sent to me by LynneJones)

        I have verified that OpSim follows the rotSkyPos = rotTelPos - paralacticAngle
        convention.

        I have verified that altAzPaFromRaDec follows the convention that objects
        east of the meridian have a negative parallactic angle.  (altAzPaFromRaDec
        uses PALPY under the hood, so it can probably be taken as correct)

        It will verify this convention for multiple random pointings.
        """

        epoch = 2000.0
        mjd = 42350.0
        np.random.seed(42)
        raList = np.random.random_sample(10)*360.0
        decList = np.random.random_sample(10)*180.0 - 90.0


        for rotSkyPos in np.arange(-90.0, 181.0, 90.0):
            for ra, dec in zip(raList, decList):
                obs = ObservationMetaData(pointingRA=ra,
                                          pointingDec=dec,
                                          mjd=mjd,
                                          rotSkyPos=rotSkyPos)

                ra_obs, dec_obs = _observedFromICRS(np.radians([ra]), np.radians([dec]),
                                                    obs_metadata=obs, epoch=2000.0,
                                                    includeRefraction=True)

                # test points that are displaced just to the (E, W, N, S) of the pointing
                # in observed geocentric RA, Dec; verify that the pupil coordinates
                # change as expected
                raTest_obs = ra_obs[0] + np.array([0.01, -0.01, 0.0, 0.0])
                decTest_obs = dec_obs[0] + np.array([0.0, 0.0, 0.01, -0.01])
                raTest, decTest = _icrsFromObserved(raTest_obs, decTest_obs, obs_metadata=obs,
                                                    epoch=2000.0, includeRefraction=True)

                x, y = _pupilCoordsFromRaDec(raTest, decTest, obs_metadata=obs, epoch=epoch)

                lon, lat = _nativeLonLatFromRaDec(raTest, decTest, obs)
                rr = np.abs(np.cos(lat)/np.sin(lat))

                if np.abs(rotSkyPos)<0.01:
                    control_x = np.array([-1.0*rr[0], 1.0*rr[1], 0.0, 0.0])
                    control_y = np.array([0.0, 0.0, 1.0*rr[2], -1.0*rr[3]])
                elif np.abs(rotSkyPos+90.0)<0.01:
                    control_x = np.array([0.0, 0.0, 1.0*rr[2], -1.0*rr[3]])
                    control_y = np.array([1.0*rr[0], -1.0*rr[1], 0.0, 0.0])
                elif np.abs(rotSkyPos-90.0)<0.01:
                    control_x = np.array([0.0, 0.0, -1.0*rr[2], 1.0*rr[3]])
                    control_y = np.array([-1.0*rr[0], 1.0*rr[1], 0.0, 0.0])
                elif np.abs(rotSkyPos-180.0)<0.01:
                    control_x = np.array([1.0*rr[0], -1.0*rr[1], 0.0, 0.0])
                    control_y = np.array([0.0, 0.0, -1.0*rr[2], 1.0*rr[3]])

                dx = np.array([xx/cc if np.abs(cc)>1.0e-10 else 1.0-xx for xx, cc in zip(x, control_x)])
                dy = np.array([yy/cc if np.abs(cc)>1.0e-10 else 1.0-yy for yy, cc in zip(y, control_y)])
                np.testing.assert_array_almost_equal(dx, np.ones(4), decimal=4)
                np.testing.assert_array_almost_equal(dy, np.ones(4), decimal=4)


    def testRaDecFromPupil(self):
        """
        Test conversion from pupil coordinates back to Ra, Dec
        """

        mjd = ModifiedJulianDate(TAI=52000.0)
        solarRA, solarDec = solarRaDec(mjd)

        # to make sure that we are more than 45 degrees from the Sun as required
        # for _icrsFromObserved to be at all accurate
        raCenter = solarRA + 100.0
        decCenter = solarDec - 30.0

        obs = ObservationMetaData(pointingRA=raCenter,
                                  pointingDec=decCenter,
                                  boundType='circle',
                                  boundLength=0.1,
                                  rotSkyPos=23.0,
                                  mjd=mjd)

        nSamples = 1000
        np.random.seed(42)
        ra = (np.random.random_sample(nSamples)*0.1-0.2) + np.radians(raCenter)
        dec = (np.random.random_sample(nSamples)*0.1-0.2) + np.radians(decCenter)
        xp, yp = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs, epoch=2000.0)
        raTest, decTest = _raDecFromPupilCoords(xp, yp, obs_metadata=obs, epoch=2000.0)
        distance = arcsecFromRadians(haversine(ra, dec, raTest, decTest))
        dex = np.argmax(distance)
        worstSolarDistance = distanceToSun(np.degrees(ra[dex]), np.degrees(dec[dex]), mjd)
        msg = "_raDecFromPupilCoords off by %e arcsec at distance to Sun of %e degrees" % \
        (distance.max(), worstSolarDistance)
        self.assertLess(distance.max(), 0.005, msg=msg)


    def testNaNs(self):
        """
        Test how _pupilCoordsFromRaDec handles improper values
        """
        obs = ObservationMetaData(pointingRA=42.0, pointingDec=-28.0,
                                  rotSkyPos=111.0, mjd=42356.0)
        nSamples = 100
        np.random.seed(42)
        raList = np.radians(np.random.random_sample(nSamples)*2.0 + 42.0)
        decList = np.radians(np.random.random_sample(nSamples)*2.0 -28.0)

        xControl, yControl = _pupilCoordsFromRaDec(raList, decList,
                                                       obs_metadata=obs,
                                                       epoch=2000.0)

        raList[5] = np.NaN
        decList[5] = np.NaN
        raList[15] = np.NaN
        decList[20] = np.NaN
        raList[30] = np.radians(42.0) + np.pi

        xTest, yTest = _pupilCoordsFromRaDec(raList, decList,
                                                 obs_metadata=obs,
                                                 epoch=2000.0)

        for ix, (xc, yc, xt, yt) in \
        enumerate(zip(xControl, yControl, xTest, yTest)):
            if ix!=5 and ix!=15 and ix!=20 and ix!=30:
                self.assertAlmostEqual(xc, xt, 10)
                self.assertAlmostEqual(yc, yt, 10)
                self.assertFalse(np.isnan(xt))
                self.assertFalse(np.isnan(yt))
            else:
                self.assertTrue(np.isnan(xt))
                self.assertTrue(np.isnan(yt))


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(PupilCoordinateUnitTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)
