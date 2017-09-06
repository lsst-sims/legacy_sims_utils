from __future__ import division
from builtins import zip
from builtins import range
import numpy as np
import unittest
import lsst.utils.tests

from lsst.sims.utils import ObservationMetaData, _nativeLonLatFromRaDec
from lsst.sims.utils import _pupilCoordsFromRaDec, pupilCoordsFromRaDec
from lsst.sims.utils import _raDecFromPupilCoords
from lsst.sims.utils import _observedFromICRS, _icrsFromObserved
from lsst.sims.utils import haversine, arcsecFromRadians, solarRaDec, ModifiedJulianDate, distanceToSun
from lsst.sims.utils import raDecFromAltAz, observedFromICRS, icrsFromObserved
from lsst.sims.utils import radiansFromArcsec


def setup_module(module):
    lsst.utils.tests.init()


class PupilCoordinateUnitTest(unittest.TestCase):

    longMessage = True

    def testExceptions(self):
        """
        Test that exceptions are raised when they ought to be
        """
        obs_metadata = ObservationMetaData(pointingRA=25.0,
                                           pointingDec=25.0,
                                           rotSkyPos=25.0,
                                           mjd=52000.0)

        rng = np.random.RandomState(42)
        ra = rng.random_sample(10) * np.radians(1.0) + np.radians(obs_metadata.pointingRA)
        dec = rng.random_sample(10) * np.radians(1.0) + np.radians(obs_metadata.pointingDec)
        raShort = np.array([1.0])
        decShort = np.array([1.0])

        # test without obs_metadata
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0)

        # test without pointingRA
        dummy = ObservationMetaData(pointingDec=obs_metadata.pointingDec,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        # test without pointingDec
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    rotSkyPos=obs_metadata.rotSkyPos,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        # test without rotSkyPos
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    pointingDec=obs_metadata.pointingDec,
                                    mjd=obs_metadata.mjd)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        # test without mjd
        dummy = ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                    pointingDec=obs_metadata.pointingDec,
                                    rotSkyPos=obs_metadata.rotSkyPos)
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          epoch=2000.0, obs_metadata=dummy)

        # test for mismatches
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

        if rotSkyPos = 0, then north is +y the camera and east is +x

        if rotSkyPos = -90, then north is -x on the camera and east is +y

        if rotSkyPos = 90, then north is +x on the camera and east is -y

        if rotSkyPos = 180, then north is -y on the camera and east is -x

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
        rng = np.random.RandomState(42)
        raList = rng.random_sample(10) * 360.0
        decList = rng.random_sample(10) * 180.0 - 90.0

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
                rr = np.abs(np.cos(lat) / np.sin(lat))

                if np.abs(rotSkyPos) < 0.01:  # rotSkyPos == 0
                    control_x = np.array([1.0 * rr[0], -1.0 * rr[1], 0.0, 0.0])
                    control_y = np.array([0.0, 0.0, 1.0 * rr[2], -1.0 * rr[3]])
                elif np.abs(rotSkyPos + 90.0) < 0.01:  # rotSkyPos == -90
                    control_x = np.array([0.0, 0.0, -1.0 * rr[2], 1.0 * rr[3]])
                    control_y = np.array([1.0 * rr[0], -1.0 * rr[1], 0.0, 0.0])
                elif np.abs(rotSkyPos - 90.0) < 0.01:  # rotSkyPos == 90
                    control_x = np.array([0.0, 0.0, 1.0 * rr[2], -1.0 * rr[3]])
                    control_y = np.array([-1.0 * rr[0], +1.0 * rr[1], 0.0, 0.0])
                elif np.abs(rotSkyPos - 180.0) < 0.01:  # rotSkyPos == 180
                    control_x = np.array([-1.0 * rr[0], +1.0 * rr[1], 0.0, 0.0])
                    control_y = np.array([0.0, 0.0, -1.0 * rr[2], 1.0 * rr[3]])

                msg = 'failed on rotSkyPos == %e\n' % rotSkyPos
                msg += 'control_x %s\n' % str(control_x)
                msg += 'test_x %s\n' % str(x)
                msg += 'control_y %s\n' % str(control_y)
                msg += 'test_y %s\n' % str(y)

                dx = np.array([xx / cc if np.abs(cc) > 1.0e-10 else 1.0 - xx for xx, cc in zip(x, control_x)])
                dy = np.array([yy / cc if np.abs(cc) > 1.0e-10 else 1.0 - yy for yy, cc in zip(y, control_y)])
                self.assertLess(np.abs(dx-np.ones(4)).max(), 0.001, msg=msg)
                self.assertLess(np.abs(dy-np.ones(4)).max(), 0.001, msg=msg)

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
        rng = np.random.RandomState(42)
        ra = (rng.random_sample(nSamples) * 0.1 - 0.2) + np.radians(raCenter)
        dec = (rng.random_sample(nSamples) * 0.1 - 0.2) + np.radians(decCenter)
        xp, yp = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs, epoch=2000.0)
        raTest, decTest = _raDecFromPupilCoords(
            xp, yp, obs_metadata=obs, epoch=2000.0)
        distance = arcsecFromRadians(haversine(ra, dec, raTest, decTest))
        dex = np.argmax(distance)
        worstSolarDistance = distanceToSun(
            np.degrees(ra[dex]), np.degrees(dec[dex]), mjd)
        msg = "_raDecFromPupilCoords off by %e arcsec at distance to Sun of %e degrees" % \
              (distance.max(), worstSolarDistance)
        self.assertLess(distance.max(), 0.005, msg=msg)

        # now check that passing in the xp, yp values one at a time still gives
        # the right answer
        for ix in range(len(ra)):
            ra_f, dec_f = _raDecFromPupilCoords(xp[ix], yp[ix], obs_metadata=obs, epoch=2000.0)
            self.assertIsInstance(ra_f, np.float)
            self.assertIsInstance(dec_f, np.float)
            dist_f = arcsecFromRadians(haversine(ra_f, dec_f, raTest[ix], decTest[ix]))
            self.assertLess(dist_f, 1.0e-9)

    def testNaNs(self):
        """
        Test how _pupilCoordsFromRaDec handles improper values
        """
        obs = ObservationMetaData(pointingRA=42.0, pointingDec=-28.0,
                                  rotSkyPos=111.0, mjd=42356.0)
        nSamples = 100
        rng = np.random.RandomState(42)
        raList = np.radians(rng.random_sample(nSamples) * 2.0 + 42.0)
        decList = np.radians(rng.random_sample(nSamples) * 2.0 - 28.0)

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
            if ix != 5 and ix != 15 and ix != 20 and ix != 30:
                self.assertAlmostEqual(xc, xt, 10)
                self.assertAlmostEqual(yc, yt, 10)
                self.assertFalse(np.isnan(xt))
                self.assertFalse(np.isnan(yt))
            else:
                np.testing.assert_equal(xt, np.NaN)
                np.testing.assert_equal(yt, np.NaN)

    def test_with_proper_motion(self):
        """
        Test that calculating pupil coordinates in the presence of proper motion, parallax,
        and radial velocity is equivalent to
        observedFromICRS -> icrsFromObserved -> pupilCoordsFromRaDec
        (mostly to make surethat pupilCoordsFromRaDec is correctly calling observedFromICRS
        with non-zero proper motion, etc.)
        """
        rng = np.random.RandomState(38442)
        is_valid = False
        while not is_valid:
            mjd_tai = 59580.0 + 10000.0*rng.random_sample()
            obs = ObservationMetaData(mjd=mjd_tai)
            ra, dec = raDecFromAltAz(78.0, 112.0, obs)
            dd = distanceToSun(ra, dec, obs.mjd)
            if dd > 45.0:
                is_valid = True

        n_obj = 1000
        rr = rng.random_sample(n_obj)*2.0
        theta = rng.random_sample(n_obj)*2.0*np.pi
        ra_list = ra + rr*np.cos(theta)
        dec_list = dec + rr*np.sin(theta)
        obs = ObservationMetaData(pointingRA=ra, pointingDec=dec, mjd=mjd_tai, rotSkyPos=19.0)

        pm_ra_list = rng.random_sample(n_obj)*100.0 - 50.0
        pm_dec_list = rng.random_sample(n_obj)*100.0 - 50.0
        px_list = rng.random_sample(n_obj) + 0.05
        v_rad_list = rng.random_sample(n_obj)*600.0 - 300.0

        for includeRefraction in (True, False):

            ra_obs, dec_obs = observedFromICRS(ra_list, dec_list,
                                               pm_ra=pm_ra_list, pm_dec=pm_dec_list,
                                               parallax=px_list, v_rad=v_rad_list,
                                               obs_metadata=obs, epoch=2000.0,
                                               includeRefraction=includeRefraction)

            ra_icrs, dec_icrs = icrsFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                 epoch=2000.0, includeRefraction=includeRefraction)

            xp_control, yp_control = pupilCoordsFromRaDec(ra_icrs, dec_icrs, obs_metadata=obs,
                                                          epoch=2000.0, includeRefraction=includeRefraction)

            xp_test, yp_test = pupilCoordsFromRaDec(ra_list, dec_list,
                                                    pm_ra=pm_ra_list, pm_dec=pm_dec_list,
                                                    parallax=px_list, v_rad=v_rad_list,
                                                    obs_metadata=obs, epoch=2000.0,
                                                    includeRefraction=includeRefraction)

            distance = arcsecFromRadians(np.sqrt(np.power(xp_test-xp_control, 2) +
                                                 np.power(yp_test-yp_control, 2)))
            self.assertLess(distance.max(), 0.006)

            # now test it in radians
            xp_rad, yp_rad = _pupilCoordsFromRaDec(np.radians(ra_list), np.radians(dec_list),
                                                   pm_ra=radiansFromArcsec(pm_ra_list),
                                                   pm_dec=radiansFromArcsec(pm_dec_list),
                                                   parallax=radiansFromArcsec(px_list),
                                                   v_rad=v_rad_list,
                                                   obs_metadata=obs, epoch=2000.0,
                                                   includeRefraction=includeRefraction)

            np.testing.assert_array_equal(xp_rad, xp_test)
            np.testing.assert_array_equal(yp_rad, yp_test)

            # now test it with proper motion = 0
            ra_obs, dec_obs = observedFromICRS(ra_list, dec_list,
                                               parallax=px_list, v_rad=v_rad_list,
                                               obs_metadata=obs, epoch=2000.0,
                                               includeRefraction=includeRefraction)

            ra_icrs, dec_icrs = icrsFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                 epoch=2000.0, includeRefraction=includeRefraction)

            xp_control, yp_control = pupilCoordsFromRaDec(ra_icrs, dec_icrs, obs_metadata=obs,
                                                          epoch=2000.0, includeRefraction=includeRefraction)

            xp_test, yp_test = pupilCoordsFromRaDec(ra_list, dec_list,
                                                    parallax=px_list, v_rad=v_rad_list,
                                                    obs_metadata=obs, epoch=2000.0,
                                                    includeRefraction=includeRefraction)

            distance = arcsecFromRadians(np.sqrt(np.power(xp_test-xp_control, 2) +
                                                 np.power(yp_test-yp_control, 2)))
            self.assertLess(distance.max(), 0.006)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
