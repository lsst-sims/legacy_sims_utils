import unittest
import numpy as np
import palpy
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils

def controlAltAzFromRaDec(raRad_in, decRad_in, longRad, latRad, mjd):
    """
    Converts RA and Dec to altitude and azimuth

    @param [in] raRad is the RA in radians
    (observed geocentric)

    @param [in] decRad is the Dec in radians
    (observed geocentric)

    @param [in] longRad is the longitude of the observer in radians
    (positive east of the prime meridian)

    @param [in[ latRad is the latitude of the observer in radians
    (positive north of the equator)

    @param [in] mjd is the universal time expressed as an MJD

    @param [out] altitude in radians

    @param [out[ azimuth in radians

    see: http://www.stargazing.net/kepler/altaz.html#twig04
    """
    obs = utils.ObservationMetaData(mjd=utils.ModifiedJulianDate(UTC=mjd),
              site=utils.Site(longitude=np.degrees(longRad), latitude=np.degrees(latRad), name='LSST'))

    if hasattr(raRad_in, '__len__'):
        raRad, decRad = utils._observedFromICRS(raRad_in, decRad_in, obs_metadata=obs,
                                                epoch=2000.0, includeRefraction=True)
    else:
        raRad_temp, decRad_temp = utils._observedFromICRS(np.array([raRad_in]), np.array([decRad_in]),
                                                          obs_metadata=obs, epoch=2000.0, includeRefraction=True)
        raRad = raRad_temp[0]
        decRad = decRad_temp[0]

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
        self.mjd = 57087.0
        self.tolerance = 1.0e-5


    def testExceptions(self):
        """
        Test to make sure that methods complain when incorrect data types are passed.
        """
        obs = utils.ObservationMetaData(pointingRA=55.0, pointingDec=-72.0, mjd=53467.8)

        raFloat = 1.1
        raList = np.array([0.2, 0.3])

        decFloat = 1.1
        decList = np.array([0.2, 0.3])

        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils._altAzPaFromRaDec, raFloat, decList, obs)
        ans = utils._altAzPaFromRaDec(raFloat, decFloat, obs)
        ans = utils._altAzPaFromRaDec(raList, decList, obs)

        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils._raDecFromAltAz, raFloat, decList, obs)
        ans = utils._raDecFromAltAz(raFloat, decFloat, obs)
        ans = utils._raDecFromAltAz(raList, decList, obs)

        self.assertRaises(RuntimeError, utils.altAzPaFromRaDec, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils.altAzPaFromRaDec, raFloat, decList, obs)
        ans = utils.altAzPaFromRaDec(raFloat, decFloat, obs)
        ans = utils.altAzPaFromRaDec(raList, decList, obs)

        self.assertRaises(RuntimeError, utils.raDecFromAltAz, raList, decFloat, obs)
        self.assertRaises(RuntimeError, utils.raDecFromAltAz, raFloat, decList, obs)
        ans = utils.raDecFromAltAz(raFloat, decFloat, obs)
        ans = utils.raDecFromAltAz(raList, decList, obs)


    def test_raDecFromAltAz(self):
        """
        Test conversion of Alt, Az to Ra, Dec using data on the Sun

        This site gives the altitude and azimuth of the Sun as a function
        of time and position on the earth

        http://aa.usno.navy.mil/data/docs/AltAz.php

        This site gives the apparent geocentric RA, Dec of major celestial objects
        as a function of time

        http://aa.usno.navy.mil/data/docs/geocentric.php

        This site converts calendar dates into Julian Dates

        http://aa.usno.navy.mil/data/docs/JulianDate.php
        """

        hours = np.radians(360.0/24.0)
        minutes = hours/60.0
        seconds = minutes/60.0

        longitude_list = []
        latitude_list = []
        mjd_list = []
        alt_list = []
        az_list = []
        ra_app_list = []
        dec_app_list = []

        longitude_list.append(np.radians(-22.0-33.0/60.0))
        latitude_list.append(np.radians(11.0+45.0/60.0))
        mjd_list.append(2457364.958333-2400000.5) # 8 December 2015 11:00 UTC
        alt_list.append(np.radians(41.1))
        az_list.append(np.radians(134.7))
        ra_app_list.append(16.0*hours + 59.0*minutes + 16.665*seconds)
        dec_app_list.append(np.radians(-22.0 - 42.0/60.0 - 2.94/3600.0))

        longitude_list.append(np.radians(-22.0-33.0/60.0))
        latitude_list.append(np.radians(11.0+45.0/60.0))
        mjd_list.append(2457368.958333-2400000.5) # 12 December 2015 11:00 UTC
        alt_list.append(np.radians(40.5))
        az_list.append(np.radians(134.7))
        ra_app_list.append(17.0*hours + 16.0*minutes +51.649*seconds)
        dec_app_list.append(np.radians(-23.0-3/60.0-50.35/3600.0))

        longitude_list.append(np.radians(145.0 + 23.0/60.0))
        latitude_list.append(np.radians(-64.0-5.0/60.0))
        mjd_list.append(2456727.583333-2400000.5) # 11 March 2014, 02:00 UTC
        alt_list.append(np.radians(29.5))
        az_list.append(np.radians(8.2))
        ra_app_list.append(23.0*hours + 24.0*minutes + 46.634*seconds)
        dec_app_list.append(np.radians(-3.0-47.0/60.0 - 47.81/3600.0))

        longitude_list.append(np.radians(145.0 + 23.0/60.0))
        latitude_list.append(np.radians(-64.0-5.0/60.0))
        mjd_list.append(2456731.583333-2400000.5) # 15 March 2014, 02:00 UTC
        alt_list.append(np.radians(28.0))
        az_list.append(np.radians(7.8))
        ra_app_list.append(23.0*hours + 39.0*minutes + 27.695*seconds)
        dec_app_list.append(np.radians(-2.0 - 13.0/60.0 - 18.32/3600.0))

        for longitude, latitude, mjd, alt, az, ra_app, dec_app in \
            zip(longitude_list, latitude_list, mjd_list, alt_list, az_list,
                ra_app_list, dec_app_list):

            obs = utils.ObservationMetaData(site=utils.Site(longitude=np.degrees(longitude),
                                                            latitude=np.degrees(latitude), name='LSST'),
                                            mjd=utils.ModifiedJulianDate(UTC=mjd))


            ra_icrs, dec_icrs = utils._raDecFromAltAz(alt, az, obs)
            ra_test, dec_test = utils._appGeoFromICRS(np.array([ra_icrs]), np.array([dec_icrs]),
                                                      mjd=obs.mjd)

            distance = np.degrees(utils.haversine(ra_app, dec_app, ra_test[0], dec_test[0]))
            self.assertLess(distance, 0.1) # since that is all the precision we have in the alt, az
                                           # data taken from the USNO
            correction = np.degrees(utils.haversine(ra_test[0], dec_test[0], ra_icrs, dec_icrs))
            self.assertLess(distance, correction)



    def testAltAzRADecRoundTrip(self):
        """
        Test that altAzPaFromRaDec and raDecFromAltAz really invert each other
        """

        np.random.seed(42)
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

                obs = utils.ObservationMetaData(mjd=mjd, site=utils.Site(longitude=lon, latitude=lat, name='LSST'))

                ra_in, dec_in = utils.raDecFromAltAz(alt_in, az_in, obs)

                self.assertFalse(np.isnan(ra_in).any())
                self.assertFalse(np.isnan(dec_in).any())

                alt_out, az_out, pa_out = utils.altAzPaFromRaDec(ra_in, dec_in, obs)

                self.assertFalse(np.isnan(pa_out).any())

                for alt_c, az_c, alt_t, az_t in \
                    zip(np.radians(alt_in), np.radians(az_in), np.radians(alt_out), np.radians(az_out)):
                    distance = utils.arcsecFromRadians(utils.haversine(az_c, alt_c, az_t, alt_t))
                    self.assertLess(distance, 0.2) # not sure why 0.2 arcsec is the limiting precision of this test


    def testAltAzFromRaDec(self):
        """
        Test conversion from RA, Dec to Alt, Az
        """

        np.random.seed(32)
        nSamples = 100
        ra = np.random.sample(nSamples)*2.0*np.pi
        dec = (np.random.sample(nSamples)-0.5)*np.pi
        lon_rad = 1.467
        lat_rad = -0.234
        controlAlt, controlAz = controlAltAzFromRaDec(ra, dec, \
                                                     lon_rad, lat_rad, \
                                                     self.mjd)

        obs = utils.ObservationMetaData(mjd=utils.ModifiedJulianDate(UTC=self.mjd),
                                        site=utils.Site(longitude=np.degrees(lon_rad), latitude=np.degrees(lat_rad), name='LSST'))

        #verify parallactic angle against an expression from
        #http://www.astro.washington.edu/groups/APO/Mirror.Motions/Feb.2000.Image.Jumps/report.html#Image%20motion%20directions
        #
        ra_obs, dec_obs = utils._observedFromICRS(ra, dec, obs_metadata=obs, epoch=2000.0,
                                                  includeRefraction=True)

        lmst, last = utils.calcLmstLast(self.mjd, lon_rad)
        hourAngle = np.radians(last*15.0) - ra_obs
        controlSinPa = np.sin(hourAngle)*np.cos(lat_rad)/np.cos(controlAlt)

        testAlt, testAz, testPa = utils._altAzPaFromRaDec(ra, dec, obs)

        distance = utils.arcsecFromRadians(utils.haversine(controlAz, controlAlt, testAz, testAlt))
        self.assertLess(distance.max(), 0.0001)


        #test non-vectorized version
        for r,d in zip(ra, dec):
            controlAlt, controlAz = controlAltAzFromRaDec(r, d, lon_rad, lat_rad, self.mjd)
            testAlt, testAz, testPa = utils._altAzPaFromRaDec(r, d, obs)
            lmst, last = utils.calcLmstLast(self.mjd, lon_rad)
            r_obs, dec_obs = utils._observedFromICRS(np.array([r]), np.array([d]), obs_metadata=obs,
                                                     epoch=2000.0, includeRefraction=True)
            hourAngle = np.radians(last*15.0) - r_obs[0]
            controlSinPa = np.sin(hourAngle)*np.cos(lat_rad)/np.cos(controlAlt)
            self.assertLess(np.abs(testAz - controlAz), self.tolerance)
            self.assertLess(np.abs(testAlt - controlAlt), self.tolerance)
            self.assertLess(np.abs(np.sin(testPa) - controlSinPa), self.tolerance)



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
