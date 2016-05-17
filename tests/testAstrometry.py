"""
Some of the data in this unit test will appear abitrary.  That is
because, in addition to testing the execution of all of the functionality
provided in the sims_coordUtils package, this unit test validates
the outputs of PALPY against the outputs of pySLALIB v 1.0.2
(it was written when we were making the transition from pySLALIB to PALPY).

There will be some difference, as the two libraries are based on slightly
different conventions (for example, the prenut routine which calculates
the matrix of precession and nutation is based on the IAU 2006/2000A
standard in PALPY and on SF2001 in pySLALIB; however, the two outputs
still agree to within one part in 10^5)

"""

from __future__ import with_statement

import numpy as np

import os
import unittest
import warnings
import sys
import math
import palpy as pal
from collections import OrderedDict
import lsst.utils.tests as utilsTests
from lsst.utils import getPackageDir

from lsst.sims.utils import ObservationMetaData
from lsst.sims.utils import _getRotTelPos, _raDecFromAltAz, calcObsDefaults, \
                            radiansFromArcsec, arcsecFromRadians, Site, \
                            raDecFromAltAz, haversine,ModifiedJulianDate

from lsst.sims.utils import solarRaDec, _solarRaDec, distanceToSun, _distanceToSun
from lsst.sims.utils import _applyPrecession, _applyProperMotion
from lsst.sims.utils import _appGeoFromICRS, _observedFromAppGeo
from lsst.sims.utils import _observedFromICRS, _icrsFromObserved
from lsst.sims.utils import _appGeoFromObserved, _icrsFromAppGeo
from lsst.sims.utils import refractionCoefficients, applyRefraction

def makeObservationMetaData():
    #create a cartoon ObservationMetaData object
    mjd = 52000.0
    alt = np.pi/2.0
    az = 0.0
    band = 'r'
    testSite = Site(latitude=np.degrees(0.5), longitude=np.degrees(1.1), height=3000,
                    temperature=260.0, pressure=725.0, lapseRate=0.005, humidity=0.4)
    obsTemp = ObservationMetaData(site=testSite, mjd=mjd)
    centerRA, centerDec = _raDecFromAltAz(alt, az, obsTemp)
    rotTel = _getRotTelPos(centerRA, centerDec, obsTemp, 0.0)

    obsDict = calcObsDefaults(centerRA, centerDec, alt, az, rotTel, mjd, band,
                 testSite.longitude_rad, testSite.latitude_rad)

    obsDict['Opsim_expmjd'] = mjd
    radius = 0.1
    phoSimMetaData = OrderedDict([
                      (k, (obsDict[k],np.dtype(type(obsDict[k])))) for k in obsDict])

    obs_metadata = ObservationMetaData(boundType='circle', boundLength=2.0*radius,
                                       phoSimMetaData=phoSimMetaData, site=testSite)



    return obs_metadata

def makeRandomSample(raCenter=None, decCenter=None, radius=None):
    #create a random sample of object data

    nsamples=100
    np.random.seed(32)

    if raCenter is None or decCenter is None or radius is None:
        ra = np.random.sample(nsamples)*2.0*np.pi
        dec = (np.random.sample(nsamples)-0.5)*np.pi
    else:
        rr = np.random.sample(nsamples)*radius
        theta = np.random.sample(nsamples)*2.0*np.pi
        ra = raCenter + rr*np.cos(theta)
        dec = decCenter + rr*np.cos(theta)

    pm_ra = (np.random.sample(nsamples)-0.5)*0.1
    pm_dec = (np.random.sample(nsamples)-0.5)*0.1
    parallax = np.random.sample(nsamples)*0.01
    v_rad = np.random.sample(nsamples)*1000.0

    return ra, dec, pm_ra, pm_dec, parallax, v_rad


class astrometryUnitTest(unittest.TestCase):
    """
    The bulk of this unit test involves inputting a set list of input values
    and comparing the astrometric results to results derived from SLALIB run
    with the same input values.  We have to create a test catalog artificially (rather than
    querying the database) because SLALIB was originally run on values that did not correspond
    to any particular Opsim run.
    """

    def setUp(self):
        self.metadata={}

        #below are metadata values that need to be set in order for
        #get_getFocalPlaneCoordinates to work.  If we had been querying the database,
        #these would be set to meaningful values.  Because we are generating
        #an artificial set of inputs that must comport to the baseline SLALIB
        #inputs, these are set arbitrarily by hand
        self.metadata['pointingRA'] = (np.radians(200.0), float)
        self.metadata['pointingDec'] = (np.radians(-30.0), float)
        self.metadata['Opsim_rotskypos'] = (1.0, float)

        # these were the LSST site parameters as coded when this unit test was written
        self.test_site=Site(longitude=np.degrees(-1.2320792),
                            latitude=np.degrees(-0.517781017),
                            height=2650.0,
                            temperature=11.505,
                            pressure=749.3,
                            lapseRate=0.0065,
                            humidity=0.4)

        self.obs_metadata=ObservationMetaData(mjd=50984.371741,
                                     boundType='circle',
                                     boundLength=0.05,
                                     phoSimMetaData=self.metadata,
                                     site=self.test_site)

        self.tol=1.0e-5

    def tearDown(self):
        del self.obs_metadata
        del self.metadata
        del self.tol


    def testDistanceToSun(self):
        """
        Test _distanceToSun using solar RA, Dec calculated from

        http://aa.usno.navy.mil/data/docs/JulianDate.php
        http://aa.usno.navy.mil/data/docs/geocentric.php
        """

        hour = np.radians(360.0/24.0)
        minute = hour/60.0
        second = minute/60.0

        mjd_list = [57026.0, 57543.625]

        sun_ra_list = [18.0*hour + 56.0*minute + 51.022*second,
                       4.0*hour + 51.0*minute + 22.776*second,]

        sun_dec_list = [np.radians(-22.0-47.0/60.0-40.27/3600.0),
                        np.radians(22.0+30.0/60.0+0.73/3600.0)]

        for raS, decS, mjd in zip(sun_ra_list, sun_dec_list, mjd_list):

            # first, verify that the Sun is where we think it is to within 5 arc seconds
            self.assertLess(arcsecFromRadians(_distanceToSun(raS, decS, mjd)), 5.0)

            # find Sun's Cartesian coordinates
            sun_x = np.cos(decS)*np.cos(raS)
            sun_y = np.cos(decS)*np.sin(raS)
            sun_z = np.sin(decS)

            # now choose positions that are a set distance away from the Sun, and make sure
            # that _distanceToSun returns the expected result
            for theta in (np.pi/2.0, np.pi/4.0, -np.pi/3.0):

                # displace by rotating about z axis
                new_x = sun_x*np.cos(theta)+sun_y*np.sin(theta)
                new_y = -sun_x*np.sin(theta)+sun_y*np.cos(theta)
                new_z = sun_z

                new_ra = np.arctan2(new_y, new_x)
                new_dec = np.arctan2(new_z, np.sqrt(new_x*new_x+new_y*new_y))

                dd = _distanceToSun(new_ra, new_dec, mjd)
                hh = haversine(raS, decS, new_ra, new_dec)
                self.assertLess(np.abs(arcsecFromRadians(dd-hh)), 5.0)

                # displace by rotating about y axis
                new_x = sun_x*np.cos(theta)+sun_z*np.sin(theta)
                new_y = sun_y
                new_z = -sun_x*np.sin(theta)+sun_z*np.cos(theta)

                new_ra = np.arctan2(new_y, new_x)
                new_dec = np.arctan2(new_z, np.sqrt(new_x*new_x+new_y*new_y))
                dd = _distanceToSun(new_ra, new_dec, mjd)
                hh = haversine(raS, decS, new_ra, new_dec)
                self.assertLess(np.abs(arcsecFromRadians(dd-hh)), 5.0)

                # displace by rotating about x axis
                new_x = sun_x
                new_y = sun_y*np.cos(theta)+sun_z*np.sin(theta)
                new_z = -sun_y*np.sin(theta)+sun_z*np.cos(theta)

                new_ra = np.arctan2(new_y, new_x)
                new_dec = np.arctan2(new_z, np.sqrt(new_x*new_x+new_y*new_y))
                dd = _distanceToSun(new_ra, new_dec, mjd)
                hh = haversine(raS, decS, new_ra, new_dec)
                self.assertLess(np.abs(arcsecFromRadians(dd-hh)), 5.0)

        # Test passing in numpy arrays of RA, Dec
        np.random.seed(87)
        nSamples = 100
        ra = np.random.random_sample(nSamples)*2.0*np.pi
        dec = (np.random.random_sample(nSamples)-0.5)*np.pi
        mjd = 59580.0
        control_distance = _distanceToSun(ra, dec, mjd)
        self.assertIsInstance(control_distance, np.ndarray)
        for ix, (rr, dd) in enumerate(zip(ra, dec)):
            dd = _distanceToSun(rr, dd, mjd)
            self.assertIsInstance(dd, np.float)
            self.assertAlmostEqual(dd, control_distance[ix], 12)


    def testDistanceToSunDeg(self):
        """
        Test that distanceToSun is consistent with _distanceToSun
        """

        for mjd, ra, dec in zip((57632.1, 45623.4, 55682.3), (112.0, 24.0, 231.2), (-25.0, 23.4, -60.0)):
             dd_deg = distanceToSun(ra, dec, mjd)
             dd_rad = _distanceToSun(np.radians(ra), np.radians(dec), mjd)
             self.assertAlmostEqual(np.radians(dd_deg), dd_rad, 10)


    def testSolarRaDecDeg(self):
        """
        Test that solarRaDec is consistent with _solarRaDec
        """

        for mjd in (57664.2, 53478.9, 45672.1):
            ra_deg, dec_deg = solarRaDec(mjd)
            ra_rad, dec_rad = _solarRaDec(mjd)
            self.assertAlmostEqual(np.radians(ra_deg), ra_rad, 10)
            self.assertAlmostEqual(np.radians(dec_deg), dec_rad, 10)


    def testDistanceToSunArray(self):
        """
        Test _distanceToSun on numpy arrays of RA, Dec using solar RA, Dec calculated from

        http://aa.usno.navy.mil/data/docs/JulianDate.php
        http://aa.usno.navy.mil/data/docs/geocentric.php
        """

        np.random.seed(77)
        nStars = 100

        hour = np.radians(360.0/24.0)
        minute = hour/60.0
        second = minute/60.0

        mjd_list = [57026.0, 57543.625]

        sun_ra_list = [18.0*hour + 56.0*minute + 51.022*second,
                       4.0*hour + 51.0*minute + 22.776*second,]

        sun_dec_list = [np.radians(-22.0-47.0/60.0-40.27/3600.0),
                        np.radians(22.0+30.0/60.0+0.73/3600.0)]

        for mjd, raS, decS in zip(mjd_list, sun_ra_list, sun_dec_list):

            ra_list = np.random.random_sample(nStars)*2.0*np.pi
            dec_list =(np.random.random_sample(nStars)-0.5)*np.pi
            distance_list = _distanceToSun(ra_list, dec_list, mjd)
            distance_control = haversine(ra_list, dec_list, np.array([raS]*nStars), np.array([decS]*nStars))
            np.testing.assert_array_almost_equal(distance_list, distance_control, 5)


    def testAstrometryExceptions(self):
        """
        Test to make sure that stand-alone astrometry methods raise an exception when they are called without
        the necessary arguments
        """
        obs_metadata = makeObservationMetaData()
        ra, dec, pm_ra, pm_dec, parallax, v_rad = makeRandomSample()

        raShort = np.array([1.0])
        decShort = np.array([1.0])


        ##########test refractionCoefficients
        self.assertRaises(RuntimeError, refractionCoefficients)
        site = obs_metadata.site
        x, y = refractionCoefficients(site=site)

        ##########test applyRefraction
        zd = 0.1
        rzd = applyRefraction(zd, x, y)

        zd = [0.1, 0.2]
        self.assertRaises(RuntimeError, applyRefraction, zd, x, y)

        zd = np.array([0.1, 0.2])
        rzd = applyRefraction(zd, x, y)

        ##########test _applyPrecession
        #test without mjd
        self.assertRaises(RuntimeError, _applyPrecession, ra, dec)

        #test mismatches
        self.assertRaises(RuntimeError, _applyPrecession, raShort, dec,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyPrecession, ra, decShort,
                          mjd=ModifiedJulianDate(TAI=52000.0))

        #test that it runs
        _applyPrecession(ra, dec, mjd=ModifiedJulianDate(TAI=52000.0))

        ##########test _applyProperMotion
        raList = list(ra)
        decList = list(dec)
        pm_raList = list(pm_ra)
        pm_decList = list(pm_dec)
        parallaxList = list(parallax)
        v_radList = list(v_rad)

        pm_raShort = np.array([pm_ra[0]])
        pm_decShort = np.array([pm_dec[0]])
        parallaxShort = np.array([parallax[0]])
        v_radShort = np.array([v_rad[0]])

        #test without mjd
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallax, v_rad)

        #test passing lists
        self.assertRaises(RuntimeError, _applyProperMotion,
                          raList, dec, pm_ra, pm_dec, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, decList, pm_ra, pm_dec, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_raList, pm_dec, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_decList, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallaxList, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallax, v_radList,
                          mjd=ModifiedJulianDate(TAI=52000.0))

        #test mismatches
        self.assertRaises(RuntimeError, _applyProperMotion,
                          raShort, dec, pm_ra, pm_dec, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, decShort, pm_ra, pm_dec, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_raShort, pm_dec, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_decShort, parallax, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallaxShort, v_rad,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _applyProperMotion,
                          ra, dec, pm_ra, pm_dec, parallax, v_radShort,
                          mjd=ModifiedJulianDate(TAI=52000.0))

        #test that it actually runs
        _applyProperMotion(ra, dec, pm_ra, pm_dec, parallax, v_rad,
                           mjd=ModifiedJulianDate(TAI=52000.0))
        _applyProperMotion(ra[0], dec[0], pm_ra[0], pm_dec[0], parallax[0], v_rad[0],
                          mjd=ModifiedJulianDate(TAI=52000.0))

        ##########test _appGeoFromICRS
        #test without mjd
        self.assertRaises(RuntimeError, _appGeoFromICRS, ra, dec)

        #test with mismatched ra, dec
        self.assertRaises(RuntimeError, _appGeoFromICRS, ra, decShort,
                          mjd=ModifiedJulianDate(TAI=52000.0))
        self.assertRaises(RuntimeError, _appGeoFromICRS, raShort, dec,
                          mjd=ModifiedJulianDate(TAI=52000.0))

        #test that it actually urns
        test=_appGeoFromICRS(ra, dec, mjd=obs_metadata.mjd)

        ##########test _observedFromAppGeo
        #test without obs_metadata
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec)

        #test without site
        dummy=ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                  pointingDec=obs_metadata.pointingDec,
                                  mjd=obs_metadata.mjd,
                                  site=None)
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec, obs_metadata=dummy)

        #test without mjd
        dummy=ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                  pointingDec=obs_metadata.pointingDec,
                                  site=Site(name='LSST'))
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec, obs_metadata=dummy)

        #test mismatches
        dummy=ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                  pointingDec=obs_metadata.pointingDec,
                                  mjd=obs_metadata.mjd,
                                  site=Site(name='LSST'))

        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, decShort, obs_metadata=dummy)
        self.assertRaises(RuntimeError, _observedFromAppGeo, raShort, dec, obs_metadata=dummy)

        #test that it actually runs
        test = _observedFromAppGeo(ra, dec, obs_metadata=dummy)

        ##########test _observedFromICRS
        #test without epoch
        self.assertRaises(RuntimeError, _observedFromICRS, ra, dec, obs_metadata=obs_metadata)

        #test without obs_metadata
        self.assertRaises(RuntimeError, _observedFromICRS, ra, dec, epoch=2000.0)

        #test without mjd
        dummy=ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                  pointingDec=obs_metadata.pointingDec,
                                  site=obs_metadata.site)
        self.assertRaises(RuntimeError, _observedFromICRS, ra, dec, epoch=2000.0, obs_metadata=dummy)

        #test that it actually runs
        dummy=ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                  pointingDec=obs_metadata.pointingDec,
                                  site=obs_metadata.site,
                                  mjd=obs_metadata.mjd)

        #test mismatches
        self.assertRaises(RuntimeError, _observedFromICRS, ra, decShort, epoch=2000.0, obs_metadata=dummy)
        self.assertRaises(RuntimeError, _observedFromICRS, raShort, dec, epoch=2000.0, obs_metadata=dummy)

        #test that it actually runs
        test = _observedFromICRS(ra, dec, obs_metadata=dummy, epoch=2000.0)



    def test_applyPrecession(self):

        ra=np.zeros((3),dtype=float)
        dec=np.zeros((3),dtype=float)

        ra[0]=2.549091039839124218e+00
        dec[0]=5.198752733024248895e-01
        ra[1]=8.693375673649429425e-01
        dec[1]=1.038086165642298164e+00
        ra[2]=7.740864769302191473e-01
        dec[2]=2.758053025017753179e-01

        self.assertRaises(RuntimeError, _applyPrecession, ra, dec)

        #just make sure it runs
        output=_applyPrecession(ra,dec, mjd=ModifiedJulianDate(TAI=57388.0))


    def test_applyProperMotion(self):
        """
        Compare the output of _applyProperMotion to control outputs
        generated by recreating the 'space motion' section of code
        from palMapqk.c in palpy/cextern/pal
        """
        VF=0.21094502
        pal_das2r=4.8481368110953599358991410235794797595635330237270e-6;

        np.random.seed(18)
        nSamples = 1000

        mjdList = np.random.random_sample(20)*20000.0 + 45000.0

        for mjd in mjdList:

            raList_icrs = np.random.random_sample(nSamples)*2.0*np.pi
            decList_icrs = (np.random.random_sample(nSamples)-0.5)*np.pi

            # stars' original position in Cartesian space
            x_list_icrs = np.cos(decList_icrs)*np.cos(raList_icrs)
            y_list_icrs = np.cos(decList_icrs)*np.sin(raList_icrs)
            z_list_icrs = np.sin(decList_icrs)


            pm_ra = (np.random.random_sample(nSamples)-0.5)*radiansFromArcsec(1.0)
            pm_dec = (np.random.random_sample(nSamples)-0.5)*radiansFromArcsec(1.0)
            px = np.random.random_sample(nSamples)*radiansFromArcsec(1.0)
            v_rad = np.random.random_sample(nSamples)*200.0


            ra_list_pm, dec_list_pm = _applyProperMotion(raList_icrs, decList_icrs,
                                                         pm_ra*np.cos(decList_icrs),
                                                         pm_dec, px, v_rad, mjd=ModifiedJulianDate(TAI=mjd))

            # stars' Cartesian position after proper motion is applied
            x_list_pm = np.cos(dec_list_pm)*np.cos(ra_list_pm)
            y_list_pm = np.cos(dec_list_pm)*np.sin(ra_list_pm)
            z_list_pm = np.sin(dec_list_pm)

            ###############################################################
            # The code below is copied from palMapqk.c in palpy/cextern/pal
            params = pal.mappa(2000.0, mjd)
            pmt = params[0]
            eb = np.array([params[1], params[2], params[3]])

            pxr = px*pal_das2r

            w = VF*v_rad*pxr

            motion_per_year = np.array([-1.0*pm_ra*y_list_icrs - pm_dec*np.cos(raList_icrs)*np.sin(decList_icrs) + w*x_list_icrs,
                                     pm_ra*x_list_icrs - pm_dec*np.sin(raList_icrs)*np.sin(decList_icrs) + w*y_list_icrs,
                                     pm_dec*np.cos(decList_icrs) + w*z_list_icrs])


            xyz_control = np.array([
                                      x_list_icrs + pmt*motion_per_year[0] - pxr*eb[0],
                                      y_list_icrs + pmt*motion_per_year[1] - pxr*eb[1],
                                      z_list_icrs + pmt*motion_per_year[2] - pxr*eb[2]
                                      ])

            xyz_norm = np.sqrt(np.power(xyz_control[0],2) + np.power(xyz_control[1],2) + np.power(xyz_control[2],2))

            # stars' Cartesian position after applying the control proper motion method
            xyz_control[0] = xyz_control[0]/xyz_norm
            xyz_control[1] = xyz_control[1]/xyz_norm
            xyz_control[2] = xyz_control[2]/xyz_norm

            # this is the Cartesian distance between the stars' positions as found by _applyProperMotion
            # and the distance as found by the control proper motion code above
            distance = np.sqrt(np.power(x_list_pm-xyz_control[0],2) + np.power(y_list_pm-xyz_control[1],2) +
                                  np.power(z_list_pm-xyz_control[2],2))

            # this is the Cartesian distance between the stars' original positions on the celestial sphere
            # and their positions after the control proper motion was applied
            correction = np.sqrt(np.power(xyz_control[0]-x_list_icrs,2) + np.power(xyz_control[1]-y_list_icrs,2) +
                                    np.power(xyz_control[2]-z_list_icrs,2))

            dex = np.argmax(distance)
            msg = 'pm %e %e vr %e px %e; time %e; err %e arcsec; corr %e' % \
            (arcsecFromRadians(pm_ra[dex]), arcsecFromRadians(pm_dec[dex]),
             v_rad[dex], arcsecFromRadians(px[dex]), pmt, arcsecFromRadians(distance[dex]),
             arcsecFromRadians(correction[dex]))

            self.assertLess((distance/correction).max(), 0.01, msg=msg)
            # demand that the two methods agree on the stars' new positions to within one part in 100


    def test_appGeoFromICRS(self):
        """
        Test conversion between ICRS RA, Dec and apparent geocentric ICRS.

        Apparent, geocentric RA, Dec of objects will be taken from this website

        http://aa.usno.navy.mil/data/docs/geocentric.php

        dates converted to JD using this website

        http://aa.usno.navy.mil/data/docs/geocentric.php

        """

        hours = np.radians(360.0/24.0)
        minutes = hours/60.0
        seconds = minutes/60.0

        # test on Arcturus
        # data taken from
        # http://aa.usno.navy.mil/data/docs/geocentric.php
        ra_icrs = 14.0*hours + 15.0*minutes + 39.67207*seconds
        dec_icrs = np.radians(19.0 + 10.0/60.0 + 56.673/3600.0)
        pm_ra = radiansFromArcsec(-1.0939)
        pm_dec = radiansFromArcsec(-2.00006)
        v_rad = -5.19
        px = radiansFromArcsec(0.08883)

        mjd_list = []
        ra_app_list = []
        dec_app_list = []

        #jd (UT)
        jd = 2457000.375000
        mjd = jd-2400000.5

        mjd_list.append(mjd)
        ra_app_list.append(14.0*hours + 16.0*minutes + 19.59*seconds)
        dec_app_list.append(np.radians(19.0 + 6.0/60.0 + 19.56/3600.0))

        jd = 2457187.208333
        mjd = jd-2400000.5
        mjd_list.append(mjd)
        ra_app_list.append(14.0*hours + 16.0*minutes + 22.807*seconds)
        dec_app_list.append(np.radians(19.0+6.0/60.0+18.12/3600.0))

        jd = 2457472.625000
        mjd = jd-2400000.5
        mjd_list.append(mjd)
        ra_app_list.append(14.0*hours + 16.0*minutes + 24.946*seconds)
        dec_app_list.append(np.radians(19.0 + 5.0/60.0 + 49.65/3600.0))

        for mjd, ra_app, dec_app in zip(mjd_list, ra_app_list, dec_app_list):
            obs = ObservationMetaData(mjd=mjd)

            ra_test, dec_test = _appGeoFromICRS(np.array([ra_icrs]), np.array([dec_icrs]),
                                                mjd=obs.mjd,
                                                pm_ra=np.array([pm_ra]),
                                                pm_dec=np.array([pm_dec]),
                                                v_rad=np.array([v_rad]),
                                                parallax=np.array([px]),
                                                epoch=2000.0)

            distance = arcsecFromRadians(haversine(ra_app, dec_app, ra_test[0], dec_test[0]))
            self.assertLess(distance, 0.1)


        # test on Sirius
        # data taken from
        # http://simbad.u-strasbg.fr/simbad/sim-id?Ident=Sirius
        ra_icrs = 6.0*hours + 45.0*minutes + 8.91728*seconds
        dec_icrs = np.radians(-16.0 - 42.0/60.0 -58.0171/3600.0)
        pm_ra = radiansFromArcsec(-0.54601)
        pm_dec = radiansFromArcsec(-1.22307)
        px = radiansFromArcsec(0.37921)
        v_rad = -5.5

        mjd_list = []
        ra_app_list = []
        dec_app_list = []

        jd = 2457247.000000
        mjd_list.append(jd-2400000.5)
        ra_app_list.append(6.0*hours + 45.0*minutes + 49.276*seconds)
        dec_app_list.append(np.radians(-16.0 - 44.0/60.0 - 18.69/3600.0))

        jd = 2456983.958333
        mjd_list.append(jd-2400000.5)
        ra_app_list.append(6.0*hours + 45.0*minutes + 49.635*seconds)
        dec_app_list.append(np.radians(-16.0 - 44.0/60.0 - 17.04/3600.0))

        jd = 2457523.958333
        mjd_list.append(jd-2400000.5)
        ra_app_list.append(6.0*hours + 45.0*minutes + 50.99*seconds)
        dec_app_list.append(np.radians(-16.0 - 44.0/60.0 - 39.76/3600.0))

        for mjd, ra_app, dec_app in zip(mjd_list, ra_app_list, dec_app_list):
            obs = ObservationMetaData(mjd=mjd)

            ra_test, dec_test = _appGeoFromICRS(np.array([ra_icrs]), np.array([dec_icrs]),
                                                mjd=obs.mjd,
                                                pm_ra=np.array([pm_ra]),
                                                pm_dec=np.array([pm_dec]),
                                                v_rad=np.array([v_rad]),
                                                parallax=np.array([px]),
                                                epoch=2000.0)

            distance = arcsecFromRadians(haversine(ra_app, dec_app, ra_test[0], dec_test[0]))
            self.assertLess(distance, 0.1)



    def test_icrsFromAppGeo(self):
        """
        Test that _icrsFromAppGeo really inverts _appGeoFromICRS.

        This test is a tricky because _appGeoFromICRS applies
        light deflection due to the sun.  _icrsFromAppGeo does not
        account for that effect, which is fine, because it is only
        meant to map pointing RA, Decs to RA, Decs on fatboy.

        _icrsFromAppGeo should invert _appGeoFromICRS to within
        0.01 arcsec at an angular distance greater than 45 degrees
        from the sun.
        """

        np.random.seed(412)
        nSamples = 100

        mjd2000 = pal.epb(2000.0) # convert epoch to mjd

        for mjd in (53000.0, 53241.6, 58504.6):

            ra_in = np.random.random_sample(nSamples)*2.0*np.pi
            dec_in = (np.random.random_sample(nSamples)-0.5)*np.pi


            ra_app, dec_app = _appGeoFromICRS(ra_in, dec_in, mjd=ModifiedJulianDate(TAI=mjd))

            ra_icrs, dec_icrs = _icrsFromAppGeo(ra_app, dec_app,
                                                epoch=2000.0, mjd=ModifiedJulianDate(TAI=mjd))

            self.assertFalse(np.isnan(ra_icrs).any())
            self.assertFalse(np.isnan(dec_icrs).any())

            valid_pts = np.where(_distanceToSun(ra_in, dec_in, mjd)>0.25*np.pi)[0]

            self.assertGreater(len(valid_pts), 0)

            distance = arcsecFromRadians(pal.dsepVector(ra_in[valid_pts], dec_in[valid_pts],
                                         ra_icrs[valid_pts], dec_icrs[valid_pts]))

            self.assertLess(distance.max(), 0.01)


    def test_icrsFromObserved(self):
        """
        Test that _icrsFromObserved really inverts _observedFromAppGeo.

        In this case, the method is only reliable at distances of more than
        45 degrees from the sun and at zenith distances less than 70 degrees.
        """

        np.random.seed(412)
        nSamples = 100

        mjd2000 = pal.epb(2000.0) # convert epoch to mjd

        site = Site(name='LSST')

        for mjd in (53000.0, 53241.6, 58504.6):
            for includeRefraction in (True, False):
                for raPointing in (23.5, 256.9, 100.0):
                    for decPointing in (-12.0, 45.0, 66.8):

                        obs = ObservationMetaData(mjd=mjd, site=site)

                        raZenith, decZenith = _raDecFromAltAz(0.5*np.pi, 0.0, obs)

                        obs = ObservationMetaData(pointingRA=raPointing, pointingDec=decPointing,
                                                  mjd=mjd, site=site)

                        rr = np.random.random_sample(nSamples)*np.radians(50.0)
                        theta = np.random.random_sample(nSamples)*2.0*np.pi

                        ra_in = raZenith + rr*np.cos(theta)
                        dec_in = decZenith + rr*np.sin(theta)

                        # test a round-trip between observedFromICRS and icrsFromObserved
                        ra_obs, dec_obs = _observedFromICRS(ra_in, dec_in, obs_metadata=obs,
                                                            includeRefraction=includeRefraction,
                                                            epoch=2000.0)

                        ra_icrs, dec_icrs = _icrsFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                              includeRefraction=includeRefraction,
                                                              epoch=2000.0)

                        valid_pts = np.where(_distanceToSun(ra_in, dec_in, mjd)>0.25*np.pi)[0]

                        self.assertGreater(len(valid_pts), 0)

                        distance = arcsecFromRadians(pal.dsepVector(ra_in[valid_pts], dec_in[valid_pts],
                                                     ra_icrs[valid_pts], dec_icrs[valid_pts]))

                        self.assertLess(distance.max(), 0.01)


                        # test a round-trip between observedFromAppGeo and appGeoFromObserved
                        ra_obs, dec_obs = _observedFromAppGeo(ra_in, dec_in, obs_metadata=obs,
                                                              includeRefraction=includeRefraction)
                        ra_app, dec_app = _appGeoFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                                includeRefraction=includeRefraction)

                        distance = arcsecFromRadians(pal.dsepVector(ra_in[valid_pts], dec_in[valid_pts],
                                                     ra_app[valid_pts], dec_app[valid_pts]))

                        self.assertLess(distance.max(), 0.01)


    def test_icrsFromObservedExceptions(self):
        """
        Test that _icrsFromObserved raises exceptions when it is supposed to.
        """
        np.random.seed(33)
        ra_in = np.random.random_sample(10)
        dec_in = np.random.random_sample(10)
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrsFromObserved(ra_in, dec_in, epoch=2000.0)
        self.assertEqual(context.exception.args[0],
                         "cannot call icrsFromObserved; obs_metadata is None")

        obs = ObservationMetaData(pointingRA=23.0, pointingDec=-19.0)
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrsFromObserved(ra_in, dec_in, epoch=2000.0, obs_metadata=obs)
        self.assertEqual(context.exception.args[0],
                         "cannot call icrsFromObserved; obs_metadata.mjd is None")

        obs = ObservationMetaData(pointingRA=23.0, pointingDec=-19.0,
                                  mjd=ModifiedJulianDate(TAI=52344.0))
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrsFromObserved(ra_in, dec_in, obs_metadata=obs)
        self.assertEqual(context.exception.args[0],
                         "cannot call icrsFromObserved; you have not specified an epoch")

        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _icrsFromObserved(ra_in[:3], dec_in, obs_metadata=obs, epoch=2000.0)
        self.assertEqual(context.exception.args[0],
                         "You passed 3 RAs but 10 Decs to icrsFromObserved")


    def test_appGeoFromObserved(self):
        """
        Test that _appGeoFromObserved really does invert _observedFromAppGeo
        """
        mjd = 58350.0
        site = Site(longitude=np.degrees(0.235), latitude=np.degrees(-1.2), name='LSST')
        raCenter, decCenter = raDecFromAltAz(90.0, 0.0,
                                             ObservationMetaData(mjd=mjd, site=site))

        obs = ObservationMetaData(pointingRA=raCenter, pointingDec=decCenter,
                                  mjd=ModifiedJulianDate(TAI=58350.0),
                                  site=site)

        np.random.seed(125543)
        nSamples = 200

        # Note: the PALPY routines in question start to become inaccurate at
        # a zenith distance of about 75 degrees, so we restrict our test points
        # to be within 50 degrees of the telescope pointing, which is at zenith
        # in a flat sky approximation
        rr = np.random.random_sample(nSamples)*np.radians(50.0)
        theta = np.random.random_sample(nSamples)*2.0*np.pi
        ra_in = np.radians(raCenter) + rr*np.cos(theta)
        dec_in = np.radians(decCenter) + rr*np.sin(theta)

        xx_in = np.cos(dec_in)*np.cos(ra_in)
        yy_in = np.cos(dec_in)*np.sin(ra_in)
        zz_in = np.sin(dec_in)

        for includeRefraction in [True, False]:
            for wavelength in (0.5, 0.3, 0.7):
                ra_obs, dec_obs = _observedFromAppGeo(ra_in, dec_in, obs_metadata=obs,
                                                      wavelength=wavelength,
                                                      includeRefraction=includeRefraction)

                ra_out, dec_out = _appGeoFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                      wavelength=wavelength,
                                                      includeRefraction=includeRefraction)


                xx_out = np.cos(dec_out)*np.cos(ra_out)
                yy_out = np.cos(dec_out)*np.sin(ra_out)
                zz_out = np.sin(dec_out)

                distance = np.sqrt(np.power(xx_in-xx_out,2) +
                                      np.power(yy_in-yy_out,2) +
                                      np.power(zz_in-zz_out,2))

                self.assertLess(distance.max(), 1.0e-12)


    def test_appGeoFromObservedExceptions(self):
        """
        Test that _appGeoFromObserved raises exceptions where expected
        """
        np.random.seed(12)
        ra_in = np.random.random_sample(10)*2.0*np.pi
        dec_in = (np.random.random_sample(10)-0.5)*np.pi

        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _appGeoFromObserved(ra_in, dec_in)
        self.assertEqual(context.exception.args[0],
                         "Cannot call appGeoFromObserved without an obs_metadata")

        obs = ObservationMetaData(pointingRA=25.0, pointingDec=-12.0,
                                  site=None, mjd=ModifiedJulianDate(TAI=52000.0))

        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _appGeoFromObserved(ra_in, dec_in, obs_metadata=obs)
        self.assertEqual(context.exception.args[0],
                         "Cannot call appGeoFromObserved: obs_metadata has no site info")

        obs = ObservationMetaData(pointingRA=25.0, pointingDec=-12.0)
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _appGeoFromObserved(ra_in, dec_in, obs_metadata=obs)
        self.assertEqual(context.exception.args[0],
                         "Cannot call appGeoFromObserved: obs_metadata has no mjd")

        obs = ObservationMetaData(pointingRA=25.0, pointingDec=-12.0,
                                  mjd=ModifiedJulianDate(TAI=52000.0))
        with self.assertRaises(RuntimeError) as context:
            ra_out, dec_out = _appGeoFromObserved(ra_in[:2], dec_in, obs_metadata=obs)
        self.assertEqual(context.exception.args[0],
                         "You passed 2 RAs but 10 Decs to appGeoFromObserved")


    def testRefractionCoefficients(self):
        output=refractionCoefficients(wavelength=5000.0, site=self.obs_metadata.site)

        self.assertAlmostEqual(output[0],2.295817926320665320e-04,6)
        self.assertAlmostEqual(output[1],-2.385964632924575670e-07,6)

    def testApplyRefraction(self):
        coeffs=refractionCoefficients(wavelength=5000.0, site=self.obs_metadata.site)

        output=applyRefraction(0.25*np.pi,coeffs[0],coeffs[1])

        self.assertAlmostEqual(output,7.851689251070859132e-01,6)



def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(astrometryUnitTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)
