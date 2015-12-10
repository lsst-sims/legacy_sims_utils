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

import numpy

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

from lsst.sims.utils import _applyPrecession, _applyProperMotion
from lsst.sims.utils import _appGeoFromICRS, _observedFromAppGeo
from lsst.sims.utils import _observedFromICRS, _icrsFromObserved
from lsst.sims.utils import _appGeoFromObserved, _icrsFromAppGeo
from lsst.sims.utils import refractionCoefficients, applyRefraction

def makeObservationMetaData():
    #create a cartoon ObservationMetaData object
    mjd = 52000.0
    alt = numpy.pi/2.0
    az = 0.0
    band = 'r'
    testSite = Site(latitude=0.5, longitude=1.1, height=3000, meanTemperature=260.0,
                    meanPressure=725.0, lapseRate=0.005)

    obsTemp = ObservationMetaData(site=testSite, mjd=mjd)

    centerRA, centerDec = _raDecFromAltAz(alt, az, obsTemp)

    rotTel = _getRotTelPos(centerRA, centerDec, obsTemp, 0.0)

    obsDict = calcObsDefaults(centerRA, centerDec, alt, az, rotTel, mjd, band,
                 testSite.longitude, testSite.latitude)

    obsDict['Opsim_expmjd'] = mjd
    radius = 0.1
    phoSimMetaData = OrderedDict([
                      (k, (obsDict[k],numpy.dtype(type(obsDict[k])))) for k in obsDict])

    obs_metadata = ObservationMetaData(boundType='circle', boundLength=2.0*radius,
                                       phoSimMetaData=phoSimMetaData, site=testSite)

    return obs_metadata

def makeRandomSample(raCenter=None, decCenter=None, radius=None):
    #create a random sample of object data

    nsamples=100
    numpy.random.seed(32)

    if raCenter is None or decCenter is None or radius is None:
        ra = numpy.random.sample(nsamples)*2.0*numpy.pi
        dec = (numpy.random.sample(nsamples)-0.5)*numpy.pi
    else:
        rr = numpy.random.sample(nsamples)*radius
        theta = numpy.random.sample(nsamples)*2.0*numpy.pi
        ra = raCenter + rr*numpy.cos(theta)
        dec = decCenter + rr*numpy.cos(theta)

    pm_ra = (numpy.random.sample(nsamples)-0.5)*0.1
    pm_dec = (numpy.random.sample(nsamples)-0.5)*0.1
    parallax = numpy.random.sample(nsamples)*0.01
    v_rad = numpy.random.sample(nsamples)*1000.0

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
        self.metadata['pointingRA'] = (numpy.radians(200.0), float)
        self.metadata['pointingDec'] = (numpy.radians(-30.0), float)
        self.metadata['Opsim_rotskypos'] = (1.0, float)

        self.obs_metadata=ObservationMetaData(mjd=50984.371741,
                                     boundType='circle',
                                     boundLength=0.05,
                                     phoSimMetaData=self.metadata)

        self.tol=1.0e-5

    def tearDown(self):
        del self.obs_metadata
        del self.metadata
        del self.tol


    def testAstrometryExceptions(self):
        """
        Test to make sure that stand-alone astrometry methods raise an exception when they are called without
        the necessary arguments
        """
        obs_metadata = makeObservationMetaData()
        ra, dec, pm_ra, pm_dec, parallax, v_rad = makeRandomSample()

        raShort = numpy.array([1.0])
        decShort = numpy.array([1.0])


        ##########test refractionCoefficients
        self.assertRaises(RuntimeError, refractionCoefficients)
        site = obs_metadata.site
        x, y = refractionCoefficients(site=site)

        ##########test applyRefraction
        zd = 0.1
        rzd = applyRefraction(zd, x, y)

        zd = [0.1, 0.2]
        self.assertRaises(RuntimeError, applyRefraction, zd, x, y)

        zd = numpy.array([0.1, 0.2])
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

        pm_raShort = numpy.array([pm_ra[0]])
        pm_decShort = numpy.array([pm_dec[0]])
        parallaxShort = numpy.array([parallax[0]])
        v_radShort = numpy.array([v_rad[0]])

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
                                  site=Site())
        self.assertRaises(RuntimeError, _observedFromAppGeo, ra, dec, obs_metadata=dummy)

        #test mismatches
        dummy=ObservationMetaData(pointingRA=obs_metadata.pointingRA,
                                  pointingDec=obs_metadata.pointingDec,
                                  mjd=obs_metadata.mjd,
                                  site=Site())

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

        ra=numpy.zeros((3),dtype=float)
        dec=numpy.zeros((3),dtype=float)

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

        numpy.random.seed(18)
        nSamples = 1000

        mjdList = numpy.random.random_sample(20)*20000.0 + 45000.0

        for mjd in mjdList:

            raList_icrs = numpy.random.random_sample(nSamples)*2.0*numpy.pi
            decList_icrs = (numpy.random.random_sample(nSamples)-0.5)*numpy.pi

            # stars' original position in Cartesian space
            x_list_icrs = numpy.cos(decList_icrs)*numpy.cos(raList_icrs)
            y_list_icrs = numpy.cos(decList_icrs)*numpy.sin(raList_icrs)
            z_list_icrs = numpy.sin(decList_icrs)


            pm_ra = (numpy.random.random_sample(nSamples)-0.5)*radiansFromArcsec(1.0)
            pm_dec = (numpy.random.random_sample(nSamples)-0.5)*radiansFromArcsec(1.0)
            px = numpy.random.random_sample(nSamples)*radiansFromArcsec(1.0)
            v_rad = numpy.random.random_sample(nSamples)*200.0


            ra_list_pm, dec_list_pm = _applyProperMotion(raList_icrs, decList_icrs,
                                                         pm_ra*numpy.cos(decList_icrs),
                                                         pm_dec, px, v_rad, mjd=mjd)

            # stars' Cartesian position after proper motion is applied
            x_list_pm = numpy.cos(dec_list_pm)*numpy.cos(ra_list_pm)
            y_list_pm = numpy.cos(dec_list_pm)*numpy.sin(ra_list_pm)
            z_list_pm = numpy.sin(dec_list_pm)

            ###############################################################
            # The code below is copied from palMapqk.c in palpy/cextern/pal
            params = pal.mappa(2000.0, mjd)
            pmt = params[0]
            eb = numpy.array([params[1], params[2], params[3]])

            pxr = px*pal_das2r

            w = VF*v_rad*pxr

            motion_per_year = numpy.array([-1.0*pm_ra*y_list_icrs - pm_dec*numpy.cos(raList_icrs)*numpy.sin(decList_icrs) + w*x_list_icrs,
                                     pm_ra*x_list_icrs - pm_dec*numpy.sin(raList_icrs)*numpy.sin(decList_icrs) + w*y_list_icrs,
                                     pm_dec*numpy.cos(decList_icrs) + w*z_list_icrs])


            xyz_control = numpy.array([
                                      x_list_icrs + pmt*motion_per_year[0] - pxr*eb[0],
                                      y_list_icrs + pmt*motion_per_year[1] - pxr*eb[1],
                                      z_list_icrs + pmt*motion_per_year[2] - pxr*eb[2]
                                      ])

            xyz_norm = numpy.sqrt(numpy.power(xyz_control[0],2) + numpy.power(xyz_control[1],2) + numpy.power(xyz_control[2],2))

            # stars' Cartesian position after applying the control proper motion method
            xyz_control[0] = xyz_control[0]/xyz_norm
            xyz_control[1] = xyz_control[1]/xyz_norm
            xyz_control[2] = xyz_control[2]/xyz_norm

            # this is the Cartesian distance between the stars' positions as found by _applyProperMotion
            # and the distance as found by the control proper motion code above
            distance = numpy.sqrt(numpy.power(x_list_pm-xyz_control[0],2) + numpy.power(y_list_pm-xyz_control[1],2) +
                                  numpy.power(z_list_pm-xyz_control[2],2))

            # this is the Cartesian distance between the stars' original positions on the celestial sphere
            # and their positions after the control proper motion was applied
            correction = numpy.sqrt(numpy.power(xyz_control[0]-x_list_icrs,2) + numpy.power(xyz_control[1]-y_list_icrs,2) +
                                    numpy.power(xyz_control[2]-z_list_icrs,2))

            dex = numpy.argmax(distance)
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

        hours = numpy.radians(360.0/24.0)
        minutes = hours/60.0
        seconds = minutes/60.0

        # test on Arcturus
        # data taken from
        # http://aa.usno.navy.mil/data/docs/geocentric.php
        ra_icrs = 14.0*hours + 15.0*minutes + 39.67207*seconds
        dec_icrs = numpy.radians(19.0 + 10.0/60.0 + 56.673/3600.0)
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
        dec_app_list.append(numpy.radians(19.0 + 6.0/60.0 + 19.56/3600.0))

        jd = 2457187.208333
        mjd = jd-2400000.5
        mjd_list.append(mjd)
        ra_app_list.append(14.0*hours + 16.0*minutes + 22.807*seconds)
        dec_app_list.append(numpy.radians(19.0+6.0/60.0+18.12/3600.0))

        jd = 2457472.625000
        mjd = jd-2400000.5
        mjd_list.append(mjd)
        ra_app_list.append(14.0*hours + 16.0*minutes + 24.946*seconds)
        dec_app_list.append(numpy.radians(19.0 + 5.0/60.0 + 49.65/3600.0))

        for mjd, ra_app, dec_app in zip(mjd_list, ra_app_list, dec_app_list):
            obs = ObservationMetaData(mjd=mjd)

            ra_test, dec_test = _appGeoFromICRS(numpy.array([ra_icrs]), numpy.array([dec_icrs]),
                                                mjd=obs.mjd,
                                                pm_ra=numpy.array([pm_ra]),
                                                pm_dec=numpy.array([pm_dec]),
                                                v_rad=numpy.array([v_rad]),
                                                parallax=numpy.array([px]),
                                                epoch=2000.0)

            distance = arcsecFromRadians(haversine(ra_app, dec_app, ra_test[0], dec_test[0]))
            self.assertLess(distance, 0.1)


        # test on Sirius
        # data taken from
        # http://simbad.u-strasbg.fr/simbad/sim-id?Ident=Sirius
        ra_icrs = 6.0*hours + 45.0*minutes + 8.91728*seconds
        dec_icrs = numpy.radians(-16.0 - 42.0/60.0 -58.0171/3600.0)
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
        dec_app_list.append(numpy.radians(-16.0 - 44.0/60.0 - 18.69/3600.0))

        jd = 2456983.958333
        mjd_list.append(jd-2400000.5)
        ra_app_list.append(6.0*hours + 45.0*minutes + 49.635*seconds)
        dec_app_list.append(numpy.radians(-16.0 - 44.0/60.0 - 17.04/3600.0))

        jd = 2457523.958333
        mjd_list.append(jd-2400000.5)
        ra_app_list.append(6.0*hours + 45.0*minutes + 50.99*seconds)
        dec_app_list.append(numpy.radians(-16.0 - 44.0/60.0 - 39.76/3600.0))

        for mjd, ra_app, dec_app in zip(mjd_list, ra_app_list, dec_app_list):
            obs = ObservationMetaData(mjd=mjd)

            ra_test, dec_test = _appGeoFromICRS(numpy.array([ra_icrs]), numpy.array([dec_icrs]),
                                                mjd=obs.mjd,
                                                pm_ra=numpy.array([pm_ra]),
                                                pm_dec=numpy.array([pm_dec]),
                                                v_rad=numpy.array([v_rad]),
                                                parallax=numpy.array([px]),
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

        numpy.random.seed(412)
        nSamples = 100

        mjd2000 = pal.epb(2000.0) # convert epoch to mjd

        for mjd in (53000.0, 53241.6, 58504.6):

            params = pal.mappa(2000.0, mjd)
            sunToEarth = params[4:7] # unit vector pointing from Sun to Earth

            ra_in = numpy.random.random_sample(nSamples)*2.0*numpy.pi
            dec_in = (numpy.random.random_sample(nSamples)-0.5)*numpy.pi

            earthToStar = pal.dcs2cVector(ra_in, dec_in) # each row is a unit vector pointing to the star

            solarDotProduct = numpy.array([(-1.0*sunToEarth*earthToStar[ii]).sum()
                                           for ii in range(earthToStar.shape[0])])

            ra_app, dec_app = _appGeoFromICRS(ra_in, dec_in, mjd=ModifiedJulianDate(TAI=mjd))

            ra_icrs, dec_icrs = _icrsFromAppGeo(ra_app, dec_app,
                                                epoch=2000.0, mjd=ModifiedJulianDate(TAI=mjd))

            self.assertFalse(numpy.isnan(ra_icrs).any())
            self.assertFalse(numpy.isnan(dec_icrs).any())

            valid_pts = numpy.where(solarDotProduct<numpy.cos(0.25*numpy.pi))[0]
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

        numpy.random.seed(412)
        nSamples = 100

        mjd2000 = pal.epb(2000.0) # convert epoch to mjd

        site = Site()

        for mjd in (53000.0, 53241.6, 58504.6):
            for includeRefraction in (True, False):
                for raPointing in (23.5, 256.9, 100.0):
                    for decPointing in (-12.0, 45.0, 66.8):

                        obs = ObservationMetaData(mjd=mjd, site=site)

                        raZenith, decZenith = _raDecFromAltAz(0.5*numpy.pi, 0.0, obs)

                        obs = ObservationMetaData(pointingRA=raPointing, pointingDec=decPointing,
                                                  mjd=mjd, site=site)

                        params = pal.mappa(2000.0, mjd)
                        sunToEarth = params[4:7] # unit vector pointing from Sun to Earth

                        rr = numpy.random.random_sample(nSamples)*numpy.radians(50.0)
                        theta = numpy.random.random_sample(nSamples)*2.0*numpy.pi

                        ra_in = raZenith + rr*numpy.cos(theta)
                        dec_in = decZenith + rr*numpy.sin(theta)

                        earthToStar = pal.dcs2cVector(ra_in, dec_in) # each row is a unit vector pointing to the star

                        solarDotProduct = numpy.array([(-1.0*sunToEarth*earthToStar[ii]).sum()
                                                       for ii in range(earthToStar.shape[0])])


                        # test a round-trip between observedFromICRS and icrsFromObserved
                        ra_obs, dec_obs = _observedFromICRS(ra_in, dec_in, obs_metadata=obs,
                                                            includeRefraction=includeRefraction,
                                                            epoch=2000.0)

                        ra_icrs, dec_icrs = _icrsFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                              includeRefraction=includeRefraction,
                                                              epoch=2000.0)

                        valid_pts = numpy.where(solarDotProduct<numpy.cos(0.25*numpy.pi))[0]
                        self.assertGreater(len(valid_pts), 0)

                        distance = arcsecFromRadians(pal.dsepVector(ra_in[valid_pts], dec_in[valid_pts],
                                                     ra_icrs[valid_pts], dec_icrs[valid_pts]))

                        self.assertLess(distance.max(), 0.01)


                        # test a round-trip between observedFromAppGeo and appGeoFromObserved
                        ra_obs, dec_obs = _observedFromAppGeo(ra_in, dec_in, obs_metadata=obs,
                                                              includeRefraction=includeRefraction)
                        ra_app, dec_app = _appGeoFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                                includeRefraction=includeRefraction)

                        valid_pts = numpy.where(solarDotProduct<numpy.cos(0.25*numpy.pi))[0]
                        self.assertGreater(len(valid_pts), 0)

                        distance = arcsecFromRadians(pal.dsepVector(ra_in[valid_pts], dec_in[valid_pts],
                                                     ra_app[valid_pts], dec_app[valid_pts]))

                        self.assertLess(distance.max(), 0.01)


    def test_icrsFromObservedExceptions(self):
        """
        Test that _icrsFromObserved raises exceptions when it is supposed to.
        """
        numpy.random.seed(33)
        ra_in = numpy.random.random_sample(10)
        dec_in = numpy.random.random_sample(10)
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
        site = Site(longitude=0.235, latitude=-1.2)
        raCenter, decCenter = raDecFromAltAz(90.0, 0.0,
                                             ObservationMetaData(mjd=mjd, site=site))

        obs = ObservationMetaData(pointingRA=raCenter, pointingDec=decCenter,
                                  mjd=ModifiedJulianDate(TAI=58350.0),
                                  site=site)

        numpy.random.seed(125543)
        nSamples = 200

        # Note: the PALPY routines in question start to become inaccurate at
        # a zenith distance of about 75 degrees, so we restrict our test points
        # to be within 50 degrees of the telescope pointing, which is at zenith
        # in a flat sky approximation
        rr = numpy.random.random_sample(nSamples)*numpy.radians(50.0)
        theta = numpy.random.random_sample(nSamples)*2.0*numpy.pi
        ra_in = numpy.radians(raCenter) + rr*numpy.cos(theta)
        dec_in = numpy.radians(decCenter) + rr*numpy.sin(theta)

        xx_in = numpy.cos(dec_in)*numpy.cos(ra_in)
        yy_in = numpy.cos(dec_in)*numpy.sin(ra_in)
        zz_in = numpy.sin(dec_in)

        for includeRefraction in [True, False]:
            for wavelength in (0.5, 0.3, 0.7):
                ra_obs, dec_obs = _observedFromAppGeo(ra_in, dec_in, obs_metadata=obs,
                                                      wavelength=wavelength,
                                                      includeRefraction=includeRefraction)

                ra_out, dec_out = _appGeoFromObserved(ra_obs, dec_obs, obs_metadata=obs,
                                                      wavelength=wavelength,
                                                      includeRefraction=includeRefraction)


                xx_out = numpy.cos(dec_out)*numpy.cos(ra_out)
                yy_out = numpy.cos(dec_out)*numpy.sin(ra_out)
                zz_out = numpy.sin(dec_out)

                distance = numpy.sqrt(numpy.power(xx_in-xx_out,2) +
                                      numpy.power(yy_in-yy_out,2) +
                                      numpy.power(zz_in-zz_out,2))

                self.assertLess(distance.max(), 1.0e-12)


    def test_appGeoFromObservedExceptions(self):
        """
        Test that _appGeoFromObserved raises exceptions where expected
        """
        numpy.random.seed(12)
        ra_in = numpy.random.random_sample(10)*2.0*numpy.pi
        dec_in = (numpy.random.random_sample(10)-0.5)*numpy.pi

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

        output=applyRefraction(0.25*numpy.pi,coeffs[0],coeffs[1])

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
