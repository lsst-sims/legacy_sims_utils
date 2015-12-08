import unittest
import numpy
import lsst.utils.tests as utilsTests

from lsst.sims.utils import raDecFromNativeLonLat, nativeLonLatFromRaDec
from lsst.sims.utils import observedFromICRS, icrsFromObserved
from lsst.sims.utils import ObservationMetaData, haversine
from lsst.sims.utils import arcsecFromRadians, raDecFromAltAz, Site

class NativeLonLatTest(unittest.TestCase):

    def testNativeLonLat(self):
        """
        Test that nativeLonLatFromRaDec works by considering stars and pointings
        at intuitive locations
        """

        mjd = 53855.0

        raList_obs = [0.0, 0.0, 0.0, 270.0]
        decList_obs = [90.0, 90.0, 0.0, 0.0]

        raPointList_obs = [0.0, 270.0, 270.0, 0.0]
        decPointList_obs = [0.0, 0.0,0.0, 0.0]

        lonControlList = [180.0, 180.0, 90.0, 270.0]
        latControlList = [0.0, 0.0, 0.0, 0.0]

        for rr_obs, dd_obs, rp_obs, dp_obs, lonc, latc in \
        zip(raList_obs, decList_obs, raPointList_obs, decPointList_obs, lonControlList, latControlList):

            obsTemp = ObservationMetaData(mjd=mjd)

            rr, dd = icrsFromObserved(numpy.array([rr_obs, rp_obs]),
                                      numpy.array([dd_obs, dp_obs]),
                                      obs_metadata=obsTemp,
                                      epoch=2000.0, includeRefraction=True)

            obs = ObservationMetaData(pointingRA=rr[1], pointingDec=dd[1], mjd=mjd)
            lon, lat = nativeLonLatFromRaDec(rr[0], dd[0], obs)
            distance = arcsecFromRadians(haversine(lon, lat, lonc, latc))
            self.assertLess(distance, 1.0)


    def testNativeLongLatComplicated(self):
        """
        Test that nativeLongLatFromRaDec works by considering stars and pointings
        at non-intuitive locations.
        """

        numpy.random.seed(42)
        nPointings = 10
        raPointingList_icrs = numpy.random.random_sample(nPointings)*360.0
        decPointingList_icrs = numpy.random.random_sample(nPointings)*180.0 - 90.0
        mjdList = numpy.random.random_sample(nPointings)*10000.0 + 43000.0

        nStars = 10
        for raPointing_icrs, decPointing_icrs, mjd in \
            zip(raPointingList_icrs, decPointingList_icrs, mjdList):

            obs = ObservationMetaData(pointingRA=raPointing_icrs, pointingDec=decPointing_icrs, mjd=mjd)
            raList_icrs = numpy.random.random_sample(nStars)*360.0
            decList_icrs = numpy.random.random_sample(nStars)*180.0 - 90.0
            raList_obs, decList_obs = observedFromICRS(raList_icrs, decList_icrs, obs_metadata=obs,
                                                       epoch=2000.0, includeRefraction=True)

            obsTemp = ObservationMetaData(mjd=mjd)
            ra_temp, dec_temp = observedFromICRS(numpy.array([raPointing_icrs]),
                                                 numpy.array([decPointing_icrs]),
                                                 obs_metadata=obsTemp, epoch=2000.0,
                                                 includeRefraction=True)

            raPointing_obs = ra_temp[0]
            decPointing_obs = dec_temp[0]

            for ra_obs, dec_obs, ra_icrs, dec_icrs in \
                zip(raList_obs, decList_obs, raList_icrs, decList_icrs):

                raRad = numpy.radians(ra_obs)
                decRad = numpy.radians(dec_obs)
                sinRa = numpy.sin(raRad)
                cosRa = numpy.cos(raRad)
                sinDec = numpy.sin(decRad)
                cosDec = numpy.cos(decRad)

                # the three dimensional position of the star
                controlPosition = numpy.array([-cosDec*sinRa, cosDec*cosRa, sinDec])

                # calculate the rotation matrices needed to transform the
                # x, y, and z axes into the local x, y, and z axes
                # (i.e. the axes with z lined up with raPointing_obs, decPointing_obs)
                alpha = 0.5*numpy.pi - numpy.radians(decPointing_obs)
                ca = numpy.cos(alpha)
                sa = numpy.sin(alpha)
                rotX = numpy.array([[1.0, 0.0, 0.0],
                                    [0.0, ca, sa],
                                    [0.0, -sa, ca]])

                cb = numpy.cos(numpy.radians(raPointing_obs))
                sb = numpy.sin(numpy.radians(raPointing_obs))
                rotZ = numpy.array([[cb, -sb, 0.0],
                                    [sb, cb, 0.0],
                                    [0.0, 0.0, 1.0]])

                # rotate the coordinate axes into the local basis
                xAxis = numpy.dot(rotZ, numpy.dot(rotX, numpy.array([1.0, 0.0, 0.0])))
                yAxis = numpy.dot(rotZ, numpy.dot(rotX, numpy.array([0.0, 1.0, 0.0])))
                zAxis = numpy.dot(rotZ, numpy.dot(rotX, numpy.array([0.0, 0.0, 1.0])))

                # calculate the local longitude and latitude of the star
                lon, lat = nativeLonLatFromRaDec(ra_icrs, dec_icrs, obs)
                cosLon = numpy.cos(numpy.radians(lon))
                sinLon = numpy.sin(numpy.radians(lon))
                cosLat = numpy.cos(numpy.radians(lat))
                sinLat = numpy.sin(numpy.radians(lat))

                # the x, y, z position of the star in the local coordinate basis
                transformedPosition = numpy.array([-cosLat*sinLon,
                                                   cosLat*cosLon,
                                                   sinLat])

                # convert that position back into the un-rotated bases
                testPosition = transformedPosition[0]*xAxis + \
                               transformedPosition[1]*yAxis + \
                               transformedPosition[2]*zAxis

                # assert that testPosition and controlPosition should be equal
                distance = numpy.sqrt(numpy.power(controlPosition-testPosition, 2).sum())
                self.assertLess(distance, 1.0e-12)



    def testNativeLonLatVector(self):
        """
        Test that nativeLonLatFromRaDec works in a vectorized way; we do this
        by performing a bunch of tansformations passing in ra and dec as numpy arrays
        and then comparing them to results computed in an element-wise way
        """

        obs = ObservationMetaData(pointingRA=123.0, pointingDec=43.0, mjd=53467.2)

        raPoint = 145.0
        decPoint = -35.0

        nSamples = 100
        numpy.random.seed(42)
        raList = numpy.random.random_sample(nSamples)*360.0
        decList = numpy.random.random_sample(nSamples)*180.0 - 90.0

        lonList, latList = nativeLonLatFromRaDec(raList, decList, obs)

        for rr, dd, lon, lat in zip(raList, decList, lonList, latList):
            lonControl, latControl = nativeLonLatFromRaDec(rr, dd, obs)
            distance = arcsecFromRadians(haversine(numpy.radians(lon), numpy.radians(lat),
                                                   numpy.radians(lonControl), numpy.radians(latControl)))

            self.assertLess(distance, 0.0001)


    def testRaDec(self):
        """
        Test that raDecFromNativeLonLat does invert
        nativeLonLatFromRaDec
        """
        numpy.random.seed(42)
        nSamples = 100
        rrList = numpy.random.random_sample(nSamples)*50.0 # because raDecFromNativeLonLat is only good
                                                           # out to a zenith distance of ~ 70 degrees
        thetaList = numpy.random.random_sample(nSamples)*2.0*numpy.pi

        rrPointingList = numpy.random.random_sample(10)*50.0
        thetaPointingList = numpy.random.random_sample(10)*2.0*numpy.pi
        mjdList = numpy.random.random_sample(nSamples)*10000.0 + 43000.0

        for rrp, thetap, mjd in \
        zip(rrPointingList, thetaPointingList, mjdList):
            site = Site()
            raZenith, decZenith = raDecFromAltAz(180.0, 0.0,
                                                  numpy.degrees(site.longitude),
                                                  numpy.degrees(site.latitude),
                                                  mjd)

            rp = raZenith + rrp*numpy.cos(thetap)
            dp = decZenith + rrp*numpy.sin(thetap)
            obs = ObservationMetaData(pointingRA=rp, pointingDec=dp, mjd=mjd, site=site)



            raList_icrs = (raZenith + rrList*numpy.cos(thetaList)) % 360.0
            decList_icrs = decZenith + rrList*numpy.sin(thetaList)

            raList_obs, decList_obs = observedFromICRS(raList_icrs, decList_icrs,
                                                       obs_metadata=obs,
                                                       epoch=2000.0, includeRefraction=True)

            # calculate the distance between the ICRS position and the observed
            # geocentric position
            dd_icrs_obs_list = arcsecFromRadians(haversine(numpy.radians(raList_icrs),
                                                           numpy.radians(decList_icrs),
                                                           numpy.radians(raList_obs),
                                                           numpy.radians(decList_obs)))

            for rr, dd, dd_icrs_obs in zip(raList_icrs, decList_icrs, dd_icrs_obs_list):
                lon, lat = nativeLonLatFromRaDec(rr, dd, obs)
                r1, d1 = raDecFromNativeLonLat(lon, lat, obs)

                # the distance between the input RA, Dec and the round-trip output
                # RA, Dec
                distance = arcsecFromRadians(haversine(numpy.radians(r1), numpy.radians(d1),
                                                       numpy.radians(rr), numpy.radians(dd)))



                rr_obs, dec_obs = observedFromICRS(numpy.array([rr]), numpy.array([dd]),
                                                   obs_metadata=obs, epoch=2000.0, includeRefraction=True)


                # verify that the round trip through nativeLonLat only changed
                # RA, Dec by less than an arcsecond
                self.assertLess(distance, 1.0)

                # verify that any difference in the round trip is much less
                # than the distance between the ICRS and the observed geocentric
                # RA, Dec
                self.assertLess(distance, dd_icrs_obs*0.01)



    def testRaDecVector(self):
        """
        Test that raDecFromNativeLonLat does invert
        nativeLonLatFromRaDec (make sure it works in a vectorized way)
        """
        numpy.random.seed(42)
        nSamples = 100
        latList = numpy.random.random_sample(nSamples)*360.0
        lonList = numpy.random.random_sample(nSamples)*180.0 - 90.0
        raPoint = 95.0
        decPoint = 75.0

        obs = ObservationMetaData(pointingRA=raPoint, pointingDec=decPoint, mjd=53467.89)

        raList, decList = raDecFromNativeLonLat(lonList, latList, obs)

        for lon, lat, ra0, dec0 in zip(lonList, latList, raList, decList):
            ra1, dec1 = raDecFromNativeLonLat(lon, lat, obs)
            distance = arcsecFromRadians(haversine(numpy.radians(ra0), numpy.radians(dec0),
                                                   numpy.radians(ra1), numpy.radians(dec1)))
            self.assertLess(distance, 0.1)



def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(NativeLonLatTest)

    return unittest.TestSuite(suites)

def run(shouldExit = False):
    utilsTests.run(suite(), shouldExit)
if __name__ == "__main__":
    run(True)
