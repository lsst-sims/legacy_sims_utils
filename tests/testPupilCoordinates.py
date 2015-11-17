import numpy
import unittest
import lsst.utils.tests as utilsTests

from lsst.sims.utils import ObservationMetaData, _nativeLonLatFromRaDec
from lsst.sims.utils import _pupilCoordsFromRaDec
from lsst.sims.utils import _raDecFromPupilCoords

class PupilCoordinateUnitTest(unittest.TestCase):

    def testExceptions(self):
        """
        Test that exceptions are raised when they ought to be
        """
        obs_metadata = ObservationMetaData(pointingRA=25.0,
                                           pointingDec=25.0,
                                           rotSkyPos=25.0,
                                           mjd=52000.0)

        numpy.random.seed(42)
        ra = numpy.random.random_sample(10)*numpy.radians(1.0) + numpy.radians(obs_metadata.pointingRA)
        dec = numpy.random.random_sample(10)*numpy.radians(1.0) + numpy.radians(obs_metadata.pointingDec)
        raShort = numpy.array([1.0])
        decShort = numpy.array([1.0])

        #test without epoch
        self.assertRaises(RuntimeError, _pupilCoordsFromRaDec, ra, dec,
                          obs_metadata=obs_metadata)

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

        #test that it actually runs
        test = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs_metadata, epoch=2000.0)


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
        numpy.random.seed(42)
        raList = numpy.random.random_sample(10)*360.0
        decList = numpy.random.random_sample(10)*180.0 - 90.0


        for rotSkyPos in numpy.arange(-90.0, 181.0, 90.0):
            for ra, dec in zip(raList, decList):
                obs = ObservationMetaData(pointingRA=ra,
                                          pointingDec=dec,
                                          mjd=mjd,
                                          rotSkyPos=rotSkyPos)

                #test order E, W, N, S
                raTest = numpy.radians(ra) + numpy.array([0.01, -0.01, 0.0, 0.0])
                decTest = numpy.radians(dec) + numpy.array([0.0, 0.0, 0.01, -0.01])
                x, y = _pupilCoordsFromRaDec(raTest, decTest, obs_metadata=obs, epoch=epoch)

                lon, lat = _nativeLonLatFromRaDec(raTest, decTest, numpy.radians(ra), numpy.radians(dec))
                rr = numpy.abs(numpy.cos(lat)/numpy.sin(lat))

                if numpy.abs(rotSkyPos)<0.01:
                    control_x = numpy.array([-1.0*rr[0], 1.0*rr[1], 0.0, 0.0])
                    control_y = numpy.array([0.0, 0.0, 1.0*rr[2], -1.0*rr[3]])
                elif numpy.abs(rotSkyPos+90.0)<0.01:
                    control_x = numpy.array([0.0, 0.0, 1.0*rr[2], -1.0*rr[3]])
                    control_y = numpy.array([1.0*rr[0], -1.0*rr[1], 0.0, 0.0])
                elif numpy.abs(rotSkyPos-90.0)<0.01:
                    control_x = numpy.array([0.0, 0.0, -1.0*rr[2], 1.0*rr[3]])
                    control_y = numpy.array([-1.0*rr[0], 1.0*rr[1], 0.0, 0.0])
                elif numpy.abs(rotSkyPos-180.0)<0.01:
                    control_x = numpy.array([1.0*rr[0], -1.0*rr[1], 0.0, 0.0])
                    control_y = numpy.array([0.0, 0.0, -1.0*rr[2], 1.0*rr[3]])

                dx = numpy.array([xx/cc if numpy.abs(cc)>1.0e-10 else 1.0-xx for xx, cc in zip(x, control_x)])
                dy = numpy.array([yy/cc if numpy.abs(cc)>1.0e-10 else 1.0-yy for yy, cc in zip(y, control_y)])
                numpy.testing.assert_array_almost_equal(dx, numpy.ones(4), decimal=4)
                numpy.testing.assert_array_almost_equal(dy, numpy.ones(4), decimal=4)



    def testRaDecFromPupil(self):
        """
        Test conversion from pupil coordinates back to Ra, Dec
        """
        raCenter = 25.0
        decCenter = -10.0
        obs = ObservationMetaData(pointingRA=raCenter,
                                  pointingDec=decCenter,
                                  boundType='circle',
                                  boundLength=0.1,
                                  rotSkyPos=23.0,
                                  mjd=52000.0)

        nSamples = 100
        numpy.random.seed(42)
        ra = (numpy.random.random_sample(nSamples)*0.1-0.2) + numpy.radians(raCenter)
        dec = (numpy.random.random_sample(nSamples)*0.1-0.2) + numpy.radians(decCenter)
        xp, yp = _pupilCoordsFromRaDec(ra, dec, obs_metadata=obs, epoch=2000.0)
        raTest, decTest = _raDecFromPupilCoords(xp, yp, obs_metadata=obs, epoch=2000.0)
        numpy.testing.assert_array_almost_equal(raTest, ra, decimal=10)
        numpy.testing.assert_array_almost_equal(decTest, dec, decimal=10)


    def testNaNs(self):
        """
        Test how _pupilCoordsFromRaDec handles improper values
        """
        obs = ObservationMetaData(pointingRA=42.0, pointingDec=-28.0,
                                  rotSkyPos=111.0, mjd=42356.0)
        nSamples = 100
        numpy.random.seed(42)
        raList = numpy.radians(numpy.random.random_sample(nSamples)*2.0 + 42.0)
        decList = numpy.radians(numpy.random.random_sample(nSamples)*2.0 -28.0)

        xControl, yControl = _pupilCoordsFromRaDec(raList, decList,
                                                       obs_metadata=obs,
                                                       epoch=2000.0)

        raList[5] = numpy.NaN
        decList[5] = numpy.NaN
        raList[15] = numpy.NaN
        decList[20] = numpy.NaN
        raList[30] = numpy.radians(42.0) + numpy.pi

        xTest, yTest = _pupilCoordsFromRaDec(raList, decList,
                                                 obs_metadata=obs,
                                                 epoch=2000.0)

        for ix, (xc, yc, xt, yt) in \
        enumerate(zip(xControl, yControl, xTest, yTest)):
            if ix!=5 and ix!=15 and ix!=20 and ix!=30:
                self.assertAlmostEqual(xc, xt, 10)
                self.assertAlmostEqual(yc, yt, 10)
                self.assertFalse(numpy.isnan(xt))
                self.assertFalse(numpy.isnan(yt))
            else:
                self.assertTrue(numpy.isnan(xt))
                self.assertTrue(numpy.isnan(yt))


def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(PupilCoordinateUnitTest)
    return unittest.TestSuite(suites)

def run(shouldExit=False):
    utilsTests.run(suite(),shouldExit)

if __name__ == "__main__":
    run(True)
