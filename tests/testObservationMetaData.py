from __future__ import with_statement

import os
import numpy
import unittest
import lsst.utils.tests as utilsTests
from collections import OrderedDict
from lsst.sims.utils import ObservationMetaData, ModifiedJulianDate
from lsst.sims.utils import Site, BoxBounds, CircleBounds

class ObservationMetaDataTest(unittest.TestCase):
    """
    This class will test that ObservationMetaData correctly assigns
    and returns its class variables (pointingRA, pointingDec, etc.)

    It will also test the behavior of the m5 member variable.
    """

    def testExceptions(self):
        """
        Test that errors are produced whenever ObservationMetaData
        parameters are overwritten in an unintentional way
        """

        metadata = {'pointingRA':[1.5], 'pointingDec':[0.5],
                    'Opsim_expmjd':[52000.0],
                    'Opsim_rotskypos':[1.3],
                    'Opsim_filter':[2],
                    'Opsim_rawseeing':[0.7]}

        obs_metadata = ObservationMetaData(phoSimMetaData=metadata,
                                           boundType='circle',
                                           boundLength=0.1)

        with self.assertRaises(RuntimeError):
            obs_metadata.pointingRA=1.2

        with self.assertRaises(RuntimeError):
            obs_metadata.pointingDec=1.2

        with self.assertRaises(RuntimeError):
            obs_metadata.rotSkyPos=1.5

        with self.assertRaises(RuntimeError):
            obs_metadata.seeing=0.5

        with self.assertRaises(RuntimeError):
            obs_metadata.setBandpassM5andSeeing()

        obs_metadata = ObservationMetaData(pointingRA=1.5,
                                           pointingDec=1.5)


    def testM5(self):
        """
        Test behavior of ObservationMetaData's m5 member variable
        """

        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName='u', m5=[12.0, 13.0])
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], m5=15.0)
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], m5=[12.0, 13.0, 15.0])

        obsMD = ObservationMetaData()
        self.assertIsNone(obsMD.m5)

        obsMD = ObservationMetaData(bandpassName='g', m5=12.0)
        self.assertAlmostEqual(obsMD.m5['g'], 12.0, 10)

        obsMD = ObservationMetaData(bandpassName=['u','g','r'], m5=[10,11,12])
        self.assertEqual(obsMD.m5['u'], 10)
        self.assertEqual(obsMD.m5['g'], 11)
        self.assertEqual(obsMD.m5['r'], 12)


    def testSeeing(self):
        """
        Test behavior of ObservationMetaData's seeing member variable
        """

        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName='u', seeing=[0.7, 0.6])
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], seeing=0.7)
        self.assertRaises(RuntimeError, ObservationMetaData, bandpassName=['u', 'g'], seeing=[0.8, 0.7, 0.6])

        obsMD = ObservationMetaData()
        self.assertIsNone(obsMD.seeing)

        obsMD = ObservationMetaData(bandpassName='g', seeing=0.7)
        self.assertAlmostEqual(obsMD.seeing['g'], 0.7, 10)

        obsMD = ObservationMetaData(bandpassName=['u','g','r'], seeing=[0.7,0.6,0.5])
        self.assertEqual(obsMD.seeing['u'], 0.7)
        self.assertEqual(obsMD.seeing['g'], 0.6)
        self.assertEqual(obsMD.seeing['r'], 0.5)


    def testM5andSeeingAssignment(self):
        """
        Test assignment of m5 and seeing seeing and bandpass in ObservationMetaData
        """
        obsMD = ObservationMetaData(bandpassName=['u','g'], m5=[15.0, 16.0], seeing=[0.7, 0.6])
        self.assertAlmostEqual(obsMD.m5['u'], 15.0, 10)
        self.assertAlmostEqual(obsMD.m5['g'], 16.0, 10)
        self.assertAlmostEqual(obsMD.seeing['u'], 0.7, 10)
        self.assertAlmostEqual(obsMD.seeing['g'], 0.6, 10)

        obsMD.setBandpassM5andSeeing(bandpassName=['i','z'], m5=[25.0, 22.0], seeing=[0.5, 0.4])
        self.assertAlmostEqual(obsMD.m5['i'], 25.0, 10)
        self.assertAlmostEqual(obsMD.m5['z'], 22.0, 10)
        self.assertAlmostEqual(obsMD.seeing['i'], 0.5, 10)
        self.assertAlmostEqual(obsMD.seeing['z'], 0.4, 10)

        with self.assertRaises(KeyError):
            obsMD.m5['u']

        with self.assertRaises(KeyError):
            obsMD.m5['g']

        obsMD.m5 = [13.0, 14.0]
        obsMD.seeing = [0.2, 0.3]
        self.assertAlmostEqual(obsMD.m5['i'], 13.0, 10)
        self.assertAlmostEqual(obsMD.m5['z'], 14.0, 10)
        self.assertAlmostEqual(obsMD.seeing['i'], 0.2, 10)
        self.assertAlmostEqual(obsMD.seeing['z'], 0.3, 10)

        obsMD.setBandpassM5andSeeing(bandpassName=['k', 'j'], m5=[21.0, 23.0])
        self.assertAlmostEqual(obsMD.m5['k'], 21.0, 10)
        self.assertAlmostEqual(obsMD.m5['j'], 23.0, 10)
        self.assertIsNone(obsMD.seeing)

        obsMD.setBandpassM5andSeeing(bandpassName=['w', 'x'], seeing=[0.9, 1.1])
        self.assertAlmostEqual(obsMD.seeing['w'], 0.9, 10)
        self.assertAlmostEqual(obsMD.seeing['x'], 1.1, 10)

        phoSimMD = {'Opsim_filter':[4]}
        obsMD.phoSimMetaData = phoSimMD
        self.assertEqual(obsMD.bandpass, 4)
        self.assertTrue(obsMD.m5 is None)
        self.assertTrue(obsMD.seeing is None)


    def testDefault(self):
        """
        Test that ObservationMetaData's default variables are properly set
        """

        testObsMD = ObservationMetaData()

        self.assertEqual(testObsMD.pointingRA, None)
        self.assertEqual(testObsMD.pointingDec, None)
        self.assertEqual(testObsMD.rotSkyPos, None)
        self.assertEqual(testObsMD.bandpass, 'r')
        self.assertEqual(testObsMD.m5, None)
        self.assertEqual(testObsMD.seeing, None)
        self.assertAlmostEqual(testObsMD.site.longitude, -1.2320792,10)
        self.assertAlmostEqual(testObsMD.site.latitude, -0.517781017,10)
        self.assertAlmostEqual(testObsMD.site.height, 2650, 10)
        self.assertAlmostEqual(testObsMD.site.xPolar, 0, 10)
        self.assertAlmostEqual(testObsMD.site.yPolar, 0, 10)
        self.assertAlmostEqual(testObsMD.site.meanTemperature, 284.655, 10)
        self.assertAlmostEqual(testObsMD.site.meanPressure, 749.3, 10)
        self.assertAlmostEqual(testObsMD.site.meanHumidity, 0.4, 10)
        self.assertAlmostEqual(testObsMD.site.lapseRate, 0.0065, 10)

    def testSite(self):
        """
        Test that site data gets passed correctly when it is not default
        """
        testSite = Site(longitude=2.0, latitude=-1.0, height=4.0,
            xPolar=0.5, yPolar=-0.5, meanTemperature=100.0,
            meanPressure=500.0, meanHumidity=0.1, lapseRate=0.1)

        testObsMD = ObservationMetaData(site=testSite)

        self.assertAlmostEqual(testObsMD.site.longitude, 2.0, 10)
        self.assertAlmostEqual(testObsMD.site.latitude, -1.0, 10)
        self.assertAlmostEqual(testObsMD.site.height, 4.0, 10)
        self.assertAlmostEqual(testObsMD.site.xPolar, 0.5, 10)
        self.assertAlmostEqual(testObsMD.site.yPolar, -0.5, 10)
        self.assertAlmostEqual(testObsMD.site.meanTemperature, 100.0, 10)
        self.assertAlmostEqual(testObsMD.site.meanPressure, 500.0, 10)
        self.assertAlmostEqual(testObsMD.site.meanHumidity, 0.1, 10)
        self.assertAlmostEqual(testObsMD.site.lapseRate, 0.1, 10)

    def testAssignment(self):
        """
        Test that ObservationMetaData member variables get passed correctly
        """

        mjd = 5120.0
        RA = 1.5
        Dec = -1.1
        rotSkyPos = -10.0
        skyBrightness = 25.0

        testObsMD = ObservationMetaData()
        testObsMD.pointingRA = RA
        testObsMD.pointingDec = Dec
        testObsMD.rotSkyPos = rotSkyPos
        testObsMD.skyBrightness = skyBrightness
        testObsMD.mjd = mjd
        testObsMD.boundType = 'box'
        testObsMD.boundLength = [1.2, 3.0]

        self.assertAlmostEqual(testObsMD.pointingRA, RA, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, Dec, 10)
        self.assertAlmostEqual(testObsMD.rotSkyPos, rotSkyPos, 10)
        self.assertAlmostEqual(testObsMD.skyBrightness, skyBrightness, 10)
        self.assertEqual(testObsMD.boundType, 'box')
        self.assertAlmostEqual(testObsMD.boundLength[0], 1.2, 10)
        self.assertAlmostEqual(testObsMD.boundLength[1], 3.0, 10)
        self.assertAlmostEqual(testObsMD.mjd.TAI, mjd, 10)

        #test reassignment

        testObsMD.pointingRA = RA+1.0
        testObsMD.pointingDec = Dec+1.0
        testObsMD.rotSkyPos = rotSkyPos+1.0
        testObsMD.skyBrightness = skyBrightness+1.0
        testObsMD.boundLength = 2.2
        testObsMD.boundType = 'circle'
        testObsMD.mjd = mjd + 10.0

        self.assertAlmostEqual(testObsMD.pointingRA, RA+1.0, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, Dec+1.0, 10)
        self.assertAlmostEqual(testObsMD.rotSkyPos, rotSkyPos+1.0, 10)
        self.assertAlmostEqual(testObsMD.skyBrightness, skyBrightness+1.0, 10)
        self.assertEqual(testObsMD.boundType, 'circle')
        self.assertAlmostEqual(testObsMD.boundLength,2.2, 10)
        self.assertAlmostEqual(testObsMD.mjd.TAI, mjd+10.0, 10)

        phosimMD = OrderedDict([('pointingRA', (-2.0,float)),
                                ('pointingDec', (0.9,float)),
                                ('Opsim_rotskypos', (1.1,float)),
                                ('Opsim_expmjd',(4000.0,float)),
                                ('Opsim_filter',('g',str))])


        testObsMD.phoSimMetaData = phosimMD
        self.assertAlmostEqual(testObsMD.pointingRA, numpy.degrees(-2.0), 10)
        self.assertAlmostEqual(testObsMD.pointingDec, numpy.degrees(0.9), 10)
        self.assertAlmostEqual(testObsMD.rotSkyPos, numpy.degrees(1.1))
        self.assertAlmostEqual(testObsMD.mjd.TAI, 4000.0, 10)
        self.assertAlmostEqual(testObsMD.bandpass, 'g')

        testObsMD = ObservationMetaData(mjd=mjd, pointingRA=RA,
            pointingDec=Dec, rotSkyPos=rotSkyPos, bandpassName='z',
            skyBrightness=skyBrightness)

        self.assertAlmostEqual(testObsMD.mjd.TAI,5120.0,10)
        self.assertAlmostEqual(testObsMD.pointingRA,1.5,10)
        self.assertAlmostEqual(testObsMD.pointingDec,-1.1,10)
        self.assertAlmostEqual(testObsMD.rotSkyPos,-10.0,10)
        self.assertEqual(testObsMD.bandpass,'z')
        self.assertAlmostEqual(testObsMD.skyBrightness, skyBrightness, 10)

        testObsMD = ObservationMetaData()
        testObsMD.phoSimMetaData = phosimMD

        self.assertAlmostEqual(testObsMD.mjd.TAI,4000.0,10)

        #recall that pointingRA/Dec are stored as radians in phoSim metadata
        self.assertAlmostEqual(testObsMD.pointingRA,numpy.degrees(-2.0),10)
        self.assertAlmostEqual(testObsMD.pointingDec,numpy.degrees(0.9),10)
        self.assertAlmostEqual(testObsMD.rotSkyPos,numpy.degrees(1.1),10)
        self.assertEqual(testObsMD.bandpass,'g')

        testObsMD = ObservationMetaData()
        testObsMD.phoSimMetaData = phosimMD

        self.assertAlmostEqual(testObsMD.mjd.TAI,4000.0,10)

        #recall that pointingRA/Dec are stored as radians in phoSim metadata
        self.assertAlmostEqual(testObsMD.pointingRA,numpy.degrees(-2.0),10)
        self.assertAlmostEqual(testObsMD.pointingDec,numpy.degrees(0.9),10)
        self.assertAlmostEqual(testObsMD.rotSkyPos,numpy.degrees(1.1),10)
        self.assertEqual(testObsMD.bandpass,'g')


        # test assigning ModifiedJulianDate
        obs = ObservationMetaData()
        mjd = ModifiedJulianDate(TAI=57388.0)
        obs.mjd = mjd
        self.assertEqual(obs.mjd, mjd)

        mjd2 = ModifiedJulianDate(TAI=45000.0)
        obs.mjd = mjd2
        self.assertEqual(obs.mjd, mjd2)
        self.assertNotEqual(obs.mjd, mjd)


    def testBoundBuilding(self):
        """
        Make sure ObservationMetaData can build bounds
        """
        boxBounds = [0.1, 0.3]
        circObs = ObservationMetaData(boundType='circle', pointingRA=0.0, pointingDec=0.0, boundLength=1.0, mjd=53580.0)
        boundControl = CircleBounds(0.0, 0.0, numpy.radians(1.0))
        self.assertEqual(circObs.bounds, boundControl)

        squareObs = ObservationMetaData(boundType = 'box', pointingRA=0.0, pointingDec=0.0, boundLength=1.0, mjd=53580.0)
        boundControl = BoxBounds(0.0, 0.0, numpy.radians(1.0))
        self.assertEqual(squareObs.bounds, boundControl)

        boxObs = ObservationMetaData(boundType = 'box', pointingRA=0.0, pointingDec=0.0, boundLength=boxBounds, mjd=53580.0)
        boundControl = BoxBounds(0.0, 0.0, numpy.radians([0.1, 0.3]))
        self.assertEqual(boxObs.bounds, boundControl)

    def testBounds(self):
        """
        Test if ObservationMetaData correctly assigns the pointing[RA,Dec]
        when circle and box bounds are specified
        """

        circRA = 25.0
        circDec = 50.0
        radius = 5.0

        boxRA = 15.0
        boxDec = 0.0
        boxLength = numpy.array([5.0,10.0])

        testObsMD = ObservationMetaData(boundType='circle',
                     pointingRA = circRA, pointingDec=circDec, boundLength = radius, mjd=53580.0)
        self.assertAlmostEqual(testObsMD.pointingRA, 25.0, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, 50.0, 10)

        testObsMD = ObservationMetaData(boundType='box',
                                        pointingRA=boxRA, pointingDec=boxDec, boundLength=boxLength,
                                        mjd=53580.0)
        self.assertAlmostEqual(testObsMD.pointingRA, 15.0, 10)
        self.assertAlmostEqual(testObsMD.pointingDec, 0.0, 10)


    def testSummary(self):
        """
        Make sure summary is safe even when no parameters have been set
        """
        obs = ObservationMetaData()
        obs.summary


def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(ObservationMetaDataTest)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
