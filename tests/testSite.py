import numpy as np
import unittest
import lsst.utils.tests as utilsTests

from lsst.sims.utils import Site

class SiteTest(unittest.TestCase):

    def setUp(self):
        # LSST default values taken from LSE-30
        self.height = 2650.0
        self.longitude = -70.7494
        self.latitude = -30.2444
        self.temperature = 11.5
        self.humidity = 0.4
        self.pressure = 750.0
        self.lapseRate = 0.0065


    def testLSST_values(self):
        """
        Test that LSST values are set correctly
        """
        site = Site(name='LSST')
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature+273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)


def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(SiteTest)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
