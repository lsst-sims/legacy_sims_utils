from __future__ import with_statement
from __future__ import division
from builtins import str
import numpy as np
import unittest
import warnings
import lsst.utils.tests

from lsst.sims.utils import Site


def setup_module(module):
    lsst.utils.tests.init()


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
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, self.height)

    def testNoDefaults(self):
        """
        Test that, if name is not 'LSST', values are set to None
        """
        with warnings.catch_warnings(record=True) as ww:
            site = Site(name='bob')

        msg = str(ww[0].message)

        self.assertIn('longitude', msg)
        self.assertIn('latitude', msg)
        self.assertIn('temperature', msg)
        self.assertIn('pressure', msg)
        self.assertIn('height', msg)
        self.assertIn('lapseRate', msg)
        self.assertIn('humidity', msg)

        self.assertEqual(site.name, 'bob')
        self.assertIsNone(site.longitude)
        self.assertIsNone(site.longitude_rad)
        self.assertIsNone(site.latitude)
        self.assertIsNone(site.latitude_rad)
        self.assertIsNone(site.temperature)
        self.assertIsNone(site.temperature_kelvin)
        self.assertIsNone(site.pressure)
        self.assertIsNone(site.humidity)
        self.assertIsNone(site.lapseRate)
        self.assertIsNone(site.height)

    def testOverrideLSSTdefaults(self):
        """
        Test that, even if LSST is specified, we are capable of overriding
        defaults
        """
        site = Site(name='LSST', longitude=26.0)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, 26.0)
        self.assertEqual(site.longitude_rad, np.radians(26.0))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, self.height)

        site = Site(name='LSST', latitude=88.0)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, 88.0)
        self.assertEqual(site.latitude_rad, np.radians(88.0))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, self.height)

        site = Site(name='LSST', height=4.0)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, 4.0)

        site = Site(name='LSST', temperature=7.0)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, 7.0)
        self.assertEqual(site.temperature_kelvin, 280.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, self.height)

        site = Site(name='LSST', pressure=14.0)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, 14.0)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, self.height)

        site = Site(name='LSST', humidity=2.1)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, 2.1)
        self.assertEqual(site.lapseRate, self.lapseRate)
        self.assertEqual(site.height, self.height)

        site = Site(name='LSST', lapseRate=3.2)
        self.assertEqual(site.name, 'LSST')
        self.assertEqual(site.longitude, self.longitude)
        self.assertEqual(site.longitude_rad, np.radians(self.longitude))
        self.assertEqual(site.latitude, self.latitude)
        self.assertEqual(site.latitude_rad, np.radians(self.latitude))
        self.assertEqual(site.temperature, self.temperature)
        self.assertEqual(site.temperature_kelvin, self.temperature + 273.15)
        self.assertEqual(site.pressure, self.pressure)
        self.assertEqual(site.humidity, self.humidity)
        self.assertEqual(site.lapseRate, 3.2)
        self.assertEqual(site.height, self.height)

    def testPartialParams(self):
        """
        test that unspecified parameters get set to None
        """
        with warnings.catch_warnings(record=True) as ww:
            site = Site(longitude=45.0, temperature=20.0)

        msg = str(ww[0].message)
        self.assertIn('latitude', msg)
        self.assertIn('height', msg)
        self.assertIn('pressure', msg)
        self.assertIn('lapseRate', msg)
        self.assertIn('humidity', msg)
        self.assertNotIn('longitue', msg)
        self.assertNotIn('temperature', msg)

        self.assertIsNone(site.name)
        self.assertIsNone(site.latitude)
        self.assertIsNone(site.latitude_rad)
        self.assertIsNone(site.height)
        self.assertIsNone(site.pressure)
        self.assertIsNone(site.humidity)
        self.assertIsNone(site.lapseRate)
        self.assertEqual(site.longitude, 45.0)
        self.assertEqual(site.longitude_rad, np.pi / 4.0)
        self.assertEqual(site.temperature, 20.0)
        self.assertEqual(site.temperature_kelvin, 293.15)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
