import unittest
import numpy as np
import lsst.utils.tests as utilsTests

import lsst.sims.utils as utils
from lsst.sims.utils import ModifiedJulianDate

class MjdTest(unittest.TestCase):
    """
    This unit test TestCase will just verify that the contents
    of ModifiedJulianDate agree with results generated 'by hand'.
    The 'by hand' transformations will have been tested by
    testTimeTransformations.py
    """

    def test_tai(self):
        """
        Test a ModifiedJulianDate initialized with a TAI
        """
        np.random.seed(88)
        tai_list = np.random.random_sample(100)*15000.0 + 42000.0

        ct_btwn = 0

        for tai in tai_list:
            if tai<57000.0 and tai>48000.0:
                ct_btwn += 1
                # make sure not all cases were outside of interpolation bounds
                # for UT1-UTC

            mjd = ModifiedJulianDate(TAI=tai)
            utc = utils.utcFromTai(tai)
            self.assertEqual(mjd.TAI, tai)
            self.assertEqual(mjd.UTC, utc)
            self.assertEqual(mjd.dut, utils.dutFromUtc(utc))
            self.assertEqual(mjd.TT, utils.ttFromTai(tai))
            self.assertEqual(mjd.dtt, utils.dttFromUtc(utc))
            self.assertEqual(mjd.TDB, utils.tdbFromTt(utils.ttFromTai(tai)))

        self.assertGreater(ct_btwn, 0)
        self.assertLess(ct_btwn, len(tai_list))


    def test_utc(self):
        """
        Test ModifiedJulianDates initialized with utc
        """
        np.random.seed(77)
        utc_list = np.random.random_sample(100)*13000.0 + 42000.0
        ct_btwn = 0

        for utc in utc_list:
            tai = utils.taiFromUtc(utc)
            if tai<57000.0 and tai>48000.0:
                ct_btwn += 1
                # make sure not all cases were outside of interpolation bounds
                # for UT1-UTC

            mjd = ModifiedJulianDate(UTC=utc)
            self.assertEqual(mjd.TAI, tai)
            self.assertEqual(mjd.UTC, utc)
            self.assertEqual(mjd.dut, utils.dutFromUtc(utc))
            self.assertEqual(mjd.TT, utils.ttFromTai(tai))
            self.assertEqual(mjd.dtt, utils.dttFromUtc(utc))
            self.assertEqual(mjd.TDB, utils.tdbFromTt(utils.ttFromTai(tai)))

        self.assertGreater(ct_btwn, 0)
        self.assertLess(ct_btwn, len(utc_list))


    def test_round_trip(self):
        """
        Test that ModifiedJulianDates assigned with TAIs are equal
        to ModifiedJulianDates assigned with equivalent UTCs
        """
        np.random.seed(77)
        utc_list = np.random.random_sample(100)*13000.0 + 42000.0
        ct_btwn = 0

        for utc in utc_list:
            tai = utils.taiFromUtc(utc)
            if tai<57000.0 and tai>48000.0:
                ct_btwn += 1
                # make sure not all cases were outside of interpolation bounds
                # for UT1-UTC

            mjd1 = ModifiedJulianDate(UTC=utc)
            mjd2 = ModifiedJulianDate(TAI=tai)

            self.assertEqual(mjd1.TAI, mjd2.TAI)
            self.assertEqual(mjd1.UTC, mjd2.UTC)
            self.assertEqual(mjd1.dut, mjd2.dut)
            self.assertEqual(mjd1.TT, mjd2.TT)
            self.assertEqual(mjd1.TDB, mjd2.TDB)
            self.assertEqual(mjd1.dtt, mjd2.dtt)

        self.assertGreater(ct_btwn, 0)
        self.assertLess(ct_btwn, len(utc_list))


    def test_eq(self):
        mjd1 = ModifiedJulianDate(TAI=43000.0)
        mjd2 = ModifiedJulianDate(TAI=43000.0)
        self.assertEqual(mjd1, mjd2)
        mjd3 = ModifiedJulianDate(TAI=43000.01)
        self.assertNotEqual(mjd1, mjd3)


def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(MjdTest)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
