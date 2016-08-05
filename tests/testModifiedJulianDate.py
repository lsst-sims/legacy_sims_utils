from __future__ import with_statement
import astropy
import unittest
import warnings
import numpy as np
import os
import lsst.utils.tests as utilsTests

from lsst.utils import getPackageDir
from lsst.sims.utils import ModifiedJulianDate, UTCtoUT1Warning


class MjdTest(unittest.TestCase):
    """
    This unit test TestCase will just verify that the contents
    of ModifiedJulianDate agree with results generated 'by hand'.
    The 'by hand' transformations will have been tested by
    testTimeTransformations.py
    """

    longMessage = True

    def test_tai_from_utc(self):
        """
        Load a table of UTC vs. TAI (as JD) generated directly
        with ERFA.  Verify our ModifiedJulianDate wrapper against
        this data.  This is mostly so that we can catch any major
        API changes in astropy.
        """

        file_name = os.path.join(getPackageDir('sims_utils'), 'tests')
        file_name = os.path.join(file_name, 'testData', 'utc_tai_comparison_data.txt')

        dtype = np.dtype([('utc', np.float), ('tai', np.float)])
        data = np.genfromtxt(file_name, dtype=dtype)

        msg = "\n\nIt is possible you are using an out-of-date astropy.\n" + \
              "Try running 'conda update astropy' and restarting the build."

        for uu, tt in zip(data['utc']-2400000.5, data['tai']-2400000.5):
            mjd = ModifiedJulianDate(UTC=uu)
            dd_sec = np.abs(mjd.TAI-tt)*86400.0
            self.assertLess(dd_sec, 5.0e-5, msg=msg)
            self.assertAlmostEqual(mjd.UTC, uu, 15, msg=msg)
            mjd = ModifiedJulianDate(TAI=tt)
            dd_sec = np.abs(mjd.UTC-uu)*86400.0
            self.assertLess(dd_sec, 5.0e-5, msg=msg)
            self.assertAlmostEqual(mjd.TAI, tt, 15, msg=msg)

    def test_tt(self):
        """
        Verify that Terrestrial Time is TAI + 32.184 seconds
        as in equation 2.223-6 of

        Explanatory Supplement to the Astrnomical Almanac
        ed. Seidelmann, Kenneth P.
        1992, University Science Books

        Mostly, this test exists to catch any major API
        changes in astropy.time
        """

        np.random.seed(115)
        tai_list = np.random.random_sample(1000)*7000.0+50000.0
        for tai in tai_list:
            mjd = ModifiedJulianDate(TAI=tai)
            self.assertAlmostEqual(mjd.TT, tai+32.184/86400.0, 15)

    def test_tdb(self):
        """
        Verify that TDB is within a few tens of microseconds of the value given
        by the approximation given by equation 2.222-1 of

        Explanatory Supplement to the Astrnomical Almanac
        ed. Seidelmann, Kenneth P.
        1992, University Science Books

        Mostly, this test exists to catch any major API
        changes in astropy.time
        """

        np.random.seed(117)
        tai_list = np.random.random_sample(1000)*10000.0 + 46000.0
        for tai in tai_list:
            mjd = ModifiedJulianDate(TAI=tai)
            g = np.radians(357.53 + 0.9856003*(np.round(tai-51544.5)))
            tdb_test = mjd.TT + (0.001658*np.sin(g) + 0.000014*np.sin(2.0*g))/86400.0
            dt = np.abs(tdb_test-mjd.TDB)*8.64*1.0e10  # convert to microseconds
            self.assertLess(dt, 50)

    def test_dut1(self):
        """
        Test that UT1 is within 0.9 seconds of UTC.

        (Because calculating UT1-UTC requires loading a lookup
        table, we will just do this somewhat gross unit test to
        make sure that the astropy.time API doesn't change out
        from under us in some weird way... for instance, returning
        dut in units of days rather than seconds, etc.)
        """

        np.random.seed(117)

        utc_list = np.random.random_sample(1000)*10000.0 + 43000.0
        for utc in utc_list:
            mjd = ModifiedJulianDate(UTC=utc)

            # first, test the self-consistency of ModifiedJulianData.dut1
            # and ModifiedJulianData.UT1-ModifiedJulianData.UTC
            #
            # this only works for days on which a leap second is not applied
            dt = (mjd.UT1-mjd.UTC)*86400.0

            self.assertAlmostEqual(dt, mjd.dut1, 6)

            self.assertLess(np.abs(mjd.dut1), 0.9)

    def test_dut1_future(self):
        """
        Test that UT1 is within 0.9 seconds of UTC.  Consider times far
        in the future.

        (Because calculating UT1-UTC requires loading a lookup
        table, we will just do this somewhat gross unit test to
        make sure that the astropy.time API doesn't change out
        from under us in some weird way... for instance, returning
        dut in units of days rather than seconds, etc.)
        """

        np.random.seed(117)

        utc_list = np.random.random_sample(1000)*10000.0 + 63000.0
        for utc in utc_list:
            mjd = ModifiedJulianDate(UTC=utc)

            # first, test the self-consistency of ModifiedJulianData.dut1
            # and ModifiedJulianData.UT1-ModifiedJulianData.UTC
            #
            # this only works for days on which a leap second is not applied
            dt = (mjd.UT1-mjd.UTC)*86400.0

            self.assertAlmostEqual(dt, mjd.dut1, 6)

            self.assertLess(np.abs(mjd.dut1), 0.9)

    def test_eq(self):
        mjd1 = ModifiedJulianDate(TAI=43000.0)
        mjd2 = ModifiedJulianDate(TAI=43000.0)
        self.assertEqual(mjd1, mjd2)
        mjd3 = ModifiedJulianDate(TAI=43000.01)
        self.assertNotEqual(mjd1, mjd3)

    @unittest.skipIf(astropy.__version__ >= '1.2',
                     "astropy 1.2 handles cases of dates too far in the future "
                     "on its own in a graceful manner. Our warning classes are not needed")
    def test_warnings(self):
        """
        Test that warnings raised when trying to interpolate UT1-UTC
        for UTC too far in the future are of the type UTCtoUT1Warning
        """

        with warnings.catch_warnings(record=True) as w_list:
            mjd = ModifiedJulianDate(1000000.0)
            # clear the warning registry, in case a previous test raised the warnings
            # we are looking for
            mjd._warn_utc_out_of_bounds.__globals__['__warningregistry__'].clear()
            mjd.UT1
        expected_warnings = 0
        for ww in w_list:
            if isinstance(ww.message, UTCtoUT1Warning):
                if 'ModifiedJulianDate.UT1' in ww.message.message:
                    expected_warnings += 1
        self.assertGreater(expected_warnings, 0, msg="UT1 did not emit a UTCtoUT1Warning")

        expected_warnings = 0
        with warnings.catch_warnings(record=True) as w_list:
            warnings.filterwarnings('always')
            mjd = ModifiedJulianDate(1000000.0)
            mjd.dut1
        for ww in w_list:
            if isinstance(ww.message, UTCtoUT1Warning):
                if 'ModifiedJulianDate.dut1' in ww.message.message:
                    expected_warnings += 1
        self.assertGreater(expected_warnings, 0, msg="dut1 did not emit a UTCtoUT1Warning")

    def test_force_values(self):
        """
        Test that we can force the properties of a ModifiedJulianDate to have
        specific values
        """
        tt = ModifiedJulianDate(TAI=59580.0)
        values = np.arange(6)
        tt._force_values(values)
        self.assertEqual(tt.TAI, 0.0)
        self.assertEqual(tt.UTC, 1.0)
        self.assertEqual(tt.TT, 2.0)
        self.assertEqual(tt.TDB, 3.0)
        self.assertEqual(tt.UT1, 4.0)
        self.assertEqual(tt.dut1, 5.0)

        tt = ModifiedJulianDate(UTC=59580.0)
        values = 2.0*np.arange(6)
        tt._force_values(values)
        self.assertEqual(tt.TAI, 0.0)
        self.assertEqual(tt.UTC, 2.0)
        self.assertEqual(tt.TT, 4.0)
        self.assertEqual(tt.TDB, 6.0)
        self.assertEqual(tt.UT1, 8.0)
        self.assertEqual(tt.dut1, 10.0)

    def test_list(self):
        """
        Test that ModifiedJulianDate.get_list() gets results that are consistent
        with creating a list of ModifiedJulianDates by hand.
        """

        rng = np.random.RandomState(88)
        tol = 10  # decimal place tolerance

        tai_list = 40000.0 + 10000.0*rng.random_sample(20)
        tai_list = np.append(tai_list, 59580.0 + 10000.0*rng.random_sample(20))
        mjd_list = ModifiedJulianDate.get_list(TAI=tai_list)
        for tai, mjd in zip(tai_list, mjd_list):
            msg = "Offending TAI: %f" % tai
            control = ModifiedJulianDate(TAI=tai)
            self.assertAlmostEqual(mjd.TAI, tai, 11, msg=msg)
            self.assertAlmostEqual(mjd.TAI, control.TAI, tol, msg=msg)
            self.assertAlmostEqual(mjd.UTC, control.UTC, tol, msg=msg)
            self.assertAlmostEqual(mjd.UT1, control.UT1, tol, msg=msg)
            self.assertAlmostEqual(mjd.TT, control.TT, tol, msg=msg)
            self.assertAlmostEqual(mjd.TDB, control.TDB, tol, msg=msg)
            self.assertAlmostEqual(mjd.dut1, control.dut1, tol, msg=msg)

        utc_list = 40000.0 + 10000.0*rng.random_sample(20)
        utc_list = np.append(utc_list, 59580.0 + 10000.0*rng.random_sample(20))
        mjd_list = ModifiedJulianDate.get_list(UTC=utc_list)
        for utc, mjd in zip(utc_list, mjd_list):
            msg = "Offending UTC: %f" % utc
            control = ModifiedJulianDate(UTC=utc)
            self.assertAlmostEqual(mjd.UTC, utc, tol, msg=msg)
            self.assertAlmostEqual(mjd.TAI, control.TAI, tol, msg=msg)
            self.assertAlmostEqual(mjd.UTC, control.UTC, tol, msg=msg)
            self.assertAlmostEqual(mjd.UT1, control.UT1, tol, msg=msg)
            self.assertAlmostEqual(mjd.TT, control.TT, tol, msg=msg)
            self.assertAlmostEqual(mjd.TDB, control.TDB, tol, msg=msg)
            self.assertAlmostEqual(mjd.dut1, control.dut1, tol, msg=msg)


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
