import numpy as np
import warnings
import os

from astropy.time import Time

from lsst.utils import getPackageDir

__all__ = ["ModifiedJulianDate"]

class ModifiedJulianDate(object):

    _ls_data_dir = os.path.join(getPackageDir('sims_utils'))
    _ls_data_dir = os.path.join(_ls_data_dir, 'python', 'lsst', 'sims', 'utils', 'data')

    _ls_data = np.genfromtxt(os.path.join(_ls_data_dir, 'leap_seconds.dat')).transpose()

    _ls_utc = _ls_data[0]-2400000.5
    _ls_c0 = _ls_data[1]
    _ls_c1 = _ls_data[2]
    _ls_c2 = _ls_data[3]

    _ls_dt = _ls_c0 + (_ls_utc - _ls_c1)*_ls_c2 # the nominal offset in seconds when each leap second is added

    # To find the number of seconds in a leap second day, take _ls_dt at that leap second, then subtract what
    # _ls_dt would have been if calculated with the leap second parameters as they were before the new leap second
    # was added.  This number of seconds is added to the day that extends from _ls_utc[ix]-1.0 to _ls_utc[ix].
    # This is trivial in the modern era where leap seconds are literally just single seconds added to a day
    # (_ls_extra_seconds is 1).  For MJD before 41317.5, we must consider the other parameters in the
    # TAI-UTC expression defined in leap_seconds.dat

    _ls_extra_seconds = np.array([
                                  _ls_dt[ix] - (_ls_c0[ix-1]+(_ls_utc[ix]-_ls_c1[ix-1])*_ls_c2[ix-1]) if ix>0
                                  else _ls_dt[ix]
                                  for ix in range(len(_ls_utc))
                                 ])

    _ls_length_of_day = 86400.0 + _ls_extra_seconds


    def __init__(self, TAI=None, UTC=None):
        """
        Must specify either:

        @param [in] TAI = the International Atomic Time as an MJD

        or

        @param [in] UTC = Universal Coordinate Time as an MJD
        """

        if TAI is None and UTC is None:
            raise RuntimeError("You must specify either TAI or UTC to "
                               "instantiate ModifiedJulianDate")

        if TAI is not None:
            self._time = Time(TAI, scale='tai', format='mjd')
            self._tai = TAI
            self._utc = None
        else:
            self._time = Time(UTC, scale='utc', format='mjd')
            self._utc = UTC
            self._tai = None

        self._tt = None
        self._tdb = None
        self._ut1 = None
        self._dut1 = None


    def __eq__(self, other):
        return self._time == other._time


    @property
    def TAI(self):
        """
        International Atomic Time as an MJD
        """
        if self._tai is None:
            intermediate_value = self._time.tai

            try:
                self._tai = intermediate_value.value
            except:
                self._tai = intermediate_value


        return self._tai


    @property
    def UTC(self):
        """
        Universal Coordinate Time as an MJD
        """
        if self._utc is None:
            intermediate_value = self._time.utc

            try:
                self._utc = intermediate_value.value
            except:
                self._utc = intermediate_value

        return self._utc



    @property
    def UT1(self):
        """
        Universal Time as an MJD
        """
        if self._ut1 is None:
            try:
                intermediate_value = self._time.ut1
            except:
                warnings.warn("UTC %e is outside of IERS table for UT1-UTC.\n" % self.UTC
                              + "Returning UT1 = UTC for lack of a better idea")
                intermediate_value = self.UTC

            try:
                self._ut1 = intermediate_value.value
            except:
                self._ut1 = intermediate_value


        return self._ut1



    def _calculate_dut1(self):
        """
        Calculate UT1-UTC in seconds
        """

        ls_dex = np.searchsorted(self._ls_utc, self.UTC, side='left') #find the nearest leap second

        if self._ls_utc[ls_dex]-self.UTC>1.0:
            # we are not on a leap second day; astropy can be trusted
            return self._time.get_delta_ut1_utc()

        return (self.UT1-self.UTC)*self._ls_length_of_day[ls_dex]


    @property
    def dut1(self):
        """
        UT1-UTC in seconds
        """

        # 2015 18 December
        # astropy.time.Time provides a method get_delta_ut1_utc(), which is
        # supposed to calculate UT1-UTC in seconds.  As of version 1.1 of
        # astropy, this method does not agree (they are off by a minus sign
        # at least) with just taking Time.ut1.value-Time.utc.value on days
        # when leap seconds are added (see astropy issue #4409).
        #
        # https://github.com/astropy/astropy/issues/4409
        #
        # I suspect this is because they are interpolating the DUT1 table
        # differently in each case.  Until this bug is fixed and it is determined
        # which method is correct, I am going to just return
        # self._time.ut1-self._time.utc here, so that our class can at least remain
        # consistent.  This will require us loading a list of leap seconds so that
        # we know what the length of the day in question is (86400 seconds or 86401 seconds).
        #
        # Once the bug in astropy is fixed, we can replace _calculate_dut1() below with
        # self._time.get_delta_ut1_utc()

        if self._dut1 is None:
            try:
                intermediate_value = self._calculate_dut1()
            except:
                warnings.warn("UTC %e is outside of IERS table for UT1-UTC.\n" % self.UTC
                              + "Returning UT1 = UTC for lack of a better idea")
                intermediate_value = 0.0

            try:
                self._dut1 = intermediate_value.value
            except:
                self._dut1 = intermediate_value


        return self._dut1


    @property
    def TT(self):
        """
        Terrestrial Time (aka Terrestrial Dynamical Time) as an MJD
        """
        if self._tt is None:
            intermediate_value = self._time.tt

            try:
                self._tt = intermediate_value.value
            except:
                self._tt = intermediate_value

        return self._tt


    @property
    def TDB(self):
        """
        Barycentric Dynamical Time as an MJD
        """
        if self._tdb is None:
            intermediate_value = self._time.tdb

            try:
                self._tdb = intermediate_value.value
            except:
                self._tdb = intermediate_value

        return self._tdb

