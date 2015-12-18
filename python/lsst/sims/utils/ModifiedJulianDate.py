import numpy as np
import warnings
import os

from astropy.time import Time

from lsst.utils import getPackageDir

__all__ = ["ModifiedJulianDate"]

class ModifiedJulianDate(object):


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


    @property
    def dut1(self):
        """
        UT1-UTC in seconds
        """

        if self._dut1 is None:
            try:
                intermediate_value = self._time.get_delta_ut1_utc()
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

