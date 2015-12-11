"""
This file will load the MJD and UT1-UTC from
sims_data/lookUpTables/dut1_table.txt for use in numpy.interp
later on
"""

import numpy as np
import os
import warnings
from lsst.utils import getPackageDir

__all__ = ["Ut1MinusUtcData"]


class Ut1MinusUtcData(object):

    file_name = os.path.join(getPackageDir('sims_data'), 'lookUpTables',
                             'dut1_table.txt')

    dtype = np.dtype([('utc', np.float), ('dut', np.float)])

    arr = np.genfromtxt(file_name, dtype=dtype).transpose()
    _utc_arr = arr['utc']
    _dut_arr = arr['dut']
    _ut1_arr = arr['utc']+arr['dut']/86400.0


    @classmethod
    def d_ut1_from_utc(cls, utc):
        """
        return UT1-UTC as a function of UTC

        @param [in] UTC as an MJD

        @param [out] UT1-UTC (in seconds)
        """

        if utc<cls._utc_arr[0] or utc>cls._utc_arr[-1]:
            msg = "UTC = %e is outside of the the bounds " % utc \
                   + "for which we have UT1-UTC "\
                   + "data (%e <= utc <= %e)\n" % (cls._utc_arr[0], cls._utc_arr[-1]) \
                   + "We will return UT1-UTC = 0, for lack of a better idea"
            warnings.warn(msg)

            return 0.0

        min_dex = np.searchsorted(cls._utc_arr, utc, side='right')-1
        return np.interp(utc, cls._utc_arr[min_dex:min_dex+2], cls._dut_arr[min_dex:min_dex+2])


    @classmethod
    def d_ut1_from_ut1(cls, ut1):
        """
        return UT1-UTC as a function of UT1

        @param [in] UT1 as an MJD

        @param [out] UT1-UTC (in seconds)
        """
        if ut1<cls._ut1_arr[0] or ut1>cls._ut1_arr[-1]:
            msg = "UT1 = %e is outside of the the bounds " % ut1 \
                   + "for which we have UT1-UTC "\
                   + "data (%e <= utc <= %e)\n" % (cls._ut1_arr[0], cls._ut1_arr[-1]) \
                   + "We will return UT1-UTC = 0, for lack of a better idea"
            warnings.warn(msg)

            return 0.0

        min_dex = np.searchsorted(cls._ut1_arr, ut1, side='right')-1
        return np.interp(ut1, cls._ut1_arr[min_dex:min_dex+2], cls._dut_arr[min_dex:min_dex+2])
