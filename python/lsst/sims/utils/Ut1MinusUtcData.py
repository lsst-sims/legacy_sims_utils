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

    arr = np.genfromtxt(file_name).transpose()
    _utc_arr = arr[0]
    _dut_arr = arr[1]
    _ut1_arr = arr[0]+arr[1]/86400.0
    _leap_second_indices = np.where(np.diff(_dut_arr)>0.1)[0]


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

        # find the index of the data largest utc that is less than the input
        # utc
        min_dex = np.searchsorted(cls._utc_arr, utc, side='right')-1

        # if we would have to interpolate across a leap second, reduce min_dex
        # by 1
        if min_dex in cls._leap_second_indices:
            min_dex -= 1

        return cls._dut_arr[min_dex] + \
              (utc-cls._utc_arr[min_dex])*(cls._dut_arr[min_dex]-cls._dut_arr[min_dex+1])/(cls._utc_arr[min_dex]-cls._utc_arr[min_dex+1])



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

        # find the index of the largest data ut1 that is less than the input
        # ut1
        min_dex = np.searchsorted(cls._ut1_arr, ut1, side='right')-1

        # if we would have to interpolate across a leap second, reduce min_dex
        # by 1
        if min_dex in cls._leap_second_indices:
            min_dex -= 1

        return cls._dut_arr[min_dex] + \
               (ut1-cls._ut1_arr[min_dex])*(cls._dut_arr[min_dex]-cls._dut_arr[min_dex+1])/(cls._ut1_arr[min_dex]-cls._ut1_arr[min_dex+1])
