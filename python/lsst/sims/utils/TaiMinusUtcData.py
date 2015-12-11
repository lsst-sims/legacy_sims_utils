"""
This fill will load the TAI-UTC leap second data from
sims_data/lookUpTables/tai-utc.dat
"""

import numpy as np
import os
from lsst.utils import getPackageDir

__all__ = ["TaiMinusUtcData"]

class TaiMinusUtcData(object):

    file_name = os.path.join(getPackageDir('sims_data'), 'lookUpTables',
                             'tai-utc.dat')


    arr = np.genfromtxt(file_name).transpose()
    _utc_arr = arr[0] - 2400000.5
    _leap_second_arr = arr[1]
    _zero_pt_arr = arr[2]
    _coeff_arr = arr[3]

    # calculate the values of TAI at the UTC where leap seconds are added
    _tai_arr = _utc_arr + (_leap_second_arr + (_utc_arr-_zero_pt_arr)*_coeff_arr)/86400.0

    @classmethod
    def d_tai_from_utc(cls, utc):
        """
        return TAI-UTC as a function of utc

        @param [in] UTC as an MJD

        @param [out] TAI-UTC (in seconds)
        """

        if utc<cls._utc_arr[0]:
            return 0.0

        # the index of the largest value in _utc_arr that
        # is less than utc
        min_dex = np.searchsorted(cls._utc_arr, utc, side='right')-1

        if min_dex>12:
            # we are in the regime where TAI-UTC is an integer number
            # of seconds
            return cls._leap_second_arr[min_dex]

        return cls._leap_second_arr[min_dex] + (utc - cls._zero_pt_arr[min_dex])*cls._coeff_arr[min_dex]


    @classmethod
    def d_tai_from_tai(cls, tai):
        """
        return tai-uts as a function of tai

        @param [in] TAI as an MJD

        @param [out] TAI-UTC (in seconds)
        """

        if tai<cls._tai_arr[0]:
            return 0.0

        # the index of the larges value in _tai_arr that
        # is less than tai
        min_dex = np.searchsorted(cls._tai_arr, tai, side='right')-1

        if min_dex>12:
            dt_out = cls._leap_second_arr[min_dex]
        else:
            dt_out = cls._leap_second_arr[min_dex] + (tai - cls._zero_pt_arr[min_dex])*cls._coeff_arr[min_dex]
            dt_out = dt_out/(1.0 + cls._coeff_arr[min_dex]/86400.0)

        return dt_out
