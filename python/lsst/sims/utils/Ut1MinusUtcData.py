"""
This file will load the MJD and UT1-UTC from
sims_data/lookUpTables/dut1_table.txt for use in numpy.interp
later on
"""

import numpy as np
import os
from lsst.utils import getPackageDir

__all__ = ["Ut1MinusUtcData"]


class Ut1MinusUtcData(object):

    file_name = os.path.join(getPackageDir('sims_data'), 'lookUpTables',
                             'dut1_table.txt')

    arr = np.genfromtxt(file_name).transpose()
    _mjd_arr = arr[0]
    _dut_arr = arr[1]
