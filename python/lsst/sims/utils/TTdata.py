"""
This file will oad the MJD and TT - TAI - 32.184 sec data from
sims_data/lookUpTables/TTBIPM.14 for use in numpy.interp later on
"""

import numpy as np
import os
from lsst.utils import getPackageDir

__all__ = ["TT_from_TAI_Data"]

class TT_from_TAI_Data(object):

    file_name = os.path.join(getPackageDir('sims_data'), 'lookUpTables',
                             'TTBIPM.14')


    arr = np.genfromtxt(file_name).transpose()
    _mjd_arr = arr[0] # UTC (as an MJD)
    _dt_arr = arr[2] # TT - TAI - 32.184 sec (in microseconds)
