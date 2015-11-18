import numpy as np
import warnings
import dutLookup as dut

__all__ = ["ut1FromUtc"]

def ut1FromUtc(utc):
    """
    Use data downloaded from

    ftp://cddis.gsfc.nasa.gov/pub/products/iers/

    to transform UTC into UT1

    @param [in] UTC as an MJD

    @param [out] UT1 as an MJD
    """

    if utc<dut._mjd_arr[0] or utc>dut._mjd_arr[-1]:
        warnings.warn("UTC = %e is outside of the the bounds " % utc
                      + "for which we have UT1-UTC "
                      + "data (%e <= utc <= %e)\n" % (dut._mjd_arr[0], dut._mjd_arr[-1])
                      + "We will return 0.0 as dut, for lack of a better idea")

        return 0.0

    sec_to_days = 1.0/86400.0

    dt = np.interp(utc, dut._mjd_arr, dut._dut_arr)
    return utc + dt*sec_to_days
