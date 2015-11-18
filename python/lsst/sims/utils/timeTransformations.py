import numpy as np
import warnings
import dutLookup as dut

__all__ = ["ut1FromUtc", "utcFromUt1"]


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


def utcFromUt1(ut1):
    """
   Use data downloaded from

    ftp://cddis.gsfc.nasa.gov/pub/products/iers/

    to transform UT1 into UTC

    @param [in] UT1 as an MJD

    @param [out] UTC as an MJD

    Note: because we only have data for UT1-UTC as a function
    of UTC, this method operates by creating arrays of UT1 and UTC
    from ut1FromUtc() and interpolating them.
    """

    utc_arr = np.arange(ut1-1.0, ut1+1.0, 0.25)
    ut1_arr = np.array([ut1FromUtc(utc) for utc in utc_arr])
    return np.interp(ut1, ut1_arr, utc_arr)
