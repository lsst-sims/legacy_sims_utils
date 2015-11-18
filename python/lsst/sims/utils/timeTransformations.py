import numpy as np
import warnings
import palpy
import dutLookup as dut

__all__ = ["taiFromUtc", "utcFromTai",
           "ut1FromUtc", "utcFromUt1"]


def taiFromUtc(utc):
    """
    Use the palpy method palDat to convert from Coordinated Universal Time
    (UTC) to International Atomic Time (TAI)

    @param [in] utc is the UTC time as an MJD

    @param [out] TAI time as an MJD
    """

    dt = palpy.dat(utc) # returns TAI-UTC in seconds
    return utc + dt/86400.0


def utcFromTai(tai):
    """
    Use the palpy method palDat to convert from International Atomic Time
    (TAI) to Coordinated Universal Time (UTC)

    @param [in] tai is the TAI time as an MJD

    @param [out] utc time as an MJD
    """

    # because the PALPY method only returns TAI-UTC as a function
    # of UTC, we will have to construct an array of TAI and UTC
    # values using taiFromUtc and interpolate along them

    sec_to_day = 1.0/86400.0

    dt_approx = palpy.dat(tai)*sec_to_day
    utc_arr = np.arange(tai - 1.0*dt_approx, tai+1.0*dt_approx, 1.0e-6)
    tai_arr = np.array([taiFromUtc(utc) for utc in utc_arr])

    return np.interp(tai, tai_arr, utc_arr)


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
                      + "We will return ut1 = utc, for lack of a better idea")

        return utc

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
