"""
For definitions of all (or, at least, most of) the time systems handled in this file, see

https://www.cv.nrao.edu/~rfisher/Ephemerides/times.html
"""

import numpy as np
import warnings
import palpy
from lsst.sims.utils import Ut1MinusUtcData
from lsst.sims.utils import TT_from_TAI_Data

__all__ = ["taiFromUtc", "utcFromTai",
           "dutFromUtc",
           "ut1FromUtc", "utcFromUt1",
           "dttFromUtc", "ttFromTai",
           "tdbFromTt"]


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


def dutFromUtc(utc):
    """
    Use data downloaded from

    ftp://cddis.gsfc.nasa.gov/pub/products/iers/

    to return UT1-UTC as a function of UTC

    @param [in] UTC as an MJD

    @param [out] UT1-UTC in seconds
    """

    if utc<Ut1MinusUtcData._mjd_arr[0] or utc>Ut1MinusUtcData._mjd_arr[-1]:
        warnings.warn("UTC = %e is outside of the the bounds " % utc
                      + "for which we have UT1-UTC "
                      + "data (%e <= utc <= %e)\n" % (Ut1MinusUtcData._mjd_arr[0], Ut1MinusUtcData._mjd_arr[-1])
                      + "We will return UT1-UTC = 0, for lack of a better idea")

        return 0.0

    return np.interp(utc, Ut1MinusUtcData._mjd_arr, Ut1MinusUtcData._dut_arr)


def ut1FromUtc(utc):
    """
    Use data downloaded from

    ftp://cddis.gsfc.nasa.gov/pub/products/iers/

    to transform UTC into UT1

    @param [in] UTC as an MJD

    @param [out] UT1 as an MJD
    """

    sec_to_days = 1.0/86400.0
    dt = dutFromUtc(utc)
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


def dttFromUtc(utc):
    """
    Use data downloaded from

    ftp://tai.bipm.org/TFG/TT(BIPM)/TTBIPM.14

    to to calculate TT-TAI in seconds as a function of UTC

    @param [in] UTC as an MJD

    @param [out] TT-TAI in seconds

    Note: The bounds of our data are 42589.0 < UTC < 57019.0

    for UTC < 42589.0 this method returns 32.184 seconds

    for UTC > 57019.0 this method returns 32.184 seconds + 27.697 microseconds

    Discussion of the precision of this method can be found at the URL above.
    Roughly speaking, the precision is 1 ns before UTC=52729; 0.1 ns after that.
    """

    if utc>57019.0:
        dt = 27.6970
    elif utc<42589.0:
        dt = 0.0
    else:
        dt = np.interp(utc, TT_from_TAI_Data._mjd_arr, TT_from_TAI_Data._dt_arr)

    return 32.1840 + dt*1.0e-6


def ttFromTai(tai):
    """
    Return Terestrial Time (TT) as a function of TAI.

    Because of numerical precision, this method assumes

    TT = TAI + 32.184 seconds

    for better precision, find TT-TAI using the method dttFromUtc
    """
    return tai + 0.00037250


def tdbFromTt(tt):
    """
    Return TDB (Barycentric Dynamical Time) from TT (Terrestrial Time)
    using equation 2.222-1 from

    Seidelmann 1992, "Explanatory Supplement to the Astronomical Almanac"
    Unviversity Science Books

    @param [in] tt as an MJD

    @param [out] TDB as an MJD
    """

    sec_to_day = 1.0/86400.0
    jd_minus_mjd = 2400000.5

    tt_jd = tt + jd_minus_mjd
    julian_date = round(tt_jd, 2)
    jd_minus = julian_date - 2451545.0
    gg = np.radians(357.53 + 0.9856003*jd_minus)
    return tt + (0.001658*np.sin(gg) + 0.000014*np.sin(2*gg))*sec_to_day
