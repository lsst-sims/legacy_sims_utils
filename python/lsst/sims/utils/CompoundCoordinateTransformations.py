"""
This file contains coordinate transformations that rely on both
palpy and the contents of AstrometryUtils.py (basically, coordinate
transformations that need to transform between observed geocentric RA, DEC
and ICRS RA, Dec)
"""

import numpy as np
import palpy
from lsst.sims.utils import _icrsFromObserved, calcLmstLast

__all__ = ["_altAzPaFromRaDec", "altAzPaFromRaDec",
           "_raDecFromAltAz", "raDecFromAltAz"]

def altAzPaFromRaDec(ra, dec, lon, lat, mjd):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.

    @param [in] lon is the longitude of the observer in degrees
    (positive east of the prime meridian).  Must be a single value.

    @param [in] lat is the latitude of the observer in degrees
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MJD.
    Can be a numpy array or a single value.  If a numpy array, should have the
    same number of entries as ra and dec.  In this case,
    each mjd will be applied to its corresponding ra, dec pair.

    @param [out] altitude in degrees

    @param [out] azimuth in degrees

    @param [out] parallactic angle in degrees
    """

    alt, az, pa = _altAzPaFromRaDec(np.radians(ra), np.radians(dec),
                                    np.radians(lon), np.radians(lat),
                                    mjd)

    return np.degrees(alt), np.degrees(az), np.degrees(pa)



def _altAzPaFromRaDec(raRad, decRad, longRad, latRad, mjd):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] raRad is RA in radians.  Can be a numpy array or a single value.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.

    @param [in] longRad is the longitude of the observer in radians
    (positive east of the prime meridian).  Must be a single value.

    @param [in] latRad is the latitude of the observer in radians
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MJD.
    Can be a numpy array or a single value.  If a numpy array, should have the
    same number of entries as ra and dec.  In this case,
    each mjd will be applied to its corresponding ra, dec pair.

    @param [out] altitude in radians

    @param [out] azimuth in radians

    @param [out] parallactic angle in radians
    """

    if isinstance(longRad, np.ndarray):
        raise RuntimeError('cannot pass numpy array of longitudes to altAzPaFromRaDec')

    if isinstance(latRad, np.ndarray):
        raise RuntimeError('cannot pass numpy array of latitudes to altAzPaFromRaDec')

    raIsArray = False
    decIsArray = False
    mjdIsArray = False
    if isinstance(raRad, np.ndarray):
        raIsArray = True

    if isinstance(decRad, np.ndarray):
        decIsArray = True

    if isinstance(mjd, np.ndarray):
        mjdIsArray = True

    if raIsArray and not decIsArray:
        raise RuntimeError('passed numpy array of RA to altAzPaFromRaDec; but only one Dec')

    if decIsArray and not raIsArray:
        raise RuntimeError('passed numpy array of Dec to altAzPaFromRaDec; but only one RA')

    if raIsArray and decIsArray and len(raRad) != len(decRad):
        raise RuntimeError('in altAzPaFromRaDec length of RA numpy array does not match length of Dec numpy array')

    if mjdIsArray and not raIsArray:
        raise RuntimeError('passed numpy array of mjd to altAzPaFromRaDec; but only one RA, Dec')

    if mjdIsArray and len(mjd) != len(raRad):
        raise RuntimeError('in altAzPaFromRaDec length of mjd numpy array is not the same as length of RA numpy array')

    lst = calcLmstLast(mjd, longRad)
    last = lst[1]
    haRad = np.radians(last*15.0) - raRad

    if isinstance(haRad, np.ndarray):
        az, azd, azdd, \
        alt, altd, altdd, \
        pa, pad, padd = palpy.altazVector(haRad, decRad, latRad)
    else:
        az, azd, azdd, \
        alt, altd, altdd, \
        pa, pad, padd = palpy.altaz(haRad, decRad, latRad)

    return alt, az, pa


def raDecFromAltAz(alt, az, lon, lat, mjd):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] alt is the altitude in degrees.  Can be a numpy array or a single value.

    @param [in] az is the azimuth in degrees.  Cant be a numpy array or a single value.

    @param [in] lon is the observatory longitude in degrees
    (positive east of the prime meridian).  Must be a single value.

    @param [in] lat is the latitude in degrees
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MD.
    Can be a numpy array or a single value.  If a single value, must have the same length as
    the numpy array of alt and az.  In this case, each MJD will be associated with the corresponding
    alt, az pair.

    @param [out] RA in degrees

    @param [out] Dec in degrees

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    ra, dec = _raDecFromAltAz(np.radians(alt), np.radians(az),
                              np.radians(lon), np.radians(lat), mjd)

    return np.degrees(ra), np.degrees(dec)


def _raDecFromAltAz(altRad, azRad, longRad, latRad, mjd):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] altRad is the altitude in radians.  Can be a numpy array or a single value.

    @param [in] azRad is the azimuth in radians.  Cant be a numpy array or a single value.

    @param [in] longRad is the observatory longitude in radians
    (positive east of the prime meridian).  Must be a single value.

    @param [in] latRad is the latitude in radians
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MD.
    Can be a numpy array or a single value.  If a single value, must have the same length as
    the numpy array of alt and az.  In this case, each MJD will be associated with the corresponding
    alt, az pair.

    @param [out] RA in radians

    @param [out] Dec in radians

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """
    if isinstance(longRad, np.ndarray):
        raise RuntimeError('cannot pass a numpy array of longitudes to raDecFromAltAz')

    if isinstance(latRad, np.ndarray):
        raise RuntimeError('cannot pass a numpy array of latitudes to raDecFromAltAz')

    mjdIsArray = False
    altIsArray = False
    azIsArray = False
    if isinstance(mjd, np.ndarray):
        mjdIsArray = True

    if isinstance(altRad, np.ndarray):
        altIsArray = True

    if isinstance(azRad, np.ndarray):
        azIsArray = True

    if altIsArray and not azIsArray:
        raise RuntimeError('passed a numpy array of alt to raDecFromAltAz, but only one az')

    if azIsArray and not altIsArray:
        raise RuntimeError('passed a numpy array of az to raDecFromAltAz, but only one alt')

    if azIsArray and altIsArray and len(altRad)!=len(azRad):
        raise RuntimeError('in raDecFromAltAz, length of alt numpy array does not match length of az numpy array')

    if mjdIsArray and not azIsArray:
        raise RuntimeError('passed a numpy array of mjd to raDecFromAltAz, but only one alt, az pair')

    if mjdIsArray and len(mjd) != len(azRad):
        raise RuntimeError('in raDecFromAltAz length of mjd numpy array does not match length of az numpy array')

    lst = calcLmstLast(mjd, longRad)
    last = lst[1]
    sinAlt = np.sin(altRad)
    cosLat = np.cos(latRad)
    sinLat = np.sin(latRad)
    decRad = np.arcsin(sinLat*sinAlt+ cosLat*np.cos(altRad)*np.cos(azRad))
    costheta = (sinAlt - np.sin(decRad)*sinLat)/(np.cos(decRad)*cosLat)
    if altIsArray:
        haRad0 =  np.arccos(costheta)
        # Make sure there were no NaNs
        nanSpots = np.where(np.isnan(haRad0))[0]
        if np.size(nanSpots) > 0:
            haRad0[nanSpots] = 0.5*np.pi*(1.0-np.sign(costheta[nanSpots]))
    else:
        haRad0 = np.arccos(costheta)
        if np.isnan(haRad0):
            if np.sign(costheta)>0.0:
                haRad0 = 0.0
            else:
                haRad0 = np.pi

    haRad = np.where(np.sin(azRad)>=0.0, -1.0*haRad0, haRad0)
    raRad = np.radians(last*15.) - haRad
    return raRad, decRad

