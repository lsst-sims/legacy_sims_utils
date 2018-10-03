import numpy as np
from lsst.sims.utils import calcLmstLast

__all__ = ['_approx_altAz2RaDec', '_approx_RaDec2AltAz', 'approx_altAz2RaDec', 'approx_RaDec2AltAz']


def approx_altAz2RaDec(alt, az, lat, lon, mjd, lmst=None):
    """
    Convert alt, az to RA, Dec without taking into account abberation, precesion, diffraction, etc.

    Parameters
    ----------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Degrees.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Must be same length as `alt`. Degrees.
    lat : float
        Latitude of the observatory in degrees.
    lon : float
        Longitude of the observatory in degrees.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    ra : array_like
        RA, in degrees.
    dec : array_like
        Dec, in degrees.
    """
    ra, dec = _approx_altAz2RaDec(np.radians(alt), np.radians(az), np.radians(lat),
                                  np.radians(lon), mjd, lmst=lmst)
    return np.degrees(ra), np.degrees(dec)


def _approx_altAz2RaDec(alt, az, lat, lon, mjd, lmst=None):
    """
    Convert alt, az to RA, Dec without taking into account abberation, precesion, diffraction, etc.

    Parameters
    ----------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Must be same length as `alt`. Radians.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians.
    """
    if lmst is None:
        lmst, last = calcLmstLast(mjd, lon)
    lmst = lmst/12.*np.pi  # convert to rad
    sindec = np.sin(lat)*np.sin(alt) + np.cos(lat)*np.cos(alt)*np.cos(az)
    sindec = np.clip(sindec, -1, 1)
    dec = np.arcsin(sindec)
    ha = np.arctan2(-np.sin(az)*np.cos(alt), -np.cos(az)*np.sin(lat)*np.cos(alt)+np.sin(alt)*np.cos(lat))
    ra = (lmst-ha)
    raneg = np.where(ra < 0)
    ra[raneg] = ra[raneg] + 2.*np.pi
    raover = np.where(ra > 2.*np.pi)
    ra[raover] -= 2.*np.pi
    return ra, dec


def approx_RaDec2AltAz(ra, dec, lat, lon, mjd, lmst=None):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore abberation, precesion, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in degrees.
    dec : array_like
        Dec, in degrees. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in degrees.
    lon : float
        Longitude of the observatory in degrees.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. degrees.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. degrees.
    """
    alt, az = _approx_RaDec2AltAz(np.radians(ra), np.radians(dec), np.radians(lat),
                                  np.radians(lon), mjd, lmst=lmst)
    return np.degrees(alt), np.degrees(az)


def _approx_RaDec2AltAz(ra, dec, lat, lon, mjd, lmst=None):
    """
    Convert Ra,Dec to Altitude and Azimuth.

    Coordinate transformation is killing performance. Just use simple equations to speed it up
    and ignore abberation, precesion, nutation, nutrition, etc.

    Parameters
    ----------
    ra : array_like
        RA, in radians.
    dec : array_like
        Dec, in radians. Must be same length as `ra`.
    lat : float
        Latitude of the observatory in radians.
    lon : float
        Longitude of the observatory in radians.
    mjd : float
        Modified Julian Date.
    lmst : float (None)
        The local mean sidereal time (computed if not given). (hours)

    Returns
    -------
    alt : numpy.array
        Altitude, same length as `ra` and `dec`. Radians.
    az : numpy.array
        Azimuth, same length as `ra` and `dec`. Radians.
    """
    if lmst is None:
        lmst, last = calcLmstLast(mjd, lon)
    lmst = lmst/12.*np.pi  # convert to rad
    ha = lmst-ra
    sindec = np.sin(dec)
    sinlat = np.sin(lat)
    coslat = np.cos(lat)
    sinalt = sindec*sinlat+np.cos(dec)*coslat*np.cos(ha)
    sinalt = np.clip(sinalt, -1, 1)
    alt = np.arcsin(sinalt)
    cosaz = (sindec-np.sin(alt)*sinlat)/(np.cos(alt)*coslat)
    cosaz = np.clip(cosaz, -1, 1)
    az = np.arccos(cosaz)
    signflip = np.where(np.sin(ha) > 0)
    az[signflip] = 2.*np.pi-az[signflip]
    return alt, az
