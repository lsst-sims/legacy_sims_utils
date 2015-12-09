"""
This file contains coordinate transformations that rely on both
palpy and the contents of AstrometryUtils.py (basically, coordinate
transformations that need to transform between observed geocentric RA, DEC
and ICRS RA, Dec)
"""

import numpy as np
import palpy
from lsst.sims.utils import _icrsFromObserved, _observedFromICRS, calcLmstLast, _observedFromAppGeo
from lsst.sims.utils import ObservationMetaData, Site

__all__ = ["_altAzPaFromRaDec", "altAzPaFromRaDec",
           "_raDecFromAltAz", "raDecFromAltAz",
           "getRotTelPos", "_getRotTelPos",
           "getRotSkyPos", "_getRotSkyPos"]

def altAzPaFromRaDec(ra, dec, obs):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [out] altitude in degrees

    @param [out] azimuth in degrees

    @param [out] parallactic angle in degrees
    """

    #sfd time This should be passing around Universal Time

    alt, az, pa = _altAzPaFromRaDec(np.radians(ra), np.radians(dec),
                                    obs)

    return np.degrees(alt), np.degrees(az), np.degrees(pa)



def _altAzPaFromRaDec(raRad, decRad, obs):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] raRad is RA in radians.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.
    Assumed to be in the International Celestial Reference System.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [out] altitude in radians

    @param [out] azimuth in radians

    @param [out] parallactic angle in radians
    """

    raIsArray = False
    decIsArray = False
    if isinstance(raRad, np.ndarray):
        raIsArray = True

    if isinstance(decRad, np.ndarray):
        decIsArray = True

    if raIsArray and not decIsArray:
        raise RuntimeError('passed numpy array of RA to altAzPaFromRaDec; but only one Dec')

    if decIsArray and not raIsArray:
        raise RuntimeError('passed numpy array of Dec to altAzPaFromRaDec; but only one RA')

    if raIsArray and decIsArray and len(raRad) != len(decRad):
        raise RuntimeError('in altAzPaFromRaDec length of RA numpy array does not match length of Dec numpy array')


    if not hasattr(raRad, '__len__'):
        raObs_temp, decObs_temp = _observedFromICRS(np.array([raRad]), np.array([decRad]), obs_metadata=obs,
                                                      includeRefraction=True, epoch=2000.0)

        raObs = raObs_temp[0]
        decObs = decObs_temp[0]


    else:
        raObs, decObs = _observedFromICRS(raRad, decRad, obs_metadata=obs, epoch=2000.0, includeRefraction=True)

    lst = calcLmstLast(obs.mjd, obs.site.longitude)
    last = lst[1]
    haRad = np.radians(last*15.0) - raObs


    #sfd time This should be passing around Universal Time

    if isinstance(haRad, np.ndarray):
        az, azd, azdd, \
        alt, altd, altdd, \
        pa, pad, padd = palpy.altazVector(haRad, decObs, obs.site.latitude)
    else:
        az, azd, azdd, \
        alt, altd, altdd, \
        pa, pad, padd = palpy.altaz(haRad, decObs, obs.site.latitude)

    return alt, az, pa


def raDecFromAltAz(alt, az, obs):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] alt is the altitude in degrees.  Can be a numpy array or a single value.

    @param [in] az is the azimuth in degrees.  Cant be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [out] RA in degrees (in the International Celestial Reference System)

    @param [out] Dec in degrees (in the International Celestial Reference System)

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    ra, dec = _raDecFromAltAz(np.radians(alt), np.radians(az), obs)

    return np.degrees(ra), np.degrees(dec)


def _raDecFromAltAz(altRad, azRad, obs):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] altRad is the altitude in radians.  Can be a numpy array or a single value.

    @param [in] azRad is the azimuth in radians.  Cant be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing
    the site of the telescope and the MJD of the observation

    @param [out] RA in radians

    @param [out] Dec in radians

    Note: This method is only accurate to within 0.01 arcsec near azimuth = 0 or pi
    """

    altIsArray = False
    azIsArray = False

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

    lst = calcLmstLast(obs.mjd, obs.site.longitude)
    last = lst[1]
    sinAlt = np.sin(altRad)
    cosLat = np.cos(obs.site.latitude)
    sinLat = np.sin(obs.site.latitude)
    decObs = np.arcsin(sinLat*sinAlt+ cosLat*np.cos(altRad)*np.cos(azRad))
    costheta = (sinAlt - np.sin(decObs)*sinLat)/(np.cos(decObs)*cosLat)
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
    raObs = np.radians(last*15.) - haRad

    if not hasattr(raObs, '__len__'):
        raRad, decRad = _icrsFromObserved(np.array([raObs]), np.array([decObs]),
                                          obs_metadata=obs, epoch=2000.0,
                                          includeRefraction=True)

        return raRad[0], decRad[0]


    raRad, decRad = _icrsFromObserved(raObs, decObs,
                                      obs_metadata=obs, epoch=2000.0,
                                      includeRefraction=True)

    return raRad, decRad


def getRotSkyPos(ra, dec, obs, rotTel):
    """
    @param [in] ra is the RA in degrees.  Can be a numpy array or a single value.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotTel is rotTelPos in degrees
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as ra and dec.  In this case,
    each rotTel will be associated with the corresponding ra, dec pair.

    @param [out] rotSkyPos in degrees

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """

    rotSky = _getRotSkyPos(np.radians(ra), np.radians(dec),
                           obs, np.radians(rotTel))

    return np.degrees(rotSky)



def _getRotSkyPos(raRad, decRad, obs, rotTelRad):
    """
    @param [in] raRad is the RA in radians.  Can be a numpy array or a single value.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotTelRad is rotTelPos in radians
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as raRad and decRad.  In this case,
    each rotTelRad will be associated with the corresponding raRad, decRad pair.

    @param [out] rotSkyPos in radians

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    altRad, azRad, paRad = _altAzPaFromRaDec(raRad, decRad, obs)

    #20 March 2015
    #I do not know where this expression comes from; we should validate it against
    #the definitions of rotTelPos and rotSkyPos
    return (rotTelRad - paRad)%(2.*np.pi)


def getRotTelPos(ra, dec, obs, rotSky):
    """
    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotSky is rotSkyPos in degrees
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as ra and dec.  In this case, each rotSkyPos
    will be associated with the corresponding ra, dec pair.

    @param [out] rotTelPos in degrees.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """

    rotTel = _getRotTelPos(np.radians(ra), np.radians(dec),
                           obs, np.radians(rotSky))

    return np.degrees(rotTel)


def _getRotTelPos(raRad, decRad, obs, rotSkyRad):
    """
    @param [in] raRad is RA in radians.  Can be a numpy array or a single value.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.

    @param [in] obs is an ObservationMetaData characterizing the telescope pointing
    and site.

    @param [in] rotSkyRad is rotSkyPos in radians
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as raRad and decRad.  In this case, each rotSkyPos
    will be associated with the corresponding raRad, decRad pair.

    @param [out] rotTelPos in radians.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (rotTelPos is calculated at the beginning of the exposure;
    expmjd is reckoned at the middle of the exposure).
    """
    altRad, azRad, paRad = _altAzPaFromRaDec(raRad, decRad, obs)

    #20 March 2015
    #I do not know where this expression comes from; we should validate it against
    #the definitions of rotTelPos and rotSkyPos
    return (rotSkyRad + paRad)%(2.*np.pi)


