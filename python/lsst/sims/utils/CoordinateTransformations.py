"""
This file contains coordinate transformation methods that are very thin wrappers
of palpy methods, or that have no dependence on palpy at all
"""
from __future__ import division

import numpy as np
import numbers
import palpy

from lsst.sims.utils.CodeUtilities import _validate_inputs

__all__ = ["_galacticFromEquatorial", "galacticFromEquatorial",
           "_equatorialFromGalactic", "equatorialFromGalactic",
           "sphericalFromCartesian", "cartesianFromSpherical",
           "rotationMatrixFromVectors",
           "equationOfEquinoxes", "calcGmstGast", "calcLmstLast",
           "angularSeparation", "_angularSeparation", "haversine",
           "arcsecFromRadians", "radiansFromArcsec",
           "arcsecFromDegrees", "degreesFromArcsec"]


def calcLmstLast(mjd, longRad):
    """
    calculates local mean sidereal time and local apparent sidereal time

    @param [in] mjd is the universal time (UT1) expressed as an MJD.
    This can be a numpy array or a single value.

    @param [in] longRad is the longitude in radians (positive east of the prime meridian)
    This can be numpy array or a single value.  If a numpy array, should have the same length as mjd.  In that
    case, each longRad will be applied only to the corresponding mjd.

    @param [out] lmst is the local mean sidereal time in hours

    @param [out] last is the local apparent sideral time in hours
    """
    mjdIsArray = False
    longRadIsArray = False
    if isinstance(mjd, np.ndarray):
        mjdIsArray = True

    if isinstance(longRad, np.ndarray):
        longRadIsArray = True

    if longRadIsArray and mjdIsArray:
        if len(longRad) != len(mjd):
            raise RuntimeError("In calcLmstLast mjd and longRad have different lengths")

    valid_type = False
    if isinstance(mjd, np.ndarray) and isinstance(longRad, np.ndarray):
        valid_type = True
    elif isinstance(mjd, np.ndarray) and isinstance(longRad, numbers.Number):
        valid_type = True
    elif isinstance(mjd, numbers.Number) and isinstance(longRad, numbers.Number):
        valid_type = True

    if not valid_type:
        msg = "Valid input types for calcLmstLast are:\n" \
              "mjd and longRad as numpy arrays of the same length\n" \
              "mjd as a numpy array and longRad as a number\n" \
              "mjd as a number and longRad as a number\n" \
              "You gave mjd: %s\n" % type(mjd) \
              + "and longRad: %s\n" % type(longRad)

        raise RuntimeError(msg)

    longDeg0 = np.degrees(longRad)
    longDeg0 %= 360.0

    if longRadIsArray:
        longDeg = np.where(longDeg0 > 180.0, longDeg0 - 360.0, longDeg0)
    else:
        if longDeg0 > 180.:
            longDeg = longDeg0 - 360.
        else:
            longDeg = longDeg0

    hrs = longDeg / 15.0
    gmstgast = calcGmstGast(mjd)
    lmst = gmstgast[0] + hrs
    last = gmstgast[1] + hrs
    lmst %= 24.
    last %= 24.
    return lmst, last


def galacticFromEquatorial(ra, dec):
    '''Convert RA,Dec (J2000) to Galactic Coordinates

    @param [in] ra is right ascension in degrees, either a number or a numpy array

    @param [in] dec is declination in degrees, either a number or a numpy array

    @param [out] gLong is galactic longitude in degrees

    @param [out] gLat is galactic latitude in degrees
    '''

    gLong, gLat = _galacticFromEquatorial(np.radians(ra), np.radians(dec))
    return np.degrees(gLong), np.degrees(gLat)


def _galacticFromEquatorial(ra, dec):
    '''Convert RA,Dec (J2000) to Galactic Coordinates

    All angles are in radians

    @param [in] ra is right ascension in radians, either a number or a numpy array

    @param [in] dec is declination in radians, either a number or a numpy array

    @param [out] gLong is galactic longitude in radians

    @param [out] gLat is galactic latitude in radians
    '''

    if isinstance(ra, np.ndarray):
        gLong, gLat = palpy.eqgalVector(ra, dec)
    else:
        gLong, gLat = palpy.eqgal(ra, dec)

    return gLong, gLat


def equatorialFromGalactic(gLong, gLat):
    '''Convert Galactic Coordinates to RA, dec (J2000)

    @param [in] gLong is galactic longitude in degrees, either a number or a numpy array
    (0 <= gLong <= 360.)

    @param [in] gLat is galactic latitude in degrees, either a number or a numpy array
    (-90. <= gLat <= 90.)

    @param [out] ra is right ascension in degrees

    @param [out] dec is declination in degrees
    '''

    ra, dec = _equatorialFromGalactic(np.radians(gLong), np.radians(gLat))
    return np.degrees(ra), np.degrees(dec)


def _equatorialFromGalactic(gLong, gLat):
    '''Convert Galactic Coordinates to RA, dec (J2000)

    @param [in] gLong is galactic longitude in radians, either a number or a numpy array
    (0 <= gLong <= 2*pi)

    @param [in] gLat is galactic latitude in radians, either a number or a numpy array
    (-pi/2 <= gLat <= pi/2)

    @param [out] ra is right ascension in radians (J2000)

    @param [out] dec is declination in radians (J2000)
    '''

    if isinstance(gLong, np.ndarray):
        ra, dec = palpy.galeqVector(gLong, gLat)
    else:
        ra, dec = palpy.galeq(gLong, gLat)

    return ra, dec


def cartesianFromSpherical(longitude, latitude):
    """
    Transforms between spherical and Cartesian coordinates.

    @param [in] longitude is a numpy array or a number in radians

    @param [in] latitude is a numpy array or number in radians

    @param [out] a numpy array of the (three-dimensional) cartesian coordinates on a unit sphere.

    if inputs are numpy arrays:
    output[i][0] will be the x-coordinate of the ith point
    output[i][1] will be the y-coordinate of the ith point
    output[i][2] will be the z-coordinate of the ith point

    All angles are in radians
    """

    valid_type = False
    if isinstance(longitude, np.ndarray) and isinstance(latitude, np.ndarray):
        valid_type = True
    elif isinstance(longitude, numbers.Number) and isinstance(latitude, numbers.Number):
        valid_type = True

    if not valid_type:
        raise RuntimeError("Longitude and latitude must both be either numpy arrays or numbers")

    cosDec = np.cos(latitude)
    return np.array([np.cos(longitude) * cosDec, np.sin(longitude) * cosDec, np.sin(latitude)]).transpose()


def sphericalFromCartesian(xyz):
    """
    Transforms between Cartesian and spherical coordinates

    @param [in] xyz is a numpy array of points in 3-D space.
    Each row is a different point.

    @param [out] returns longitude and latitude

    All angles are in radians
    """

    if not isinstance(xyz, np.ndarray):
        raise RuntimeError("You need to pass a numpy array to sphericalFromCartesian")

    if len(xyz.shape) > 1:
        rad = np.sqrt(np.power(xyz, 2).sum(axis=1))
        longitude = np.arctan2(xyz[:, 1], xyz[:, 0])
        latitude = np.arcsin(xyz[:, 2] / rad)
    else:
        rad = np.sqrt(np.dot(xyz, xyz))
        longitude = np.arctan2(xyz[1], xyz[0])
        latitude = np.arcsin(xyz[2] / rad)

    return longitude, latitude


def rotationMatrixFromVectors(v1, v2):
    '''
    Given two vectors v1,v2 calculate the rotation matrix for v1->v2 using the axis-angle approach

    @param [in] v1, v2 are two Cartesian unit vectors (in three dimensions)

    @param [out] rot is the rotation matrix that rotates from one to the other

    '''

    if np.abs(np.sqrt(np.dot(v1, v1)) - 1.0) > 0.01:
        raise RuntimeError("v1 in rotationMatrixFromVectors is not a unit vector")

    if np.abs(np.sqrt(np.dot(v2, v2)) - 1.0) > 0.01:
        raise RuntimeError("v2 in rotationMatrixFromVectors is not a unit vector")

    # Calculate the axis of rotation by the cross product of v1 and v2
    cross = np.cross(v1, v2)
    cross = cross / np.sqrt(np.dot(cross, cross))

    # calculate the angle of rotation via dot product
    angle = np.arccos(np.dot(v1, v2))
    sinDot = np.sin(angle)
    cosDot = np.cos(angle)

    # calculate the corresponding rotation matrix
    # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rot = [[cosDot + cross[0] * cross[0] * (1 - cosDot), -cross[2] * sinDot +
            (1 - cosDot) * cross[0] * cross[1],
            cross[1] * sinDot + (1 - cosDot) * cross[0] * cross[2]],
           [cross[2] * sinDot + (1 - cosDot) * cross[0] * cross[1], cosDot +
            (1 - cosDot) * cross[1] * cross[1],
            -cross[0] * sinDot + (1 - cosDot) * cross[1] * cross[2]],
           [-cross[1] * sinDot + (1 - cosDot) * cross[0] * cross[2], cross[0] * sinDot +
            (1 - cosDot) * cross[1] * cross[2], cosDot + (1 - cosDot) * (cross[2] * cross[2])]]

    return rot


def equationOfEquinoxes(d):
    """
    The equation of equinoxes. See http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] d is either a numpy array or a number that is Terrestrial Time
    expressed as an MJD

    @param [out] the equation of equinoxes in radians.
    """

    if isinstance(d, np.ndarray):
        return palpy.eqeqxVector(d)
    else:
        return palpy.eqeqx(d)


def calcGmstGast(mjd):
    """
    Compute Greenwich mean sidereal time and Greenwich apparent sidereal time
    see: From http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] mjd is the universal time (UT1) expressed as an MJD

    @param [out] gmst Greenwich mean sidereal time in hours

    @param [out] gast Greenwich apparent sidereal time in hours
    """

    date = np.floor(mjd)
    ut1 = mjd - date
    if isinstance(mjd, np.ndarray):
        gmst = palpy.gmstaVector(date, ut1)
    else:
        gmst = palpy.gmsta(date, ut1)

    eqeq = equationOfEquinoxes(mjd)
    gast = gmst + eqeq

    gmst = gmst * 24.0 / (2.0 * np.pi)
    gmst %= 24.0

    gast = gast * 24.0 / (2.0 * np.pi)
    gast %= 24.0

    return gmst, gast


def _angularSeparation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in radians

    Parameters
    ----------
    long1 is the first longitudinal coordinate in radians

    lat1 is the first latitudinal coordinate in radians

    long2 is the second longitudinal coordinate in radians

    lat2 is the second latitudinal coordinate in radians

    Returns
    -------
    The angular separation between the two points in radians

    Calculated based on the haversine formula
    From http://en.wikipedia.org/wiki/Haversine_formula
    """
    are_arrays_1 = _validate_inputs([long1, lat1],
                                    ['long1', 'lat1'],
                                    'angularSeparation')

    are_arrays_2 = _validate_inputs([long2, lat2],
                                    ['long2', 'lat2'],
                                    'angularSeparation')

    # The code below is necessary because the call to np.radians() in
    # angularSeparation() will automatically convert floats
    # into length 1 numpy arrays, confusing validate_inputs()
    if are_arrays_1 and len(long1) == 1:
        are_arrays_1 = False
        long1 = long1[0]
        lat1 = lat1[0]

    if are_arrays_2 and len(long2) == 1:
        are_arrays_2 = False
        long2 = long2[0]
        lat2 = lat2[0]

    if are_arrays_1 and are_arrays_2:
        if len(long1) != len(long2):
            raise RuntimeError("You need to pass arrays of the same length "
                               "as arguments to angularSeparation()")

    t1 = np.sin(lat2/2.0 - lat1/2.0)**2
    t2 = np.cos(lat1)*np.cos(lat2)*np.sin(long2/2.0 - long1/2.0)**2
    _sum = t1 + t2

    if isinstance(_sum, numbers.Number):
        if _sum<0.0:
            _sum = 0.0
    else:
        _sum = np.where(_sum<0.0, 0.0, _sum)

    return 2.0*np.arcsin(np.sqrt(_sum))

def angularSeparation(long1, lat1, long2, lat2):
    """
    Angular separation between two points in degrees

    Parameters
    ----------
    long1 is the first longitudinal coordinate in degrees

    lat1 is the first latitudinal coordinate in degrees

    long2 is the second longitudinal coordinate in degrees

    lat2 is the second latitudinal coordinate in degrees

    Returns
    -------
    The angular separation between the two points in degrees
    """
    return np.degrees(_angularSeparation(np.radians(long1),
                                         np.radians(lat1),
                                         np.radians(long2),
                                         np.radians(lat2)))


def haversine(long1, lat1, long2, lat2):
    """
    DEPRECATED; use angularSeparation() instead

    Return the angular distance between two points in radians

    @param [in] long1 is the longitude of point 1 in radians

    @param [in] lat1 is the latitude of point 1 in radians

    @param [in] long2 is the longitude of point 2 in radians

    @param [in] lat2 is the latitude of point 2 in radians

    @param [out] the angular separation between points 1 and 2 in radians
    """
    return _angularSeparation(long1, lat1, long2, lat2)


def arcsecFromRadians(value):
    """
    Convert an angle in radians to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0 * np.degrees(value)


def radiansFromArcsec(value):
    """
    Convert an angle in arcseconds to radians

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return np.radians(value / 3600.0)


def arcsecFromDegrees(value):
    """
    Convert an angle in degrees to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0 * value


def degreesFromArcsec(value):
    """
    Convert an angle in arcseconds to degrees

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return value / 3600.0
