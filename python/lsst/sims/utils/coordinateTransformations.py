import numpy
import palpy
from collections import OrderedDict

__all__ = ["_galacticFromEquatorial", "galacticFromEquatorial",
           "_equatorialFromGalactic", "equatorialFromGalactic",
           "sphericalFromCartesian", "cartesianFromSpherical",
           "rotationMatrixFromVectors",
           "equationOfEquinoxes", "calcGmstGast", "calcLmstLast",
           "_altAzPaFromRaDec", "altAzPaFromRaDec",
           "_raDecFromAltAz", "raDecFromAltAz",
           "_getRotSkyPos", "getRotSkyPos",
           "_getRotTelPos", "getRotTelPos",
           "haversine",
           "calcObsDefaults", "makeObservationMetadata", "makeObsParamsAzAltTel",
           "makeObsParamsAzAltSky", "makeObsParamsRaDecTel", "makeObsParamsRaDecSky",
           "arcsecFromRadians", "radiansFromArcsec",
           "arcsecFromDegrees", "degreesFromArcsec"]


def calcLmstLast(mjd, longRad):
    """
    calculates local mean sidereal time and local apparent sidereal time

    @param [in] mjd is the universal time expressed as an MJD.
    This can be a numpy array or a single value.

    @param [in] longRad is the longitude in radians (positive east of the prime meridian)
    This can be numpy array or a single value.  If a numpy array, should have the same length as mjd.  In that
    case, each longRad will be applied only to the corresponding mjd.

    @param [out] lmst is the local mean sidereal time in hours

    @param [out] last is the local apparent sideral time in hours
    """
    mjdIsArray = False
    longRadIsArray = False
    if isinstance(mjd, numpy.ndarray):
        mjdIsArray = True

    if isinstance(longRad, numpy.ndarray):
        longRadIsArray = True

    if longRadIsArray and mjdIsArray:
        if len(longRad) != len(mjd):
            raise RuntimeError("in calcLmstLast mjd and longRad have different lengths")

    if longRadIsArray and not mjdIsArray:
        raise RuntimeError("in calcLmstLast longRad is numpy array but mjd is not")

    longDeg0 = numpy.degrees(longRad)
    longDeg0 %= 360.0

    if longRadIsArray:
        longDeg = numpy.where(longDeg0>180.0, longDeg0-360.0, longDeg0)
    else:
        if longDeg0 > 180.:
            longDeg = longDeg0-360.
        else:
            longDeg = longDeg0

    hrs = longDeg/15.
    gmstgast = calcGmstGast(mjd)
    lmst = gmstgast[0]+hrs
    last = gmstgast[1]+hrs
    lmst %= 24.
    last %= 24.
    return lmst, last


def galacticFromEquatorial(ra, dec):
    '''Convert RA,Dec (J2000) to Galactic Coordinates

    @param [in] ra is right ascension in degrees, either a float or a numpy array

    @param [in] dec is declination in degrees, either a float or a numpy array

    @param [out] gLong is galactic longitude in degrees

    @param [out] gLat is galactic latitude in degrees
    '''

    gLong, gLat = _galacticFromEquatorial(numpy.radians(ra), numpy.radians(dec))
    return numpy.degrees(gLong), numpy.degrees(gLat)


def _galacticFromEquatorial(ra, dec):
    '''Convert RA,Dec (J2000) to Galactic Coordinates

    All angles are in radians

    @param [in] ra is right ascension in radians, either a float or a numpy array

    @param [in] dec is declination in radians, either a float or a numpy array

    @param [out] gLong is galactic longitude in radians

    @param [out] gLat is galactic latitude in radians
    '''

    if isinstance(ra, numpy.ndarray):
        gLong, gLat = palpy.eqgalVector(ra, dec)
    else:
        gLong, gLat = palpy.eqgal(ra, dec)

    return gLong, gLat


def equatorialFromGalactic(gLong, gLat):
    '''Convert Galactic Coordinates to RA, dec (J2000)

    @param [in] gLong is galactic longitude in degrees, either a float or a numpy array
    (0 <= gLong <= 360.)

    @param [in] gLat is galactic latitude in degrees, either a float or a numpy array
    (-90. <= gLat <= 90.)

    @param [out] ra is right ascension in degrees

    @param [out] dec is declination in degrees
    '''

    ra, dec = _equatorialFromGalactic(numpy.radians(gLong), numpy.radians(gLat))
    return numpy.degrees(ra), numpy.degrees(dec)


def _equatorialFromGalactic(gLong, gLat):
    '''Convert Galactic Coordinates to RA, dec (J2000)

    @param [in] gLong is galactic longitude in radians, either a float or a numpy array
    (0 <= gLong <= 2*pi)

    @param [in] gLat is galactic latitude in radians, either a float or a numpy array
    (-pi/2 <= gLat <= pi/2)

    @param [out] ra is right ascension in radians

    @param [out] dec is declination in radians
    '''

    if isinstance(gLong, numpy.ndarray):
        ra, dec = palpy.galeqVector(gLong, gLat)
    else:
        ra, dec = palpy.galeq(gLong, gLat)

    return ra, dec


def cartesianFromSpherical(longitude, latitude):
    """
    Transforms between spherical and Cartesian coordinates.

    @param [in] longitude is a numpy array of longitudes

    @param [in] latitude is a numpy array of latitudes

    @param [out] a numpy array of the (three-dimensional) cartesian coordinates on a unit sphere

    All angles are in radians
    """

    if not isinstance(longitude, numpy.ndarray) or not isinstance(latitude, numpy.ndarray):
        raise RuntimeError("you need to pass numpy arrays to cartesianFromSpherical")

    cosDec = numpy.cos(latitude)
    return numpy.array([numpy.cos(longitude)*cosDec,
                      numpy.sin(longitude)*cosDec,
                      numpy.sin(latitude)]).transpose()


def sphericalFromCartesian(xyz):
    """
    Transforms between Cartesian and spherical coordinates

    @param [in] xyz is a numpy array of points in 3-D space.
    Each row is a different point.

    @param [out] returns longitude and latitude

    All angles are in radians
    """

    if not isinstance(xyz, numpy.ndarray):
        raise RuntimeError("you need to pass a numpy array to sphericalFromCartesian")

    if len(xyz.shape)>1:
        rad = numpy.sqrt(numpy.power(xyz,2).sum(axis=1))
        longitude = numpy.arctan2( xyz[:,1], xyz[:,0])
        latitude = numpy.arcsin( xyz[:,2] / rad)
    else:
        rad = numpy.sqrt(numpy.dot(xyz,xyz))
        longitude = numpy.arctan2(xyz[1], xyz[0])
        latitude = numpy.arcsin(xyz[2]/rad)

    return longitude, latitude


def rotationMatrixFromVectors(v1, v2):
    '''
    Given two vectors v1,v2 calculate the rotation matrix for v1->v2 using the axis-angle approach

    @param [in] v1, v2 are two Cartesian unit vectors (in three dimensions)

    @param [out] rot is the rotation matrix that rotates from one to the other

    '''

    if numpy.abs(numpy.sqrt(numpy.dot(v1,v1))-1.0) > 0.01:
        raise RuntimeError("v1 in rotationMatrixFromVectors is not a unit vector")

    if numpy.abs(numpy.sqrt(numpy.dot(v2,v2))-1.0) > 0.01:
        raise RuntimeError("v2 in rotationMatrixFromVectors is not a unit vector")

    # Calculate the axis of rotation by the cross product of v1 and v2
    cross = numpy.cross(v1,v2)
    cross = cross / numpy.sqrt(numpy.dot(cross,cross))

    # calculate the angle of rotation via dot product
    angle  = numpy.arccos(numpy.dot(v1,v2))
    sinDot = numpy.sin(angle)
    cosDot = numpy.cos(angle)

    # calculate the corresponding rotation matrix
    # http://en.wikipedia.org/wiki/Rotation_matrix#Rotation_matrix_from_axis_and_angle
    rot = [[cosDot + cross[0]*cross[0]*(1-cosDot), -cross[2]*sinDot+(1-cosDot)*cross[0]*cross[1], \
            cross[1]*sinDot + (1-cosDot)*cross[0]*cross[2]],\
            [cross[2]*sinDot+(1-cosDot)*cross[0]*cross[1], cosDot + (1-cosDot)*cross[1]*cross[1], \
            -cross[0]*sinDot+(1-cosDot)*cross[1]*cross[2]], \
            [-cross[1]*sinDot+(1-cosDot)*cross[0]*cross[2], \
            cross[0]*sinDot+(1-cosDot)*cross[1]*cross[2], \
            cosDot + (1-cosDot)*(cross[2]*cross[2])]]

    return rot


def equationOfEquinoxes(d):
    """
    The equation of equinoxes. See http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] d is either a numpy array or a float that is Terrestrial Time
    expressed as an MJD

    @param [out] the equation of equinoxes in radians.
    """

    if isinstance(d, numpy.ndarray):
        return palpy.eqeqxVector(d)
    else:
        return palpy.eqeqx(d)


def calcGmstGast(mjd):
    """
    Compute Greenwich mean sidereal time and Greenwich apparent sidereal time
    see: From http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] mjd is the universal time expressed as an MJD

    @param [out] gmst Greenwich mean sidereal time in hours

    @param [out] gast Greenwich apparent sidereal time in hours
    """

    date = numpy.floor(mjd)
    ut1 = mjd-date
    if isinstance(mjd, numpy.ndarray):
        gmst = palpy.gmstaVector(date, ut1)
    else:
        gmst = palpy.gmsta(date, ut1)

    eqeq = equationOfEquinoxes(mjd)
    gast = gmst + eqeq

    gmst = gmst*24.0/(2.0*numpy.pi)
    gmst %= 24.0

    gast = gast*24.0/(2.0*numpy.pi)
    gast %= 24.0

    return gmst, gast


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

    alt, az, pa = _altAzPaFromRaDec(numpy.radians(ra), numpy.radians(dec),
                                    numpy.radians(lon), numpy.radians(lat),
                                    mjd)

    return numpy.degrees(alt), numpy.degrees(az), numpy.degrees(pa)



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

    if isinstance(longRad, numpy.ndarray):
        raise RuntimeError('cannot pass numpy array of longitudes to altAzPaFromRaDec')

    if isinstance(latRad, numpy.ndarray):
        raise RuntimeError('cannot pass numpy array of latitudes to altAzPaFromRaDec')

    raIsArray = False
    decIsArray = False
    mjdIsArray = False
    if isinstance(raRad, numpy.ndarray):
        raIsArray = True

    if isinstance(decRad, numpy.ndarray):
        decIsArray = True

    if isinstance(mjd, numpy.ndarray):
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
    haRad = numpy.radians(last*15.0) - raRad

    if isinstance(haRad, numpy.ndarray):
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
    """

    ra, dec = _raDecFromAltAz(numpy.radians(alt), numpy.radians(az),
                              numpy.radians(lon), numpy.radians(lat), mjd)

    return numpy.degrees(ra), numpy.degrees(dec)


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
    """
    if isinstance(longRad, numpy.ndarray):
        raise RuntimeError('cannot pass a numpy array of longitudes to raDecFromAltAz')

    if isinstance(latRad, numpy.ndarray):
        raise RuntimeError('cannot pass a numpy array of latitudes to raDecFromAltAz')

    mjdIsArray = False
    altIsArray = False
    azIsArray = False
    if isinstance(mjd, numpy.ndarray):
        mjdIsArray = True

    if isinstance(altRad, numpy.ndarray):
        altIsArray = True

    if isinstance(azRad, numpy.ndarray):
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
    sinAlt = numpy.sin(altRad)
    cosLat = numpy.cos(latRad)
    sinLat = numpy.sin(latRad)
    decRad = numpy.arcsin(sinLat*sinAlt+ cosLat*numpy.cos(altRad)*numpy.cos(azRad))
    haRad0 = numpy.arccos((sinAlt - numpy.sin(decRad)*sinLat)/(numpy.cos(decRad)*cosLat))
    haRad = numpy.where(numpy.sin(azRad)>=0.0, -1.0*haRad0, haRad0)
    raRad = numpy.radians(last*15.) - haRad
    return raRad, decRad


def getRotSkyPos(ra, dec, lon, lat, mjd, rotTel):
    """
    @param [in] ra is the RA in degrees.  Can be a numpy array or a single value.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.

    @param [in] lon is the observer's longitude in degrees
    (positive east of the prime meridian).  Must be a single value.

    @param [in] lat is the observer's latitude in degrees
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MJD.
    Can be a numpy array or a single value.  If a numpy array, must have the same length
    as the numpy arrays of ra and dec.  In this case, each mjd will be associated
    with the corresponding ra, dec pair.

    @param [in] rotTel is rotTelPos in degrees
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as ra and dec.  In this case,
    each rotTel will be associated with the corresponding ra, dec pair.

    @param [out] rotSkyPos in degrees

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (one is the beginning of the exposure; the
    other is the middle of the exposure).
    """

    rotSky = _getRotSkyPos(numpy.radians(ra), numpy.radians(dec),
                           numpy.radians(lon), numpy.radians(lat),
                           mjd, numpy.radians(rotTel))

    return numpy.degrees(rotSky)



def _getRotSkyPos(raRad, decRad, longRad, latRad, mjd, rotTelRad):
    """
    @param [in] raRad is the RA in radians.  Can be a numpy array or a single value.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.

    @param [in] longRad is the observer's longitude in radians
    (positive east of the prime meridian).  Must be a single value.

    @param [in] latRad is the observer's latitude in radians
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MJD.
    Can be a numpy array or a single value.  If a numpy array, must have the same length
    as the numpy arrays of raRad and decRad.  In this case, each mjd will be associated
    with the corresponding raRad, decRad pair.

    @param [in] rotTelRad is rotTelPos in radians
    (the angle of the camera rotator).  Can be a numpy array or a single value.
    If a numpy array, should have the same length as raRad and decRad.  In this case,
    each rotTelRad will be associated with the corresponding raRad, decRad pair.

    @param [out] rotSkyPos in radians

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (one is the beginning of the exposure; the
    other is the middle of the exposure).
    """
    altRad, azRad, paRad = _altAzPaFromRaDec(raRad, decRad, longRad, latRad, mjd)

    #20 March 2015
    #I do not know where this expression comes from; we should validate it against
    #the definitions of rotTelPos and rotSkyPos
    return (rotTelRad - paRad)%(2.*numpy.pi)


def getRotTelPos(ra, dec, lon, lat, mjd, rotSky):
    """
    @param [in] ra is RA in degrees.  Can be a numpy array or a single value.

    @param [in] dec is Dec in degrees.  Can be a numpy array or a single value.

    @param [in] lon is the observer's longitude in degrees
    (positive east of the prime meridian).  Must be a single value.

    @param [in] lat is the observer's latitude in degrees
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MJD.
    Can be a numpy array or a single value.  If a numpy array, must have the same length
    as ra and dec.  In this case, each MJD will be associated with the
    corresponding ra, dec pair.

    @param [in] rotSkyRad is rotSkyPos in degrees
    (the angle of the field of view relative to the South pole given a
    rotator angle).  Can be a numpy array or a single value.  If a numpy array, should
    have the same length as ra and dec.  In this case, each rotSkyPos
    will be associated with the corresponding ra, dec pair.

    @param [out] rotTelPos in degrees.

    WARNING: As of 13 April 2015, this method does not agree with OpSim on
    the relationship between rotSkyPos and rotTelPos.  This is due to a
    discrepancy between the time that OpSim uses as the MJD when calculating
    rotTelPos and the time that OpSim reports as being the actual expmjd
    of the exposure (one is the beginning of the exposure; the
    other is the middle of the exposure).
    """

    rotTel = _getRotTelPos(numpy.radians(ra), numpy.radians(dec),
                           numpy.radians(lon), numpy.radians(lat),
                           mjd, numpy.radians(rotSky))

    return numpy.degrees(rotTel)


def _getRotTelPos(raRad, decRad, longRad, latRad, mjd, rotSkyRad):
    """
    @param [in] raRad is RA in radians.  Can be a numpy array or a single value.

    @param [in] decRad is Dec in radians.  Can be a numpy array or a single value.

    @param [in] longRad is the observer's longitude in radians
    (positive east of the prime meridian).  Must be a single value.

    @param [in] latRad is the observer's latitude in radians
    (positive north of the equator).  Must be a single value.

    @param [in] mjd is the Universal Time expressed as an MJD.
    Can be a numpy array or a single value.  If a numpy array, must have the same length
    as raRad and decRad.  In this case, each MJD will be associated with the
    corresponding raRad, decRad pair.

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
    of the exposure (one is the beginning of the exposure; the
    other is the middle of the exposure).
    """
    altRad, azRad, paRad = _altAzPaFromRaDec(raRad, decRad, longRad, latRad, mjd)

    #20 March 2015
    #I do not know where this expression comes from; we should validate it against
    #the definitions of rotTelPos and rotSkyPos
    return (rotSkyRad + paRad)%(2.*numpy.pi)


def haversine(long1, lat1, long2, lat2):
    """
    Return the angular distance between two points in radians

    @param [in] long1 is the longitude of point 1 in radians

    @param [in] lat1 is the latitude of point 1 in radians

    @param [in] long2 is the longitude of point 2 in radians

    @param [in] lat2 is the latitude of point 2 in radians

    @param [out] the angular separation between points 1 and 2 in radians

    From http://en.wikipedia.org/wiki/Haversine_formula
    """
    t1 = numpy.sin(lat2/2.-lat1/2.)**2
    t2 = numpy.cos(lat1)*numpy.cos(lat2)*numpy.sin(long2/2. - long1/2.)**2
    return 2*numpy.arcsin(numpy.sqrt(t1 + t2))


def calcObsDefaults(raRad, decRad, altRad, azRad, rotTelRad, mjd, band, longRad, latRad):
    """
    Fromats input data into a dict of metadata that PhoSim expect.

    Users should probably not be calling this by hand, as it makes no effort to ensure that
    the input values are self-consistent.  This method is called by
    makeObsParamsAzAltTel
    makeObsParamsAzAltSky
    makeObsPeramsRaDecTel
    makeObsParamsRaDecSky
    which do ensure self-consistency of values.

    @param [in] raRad is RA in radians

    @param [in] decRad is Dec in radians

    @param [in] altRad is altitude in radians

    @param [in] azRad is azimuth in radians

    @param [in] rotTelRad is rotTelPos in radians

    @param [in] mjd is the Universal Time expressed as an MJD

    @param [in] band is 'u', 'g', 'r', 'i', 'z', or 'y'
    (i.e. the bandpass of the observation)

    @param [in] longRad is the observer's longitude in radians
    (positive east of the prime meridan)

    @param [in] latRad is the observer's latitude in radians
    (positive north of the equator)

    @param [out] a dict of meta data which PhoSim expects in the
    headers of its input InstanceCatalogs
    """
    obsMd = {}
    #Defaults
    moonra, moondec = _raDecFromAltAz(-numpy.pi/2., 0., longRad, latRad, mjd)
    sunalt = -numpy.pi/2.
    moonalt = -numpy.pi/2.
    dist2moon = haversine(moonra, moondec, raRad, decRad)
    obsMd['Opsim_moonra'] = moonra
    obsMd['Opsim_moondec'] = moondec
    obsMd['Opsim_sunalt'] = sunalt
    obsMd['Opsim_moonalt'] = moonalt
    obsMd['Opsim_dist2moon'] = dist2moon

    rotSkyPos = _getRotSkyPos(raRad, decRad, longRad, latRad, mjd, rotTelRad)
    obsMd['Opsim_filter'] = band
    obsMd['Unrefracted_RA'] = raRad
    obsMd['Unrefracted_Dec'] = decRad
    obsMd['Opsim_rotskypos'] = rotSkyPos
    obsMd['Opsim_rottelpos'] = rotTelRad
    obsMd['Unrefracted_Altitude'] = altRad
    obsMd['Unrefracted_Azimuth'] = azRad
    return obsMd


def makeObservationMetadata(metaData):
    return OrderedDict([(k,(metaData[k], numpy.asarray(metaData[k]).dtype))
                         for k in metaData])


def makeObsParamsAzAltTel(azRad, altRad, mjd, band, rotTelRad=0., longRad=-1.2320792, latRad=-0.517781017, **kwargs):
    '''
    Calculate a minimal set of observing parameters give the ra, dec, and time of the observation.
    altRad -- Altitude of the boresite of the observation in radians
    azRad -- Azimuth of the boresite of the observation in radians
    mjd -- MJD of the observation
    band -- bandpass of the observation e.g. 'r'
    rotTelRad -- Rotation of the camera relative to the telescope in radians Default=0.
    longRad -- Longitude of the observatory in radians Default=-1.2320792
    latRad -- Latitude of the observatory in radians Default=-0.517781017
    **kwargs -- The kwargs will be put in the returned dictionary overriding the default value if it exists
    '''

    raRad, decRad = _raDecFromAltAz(altRad, azRad, longRad, latRad, mjd)
    obsMd = calcObsDefaults(raRad, decRad, altRad, azRad, rotTelRad, mjd, band, longRad, latRad)
    obsMd.update(kwargs)
    return makeObservationMetadata(obsMd)


def makeObsParamsAzAltSky(azRad, altRad, mjd, band, rotSkyRad=numpy.pi, longRad=-1.2320792, latRad=-0.517781017, **kwargs):
    '''
    Calculate a minimal set of observing parameters give the ra, dec, and time of the observation.
    altRad -- Altitude of the boresite of the observation in radians
    azRad -- Azimuth of the boresite of the observation in radians
    mjd -- MJD of the observation
    band -- bandpass of the observation e.g. 'r'
    rotTelRad -- Rotation of the field of view relative to the North pole in radians Default=0.
    longRad -- Longitude of the observatory in radians Default=-1.2320792
    latRad -- Latitude of the observatory in radians Default=-0.517781017
    **kwargs -- The kwargs will be put in the returned dictionary overriding the default value if it exists
    '''
    raRad, decRad = _raDecFromAltAz(altRad, azRad, longRad, latRad, mjd)
    rotTelRad = _getRotTelPos(raRad, decRad, longRad, latRad, mjd, rotSkyRad)
    return makeObsParamsAzAltTel(azRad, altRad, mjd, band, rotTelRad=rotTelRad, longRad=longRad, latRad=latRad, **kwargs)


def makeObsParamsRaDecTel(raRad, decRad, mjd, band, rotTelRad=0., longRad=-1.2320792, latRad=-0.517781017, **kwargs):
    '''
    Calculate a minimal set of observing parameters give the ra, dec, and time of the observation.
    raRad -- RA of the boresite of the observation in radians
    decRad -- Dec of the boresite of the observation in radians
    mjd -- MJD of the observation
    band -- bandpass of the observation e.g. 'r'
    rotTelRad -- Rotation of the camera relative to the telescope in radians Default=0.
    longRad -- Longitude of the observatory in radians Default=-1.2320792
    latRad -- Latitude of the observatory in radians Default=-0.517781017
    **kwargs -- The kwargs will be put in the returned dictionary overriding the default value if it exists
    '''
    altRad, azRad, paRad = altAzPaFromRaDec(raRad, decRad, longRad, latRad, mjd)
    obsMd = calcObsDefaults(raRad, decRad, altRad, azRad, rotTelRad, mjd, band, longRad, latRad)
    obsMd.update(kwargs)
    return makeObservationMetadata(obsMd)


def makeObsParamsRaDecSky(raRad, decRad, mjd, band, rotSkyRad=numpy.pi, longRad=-1.2320792, latRad=-0.517781017, **kwargs):
    '''
    Calculate a minimal set of observing parameters give the ra, dec, and time of the observation.
    raRad -- RA of the boresite of the observation in radians
    decRad -- Dec of the boresite of the observation in radians
    mjd -- MJD of the observation
    band -- bandpass of the observation e.g. 'r'
    rotSkyRad -- Rotation of the field of view relative to the North pole in radians Default=0.
    longRad -- Longitude of the observatory in radians Default=-1.2320792
    latRad -- Latitude of the observatory in radians Default=-0.517781017
    **kwargs -- The kwargs will be put in the returned dictionary overriding the default value if it exists
    '''
    rotTelRad = _getRotTelPos(raRad, decRad, longRad, latRad, mjd, rotSkyRad)
    return makeObsParamsRaDecTel(raRad, decRad, mjd, band, rotTelRad=rotTelRad, longRad=longRad, latRad=latRad, **kwargs)


def arcsecFromRadians(value):
    """
    Convert an angle in radians to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0*numpy.degrees(value)


def radiansFromArcsec(value):
    """
    Convert an angle in arcseconds to radians

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return numpy.radians(value/3600.0)


def arcsecFromDegrees(value):
    """
    Convert an angle in degrees to arcseconds

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return 3600.0*value


def degreesFromArcsec(value):
    """
    Convert an angle in arcseconds to degrees

    Note: if you input None, you will get None back
    """
    if value is None:
        return None

    return value/3600.0
