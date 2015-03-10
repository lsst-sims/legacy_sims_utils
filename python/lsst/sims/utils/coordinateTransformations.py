import math
import numpy
import palpy
from collections import OrderedDict

__all__ = ["equationOfEquinoxes", "calcGmstGast", "calcLmstLast", "raDecToAltAzPa",
           "altAzToRaDec", "getRotSkyPos", "getRotTelPos", "haversine",
           "calcObsDefaults", "makeObservationMetadata", "makeObsParamsAzAltTel",
           "makeObsParamsAzAltSky", "makeObsParamsRaDecTel", "makeObsParamsRaDecSky",
           "radiansToArcsec","arcsecToRadians"]

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

def calcLmstLast(mjd, longRad):
    """
    calculates local mean sidereal time and local apparent sidereal time

    @param [in] mjd is the universal time expressed as an MJD

    @param [in] longRad is the longitude in radians (positive east of the prime meridian)

    @param [out] lmst is the local mean sidereal time in hours

    @param [out] last is hte local apparent sideral time in hours
    """
    longDeg = numpy.degrees(longRad)
    longDeg %= 360.
    if longDeg > 180.:
        longDeg -= 360.
    hrs = longDeg/15.
    gmstgast = calcGmstGast(mjd)
    lmst = gmstgast[0]+hrs
    last = gmstgast[1]+hrs
    lmst %= 24.
    last %= 24.
    return lmst, last

def raDecToAltAzPa(raRad, decRad, longRad, latRad, mjd):
    """
    Convert RA, Dec, longitude, latitude and MJD into altitude, azimuth
    and parallactic angle using PALPY

    @param [in] raRad is RA in radians

    @param [in] decRad is Dec in radians

    @param [in] longRad is the longitude of the observer in radians
    (positive east of the prime meridian)

    @param [in] latRad is the latitude of the observer in radians
    (positive north of the equator)

    @param [in] mjd is the Universal Time expressed as an MJD

    @param [out] altitude in radians

    @param [out] azimuth in radians

    @param [out] parallactic angle in radians
    """

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
        pa, pad, padd = palpy.altAz(haRad, decRad, latRad)

    return alt, az, pa

def altAzToRaDec(altRad, azRad, longRad, latRad, mjd):
    """
    Convert altitude and azimuth to RA and Dec

    @param [in] altRad is the altitude in radians

    @param [in] azRad is the azimuth in radians

    @param [in] longRad is the observatory longitude in radians
    (positive east of the prime meridian)

    @param [in] latRad is the latitude in radians
    (positive north of the equator)

    @param [in] mjd is the Universal Time expressed as an MD

    @param [out] RA in radians

    @param [out] Dec in radians
    """
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

def getRotSkyPos(raRad, decRad, longRad, latRad, mjd, rotTelRad):
    """
    @param [in] raRad is the RA in radians

    @param [in] decRad is Dec in radians

    @param [in] longRad is the observer's longitude in radians
    (positive east of the prime meridian)

    @param [in] latRad is the observer's latitude in radians
    (positive north of the equator)

    @param [in] mjd is the Universal Time expressed as an JD

    @param [in] rotTelRad is rotTelPos in radians
    (the angle of the camera rotator)

    @param [out] rotSkyPos in radians
    """
    altRad, azRad, paRad = raDecToAltAzPa(raRad, decRad, longRad, latRad, mjd)
    return (rotTelRad - paRad + math.pi)%(2.*math.pi)

def getRotTelPos(raRad, decRad, longRad, latRad, mjd, rotSkyRad):
    """
    @param [in] raRad is RA in radians

    @param [in] decRad is Dec in radians

    @param [in] longRad is the observer's longitude in radians
    (positive east of the prime meridian)

    @param [in] latRad is the observer's latitude in radians
    (positive north of the equator)

    @param [in] mjd is the Universal Time expressed as an MJD

    @parma [in] rotSkyRad is rotSkyPos in radians
    (the angle of the field of view relative to the South pole given a
    rotator angle)
    """
    altRad, azRad, paRad = raDecToAltAzPa(raRad, decRad, longRad, latRad, mjd)
    return (rotSkyRad + paRad - math.pi)%(2.*math.pi)

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
    obsMd = {}
    #Defaults
    moonra, moondec = altAzToRaDec(-math.pi/2., 0., longRad, latRad, mjd)
    sunalt = -math.pi/2.
    moonalt = -math.pi/2.
    dist2moon = haversine(moonra, moondec, raRad, decRad)
    obsMd['Opsim_moonra'] = moonra
    obsMd['Opsim_moondec'] = moondec
    obsMd['Opsim_sunalt'] = sunalt
    obsMd['Opsim_moonalt'] = moonalt
    obsMd['Opsim_dist2moon'] = dist2moon

    rotSkyPos = getRotSkyPos(azRad, decRad, latRad, rotTelRad)
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

    raRad, decRad = altAzToRaDec(altRad, azRad, longRad, latRad, mjd)
    obsMd = calcObsDefaults(raRad, decRad, altRad, azRad, rotTelRad, mjd, band, longRad, latRad)
    obsMd.update(kwargs)
    return makeObservationMetadata(obsMd)

def makeObsParamsAzAltSky(azRad, altRad, mjd, band, rotSkyRad=math.pi, longRad=-1.2320792, latRad=-0.517781017, **kwargs):
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
    raRad, decRad = altAzToRaDec(altRad, azRad, longRad, latRad, mjd)
    rotTelRad = getRotTelPos(azRad, decRad, latRad, rotSkyRad)
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
    altRad, azRad = raDecToAltAz(raRad, decRad, longRad, latRad, mjd)
    obsMd = calcObsDefaults(raRad, decRad, altRad, azRad, rotTelRad, mjd, band, longRad, latRad)
    obsMd.update(kwargs)
    return makeObservationMetadata(obsMd)

def makeObsParamsRaDecSky(raRad, decRad, mjd, band, rotSkyRad=math.pi, longRad=-1.2320792, latRad=-0.517781017, **kwargs):
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
    altRad, azRad = raDecToAltAz(raRad, decRad, longRad, latRad, mjd)
    rotTelRad = getRotTelPos(azRad, decRad, latRad, rotSkyRad)
    return makeObsParamsRaDecTel(raRad, decRad, mjd, band, rotTelRad=rotTelRad, longRad=longRad, latRad=latRad, **kwargs)

def radiansToArcsec(value):
    """
    Convert an angle in radians to arcseconds
    """

    return 3600.0*numpy.degrees(value)

def arcsecToRadians(value):
    """
    Convert an angle in arcseconds to radians
    """

    return numpy.radians(value/3600.0)
