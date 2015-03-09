import math
import numpy
import palpy
from collections import OrderedDict

__all__ = ["equationOfEquinoxes", "calcGmstGast", "calcLmstLast", "raDecToAltAz",
           "altAzToRaDec", "calcPa", "getRotSkyPos", "getRotTelPos", "haversine",
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

    @param [in] longRad is the longitude in radians

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

def raDecToAltAz(raRad, decRad, longRad, latRad, mjd):
    lst = calcLmstLast(mjd, longRad)
    last = lst['LAST']
    haRad = math.radians(last*15.) - raRad
    altRad = math.asin(math.sin(decRad)*math.sin(latRad)+math.cos(decRad)*math.cos(latRad)*math.cos(haRad))
    azRad = math.acos((math.sin(decRad) - math.sin(altRad)*math.sin(latRad))/(math.cos(altRad)*math.cos(latRad)))
    if math.sin(haRad) >= 0:
        azRad = 2.*math.pi-azRad
    return altRad, azRad

def altAzToRaDec(altRad, azRad, longRad, latRad, mjd):
    lst = calcLmstLast(mjd, longRad)
    last = lst['LAST']
    decRad = math.asin(math.sin(latRad)*math.sin(altRad)+ math.cos(latRad)*math.cos(altRad)*math.cos(azRad))
    haRad = math.acos((math.sin(altRad) - math.sin(decRad)*math.sin(latRad))/(math.cos(decRad)*math.cos(latRad)))
    raRad = math.radians(last*15.) - haRad
    return raRad, decRad

def calcPa(azRad, decRad, latRad):
    """
    Calculate the Parallactic angle
    azRad is the azimuth of the object assuming OpSim conventions (radians)
    latRad is the latitude of the observatory (radians)
    decRad is the declination of the object (radians)
    """
    try:
        paRad = math.asin(math.sin(azRad)*math.cos(latRad)/math.cos(decRad))
    except ValueError, e:
        if not math.fabs(decRad) > math.fabs(latRad):
            raise ValueError("The point is circumpolar but the Azimuth is not valid: Az=%.2f"%(math.degrees(azRad)))
        else:
            raise e
    return paRad

def getRotSkyPos(azRad, decRad, latRad, rotTelRad):
    """
    azRad is the azimuth of the object assuming opSim conventions (radians)
    decRad is the declination of the object (radians)
    latRad is the latitude of the observatory (radians)
    rotTelRad is the angle of the camera rotator assuming OpSim
    conventions (radians)
    """
    paRad = calcPa(azRad, decRad, latRad)
    return (rotTelRad - paRad + math.pi)%(2.*math.pi)

def getRotTelPos(azRad, decRad, latRad, rotSkyRad):
    """
    azRad is the azimuth of the object assuming opSim conventions (radians)
    decRad is the declination of the object (radians)
    latRad is the latitude of the observatory (radians)
    rotSkyRad is the angle of the field of view relative to the South pole given
    a rotator angle in OpSim conventions (radians)
    """
    paRad = calcPa(azRad, decRad, latRad)
    return (rotSkyRad + paRad - math.pi)%(2.*math.pi)

def haversine(long1, lat1, long2, lat2):
    #From http://en.wikipedia.org/wiki/Haversine_formula
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
