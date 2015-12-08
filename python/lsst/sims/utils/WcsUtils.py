import numpy
from lsst.sims.utils import _observedFromICRS, _icrsFromObserved

__all__ = ["_nativeLonLatFromRaDec", "_raDecFromNativeLonLat",
           "nativeLonLatFromRaDec", "raDecFromNativeLonLat"]

def _nativeLonLatFromRaDec(ra_in, dec_in, obs_metadata):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The transformation is achieved by rotating the vector pointing to the RA
    and Dec being transformed once about the x axis and once about the z axis.
    These are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    Note: RA, and Dec are assumed to be in the International Celestial Reference
    System.  Before calculating native longitude and latitude, this method will
    convert RA, and Dec to observed geocentric coordinates.

    @param [in] ra is the RA of the star being transformed in radians
    (in the International Celestial Reference System)

    @param [in] dec is the Dec of the star being transformed in radians
    (in the International Celestial Reference System)

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing of
    the telescope.

    @param [out] lonOut is the native longitude in radians

    @param [out] latOut is the native latitude in radians
    """

    if not hasattr(ra_in, '__len__'):
        ra_temp, dec_temp = _observedFromICRS(numpy.array([ra_in]), numpy.array([dec_in]),
                                              obs_metadata=obs_metadata, epoch=2000.0,
                                              includeRefraction=True)

        ra = ra_temp[0]
        dec = dec_temp[0]
    else:
        ra, dec = _observedFromICRS(ra_in, dec_in,
                                    obs_metadata=obs_metadata, epoch=2000.0,
                                    includeRefraction=True)

    ra_temp, dec_temp = _observedFromICRS(numpy.array([obs_metadata._pointingRA]),
                                          numpy.array([obs_metadata._pointingDec]),
                                          obs_metadata=obs_metadata, epoch=2000.0,
                                          includeRefraction=True)

    raPointing = ra_temp[0]
    decPointing = dec_temp[0]

    x = -1.0*numpy.cos(dec)*numpy.sin(ra)
    y = numpy.cos(dec)*numpy.cos(ra)
    z = numpy.sin(dec)

    alpha = decPointing - 0.5*numpy.pi
    beta = raPointing

    ca=numpy.cos(alpha)
    sa=numpy.sin(alpha)
    cb=numpy.cos(beta)
    sb=numpy.sin(beta)

    v2 = numpy.dot(numpy.array([
                                [1.0, 0.0, 0.0],
                                [0.0, ca, sa],
                                [0.0, -1.0*sa, ca]
                                ]),
                   numpy.dot(numpy.array([[cb, sb, 0.0],
                                          [-sb, cb, 0.0],
                                          [0.0, 0.0, 1.0]
                                          ]), numpy.array([x,y,z])))

    cc = numpy.sqrt(v2[0]*v2[0]+v2[1]*v2[1])
    latOut = numpy.arctan2(v2[2], cc)

    _y = v2[1]/numpy.cos(latOut)
    _ra_raw = numpy.arccos(_y)

    # control for _y=1.0, -1.0 but actually being stored as just outside
    # the bounds of -1.0<=_y<=1.0 because of floating point error
    if hasattr(_ra_raw, '__len__'):
        _ra = numpy.array([rr if not numpy.isnan(rr) \
                           else 0.5*numpy.pi*(1.0-numpy.sign(yy)) \
                           for rr, yy in zip(_ra_raw, _y)])
    else:
        if numpy.isnan(_ra_raw):
            if numpy.sign(_y)<0.0:
                _ra = numpy.pi
            else:
                _ra = 0.0
        else:
            _ra = _ra_raw

    _x = -numpy.sin(_ra)

    if type(_ra) is numpy.float64:
        if numpy.isnan(_ra):
            lonOut = 0.0
        elif (numpy.abs(_x)>1.0e-9 and numpy.sign(_x)!=numpy.sign(v2[0])) \
             or (numpy.abs(_y)>1.0e-9 and numpy.sign(_y)!=numpy.sign(v2[1])):
            lonOut = 2.0*numpy.pi-_ra
        else:
            lonOut = _ra
    else:
        _lonOut = [2.0*numpy.pi-rr if (numpy.abs(xx)>1.0e-9 and numpy.sign(xx)!=numpy.sign(v2_0)) \
                                   or (numpy.abs(yy)>1.0e-9 and numpy.sign(yy)!=numpy.sign(v2_1)) \
                                   else rr \
                                   for rr, xx, yy, v2_0, v2_1 in zip(_ra, _x, _y, v2[0], v2[1])]

        lonOut = numpy.array([0.0 if numpy.isnan(ll) else ll for ll in _lonOut])

    return lonOut, latOut


def nativeLonLatFromRaDec(ra, dec, obs_metadata):
    """
    Convert the RA and Dec of a star into `native' longitude and latitude.

    Native longitude and latitude are defined as what RA and Dec would be
    if the celestial pole were at the location where the telescope is pointing.
    The coordinate basis axes for this system is achieved by taking the true
    coordinate basis axes and rotating them once about the z axis and once about
    the x axis (or, by rotating the vector pointing to the RA and Dec being
    transformed once about the x axis and once about the z axis).  These
    are the Euler rotations referred to in Section 2.3 of

    Calabretta and Greisen (2002), A&A 395, p. 1077

    Note: RA, and Dec are assumed to be in the International Celestial Reference
    System.  Before calculating native longitude and latitude, this method will
    convert RA, and Dec to observed geocentric coordinates.

    @param [in] ra is the RA of the star being transformed in degrees
    (in the International Celestial Reference System)

    @param [in] dec is the Dec of the star being transformed in degrees
    (in the International Celestial Reference System)

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing of
    the telescope.

    @param [out] lonOut is the native longitude in degrees

    @param [out] latOut is the native latitude in degrees
    """

    lon, lat = _nativeLonLatFromRaDec(numpy.radians(ra), numpy.radians(dec),
                                      obs_metadata)

    return numpy.degrees(lon), numpy.degrees(lat)


def _raDecFromNativeLonLat(lon, lat, obs_metadata):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for _nativeLonLatFromRaDec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in radians

    @param [in] lat is the native latitude in radians

    @param [in] obs_metadata is an ObservationMetaData characterizing the pointing
    of the telescope

    @param [out] raOut is the RA of the star in radians
    (in the International Celestial Reference System)

    @param [in] decOut is the Dec of the star in radians
    (in the International Celestial Reference System)

    Note: Because of its reliance on icrsFromObserved, this
    method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.
    """

    ra_temp, dec_temp = _observedFromICRS(numpy.array([obs_metadata._pointingRA]),
                                          numpy.array([obs_metadata._pointingDec]),
                                          obs_metadata=obs_metadata, epoch=2000.0,
                                          includeRefraction=True)

    raPointing = ra_temp[0]
    decPointing = dec_temp[0]

    x = -1.0*numpy.cos(lat)*numpy.sin(lon)
    y = numpy.cos(lat)*numpy.cos(lon)
    z = numpy.sin(lat)

    alpha = 0.5*numpy.pi - decPointing
    beta = raPointing

    ca=numpy.cos(alpha)
    sa=numpy.sin(alpha)
    cb=numpy.cos(beta)
    sb=numpy.sin(beta)

    v2 = numpy.dot(numpy.array([[cb, -1.0*sb, 0.0],
                                [sb, cb, 0.0],
                                [0.0, 0.0, 1.0]
                                ]),
                                numpy.dot(numpy.array([[1.0, 0.0, 0.0],
                                                       [0.0, ca, sa],
                                                       [0.0, -1.0*sa, ca]
                                ]),
                                numpy.array([x,y,z])))


    cc = numpy.sqrt(v2[0]*v2[0]+v2[1]*v2[1])
    decObs = numpy.arctan2(v2[2], cc)

    _y = v2[1]/numpy.cos(decObs)
    _ra = numpy.arccos(_y)
    _x = -numpy.sin(_ra)

    if type(_ra) is numpy.float64:
        if numpy.isnan(_ra):
            raObs = 0.0
        elif (numpy.abs(_x)>1.0e-9 and numpy.sign(_x)!=numpy.sign(v2[0])) \
             or (numpy.abs(_y)>1.0e-9 and numpy.sign(_y)!=numpy.sign(v2[1])):
            raObs = 2.0*numpy.pi-_ra
        else:
            raObs = _ra
    else:
        _raObs = [2.0*numpy.pi-rr if (numpy.abs(xx)>1.0e-9 and numpy.sign(xx)!=numpy.sign(v2_0)) \
                                  or (numpy.abs(yy)>1.0e-9 and numpy.sign(yy)!=numpy.sign(v2_1)) \
                                  else rr \
                                  for rr, xx, yy, v2_0, v2_1 in zip(_ra, _x, _y, v2[0], v2[1])]

        raObs = numpy.array([0.0 if numpy.isnan(rr) else rr for rr in _raObs])


    # convert from observed geocentric coordinates to International Celestial Reference System
    # coordinates

    if hasattr(raObs,'__len__'):
        raOut, decOut = _icrsFromObserved(raObs, decObs, obs_metadata=obs_metadata,
                                          epoch=2000.0, includeRefraction=True)
    else:
        raOut, decOut = _icrsFromObserved(numpy.array([raObs]), numpy.array([decObs]),
                                          obs_metadata=obs_metadata,
                                          epoch=2000.0, includeRefraction=True)

    if not hasattr(lon, '__len__'):
        return raOut[0], decOut[0]

    return raOut, decOut


def raDecFromNativeLonLat(lon, lat, obs_metadata):
    """
    Transform a star's position in native longitude and latitude into
    RA and Dec.  See the doc string for nativeLonLatFromRaDec for definitions
    of native longitude and latitude.

    @param [in] lon is the native longitude in degrees

    @param [in] lat is the native latitude in degrees

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    pointing of the telescope

    @param [out] raOut is the RA of the star in degrees
    (in the International Celestial Reference System)

    @param [in] decOut is the Dec of the star in degrees
    (in the International Celestial Reference System)

    Note: Because of its reliance on icrsFromObserved, this
    method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.
    """

    ra, dec = _raDecFromNativeLonLat(numpy.radians(lon),
                                     numpy.radians(lat),
                                     obs_metadata)

    return numpy.degrees(ra), numpy.degrees(dec)
