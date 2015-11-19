import numpy
import palpy
from lsst.sims.utils import arcsecFromRadians, cartesianFromSpherical, sphericalFromCartesian
from lsst.sims.utils import radiansFromArcsec
from lsst.sims.utils import haversine

__all__ = ["applyRefraction", "refractionCoefficients",
           "_applyPrecession", "applyPrecession",
           "_applyProperMotion", "applyProperMotion",
           "_appGeoFromICRS", "appGeoFromICRS",
           "_icrsFromAppGeo", "icrsFromAppGeo",
           "_observedFromAppGeo", "observedFromAppGeo",
           "_appGeoFromObserved", "appGeoFromObserved",
           "_observedFromICRS", "observedFromICRS",
           "_icrsFromObserved", "icrsFromObserved",
           "_pupilCoordsFromRaDec", "pupilCoordsFromRaDec",
           "_raDecFromPupilCoords", "raDecFromPupilCoords"]



def refractionCoefficients(wavelength=0.5, site=None):
    """ Calculate the refraction using PAL's refco routine

    This calculates the refraction at 2 angles and derives a tanz and tan^3z
    coefficient for subsequent quick calculations. Good for zenith distances < 76 degrees

    @param [in] wavelength is effective wavelength in microns (default 0.5)

    @param [in] site is an instantiation of the Site class defined in
    sims_utils/../Site.py

    One should call PAL refz to apply the coefficients calculated here

    """
    precision = 1.e-10

    if site is None:
        raise RuntimeError("Cannot call refractionCoefficients; no site information")

    #TODO the latitude in refco needs to be astronomical latitude,
    #not geodetic latitude
    _refcoOutput=palpy.refco(site.height,
                        site.meanTemperature,
                        site.meanPressure,
                        site.meanHumidity,
                        wavelength ,
                        site.latitude,
                        site.lapseRate,
                        precision)

    return _refcoOutput[0], _refcoOutput[1]


def applyRefraction(zenithDistance, tanzCoeff, tan3zCoeff):
    """ Calculted refracted Zenith Distance

    uses the quick PAL refco routine which approximates the refractin calculation

    @param [in] zenithDistance is unrefracted zenith distance of the source in radians.
    Can either be a float or a numpy array (not a list).

    @param [in] tanzCoeff is the first output from refractionCoefficients (above)

    @param [in] tan3zCoeff is the second output from refractionCoefficients (above)

    @param [out] refractedZenith is the refracted zenith distance in radians

    """

    if isinstance(zenithDistance, list):
        raise RuntimeError("You passed a list of zenithDistances to " +
                           "applyRefraction.  The method won't know how to " +
                           "handle that.  Pass a numpy array.")

    if isinstance(zenithDistance, numpy.ndarray):
        refractedZenith = palpy.refzVector(zenithDistance, tanzCoeff, tan3zCoeff)
    else:
        refractedZenith=palpy.refz(zenithDistance, tanzCoeff, tan3zCoeff)

    return refractedZenith


def applyPrecession(ra, dec, epoch=2000.0, mjd=None):
    """
    applyPrecession() applies precesion and nutation to coordinates between two epochs.
    Accepts RA and dec as inputs.  Returns corrected RA and dec (in degrees).

    Assumes FK5 as the coordinate system
    units:  ra_in (degrees), dec_in (degrees)

    The precession-nutation matrix is calculated by the palpy.prenut method
    which uses the IAU 2006/2000A model

    @param [in] ra in degrees

    @param [in] dec in degrees

    @param [in] epoch is the epoch of the mean equinox (in years; default 2000)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the RA
    corrected for precession and nutation and the second row is the
    Dec corrected for precession and nutation (both in degrees)

    """

    output = _applyPrecession(numpy.radians(ra), numpy.radians(dec),
                              epoch=epoch, mjd=mjd)

    return numpy.degrees(output)



def _applyPrecession(ra, dec, epoch=2000.0, mjd=None):
    """
    _applyPrecession() applies precesion and nutation to coordinates between two epochs.
    Accepts RA and dec as inputs.  Returns corrected RA and dec (in radians).

    Assumes FK5 as the coordinate system
    units:  ra_in (radians), dec_in (radians)

    The precession-nutation matrix is calculated by the palpy.prenut method
    which uses the IAU 2006/2000A model

    @param [in] ra in radians

    @param [in] dec in radians

    @param [in] epoch is the epoch of the mean equinox (in years; default 2000)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the RA
    corrected for precession and nutation and the second row is the
    Dec corrected for precession and nutation (both in radians)
    """

    if hasattr(ra, '__len__'):
        if len(ra) != len(dec):
            raise RuntimeError("You supplied %d RAs but %d Decs to applyPrecession" %
                               (len(ra), len(dec)))

    if mjd is None:
        raise RuntimeError("You need to supply applyPrecession with an mjd")

    # Determine the precession and nutation
    #palpy.prenut takes the julian epoch for the mean coordinates
    #and the MJD for the the true coordinates
    #
    #TODO it is not specified what this MJD should be (i.e. in which
    #time system it should be reckoned)
    rmat=palpy.prenut(epoch, mjd.TT)

    # Apply rotation matrix
    xyz = cartesianFromSpherical(ra,dec)
    xyz =  numpy.dot(rmat,xyz.transpose()).transpose()

    raOut,decOut = sphericalFromCartesian(xyz)
    return numpy.array([raOut,decOut])


def applyProperMotion(ra, dec, pm_ra, pm_dec, parallax, v_rad, \
                      epoch=2000.0, mjd=None):
    """Applies proper motion between two epochs.

    units:  ra (degrees), dec (degrees), pm_ra (arcsec/year), pm_dec
    (arcsec/year), parallax (arcsec), v_rad (km/sec, positive if receding),
    epoch (Julian years)

    Returns corrected ra and dec (in radians)

    The function palpy.pm does not work properly if the parallax is below
    0.00045 arcseconds

    @param [in] ra in degrees.  Can be a float or a numpy array (not a list).

    @param [in] dec in degrees.  Can be a float or a numpy array (not a list).

    @param [in] pm_ra is ra proper motion multiplied by cos(Dec) in arcsec/year.
    Can be a float or a numpy array (not a list).

    @param [in] pm_dec is dec proper motion in arcsec/year.
    Can be a float or a numpy array (not a list).

    @param [in] parallax in arcsec. Can be a float or a numpy array (not a list).

    @param [in] v_rad is radial velocity in km/sec (positive if the object is receding).
    Can be a float or a numpy array (not a list).

    @param [in] epoch is epoch in Julian years (default: 2000.0)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the RA corrected
    for proper motion and the second row is the Dec corrected for proper motion
    (both in degrees)
    """

    output = _applyProperMotion(numpy.radians(ra), numpy.radians(dec),
                                radiansFromArcsec(pm_ra),
                                radiansFromArcsec(pm_dec),
                                radiansFromArcsec(parallax),
                                v_rad, epoch=epoch, mjd=mjd)

    return numpy.degrees(output)


def _applyProperMotion(ra, dec, pm_ra, pm_dec, parallax, v_rad, \
                      epoch=2000.0, mjd=None):
    """Applies proper motion between two epochs.

    units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
    (radians/year), parallax (radians), v_rad (km/sec, positive if receding),
    epoch (Julian years)

    Returns corrected ra and dec (in radians)

    The function palpy.pm does not work properly if the parallax is below
    0.00045 arcseconds

    @param [in] ra in radians.  Can be a float or a numpy array (not a list).

    @param [in] dec in radians.  Can be a float or a numpy array (not a list).

    @param [in] pm_ra is ra proper motion multiplied by cos(Dec) in radians/year.
    Can be a float or a numpy array (not a list).

    @param [in] pm_dec is dec proper motion in radians/year.
    Can be a float or a numpy array (not a list).

    @param [in] parallax in radians. Can be a float or a numpy array (not a list).

    @param [in] v_rad is radial velocity in km/sec (positive if the object is receding).
    Can be a float or a numpy array (not a list).

    @param [in] epoch is epoch in Julian years (default: 2000.0)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the RA corrected
    for proper motion and the second row is the Dec corrected for proper motion
    (both in radians)

    """

    if isinstance(ra, list) or isinstance(dec, list) or \
    isinstance(pm_ra, list) or isinstance(pm_dec, list) or \
    isinstance(parallax, list) or isinstance(v_rad, list):

        raise RuntimeError("You tried to pass lists to applyPm. " +
                           "The method does not know how to handle lists. " +
                           "Use numpy arrays.")

    if mjd is None:
        raise RuntimeError("cannot call applyProperMotion; mjd is None")

    parallaxArcsec=arcsecFromRadians(parallax)
    #convert to Arcsec because that is what PALPY expects

    # Generate Julian epoch from MJD
    #
    # 19 November 2015
    # I am assuming here that the time scale should be
    # Terrestrial Dynamical Time (TT), since that is used
    # as the independent variable for apparent geocentric
    # ephemerides
    julianEpoch = palpy.epj(mjd.TT)

    #because PAL and ERFA expect proper motion in terms of "coordinate
    #angle; not true angle" (as stated in erfa/starpm.c documentation)
    pm_ra_corrected = pm_ra/numpy.cos(dec)

    if isinstance(ra, numpy.ndarray):
        if len(ra) != len(dec) or \
        len(ra) != len(pm_ra) or \
        len(ra) != len(pm_dec) or \
        len(ra) != len(parallaxArcsec) or \
        len(ra) != len(v_rad):

            raise RuntimeError("You passed: " +
                               "%d RAs, " % len(ra) +
                               "%d Dec, " % len(dec) +
                               "%d pm_ras, " % len(pm_ra) +
                               "%d pm_decs, " % len(pm_dec) +
                               "%d parallaxes, " % len(parallaxArcsec)+
                               "%d v_rads " % len(v_rad) +
                               "to applyPm; those numbers need to be identical.")

        raOut, decOut = palpy.pmVector(ra,dec,pm_ra_corrected,pm_dec,parallaxArcsec,v_rad, epoch, julianEpoch)
    else:
        raOut, decOut = palpy.pm(ra, dec, pm_ra_corrected, pm_dec, parallaxArcsec, v_rad, epoch, julianEpoch)

    return numpy.array([raOut,decOut])



def appGeoFromICRS(ra, dec, pm_ra=None, pm_dec=None, parallax=None,
                   v_rad=None, epoch=2000.0, mjd = None):
    """
    Convert the mean position (RA, Dec) in the International Celestial Reference
    System (ICRS) to the mean apparent geocentric position

    units:  ra (degrees), dec (degrees), pm_ra (arcsec/year), pm_dec
    (arcsec/year), parallax (arcsec), v_rad (km/sec; positive if receding),
    epoch (Julian years)

    @param [in] ra in degrees (ICRS).  Must be a numpy array.

    @param [in] dec in degrees (ICRS).  Must be a numpy array.

    @param [in] pm_ra is ra proper motion multiplied by cos(Dec) in arcsec/year

    @param [in] pm_dec is dec proper motion in arcsec/year

    @param [in] parallax in arcsec

    @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)

    @param [in] epoch is the julian epoch (in years) of the equinox against which to
    measure RA (default: 2000.0)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the apparent
    geocentric RA and the second row is the apparent geocentric Dec (both in degrees)
    """

    if pm_ra is not None:
        pm_ra_in = radiansFromArcsec(pm_ra)
    else:
        pm_ra_in = None

    if pm_dec is not None:
        pm_dec_in = radiansFromArcsec(pm_dec)
    else:
        pm_dec_in = None

    if parallax is not None:
        px_in = radiansFromArcsec(parallax)
    else:
        px_in = None

    output = _appGeoFromICRS(numpy.radians(ra), numpy.radians(dec),
                             pm_ra=pm_ra_in, pm_dec=pm_dec_in,
                             parallax=px_in, v_rad=v_rad, epoch=epoch, mjd=mjd)


    return numpy.degrees(output)


def _appGeoFromICRS(ra, dec, pm_ra=None, pm_dec=None, parallax=None,
                   v_rad=None, epoch=2000.0, mjd = None):
    """
    Convert the mean position (RA, Dec) in the International Celestial Reference
    System (ICRS) to the mean apparent geocentric position

    units:  ra (radians), dec (radians), pm_ra (radians/year), pm_dec
    (radians/year), parallax (radians), v_rad (km/sec; positive if receding),
    epoch (Julian years)

    @param [in] ra in radians (ICRS).  Must be a numpy array.

    @param [in] dec in radians (ICRS).  Must be a numpy array.

    @param [in] pm_ra is ra proper motion multiplied by cos(Dec) in radians/year

    @param [in] pm_dec is dec proper motion in radians/year

    @param [in] parallax in radians

    @param [in] v_rad is radial velocity in km/sec (positive if the object is receding)

    @param [in] epoch is the julian epoch (in years) of the equinox against which to
    measure RA (default: 2000.0)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the apparent
    geocentric RAand the second row is the apparent geocentric Dec (both in radians)
    """

    if mjd is None:
        raise RuntimeError("cannot call appGeoFromICRS; mjd is None")

    if len(ra) != len(dec):
        raise RuntimeError('appGeoFromICRS: len(ra) %d len(dec) %d '
                        % (len(ra),len(dec)))

    if pm_ra is None:
        pm_ra=numpy.zeros(len(ra))

    if pm_dec is None:
        pm_dec=numpy.zeros(len(ra))

    if v_rad is None:
        v_rad=numpy.zeros(len(ra))

    if parallax is None:
        parallax=numpy.zeros(len(ra))

    # Define star independent mean to apparent place parameters
    # palpy.mappa calculates the star-independent parameters
    # needed to correct RA and Dec
    # e.g the Earth barycentric and heliocentric position and velocity,
    # the precession-nutation matrix, etc.
    #
    # arguments of palpy.mappa are:
    # epoch of mean equinox to be used (Julian)
    #
    # date (MJD)
    prms=palpy.mappa(epoch, mjd.TDB)

    # palpy.mapqk does a quick mean to apparent place calculation using
    # the output of palpy.mappa
    #
    # Taken from the palpy source code (palMap.c which calls both palMappa and palMapqk):
    # The accuracy is sub-milliarcsecond, limited by the
    # precession-nutation model (see palPrenut for details).

    # because PAL and ERFA expect proper motion in terms of "coordinate
    # angle; not true angle" (as stated in erfa/starpm.c documentation)
    pm_ra_corrected = pm_ra/numpy.cos(dec)

    raOut,decOut = palpy.mapqkVector(ra,dec,pm_ra_corrected,pm_dec,arcsecFromRadians(parallax),v_rad,prms)

    return numpy.array([raOut,decOut])


def _icrsFromAppGeo(ra, dec, epoch=2000.0, mjd = None):
    """
    Convert the apparent geocentric position in (RA, Dec) to
    the mean position in the International Celestial Reference
    System (ICRS)

    This method undoes the effects of precession, annual aberration,
    and nutation.  It is meant for mapping pointing RA and Dec (which
    presumably include the above effects) back to mean ICRS RA and Dec
    so that the user knows how to query a database of mean RA and Decs
    for objects observed at a given telescope pointing.

    This method works in radians.

    @param [in] ra in radians (apparent geocentric).  Must be a numpy array.

    @param [in] dec in radians (apparent geocentric).  Must be a numpy array.

    @param [in] epoch is the julian epoch (in years) of the equinox against which to
    measure RA (default: 2000.0)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the mean ICRS RA and
    the second row is the mean ICRS Dec (both in radians)
    """

    # Define star independent mean to apparent place parameters
    # palpy.mappa calculates the star-independent parameters
    # needed to correct RA and Dec
    # e.g the Earth barycentric and heliocentric position and velocity,
    # the precession-nutation matrix, etc.
    #
    # arguments of palpy.mappa are:
    # epoch of mean equinox to be used (Julian)
    #
    # date (MJD)
    params = palpy.mappa(epoch, mjd.TDB)

    raOut, decOut = palpy.ampqkVector(ra, dec, params)

    return numpy.array([raOut, decOut])


def icrsFromAppGeo(ra, dec, epoch=2000.0, mjd = None):
    """
    Convert the apparent geocentric position in (RA, Dec) to
    the mean position in the International Celestial Reference
    System (ICRS)

    This method undoes the effects of precession, annual aberration,
    and nutation.  It is meant for mapping pointing RA and Dec (which
    presumably include the above effects) back to mean ICRS RA and Dec
    so that the user knows how to query a database of mean RA and Decs
    for objects observed at a given telescope pointing.

    This method works in degrees.

    @param [in] ra in degrees (apparent geocentric).  Must be a numpy array.

    @param [in] dec in degrees (apparent geocentric).  Must be a numpy array.

    @param [in] epoch is the julian epoch (in years) of the equinox against which to
    measure RA (default: 2000.0)

    @param [in] mjd is an instantiation of the ModifiedJulianDate class
    representing the date of the observation

    @param [out] a 2-D numpy array in which the first row is the mean ICRS RA and
    the second row is the mean ICRS Dec (both in degrees)
    """

    raOut, decOut = _icrsFromAppGeo(numpy.radians(ra), numpy.radians(dec),
                                    epoch=epoch, mjd=mjd)

    return numpy.array([numpy.degrees(raOut), numpy.degrees(decOut)])


def observedFromAppGeo(ra, dec, includeRefraction = True,
                       altAzHr=False, wavelength=0.5, obs_metadata = None):
    """
    Convert apparent geocentric (RA, Dec) to observed (RA, Dec).  More
    specifically: apply refraction and diurnal aberration.

    This method works in degrees.

    @param [in] ra is geocentric apparent RA (degrees).  Must be a numpy array.

    @param [in] dec is geocentric apparent Dec (degrees).  Must be a numpy array.

    @param [in] includeRefraction is a boolean to turn refraction on and off

    @param [in] altAzHr is a boolean indicating whether or not to return altitude
    and azimuth

    @param [in] wavelength is effective wavelength in microns (default: 0.5)

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    observation.

    @param [out] a 2-D numpy array in which the first row is the observed RA
    and the second row is the observed Dec (both in degrees)

    @param [out] a 2-D numpy array in which the first row is the altitude
    and the second row is the azimuth (both in degrees).  Only returned
    if altAzHr == True.
    """

    if altAzHr:
        raDec, \
        altAz = _observedFromAppGeo(numpy.radians(ra), numpy.radians(dec),
                                            includeRefraction=includeRefraction,
                                            altAzHr=altAzHr, wavelength=wavelength,
                                            obs_metadata=obs_metadata)

        return numpy.degrees(raDec), numpy.degrees(altAz)

    else:
        output = _observedFromAppGeo(numpy.radians(ra), numpy.radians(dec),
                                            includeRefraction=includeRefraction,
                                            altAzHr=altAzHr, wavelength=wavelength,
                                            obs_metadata=obs_metadata)

        return numpy.degrees(output)


def _calculateObservatoryParameters(obs_metadata, wavelength, includeRefraction):
    """
    Computer observatory-based parameters using palpy.aoppa

    @param [in] obs_metadata is an ObservationMetaData characterizing
    the specific telescope site and pointing

    @param [in] wavelength is the effective wavelength in microns

    @param [in] includeRefraction is a boolean indicating whether or not
    to include the effects of refraction

    @param [out] the numpy array of observatory paramters calculated by
    palpy.aoppa
    """

    # Correct site longitude for polar motion slaPolmo
    #
    #17 October 2014
    #  palAop.c (which calls Aoppa and Aopqk, as we do here) says
    #  *     - The azimuths etc produced by the present routine are with
    #  *       respect to the celestial pole.  Corrections to the terrestrial
    #  *       pole can be computed using palPolmo.
    #
    #currently, palPolmo is not implemented in PAL
    #I have filed an issue with the PAL team to change that.

    #
    #palpy.aoppa computes star-independent parameters necessary for
    #converting apparent place into observed place
    #i.e. it calculates geodetic latitude, magnitude of diurnal aberration,
    #refraction coefficients and the like based on data about the observation site
    #
    #TODO: palpy.aoppa requires as its first argument
    #the UTC time expressed as an MJD.  It is not clear to me
    #how to actually calculate that.
    if (includeRefraction == True):
        obsPrms=palpy.aoppa(obs_metadata.mjd.UTC, obs_metadata.mjd.dut,
                          obs_metadata.site.longitude,
                          obs_metadata.site.latitude,
                          obs_metadata.site.height,
                          obs_metadata.site.xPolar,
                          obs_metadata.site.yPolar,
                          obs_metadata.site.meanTemperature,
                          obs_metadata.site.meanPressure,
                          obs_metadata.site.meanHumidity,
                          wavelength ,
                          obs_metadata.site.lapseRate)
    else:
        #we can discard refraction by setting pressure and humidity to zero
        obsPrms=palpy.aoppa(obs_metadata.mjd.UTC, obs_metadata.mjd.dut,
                          obs_metadata.site.longitude,
                          obs_metadata.site.latitude,
                          obs_metadata.site.height,
                          obs_metadata.site.xPolar,
                          obs_metadata.site.yPolar,
                          obs_metadata.site.meanTemperature,
                          0.0,
                          0.0,
                          wavelength ,
                          obs_metadata.site.lapseRate)


    return obsPrms


def _observedFromAppGeo(ra, dec, includeRefraction = True,
                       altAzHr=False, wavelength=0.5, obs_metadata = None):
    """
    Convert apparent geocentric (RA, Dec) to observed (RA, Dec).  More specifically:
    apply refraction and diurnal aberration.

    This method works in radians.

    @param [in] ra is geocentric apparent RA (radians).  Must be a numpy array.

    @param [in] dec is geocentric apparent Dec (radians).  Must be a numpy array.

    @param [in] includeRefraction is a boolean to turn refraction on and off

    @param [in] altAzHr is a boolean indicating whether or not to return altitude
    and azimuth

    @param [in] wavelength is effective wavelength in microns (default: 0.5)

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    observation.

    @param [out] a 2-D numpy array in which the first row is the observed RA
    and the second row is the observed Dec (both in radians)

    @param [out] a 2-D numpy array in which the first row is the altitude
    and the second row is the azimuth (both in radians).  Only returned
    if altAzHr == True.

    """

    if obs_metadata is None:
        raise RuntimeError("Cannot call observedFromAppGeo without an obs_metadata")

    if obs_metadata.site is None:
        raise RuntimeError("Cannot call observedFromAppGeo: obs_metadata has no site info")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call observedFromAppGeo: obs_metadata has no mjd")

    if len(ra)!=len(dec):
        raise RuntimeError("You passed %d RAs but %d Decs to observedFromAppGeo" % \
                           (len(ra), len(dec)))


    obsPrms = _calculateObservatoryParameters(obs_metadata, wavelength, includeRefraction)

    #palpy.aopqk does an apparent to observed place
    #correction
    #
    #it corrects for diurnal aberration and refraction
    #(using a fast algorithm for refraction in the case of
    #a small zenith distance and a more rigorous algorithm
    #for a large zenith distance)
    #

    azimuth, zenith, hourAngle, decOut, raOut = palpy.aopqkVector(ra,dec,obsPrms)

    #
    #Note: this is a choke point.  Even the vectorized version of aopqk
    #is expensive (it takes about 0.006 seconds per call)
    #
    #Actually, this is only a choke point if you are dealing with zenith
    #distances of greater than about 70 degrees

    if altAzHr == True:
        #
        #palpy.de2h converts equatorial to horizon coordinates
        #
        az, alt = palpy.de2hVector(hourAngle,decOut,obs_metadata.site.latitude)
        return numpy.array([raOut, decOut]), numpy.array([alt, az])
    return numpy.array([raOut, decOut])


def appGeoFromObserved(ra, dec, includeRefraction = True,
                        wavelength=0.5, obs_metadata = None):
    """
    Convert observed (RA, Dec) to apparent geocentric (RA, Dec).  More
    specifically: undo the effects of refraction and diurnal aberration.

    Note: This method is only accurate at zenith distances less than ~75 degrees.

    This method works in degrees.

    @param [in] ra is observed RA (degrees).  Must be a numpy array.

    @param [in] dec is observed Dec (degrees).  Must be a numpy array.

    @param [in] includeRefraction is a boolean to turn refraction on and off

    @param [in] wavelength is effective wavelength in microns (default: 0.5)

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    observation.

    @param [out] a 2-D numpy array in which the first row is the apparent
    geocentric RA and the second row is the apparentGeocentric Dec (both
    in degrees)
    """

    raOut, decOut = _appGeoFromObserved(numpy.radians(ra), numpy.radians(dec),
                                        includeRefraction=includeRefraction,
                                        wavelength=wavelength,
                                        obs_metadata=obs_metadata)

    return numpy.array([numpy.degrees(raOut), numpy.degrees(decOut)])


def _appGeoFromObserved(ra, dec, includeRefraction = True,
                        wavelength=0.5, obs_metadata = None):
    """
    Convert observed (RA, Dec) to apparent geocentric (RA, Dec).
    More specifically: undo the effects of refraction and diurnal aberration.

    Note: This method is only accurate at zenith distances less than ~ 75 degrees.

    This method works in radians.

    @param [in] ra is observed RA (radians).  Must be a numpy array.

    @param [in] dec is observed Dec (radians).  Must be a numpy array.

    @param [in] includeRefraction is a boolean to turn refraction on and off

    @param [in] wavelength is effective wavelength in microns (default: 0.5)

    @param [in] obs_metadata is an ObservationMetaData characterizing the
    observation.

    @param [out] a 2-D numpy array in which the first row is the apparent
    geocentric RA and the second row is the apparentGeocentric Dec (both
    in radians)
    """

    if obs_metadata is None:
        raise RuntimeError("Cannot call appGeoFromObserved without an obs_metadata")

    if obs_metadata.site is None:
        raise RuntimeError("Cannot call appGeoFromObserved: obs_metadata has no site info")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call appGeoFromObserved: obs_metadata has no mjd")

    if len(ra)!=len(dec):
        raise RuntimeError("You passed %d RAs but %d Decs to appGeoFromObserved" % \
                           (len(ra), len(dec)))

    obsPrms = _calculateObservatoryParameters(obs_metadata, wavelength, includeRefraction)

    raOut, decOut = palpy.oapqkVector('r', ra, dec, obsPrms)
    return numpy.array([raOut, decOut])


def observedFromICRS(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                     obs_metadata=None, epoch=None, includeRefraction=True):
    """
    Convert mean position (RA, Dec) in the International Celestial Reference Frame
    to observed (RA, Dec).

    included are precession-nutation, aberration, proper motion, parallax, refraction,
    radial velocity, diurnal aberration.

    This method works in degrees.

    @param [in] ra is the unrefracted RA in degrees (ICRS).  Must be a numpy array.

    @param [in] dec is the unrefracted Dec in degrees (ICRS).  Must be a numpy array.

    @param [in] pm_ra is proper motion in RA multiplied by cos(Dec) (arcsec/yr)

    @param [in] pm_dec is proper motion in dec (arcsec/yr)

    @param [in] parallax is parallax in arcsec

    @param [in] v_rad is radial velocity (km/s)

    @param [in] obs_metadata is an ObservationMetaData object describing the
    telescope pointing.

    @param [in] epoch is the julian epoch (in years) against which the mean
    equinoxes are measured.

    @param [in] includeRefraction toggles whether or not to correct for refraction

    @param [out] a 2-D numpy array in which the first row is the observed
    RA and the second row is the observed Dec (both in degrees)
    """

    if pm_ra is not None:
        pm_ra_in = radiansFromArcsec(pm_ra)
    else:
        pm_ra_in = None

    if pm_dec is not None:
        pm_dec_in = radiansFromArcsec(pm_dec)
    else:
        pm_dec_in = None

    if parallax is not None:
        parallax_in = radiansFromArcsec(parallax)
    else:
        parallax_in = None

    output = _observedFromICRS(numpy.radians(ra), numpy.radians(dec),
                               pm_ra=pm_ra_in, pm_dec=pm_dec_in, parallax=parallax_in,
                               v_rad=v_rad, obs_metadata=obs_metadata, epoch=epoch,
                               includeRefraction=includeRefraction)

    return numpy.degrees(output)



def _observedFromICRS(ra, dec, pm_ra=None, pm_dec=None, parallax=None, v_rad=None,
                     obs_metadata=None, epoch=None, includeRefraction=True):
    """
    Convert mean position (RA, Dec) in the International Celestial Reference Frame
    to observed (RA, Dec)-like coordinates.

    included are precession-nutation, aberration, proper motion, parallax, refraction,
    radial velocity, diurnal aberration.

    This method works in radians.

    @param [in] ra is the unrefracted RA in radians (ICRS).  Must be a numpy array.

    @param [in] dec is the unrefracted Dec in radians (ICRS).  Must be a numpy array.

    @param [in] pm_ra is proper motion in RA multiplied by cos(Dec) (radians/yr)

    @param [in] pm_dec is proper motion in dec (radians/yr)

    @param [in] parallax is parallax in radians

    @param [in] v_rad is radial velocity (km/s)

    @param [in] obs_metadata is an ObservationMetaData object describing the
    telescope pointing.

    @param [in] epoch is the julian epoch (in years) against which the mean
    equinoxes are measured.

    @param [in] includeRefraction toggles whether or not to correct for refraction

    @param [out] a 2-D numpy array in which the first row is the observed
    RA and the second row is the observed Dec (both in radians)

    """

    if obs_metadata is None:
        raise RuntimeError("cannot call observedFromICRS; obs_metadata is none")

    if obs_metadata.mjd is None:
        raise RuntimeError("cannot call observedFromICRS; obs_metadata.mjd is none")

    if epoch is None:
        raise RuntimeError("cannot call observedFromICRS; you have not specified an epoch")

    if len(ra)!=len(dec):
        raise RuntimeError("You passed %d RAs but %d Decs to observedFromICRS" % \
                           (len(ra), len(dec)))

    ra_apparent, dec_apparent = _appGeoFromICRS(ra, dec, pm_ra = pm_ra,
             pm_dec = pm_dec, parallax = parallax, v_rad = v_rad, epoch = epoch, mjd=obs_metadata.mjd)

    ra_out, dec_out = _observedFromAppGeo(ra_apparent, dec_apparent, obs_metadata=obs_metadata,
                                               includeRefraction = includeRefraction)

    return numpy.array([ra_out,dec_out])


def icrsFromObserved(ra, dec, obs_metadata=None, epoch=None, includeRefraction=True):
    """
    Convert observed RA, Dec into mean International Celestial Reference Frame (ICRS)
    RA, Dec.  This method undoes the effects of precession, nutation, aberration (annual
    and diurnal), and refraction.  It is meant so that users can take pointing RA and Decs,
    which will be in the observed reference system, and transform them into ICRS for
    purposes of querying database tables (likely to contain mean ICRS RA, Dec) for objects
    visible from a given pointing.

    Note: This method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.

    This method works in degrees.

    @param [in] ra is the observed RA in degrees.  Must be a numpy array.

    @param [in] dec is the observed Dec in degrees.  Must be a numpy array.

    @param [in] obs_metadata is an ObservationMetaData object describing the
    telescope pointing.

    @param [in] epoch is the julian epoch (in years) against which the mean
    equinoxes are measured.

    @param [in] includeRefraction toggles whether or not to correct for refraction

    @param [out] a 2-D numpy array in which the first row is the mean ICRS
    RA and the second row is the mean ICRS Dec (both in degrees)
    """

    ra_out, dec_out = _icrsFromObserved(numpy.radians(ra), numpy.radians(dec),
                                        obs_metadata=obs_metadata,
                                        epoch=epoch, includeRefraction=includeRefraction)

    return numpy.array([numpy.degrees(ra_out), numpy.degrees(dec_out)])


def _icrsFromObserved(ra, dec, obs_metadata=None, epoch=None, includeRefraction=True):
    """
    Convert observed RA, Dec into mean International Celestial Reference Frame (ICRS)
    RA, Dec.  This method undoes the effects of precession, nutation, aberration (annual
    and diurnal), and refraction.  It is meant so that users can take pointing RA and Decs,
    which will be in the observed reference system, and transform them into ICRS for
    purposes of querying database tables (likely to contain mean ICRS RA, Dec) for objects
    visible from a given pointing.

    Note: This method is only accurate at angular distances from the sun of greater
    than 45 degrees and zenith distances of less than 75 degrees.

    This method works in radians.

    @param [in] ra is the observed RA in radians.  Must be a numpy array.

    @param [in] dec is the observed Dec in radians.  Must be a numpy array.

    @param [in] obs_metadata is an ObservationMetaData object describing the
    telescope pointing.

    @param [in] epoch is the julian epoch (in years) against which the mean
    equinoxes are measured.

    @param [in] includeRefraction toggles whether or not to correct for refraction

    @param [out] a 2-D numpy array in which the first row is the mean ICRS
    RA and the second row is the mean ICRS Dec (both in radians)
    """

    if obs_metadata is None:
        raise RuntimeError("cannot call icrsFromObserved; obs_metadata is None")

    if obs_metadata.mjd is None:
        raise RuntimeError("cannot call icrsFromObserved; obs_metadata.mjd is None")

    if epoch is None:
        raise RuntimeError("cannot call icrsFromObserved; you have not specified an epoch")

    if len(ra)!=len(dec):
        raise RuntimeError("You passed %d RAs but %d Decs to icrsFromObserved" % \
                           (len(ra), len(dec)))


    ra_app, dec_app = _appGeoFromObserved(ra, dec, obs_metadata=obs_metadata,
                                          includeRefraction=includeRefraction)

    ra_icrs, dec_icrs = _icrsFromAppGeo(ra_app, dec_app, epoch=epoch,
                                        mjd=obs_metadata.mjd)

    return numpy.array([ra_icrs, dec_icrs])


def raDecFromPupilCoords(xPupil, yPupil, obs_metadata=None, epoch=None):
    """
    @param [in] xPupil -- pupil coordinates in radians

    @param [in] yPupil -- pupil coordinates in radians

    @param [in] obs_metadata -- an instantiation of ObservationMetaData characterizing
    the state of the telescope

    @param [in] epoch -- julian epoch of the mean equinox used for the coordinate
    transforations (in years)

    @param [out] a 2-D numpy array in which the first row is RA and the second
    row is Dec (both in degrees)
    """

    output = _raDecFromPupilCoords(xPupil, yPupil, obs_metadata=obs_metadata,
                                               epoch=epoch)

    return numpy.degrees(output)


def _raDecFromPupilCoords(xPupil, yPupil, obs_metadata=None, epoch=None):
    """
    @param [in] xPupil -- pupil coordinates in radians

    @param [in] yPupil -- pupil coordinates in radians

    @param [in] obs_metadata -- an instantiation of ObservationMetaData characterizing
    the state of the telescope

    @param [in] epoch -- julian epoch of the mean equinox used for the coordinate
    transforations (in years)

    @param [out] a 2-D numpy array in which the first row is RA and the second
    row is Dec (both in radians)
    """

    if obs_metadata is None:
        raise RuntimeError("Cannot call raDecFromPupilCoords without obs_metadata")

    if epoch is None:
        raise RuntimeError("Cannot call raDecFromPupilCoords; epoch is None")

    if obs_metadata.rotSkyPos is None:
        raise RuntimeError("Cannot call raDecFromPupilCoords without rotSkyPos " + \
                           "in obs_metadata")

    if obs_metadata.pointingRA is None or obs_metadata.pointingDec is None:
        raise RuntimeError("Cannot call raDecFromPupilCoords "+ \
                          "without pointingRA, pointingDec in obs_metadata")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot calculate x_pupil, y_pupil without mjd " + \
                           "in obs_metadata")

    if len(xPupil)!=len(yPupil):
        raise RuntimeError("You passed %d RAs but %d Decs into raDecFromPupilCoords" % \
                           (len(raObj), len(decObj)))


    #This is the same as theta in pupilCoordsFromRaDec, except without the minus sign.
    #This is because we will be reversing the rotation performed in that other method.
    theta = -1.0*obs_metadata._rotSkyPos

    x_g = xPupil*numpy.cos(theta) - yPupil*numpy.sin(theta)
    y_g = xPupil*numpy.sin(theta) + yPupil*numpy.cos(theta)

    x_g *= -1.0

    # x_g and y_g are now the x and y coordinates
    # can now use the PALPY method palDtp2s to convert to RA, Dec.
    # Unfortunately, that method has not yet been vectorized.
    raOut = []
    decOut = []
    for xx, yy in zip(x_g, y_g):
        rr, dd = palpy.dtp2s(xx, yy, obs_metadata._pointingRA, obs_metadata._pointingDec)
        raOut.append(rr)
        decOut.append(dd)

    return numpy.array([raOut, decOut])



def pupilCoordsFromRaDec(ra_in, dec_in, obs_metadata=None, epoch=None):
    """
    Take an input RA and dec from the sky and convert it to coordinates
    on the focal plane.

    This uses PAL's gnonomonic projection routine which assumes that the focal
    plane is perfectly flat.  The output is in Cartesian coordinates, assuming
    that the Celestial Sphere is a unit sphere.

    @param [in] ra_in is a numpy array of RAs in degrees

    @param [in] dec_in in degrees

    @param [in] obs_metadata is an ObservationMetaData instantiation characterizing the
    telescope location and pointing.

    @param [in] epoch is the epoch of mean ra and dec in julian years (optional; if not
    provided, this method will try to get it from the db_obj member variable, assuming this
    method is part of an InstanceCatalog)

    @param [out] returns a numpy array whose first row is the x coordinate on the pupil in
    radians and whose second row is the y coordinate in radians
    """

    return _pupilCoordsFromRaDec(numpy.radians(ra_in), numpy.radians(dec_in),
                                      obs_metadata=obs_metadata, epoch=epoch)


def _pupilCoordsFromRaDec(ra_in, dec_in, obs_metadata=None, epoch=None):
    """
    Take an input RA and dec from the sky and convert it to coordinates
    on the focal plane.

    This uses PAL's gnonomonic projection routine which assumes that the focal
    plane is perfectly flat.  The output is in Cartesian coordinates, assuming
    that the Celestial Sphere is a unit sphere.

    @param [in] ra_in is a numpy array of RAs in radians

    @param [in] dec_in in radians

    @param [in] obs_metadata is an ObservationMetaData instantiation characterizing the
    telescope location and pointing.

    @param [in] epoch is the epoch of mean ra and dec in julian years (optional; if not
    provided, this method will try to get it from the db_obj member variable, assuming this
    method is part of an InstanceCatalog)

    @param [out] returns a numpy array whose first row is the x coordinate on the pupil in
    radians and whose second row is the y coordinate in radians
    """

    if obs_metadata is None:
        raise RuntimeError("Cannot call pupilCoordsFromRaDec without obs_metadata")

    if obs_metadata.mjd is None:
        raise RuntimeError("Cannot call pupilCoordsFromRaDec; obs_metadata.mjd is None")

    if epoch is None:
        raise RuntimeError("Cannot call pupilCoordsFromRaDec; epoch is None")

    if len(ra_in)!=len(dec_in):
        raise RuntimeError("You passed %d RAs but %d Decs to pupilCoordsFromRaDec" % (len(ra_in), len(dec_in)))

    if obs_metadata.rotSkyPos is None:
        #there is no observation meta data on which to base astrometry
        raise RuntimeError("Cannot calculate [x,y]_focal_nominal without obs_metadata.rotSkyPos")

    if obs_metadata.pointingRA is None or obs_metadata.pointingDec is None:
        raise RuntimeError("Cannot calculate [x,y]_focal_nominal without pointingRA and Dec in obs_metadata")

    theta = obs_metadata._rotSkyPos

    #palpy.ds2tp performs the gnomonic projection on ra_in and dec_in
    #with a tangent point at (pointingRA, pointingDec)
    #
    try:
        x, y = palpy.ds2tpVector(ra_in, dec_in, obs_metadata._pointingRA, obs_metadata._pointingDec)
    except:
        # apparently, one of your ra/dec values was improper; we will have to do this
        # element-wise, putting NaN in the place of the bad values
        x = []
        y = []
        for rr, dd in zip(ra_in, dec_in):
            try:
                xx, yy = palpy.ds2tp(rr, dd, obs_metadata._pointingRA, obs_metadata._pointingDec)
            except:
                xx = numpy.NaN
                yy = numpy.NaN
            x.append(xx)
            y.append(yy)
        x = numpy.array(x)
        y = numpy.array(y)

    # The extra negative sign on x_out comes from the following:
    # The Gnomonic projection as calculated by palpy is such that,
    # if north is in the +y direction, then west is in the -x direction,
    # which is the opposite of the behavior we want (I do not know how to
    # express this analytically; I have just confirmed it numerically)
    x *= -1.0

    #rotate the result by rotskypos (rotskypos being "the angle of the sky relative to
    #camera cooridnates" according to phoSim documentation) to account for
    #the rotation of the focal plane about the telescope pointing


    x_out = x*numpy.cos(theta) - y*numpy.sin(theta)
    y_out = x*numpy.sin(theta) + y*numpy.cos(theta)

    return numpy.array([x_out, y_out])
