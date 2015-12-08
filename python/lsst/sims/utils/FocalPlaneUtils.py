import numpy as np
import palpy


__all__ = ["_pupilCoordsFromRaDec", "pupilCoordsFromRaDec",
           "_raDecFromPupilCoords", "raDecFromPupilCoords"]


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

    return _pupilCoordsFromRaDec(np.radians(ra_in), np.radians(dec_in),
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
                xx = np.NaN
                yy = np.NaN
            x.append(xx)
            y.append(yy)
        x = np.array(x)
        y = np.array(y)

    # The extra negative sign on x_out comes from the following:
    # The Gnomonic projection as calculated by palpy is such that,
    # if north is in the +y direction, then west is in the -x direction,
    # which is the opposite of the behavior we want (I do not know how to
    # express this analytically; I have just confirmed it numerically)
    x *= -1.0

    #rotate the result by rotskypos (rotskypos being "the angle of the sky relative to
    #camera cooridnates" according to phoSim documentation) to account for
    #the rotation of the focal plane about the telescope pointing


    x_out = x*np.cos(theta) - y*np.sin(theta)
    y_out = x*np.sin(theta) + y*np.cos(theta)

    return np.array([x_out, y_out])


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

    return np.degrees(output)


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

    x_g = xPupil*np.cos(theta) - yPupil*np.sin(theta)
    y_g = xPupil*np.sin(theta) + yPupil*np.cos(theta)

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

    return np.array([raOut, decOut])
