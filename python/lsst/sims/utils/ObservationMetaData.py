import numpy
import inspect
from .SpatialBounds import SpatialBounds
from lsst.sims.utils import haversine, Site, _icrsFromObserved, _observedFromICRS
from lsst.sims.utils import _raDecFromNativeLonLat
from lsst.sims.utils import ModifiedJulianDate

__all__ = ["ObservationMetaData"]

class ObservationMetaData(object):
    """Observation Metadata

    This class contains any metadata for a query which is associated with
    a particular telescope pointing, including bounds in RA and DEC, and
    the time of the observation.

    **Parameters**

        * pointing[RA,Dec] float
          The coordinates of the pointing (in degrees)

        * boundType characterizes the shape of the field of view.  Current options
          are 'box, and 'circle'

        * boundLength is the characteristic length scale of the field of view in degrees.
          If boundType is 'box', boundLength can be a float(in which case boundLength is
          half the length of the side of each box) or boundLength can be a numpy array
          in which case the first argument is
          half the width of the RA side of the box and the second argument is half the
          Dec side of the box.
          If boundType is 'circle,' this will be the radius of the circle.
          The bound will be centered on the point (pointingRA, pointingDec),
          however, because objects are stored at their mean RA, Dec in the LSST databases
          (i.e. they are stored at values of RA, Dec which neglect precession, nutation,
          aberration, and refraction), the bounds applied to database queries will be made
          slightly larger so that queries can be reasonably expected to return all of the
          objects within the desired field of view once those corrections have been applied.

        * mjd : (optional)
          Either a float (in which case, it will be assumed to be in International
          Atomic Time), or an instnatiation of the ModifiedJulianDate class representing
          the date of the observation

        * bandpassName : a char (e.g. 'u') or list (e.g. ['u', 'g', 'z'])
          denoting the bandpasses used for this particular observation

        * phoSimMetaData : dict (optional)
          a dictionary containing metadata used by PhoSim

        * m5: float or list (optional)
          this should be the 5-sigma limiting magnitude in the bandpass or
          bandpasses specified in bandpassName.  Ultimately, m5 will be stored
          in a dict keyed to the bandpassName (or Names) you passed in, i.e.
          you will be able to access m5 from outside of this class using, for
          example:

          myObservationMetaData.m5['u']

        * skyBrightness: float (optional) the magnitude of the sky in the
          filter specified by bandpassName

        * seeing float or list (optional)
          Analogous to m5, corresponds to the seeing in arcseconds in the bandpasses in
          bandpassName

        * epoch (optional) is the epoch used for converting from pointingRA, Dec to
          meanRA, Dec when constructing query bounds on a database.  This defaults
          to 2000.0 and should only be changed if you plan to use this
          ObservationMetaData to query a database with meanRA, Dec stored in a
          system that is not measured against the equinox at Julian epoch 2000.0

        * rotSkyPos float (optional)
          The orientation of the telescope in degrees.
          This is used by the Astrometry mixins in sims_coordUtils.

        The convention for rotSkyPos is as follows:

        rotSkyPos = 0 means north is in the +y direction on the camera and east is -x

        rotSkyPos = 90 means north is -x and east is -y

        rotSkyPos = -90 means north is +x and east is +y

        rotSkyPos = 180 means north is -y and east is +x

        This should be consistent with PhoSim conventions.

    **Examples**::
        >>> data = ObservationMetaData(boundType='box', pointingRA=5.0, pointingDec=15.0,
                    boundLength=5.0)

    """
    def __init__(self, boundType=None, boundLength=None,
                 mjd=None, pointingRA=None, pointingDec=None, rotSkyPos=None,
                 bandpassName=None, phoSimMetaData=None, site=Site(), m5=None, skyBrightness=None,
                 seeing=None, epoch=2000.0):

        self._bounds = None
        self._boundType = boundType
        self._bandpass = bandpassName
        self._skyBrightness = skyBrightness
        self._site = site
        self._epoch = epoch

        if mjd is not None:
            if isinstance(mjd, float) or isinstance(mjd, int):
                self._mjd = ModifiedJulianDate(TAI=mjd)
            elif isinstance(mjd, ModifiedJulianDate):
                self._mjd = mjd
            else:
                raise RuntimeError("You must pass either a float or a ModifiedJulianDate "
                                   "as the kwarg mjd to ObservationMetaData")
        else:
            self._mjd = None

        if rotSkyPos is not None:
            self._rotSkyPos = numpy.radians(rotSkyPos)
        else:
            self._rotSkyPos = None

        if pointingRA is not None:
            self._pointingRA = numpy.radians(pointingRA)
        else:
            self._pointingRA = None

        if pointingDec is not None:
            self._pointingDec = numpy.radians(pointingDec)
        else:
            self._pointingDec = None

        if boundLength is not None:
            self._boundLength = numpy.radians(boundLength)
        else:
            self._boundLength = None

        if phoSimMetaData is not None:
            self._assignPhoSimMetaData(phoSimMetaData)
        else:
            self._phoSimMetaData = None

        self._m5 = self._assignDictKeyedToBandpass(m5, 'm5')

        # 11 June 2015
        # I think it is okay to assign seeing after _phoSimMetaData has been
        # assigned.  Technically, the _phoSimMetaData contains `rawseeing`, which
        # is some idealized seeing at zenith at 500nm.  This will be different
        # from seeing.  After instantiation, I don't think users should be
        # allowed to set seeing (on the assumption that seeing and rawseeing are
        # somehow in sync).
        self._seeing = self._assignDictKeyedToBandpass(seeing, 'seeing')

        #this should be done after phoSimMetaData is assigned, just in case
        #self._assignPhoSimMetadata overwrites pointingRA/Dec
        if self._bounds is None:
            self._buildBounds()


    @property
    def summary(self):
        mydict = {}
        mydict['site'] = self.site

        mydict['boundType'] = self.boundType
        mydict['boundLength'] = self.boundLength
        mydict['pointingRA'] = self.pointingRA
        mydict['pointingDec'] = self.pointingDec
        mydict['rotSkyPos'] = self.rotSkyPos

        mydict['mjd'] = self.mjd.TAI
        mydict['bandpass'] = self.bandpass
        mydict['skyBrightness'] = self.skyBrightness
        # mydict['m5'] = self.m5

        mydict['phoSimMetaData'] = self.phoSimMetaData

        return mydict

    def _assignDictKeyedToBandpass(self, inputValue, inputName):
        """
        This method sets up a dict of either m5 or seeing values (or any other quantity
        keyed to bandpassName).  It reads in a list of values and associates them with
        the list of bandpass names in self._bandpass.

        Note: this method assumes that self._bandpass has already been set.
        It will raise an exception of self._bandpass is None.

        @param [in] inputValue is a single value or list of m5/seeing/etc. corresponding to
        the bandpasses stored in self._bandpass

        @param [in] inputName is the name of the paramter stored in inputValue
        (for constructing helpful error message)

        @param [out] returns a dict of inputValue values keed to self._bandpass
        """

        if inputValue is None:
            return None
        else:
            bandpassIsList = False
            inputIsList = False

            if self._bandpass is None:
                raise RuntimeError('You cannot set %s if you have not set ' % inputName +
                                   'bandpass in ObservationMetaData')

            if hasattr(self._bandpass, '__iter__'):
                bandpassIsList = True

            if hasattr(inputValue, '__iter__'):
                inputIsList = True

            if bandpassIsList and not inputIsList:
                raise RuntimeError('You passed a list of bandpass names' + \
                                   'but did not pass a list of %s to ObservationMetaData' % inputName)

            if inputIsList and not bandpassIsList:
                raise RuntimeError('You passed a list of %s ' % inputName + \
                                    'but did not pass a list of bandpass names to ObservationMetaData')


            if inputIsList:
                if len(inputValue) != len(self._bandpass):
                    raise RuntimeError('The list of %s you passed to ObservationMetaData ' % inputName + \
                                       'has a different length than the list of bandpass names you passed')

            #now build the dict
            if bandpassIsList:
                if len(inputValue) != len(self._bandpass):
                    raise RuntimeError('In ObservationMetaData you tried to assign bandpass ' +
                                       'and %s with lists of different length' % inputName)

                outputDict = {}
                for b, m in zip(self._bandpass, inputValue):
                    outputDict[b] = m
            else:
                outputDict = {self._bandpass:inputValue}

            return outputDict


    def _findCircularBoundLength(self):
        """
        Find the boundLength parameter that must be passed to SpatialBounds
        in the case of a circular field of view.

        Because sources are stored in the database at mean ICRS position
        but the ObservationMetaData pointingRA and pointingDec refer to
        observed position (the difference being the application of precession,
        nutation, aberration, and refraction), we must allow for the possibility
        that, in order to truly find all of the objects in the field of view,
        the boundLength passed to SpatialBounds is not strictly the
        boundLength stored in self._boundLength
        """
        raICRS, decICRS = _icrsFromObserved(numpy.array([self._pointingRA]), numpy.array([self._pointingDec]),
                                            obs_metadata=self, epoch=self._epoch)

        # We now need to determine if an offset must be applied to boundLength
        # to account for the transformation between mean and observed RA, Dec.
        # We do this by considering points whose observed position are within
        # the requested boundLength.  We will transform these points into ICRS
        # and compare their distance (in ICRS) from raICRS, decICRS.  The maximum
        # of these distances will be compared to boundLength.  If it is greater than
        # boundLength, this value will be the new boundLength.

        radius = self._boundLength

        theta_list = numpy.arange(0.0, 2.0*numpy.pi+0.01, 0.02*numpy.pi)

        ra_test_list = []
        dec_test_list = []
        for theta in theta_list:
            ra_test, dec_test = _raDecFromNativeLonLat(theta, 0.5*numpy.pi-radius, self._pointingRA, self._pointingDec)
            dd = haversine(ra_test, dec_test, self._pointingRA, self._pointingDec)
            ra_test_list.append(ra_test)
            dec_test_list.append(dec_test)

        ra_test_icrs, dec_test_icrs = _icrsFromObserved(numpy.array(ra_test_list),
                                                        numpy.array(dec_test_list),
                                                        obs_metadata=self, epoch=self._epoch)


        ddmax = haversine(ra_test_icrs, dec_test_icrs, raICRS[0], decICRS[0]).max()
        if ddmax>self._boundLength:
            return ddmax
        else:
            return self._boundLength


    def _findRectangularBoundLength(self):
        """
        Find the boundLength parameter that must be passed to SpatialBounds
        in the case of a rectangular field of view.

        Because sources are stored in the database at mean ICRS position
        but the ObservationMetaData pointingRA and pointingDec refer to
        observed position (the difference being the application of precession,
        nutation, aberration, and refraction), we must allow for the possibility
        that, in order to truly find all of the objects in the field of view,
        the boundLength passed to SpatialBounds is not strictly the
        boundLength stored in self._boundLength
        """
        raICRS, decICRS = _icrsFromObserved(numpy.array([self._pointingRA]), numpy.array([self._pointingDec]),
                                            obs_metadata=self, epoch=self._epoch)

        if hasattr(self._boundLength, "__len__"):
            raBound = self._boundLength[0]
            decBound = self._boundLength[1]
        else:
            raBound = self._boundLength
            decBound = self._boundLength


        # Here we find a set of RA, Dec pairs that fall within the nominally requested
        # field of view.  We transform those points from the Observed RA, Dec to ICRS.
        # We then compare the maximum RA and Dec difference between these test points
        # and the ICRS center of the field of view.  If those maximum difference
        # are greater than the nominal boundLength, we update boundLength accordingly

        dra = 0.1*raBound
        ra_list = numpy.arange(self._pointingRA-raBound, self._pointingRA+raBound+0.1*dra, dra)
        ddec = 0.1*decBound
        dec_list = numpy.arange(self._pointingDec-decBound, self._pointingDec+decBound+0.1*ddec, ddec)

        ra_test_list = []
        dec_test_list = []
        for rr in ra_list:
            for dd in dec_list:
                ra_test_list.append(rr)
                dec_test_list.append(dd)

        ra_test_icrs, dec_test_icrs = _icrsFromObserved(numpy.array(ra_test_list),
                                                        numpy.array(dec_test_list),
                                                        obs_metadata=self, epoch=self._epoch)

        draMax = raBound
        ddecMax = decBound

        dra_test_max = numpy.abs(ra_test_icrs - raICRS[0]).max()
        if dra_test_max>draMax:
            draMax = dra_test_max

        ddec_test_max = numpy.abs(dec_test_icrs - decICRS[0]).max()
        if ddec_test_max>ddecMax:
            ddecMax = ddec_test_max

        return [draMax, ddecMax]


    def _buildBounds(self):
        """
        Set up the member variable self._bounds.

        If self._boundType, self._boundLength, self._pointingRA, or
        self._pointingDec are None, nothing will happen.
        """

        if self._boundType is None:
            return

        if self._boundLength is None:
            return

        if self._pointingRA is None or self._pointingDec is None:
            return

        if self._mjd is None:
            raise RuntimeError("You cannot build spatial bounds with an ObservationMetaData that "
                               "does not have an MJD")

        if self._epoch is None:
            raise RuntimeError("You cannot build spatial bounds with an ObservationMetaData that "
                               "does not have an epoch")

        raICRS, decICRS = _icrsFromObserved(numpy.array([self._pointingRA]), numpy.array([self._pointingDec]),
                                            obs_metadata=self, epoch=self._epoch)

        if self._boundType == 'circle':
            trueBoundLength = self._findCircularBoundLength()
        elif self._boundType == 'box':
            trueBoundLength = self._findRectangularBoundLength()
        else:
            raise RuntimeError("ObservationMetaData does not know how to "
                               "set the BoundLength for boundType %s" % self._boundType)

        self._bounds = SpatialBounds.getSpatialBounds(self._boundType, raICRS[0], decICRS[0],
                                                      trueBoundLength)


    def _assignPhoSimMetaData(self, metaData):
        """
        Assign the dict metaData to be the associated phoSimMetaData dict of this object.

        In doing so, this method will copy pointingRA, pointingDec, rotSkyPos,
        MJD, and bandpass from the metaData (if present) to the corresponding
        member variables.  If by doing so you try to overwrite a parameter that you
        have already set by hand, this method will raise an exception.
        """

        self._phoSimMetaData = metaData

        if self._phoSimMetaData is not None:
            #overwrite member variables with values from the phoSimMetaData
            if 'Opsim_expmjd' in self._phoSimMetaData:
                if self._mjd is not None:
                    raise RuntimeError('WARNING in ObservationMetaData trying to overwrite mjd with phoSimMetaData')

                self._mjd = ModifiedJulianDate(TAI=self._phoSimMetaData['Opsim_expmjd'][0])

            if 'Opsim_rotskypos' in self._phoSimMetaData:
                if self._rotSkyPos is not None:
                    raise RuntimeError('WARNING in ObservationMetaData trying to overwrite rotSkyPos ' +
                                       'with phoSimMetaData')

                self._rotSkyPos = self._phoSimMetaData['Opsim_rotskypos'][0]


            if 'Opsim_filter' in self._phoSimMetaData:
                if self._bandpass is not None:
                    raise RuntimeError('WARNING in ObservationMetaData trying to overwrite bandpass ' +
                                       'with phoSimMetaData')

                self._bandpass = self._phoSimMetaData['Opsim_filter'][0]

            if 'Opsim_rawseeing' in self._phoSimMetaData:
                if hasattr(self, '_seeing') and self._seeing is not None:
                    raise RuntimeError('WARNING in ObservationMetaDAta trying to overwrite seeing ' +
                                       'with phoSimMetaData')

            if 'pointingDec' in self._phoSimMetaData and 'pointingRA' not in self._phoSimMetaData:
                raise RuntimeError("In ObservationMetaData, your phoSimMetaData specifies pointingDec, "
                                   "but not pointingRA")

            if 'pointingRA' in self._phoSimMetaData and 'pointingDec' not in self._phoSimMetaData:
                raise RuntimeError("In ObservationMetaData, your phoSimMetaData specifies pointingRA, "
                                   "but not pointingDec")

            if 'pointingRA' in self._phoSimMetaData and 'pointingDec' in self._phoSimMetaData:
                if self._pointingRA is not None:
                    raise RuntimeError('WARNING in ObservationMetaData trying to overwrite pointingRA ' +
                                       'with phoSimMetaData')

                if self._pointingDec is not None:
                    raise RuntimeError('WARNING in ObservationMetaData trying to overwrite pointingDec ' +
                                       'with phoSimMetaData')

                self._pointingRA = self._phoSimMetaData['pointingRA'][0]

                self._pointingDec = self._phoSimMetaData['pointingDec'][0]

        self._buildBounds()

    @property
    def pointingRA(self):
        """
        The RA of the telescope pointing in degrees.
        """
        if self._pointingRA is not None:
            return numpy.degrees(self._pointingRA)
        else:
            return None

    @pointingRA.setter
    def pointingRA(self, value):
        if self._phoSimMetaData is not None:
            if 'pointingRA' in self._phoSimMetaData:
                raise RuntimeError('WARNING overwriting pointingRA ' +
                                   'which was set by phoSimMetaData')

        self._pointingRA = numpy.radians(value)
        self._buildBounds()

    @property
    def pointingDec(self):
        """
        The Dec of the telescope pointing in degrees.
        """
        if self._pointingDec is not None:
            return numpy.degrees(self._pointingDec)
        else:
            return None

    @pointingDec.setter
    def pointingDec(self, value):
        if self._phoSimMetaData is not None:
            if 'pointingDec' in self._phoSimMetaData:
                raise RuntimeError('WARNING overwriting pointingDec ' +
                                   'which was set by phoSimMetaData')

        self._pointingDec = numpy.radians(value)
        self._buildBounds()

    @property
    def boundLength(self):
        """
        Either a list or a float indicating the size of the field
        of view associated with this ObservationMetaData.

        See the documentation in the SpatialBounds class for more
        details (specifically, the 'length' paramter).

        In degrees (Yes: the documentation in SpatialBounds says that
        the length should be in degrees.  The present class converts
        from degrees to radians before passing to SpatialBounds.
        """
        return numpy.degrees(self._boundLength)

    @boundLength.setter
    def boundLength(self, value):
        self._boundLength = numpy.radians(value)
        self._buildBounds()

    @property
    def boundType(self):
        """
        Tag indicating what sub-class of SpatialBounds should
        be instantiated for this ObservationMetaData.
        """
        return self._boundType

    @boundType.setter
    def boundType(self, value):
        self._boundType = value
        self._buildBounds()

    @property
    def bounds(self):
        """
        Instantiation of a sub-class of SpatialBounds.  This
        is what actually construct the WHERE clause of the SQL
        query associated with this ObservationMetaData.
        """
        return self._bounds

    @property
    def rotSkyPos(self):
        """
        The rotation of the telescope with respect to the sky in degrees.
        It is a parameter you should get from OpSim.
        """
        if self._rotSkyPos is not None:
            return numpy.degrees(self._rotSkyPos)
        else:
            return None

    @rotSkyPos.setter
    def rotSkyPos(self,value):
        if self._phoSimMetaData is not None:
            if 'Opsim_rotskypos' in self._phoSimMetaData:
                raise RuntimeError('WARNING overwriting rotSkyPos ' +
                                   'which was set by phoSimMetaData')

        self._rotSkyPos = numpy.radians(value)


    @property
    def m5(self):
        """
        A dict of m5 (the 5-sigma limiting magnitude) values
        associated with the bandpasses represented by this
        ObservationMetaData.
        """
        return self._m5

    @m5.setter
    def m5(self, value):
        self._m5 = self._assignDictKeyedToBandpass(value, 'm5')


    @property
    def seeing(self):
        """
        A dict of seeing values in arcseconds associated
        with the bandpasses represetned by this ObservationMetaData
        """
        return self._seeing

    @seeing.setter
    def seeing(self, value):
        if self._phoSimMetaData is not None:
            if 'Opsim_rawseeing' in self._phoSimMetaData:
                raise RuntimeError('In ObservationMetaData trying to overwrite seeing ' +
                                   'which was set by phoSimMetaData')

        self._seeing = self._assignDictKeyedToBandpass(value, 'seeing')


    @property
    def site(self):
        """
        An instantiation of the Site class containing information about
        the telescope site.
        """
        return self._site

    @site.setter
    def site(self, value):
        self._site = value

    @property
    def mjd(self):
        """
        The MJD of the observation associated with this ObservationMetaData.
        """
        return self._mjd

    @mjd.setter
    def mjd(self, value):
        """
        Either a float or a ModifiedJulianDate.  If a float, this setter
        assumes that you are passing in International Atomic Time
        """
        if self._phoSimMetaData is not None:
            if 'Opsim_expmjd' in self._phoSimMetaData:
                raise RuntimeError('WARNING overwriting mjd ' +
                                   'which was set by phoSimMetaData')

        if isinstance(value, float):
            self._mjd = ModifiedJulianDate(TAI=value)
        elif isinstance(value, ModifiedJulianDate):
            self._mjd = value
        else:
            raise RuntimeError("You can only set mjd to either a float or a ModifiedJulianDate")


    @property
    def bandpass(self):
        """
        The bandpass associated with this ObservationMetaData.
        Can be a list.
        """
        if self._bandpass is not None:
            return self._bandpass
        else:
            return 'r'

    def setBandpassM5andSeeing(self, bandpassName=None, m5=None, seeing=None):
        """
        Set the bandpasses and associated 5-sigma limiting magnitudes
        and seeing values for this ObservationMetaData.

        @param [in] bandpassName is either a char or a list of chars denoting
        the name of the bandpass associated with this ObservationMetaData.

        @param [in] m5 is the 5-sigma-limiting magnitude(s) associated
        with bandpassName

        @param [in] seeing is the seeing(s) in arcseconds associated
        with bandpassName

        Nothing is returned.  This method just sets member variables of
        this ObservationMetaData.
        """
        if self._phoSimMetaData is not None:
            if 'Opsim_filter' in self._phoSimMetaData:
                raise RuntimeError('WARNING overwriting bandpass ' +
                                   'which was set by phoSimMetaData')

            if 'Opsim_rawseeing' in self._phoSimMetaData:
                raise RuntimeError('WARNING overwriting seeing ' +
                                   'which was set by phoSimMetaData')


        self._bandpass = bandpassName
        self._m5 = self._assignDictKeyedToBandpass(m5, 'm5')
        self._seeing = self._assignDictKeyedToBandpass(seeing, 'seeing')

    @property
    def skyBrightness(self):
        """
        The sky brightness in mags per square arcsecond associated
        with this ObservationMetaData.
        """
        return self._skyBrightness

    @skyBrightness.setter
    def skyBrightness(self, value):
        self._skyBrightness = value

    @property
    def phoSimMetaData(self):
        """
        A dict of parameters expected by PhoSim characterizing this
        ObservationMetaData.  Note that setting this paramter
        could overwrite pointingRA, pointingDec, rotSkyPos,
        MJD, or bandpass and m5 (if they are present in this
        dict).
        """
        return self._phoSimMetaData

    @phoSimMetaData.setter
    def phoSimMetaData(self, value):
        if 'pointingRA' in value:
            self._pointingRA = None

        if 'pointingDec' in value:
            self._pointingDec = None

        if 'Opsim_rotskypos' in value:
            self._rotSkyPos = None

        if 'Opsim_expmjd' in value:
            self._mjd = None

        if 'Opsim_filter' in value:
            self._bandpass = None
            self._m5 = None
            self._seeing = None

        self._assignPhoSimMetaData(value)
