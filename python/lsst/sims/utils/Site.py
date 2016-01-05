""" Site Class

    Class defines the attributes of the site unless overridden
    ajc@astro 2/23/2010

    Restoring this so that the astrometry mixin in Astrometry.py
    can inherit the site information
    danielsf 1/27/2014

"""

import numpy as np

__all__ = ["Site"]

class LSST_site_parameters(object):
    """
    This is a struct containing the LSST site parameters as defined in

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    (accessed on 4 January 2016)

    This class only exists for initializing Site with LSST parameter values.
    Users should not be accessing this class directly.
    """

    def __init__(self):
        self.longitude=-70.7494 # in degrees
        self.latitude=-30.2444 # in degrees
        self.height=2650 # in meters
        self.temperature=11.5 # in centigrade
        self.pressure=750.0 # in millibars
        self.humidity=0.4 # scale 0-1
        self.lapseRate=0.0065 # in Kelvin per meter
        self.xPolar=0 # x polarmotion component in degrees
        self.yPolar=0 # y polar motion component



class Site (object):
    """
    This class will store site information for use in Catalog objects.

    Defaults values are LSST site values taken from the Observatory System Specification
    document

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    on 4 January 2016

    @param [in] name is the name of the observatory (set to 'LSST' to default
    all other parameters to LSST values defined in

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    as accessed on 4 January 2016

    @param [in] longitude in degrees

    @param [in] latitude in degrees

    @param [in] height (in meters)

    @param [in] temperature in centigrade

    @param [in] pressure in millibars

    @param [in] humidity range 0-1

    @param [in] lapseRate in Kelvin per meter

    @param [in] xPolar is the polar motion x coordinate in degrees

    @param [in] yPolar is the polar motion y coordinate in degress
    """

    def __init__(self,
                 name=None,
                 longitude=None,
                 latitude=None,
                 height=None,
                 temperature=None,
                 pressure=None,
                 humidity=None,
                 lapseRate=None,
                 xPolar=None,
                 yPolar=None):

        default_params = None
        self._name = name
        if self._name is 'LSST':
            default_params = LSST_site_parameters()

        if default_params is not None:
            if longitude is None:
                longitude = default_params.longitude

            if latitude is None:
                latitude = default_params.latitude

            if height is None:
                height = default_params.height

            if temperature is None:
                temperature = default_params.temperature

            if pressure is None:
                pressure = default_params.pressure

            if humidity is None:
                humidity = default_params.humidity

            if lapseRate is None:
                lapseRate = default_params.lapseRate

            if xPolar is None:
                xPolar = default_params.xPolar

            if yPolar is None:
                yPolar = default_params.yPolar


        if longitude is not None:
            self._longitude_rad = np.radians(longitude)
        else:
            self._longitude_rad = None

        if latitude is not None:
            self._latitude_rad = np.radians(latitude)
        else:
            self._latitude_rad = None

        self._longitude_deg = longitude
        self._latitude_deg = latitude
        self._height = height
        self._pressure = pressure

        if xPolar is not None:
            self._xPolar = np.radians(xPolar)
        else:
            self._xPolar = None

        if yPolar is not None:
            self._yPolar= np.radians(yPolar)
        else:
            self._yPolar = None

        if temperature is not None:
            self._temperature_kelvin = temperature+273.15 # in Kelvin
        else:
            self._temperature_kelvin = None

        self._temperature_centigrade = temperature
        self._humidity = humidity
        self._lapseRate = lapseRate


    @property
    def name(self):
        """
        observatory name
        """
        return self._name


    @property
    def longitude_rad(self):
        """
        observatory longitude in radians
        """
        return self._longitude_rad


    @property
    def longitude(self):
        """
        observatory longitude in degrees
        """
        return self._longitude_deg


    @property
    def latitude_rad(self):
        """
        observatory latitude in radians
        """
        return self._latitude_rad


    @property
    def latitude(self):
        """
        observatory latitude in degrees
        """
        return self._latitude_deg


    @property
    def temperature(self):
        """
        mean temperature in centigrade
        """
        return self._temperature_centigrade


    @property
    def temperature_kelvin(self):
        """
        mean temperature in Kelvin
        """
        return self._temperature_kelvin


    @property
    def height(self):
        """
        height in meters
        """
        return self._height


    @property
    def pressure(self):
        """
        mean pressure in millibars
        """
        return self._pressure


    @property
    def humidity(self):
        """
        mean humidity in the range 0-1
        """
        return self._humidity


    @property
    def lapseRate(self):
        """
        temperature lapse rate (in Kelvin per meter)
        """
        return self._lapseRate


    @property
    def xPolar(self):
        """
        x polar motion component in radians
        """
        return self._xPolar


    @property
    def yPolar(self):
        """
        y polar motion component in radians
        """
        return self._yPolar
