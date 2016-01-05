""" Site Class

    Class defines the attributes of the site unless overridden
    ajc@astro 2/23/2010

    Restoring this so that the astrometry mixin in Astrometry.py
    can inherit the site information
    danielsf 1/27/2014

"""

import numpy as np

__all__ = ["Site"]

class Site (object):
    """
    This class will store site information for use in Catalog objects.

    Defaults values are LSST site values taken from the Observatory System Specification
    document

    https://docushare.lsstcorp.org/docushare/dsweb/ImageStoreViewer/LSE-30

    on 4 January 2016

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
    def __init__(self, longitude=-70.7494,
                       latitude=-30.2444,
                       height=2650,
                       temperature=11.5,
                       pressure=750.0,
                       humidity=0.4,
                       lapseRate=0.0065,
                       xPolar=0, yPolar=0):

        self._longitude_rad = np.radians(longitude)
        self._latitude_rad = np.radians(latitude)
        self._longitude_deg = longitude
        self._latitude_deg = latitude
        self._height = height
        self._pressure = pressure
        self._xPolar = np.radians(xPolar)
        self._yPolar= np.radians(yPolar)
        self._temperature_kelvin = temperature+273.15 # in Kelvin
        self._temperature_centigrade = temperature
        self._humidity = humidity
        self._lapseRate = lapseRate


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
