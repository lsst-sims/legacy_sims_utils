""" Site Class

    Class defines the attributes of the site unless overridden
    ajc@astro 2/23/2010
    
    Restoring this so that the astrometry mixin in Astrometry.py
    can inherit the site information
    danielsf 1/27/2014

"""

__all__ = ["Site"]

class Site (object):
    """ 
    This class will store site information for use in Catalog objects.
    
    Defaults values are LSST site values
    
    @param [in] longitude
    
    @param [in] latitude
    
    @param [in] height (in meters)
    
    @param [in] xPolar is the polar motion x coordinate in radians
    
    @param [in] yPolar is the polar motion y coordinate in radians
    
    @param [in] meanTemperature
    
    @param [in] meanPressure in millibars
    
    @param [in] meanHumidity range 0-1
    
    @param [in] lapseRate in Kelvin per meter
    
    """
    def __init__(self, longitude=-1.2320792, latitude=-0.517781017, height=2650, \
                 xPolar=0, yPolar=0, meanTemperature=284.655, meanPressure=749.3, \
                 meanHumidity=0.4, lapseRate=0.0065):
        
        self.longitude=longitude
        self.latitude=latitude
        self.height=height
        self.xPolar=xPolar
        self.yPolar=yPolar
        self.meanTemperature=meanTemperature
        self.meanPressure=meanPressure
        self.meanHumidity=meanHumidity
        self.lapseRate=lapseRate

