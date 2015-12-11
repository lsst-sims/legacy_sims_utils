from lsst.sims.utils import taiFromUtc, utcFromTai
from lsst.sims.utils import ut1FromUtc, utcFromUt1
from lsst.sims.utils import dttFromUtc, ttFromTai, tdbFromTt
from lsst.sims.utils import Ut1MinusUtcData

__all__ = ["ModifiedJulianDate"]

class ModifiedJulianDate(object):

    def __init__(self, TAI=None, UTC=None):
        """
        Must specify either:

        @param [in] TAI = the International Atomic Time as an MJD

        or

        @param [in] UTC = Universal Coordinate Time as an MJD
        """

        if TAI is None and UTC is None:
            raise RuntimeError("You must specify either TAI or UTC to "
                               "instantiate ModifiedJulianDate")

        if TAI is not None:
            self._tai = TAI
            self._utc = utcFromTai(self._tai)
        else:
            self._utc = UTC
            self._tai = taiFromUtc(self._utc)

        self._dut = Ut1MinusUtcData.d_ut1_from_utc(self._utc)
        self._ut1 = ut1FromUtc(self._utc)
        self._tt = ttFromTai(self._tai)
        self._tdb = tdbFromTt(self._tt)
        self._dtt = dttFromUtc(self._utc)


    def __eq__(self, other):
        return (self._tai == other._tai) \
               and (self._utc == other._utc) \
               and (self._ut1 == other._ut1) \
               and (self._dut == other._dut) \
               and (self._tt == other._tt) \
               and (self._tdb == other._tdb) \
               and (self._dtt == other._dtt)


    @property
    def TAI(self):
        """
        International Atomic Time as an MJD
        """
        return self._tai


    @property
    def UTC(self):
        """
        Universal Coordinate Time as an MJD
        """
        return self._utc



    @property
    def UT1(self):
        """
        Universal Time as an MJD
        """
        return self._ut1


    @property
    def dut(self):
        """
        UT1-UTC in seconds
        """
        return self._dut


    @property
    def TT(self):
        """
        Terrestrial Time (aka Terrestrial Dynamical Time) as an MJD
        """
        return self._tt


    @property
    def TDB(self):
        """
        Barycentric Dynamical Time as an MJD
        """
        return self._tdb


    @property
    def dtt(self):
        """
        TT - TAI in seconds

        where TT is Terrestrial Time
        and TAI is International Atomic Time
        """
        return self._dtt
