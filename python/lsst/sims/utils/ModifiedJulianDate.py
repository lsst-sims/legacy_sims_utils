from lsst.sims.utils import taiFromUtc, utcFromTai, dutFromUtc
from lsst.sims.utils import ut1FromUtc, utcFromUt1
from lsst.sims.utils import dttFromUtc, ttFromTai, tdbFromTt

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
            _tai = TAI
            _utc = utcFromTai(_tai)
        else:
            _utc = UTC
            _tai = taiFromUtc(_utc)

        _dut = dutFromUtc(_utc)
        _ut1 = ut1FromUtc(_utc)
        _tt = ttFromTai(_tai)
        _tdb = tdbFromTt(_tt)
        _dtt = dttFromUtc(_utc)


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
