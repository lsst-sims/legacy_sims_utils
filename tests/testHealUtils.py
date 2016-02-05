import numpy as np
import unittest
import healpy as hp
import lsst.sims.utils as utils

class TestHealUtils(unittest.TestCase):

    def testRaDecs(self):
        """
        Test that the Ra Dec conversions round-trip
        """

        nside = 64
        hpids = np.arange(hp.nside2npix(nside))
        ra, dec = utils.hpid2RaDec(nside, hpids)

        hpids_return = utils.raDec2Hpid(nside, ra, dec)

        np.testing.assert_array_equal(hpids, hpids_return)

    def testBin(self):
        """
        Test that healbin returns correct values and valid healpy maps.
        """

        ra = np.zeros(3)
        dec = np.zeros(3)
        values = ra*0.+1.

        nside = 128
        hpid = utils.raDec2Hpid(nside, ra[0], dec[0])

        map1 = utils.healbin(ra,dec,values, nside=nside)
        assert(map1[hpid] == 1.)
        assert(hp.maptype(map1) == 0)
        map2 = utils.healbin(ra,dec,values, nside=nside, reduceFunc=np.sum)
        assert(map2[hpid] == 3.)
        assert(hp.maptype(map2) == 0)
        map3 = utils.healbin(ra,dec,values, nside=nside, reduceFunc=np.std)
        assert(map3[hpid] == 0.)
        assert(hp.maptype(map3) == 0)


if __name__ == "__main__":
    unittest.main()
