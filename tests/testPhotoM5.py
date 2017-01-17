import unittest
import lsst.utils.tests
from lsst.sims.utils import photo_m5


def setup_module(module):
    lsst.utils.tests.init()


class PhotoM5Test(unittest.TestCase):
    def testm5(self):
        filters = ['u', 'g', 'r', 'i', 'z', 'y']
        kwargs = {}
        # List all parameters to test, with better conditions first
        kwargs['musky'] = [23., 22.]
        kwargs['FWHMeff'] = [1., 1.5]
        kwargs['expTime'] = [60., 30.]
        kwargs['airmass'] = [1., 2.2]
        kwargs['tauCloud'] = [0., 2.2]

        k_default = {}
        for key in kwargs:
            k_default[key] = kwargs[key][0]

        for filtername in filters:
            m5_baseline = photo_m5(filtername, **k_default)
            for key in kwargs:
                k_new = k_default.copy()
                k_new[key] = kwargs[key][1]
                m5_new = photo_m5(filtername, **k_new)
                assert(m5_new < m5_baseline)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
