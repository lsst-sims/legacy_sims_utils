from __future__ import division, with_statement
from builtins import zip
from builtins import range
import numpy as np
import unittest
import lsst.utils.tests
import lsst.sims.utils as utils


def setup_module(module):
    lsst.utils.tests.init()


class KdTreeTestCase(unittest.TestCase):

    def testKDTreeAPI(self):
        """
        Make sure the API provided by scipy to the kdTree algorithm is functional.
        """
        _ra = np.linspace(0., 2.*np.pi)
        _dec = np.linspace(-np.pi, np.pi)

        Ra, Dec = np.meshgrid(_ra, _dec)
        tree = utils._buildTree(Ra.flatten(), Dec.flatten())

        x, y, z = utils._xyz_from_ra_dec(_ra, _dec)
        indx = tree.query_ball_point(list(zip(x, y, z)), utils.xyz_angular_radius())

        self.assertEqual(indx.shape, _ra.shape)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()

