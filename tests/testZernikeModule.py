import unittest
import numpy as np

import lsst.utils.tests

from lsst.sims.utils import _FactorialGenerator
from lsst.sims.utils import ZernikePolynomialGenerator


def setup_module(module):
    lsst.utils.tests.init()


class FactorialTestCase(unittest.TestCase):

    def test_factorial(self):
        gen = _FactorialGenerator()
        ii = gen.evaluate(9)
        ans = 9*8*7*6*5*4*3*2
        self.assertEqual(ii, ans)

        ii = gen.evaluate(15)
        ans = 15*14*13*12*11*10*9*8*7*6*5*4*3*2
        self.assertEqual(ii, ans)

        ii = gen.evaluate(3)
        ans = 6
        self.assertEqual(ii, ans)

        self.assertEqual(gen.evaluate(0), 1)
        self.assertEqual(gen.evaluate(1), 1)


class ZernikeTestCase(unittest.TestCase):

    longMessage = True

    @classmethod
    def setUpClass(cls):
        cls.d_r = 0.01
        cls.d_phi = 0.005*np.pi
        r_grid = np.arange(0.0, 1.0, cls.d_r)
        phi_grid = np.arange(0.0, 2.0*np.pi, cls.d_phi)
        mesh = np.meshgrid(r_grid, phi_grid)
        cls.r_grid = mesh[0].flatten()
        cls.phi_grid = mesh[1].flatten()

    def test_orthogonality(self):

        polynomials = {}
        z_gen = ZernikePolynomialGenerator()

        for n in range(3):
            for m in range(-n, n+1, 2):
                vals = np.zeros(len(self.r_grid), dtype=float)
                for ii, (rr, pp) in enumerate(zip(self.r_grid, self.phi_grid)):
                    vals[ii] = z_gen.evaluate(rr, pp, n, m)
                nm_tuple = (n,m)
                polynomials[nm_tuple] = vals

        p_keys = list(polynomials.keys())
        for ii in range(len(p_keys)):
            p1_name = p_keys[ii]
            p1 = polynomials[p1_name]
            integral = (p1*p1*self.r_grid*self.d_r*self.d_phi).sum()
            normed_integral = integral/z_gen.norm(p1_name[0], p1_name[1])
            self.assertLess(np.abs(normed_integral-1.0), 0.04)
            for jj in range(ii+1, len(p_keys)):
                p2_name = p_keys[jj]
                p2 = polynomials[p2_name]
                dot = (p1*p2*self.r_grid*self.d_r*self.d_phi).sum()
                msg = '\n%s norm %e\n dot %e\n' % (p1_name, integral, dot)
                self.assertLess(np.abs(dot/integral), 0.01, msg=msg)

    def test_zeros(self):
        rng = np.random.RandomState(88)
        z_gen = ZernikePolynomialGenerator()
        for n in range(4):
            for m in range(-(n-1), n, 2):
                r = rng.random_sample()
                phi = rng.random_sample()*2.0*np.pi
                self.assertAlmostEqual(0.0, z_gen.evaluate(r, phi, n, m), 10)

    def test_array(self):
        z_gen = ZernikePolynomialGenerator()
        n = 2
        m = -2
        val_arr = z_gen.evaluate(self.r_grid, self.phi_grid, n, m)
        self.assertEqual(len(val_arr), len(self.r_grid))
        for ii, (rr, pp) in enumerate(zip(self.r_grid, self.phi_grid)):
            vv = z_gen.evaluate(rr, pp, n, m)
            self.assertAlmostEqual(vv, val_arr[ii], 14)

    def test_xy(self):
        n = 4
        m = 2
        z_gen = ZernikePolynomialGenerator()
        x = self.r_grid*np.cos(self.phi_grid)
        y = self.r_grid*np.sin(self.phi_grid)
        val_arr = z_gen.evaluate_xy(x, y, n, m)
        self.assertGreater(np.abs(val_arr).max(), 1.0e-6)
        for ii, (rr, pp) in enumerate(zip(self.r_grid, self.phi_grid)):
            vv = z_gen.evaluate(rr, pp, n, m)
            self.assertAlmostEqual(vv, val_arr[ii], 14)


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
