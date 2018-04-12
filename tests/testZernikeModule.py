import unittest

import lsst.utils.tests

from lsst.sims.utils import _FactorialGenerator

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


class MemoryTestClass(lsst.utils.tests.MemoryTestCase):
    pass

if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
