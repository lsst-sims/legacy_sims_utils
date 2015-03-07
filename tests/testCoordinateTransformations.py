import os
import numpy
import unittest
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils

def controlEquationOfEquinoxes(mjd):
    """
    Taken from http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] mjd is a Modified Julian Date

    @param [out] the equation of equinoxes in radians
    """

    JD = mjd + 2400000.5
    D = JD - 2451545.0
    omegaDegrees = 125.04 - 0.052954*D
    Ldegrees = 280.47 + 0.98565*D
    deltaPsiHours = -0.000319*numpy.sin(numpy.radians(omegaDegrees)) \
                    - 0.000024 * numpy.sin(2.0*numpy.radians(Ldegrees))
    epsilonDegrees = 23.4393 - 0.0000004*D
    return (deltaPsiHours/24.0)*2.0*numpy.pi*numpy.cos(numpy.radians(epsilonDegrees))


class testCoordinateTransformations(unittest.TestCase):

    def testEquationOfEquinoxes(self):
        tolerance = 1.0e-5
        numpy.random.seed(32)
        ntests = 100
        mjd = 57087 - 1000.0*(numpy.random.sample(ntests))

        #test vectorized version
        control = controlEquationOfEquinoxes(mjd)
        test = utils.equationOfEquinoxes(mjd)
        self.assertTrue(numpy.abs(test-control).max() < tolerance)

        #test non-vectorized version
        for mm in mjd:
            control = controlEquationOfEquinoxes(mm)
            test = utils.equationOfEquinoxes(mm)
            self.assertTrue(numpy.abs(test-control) < tolerance)

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(testCoordinateTransformations)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
