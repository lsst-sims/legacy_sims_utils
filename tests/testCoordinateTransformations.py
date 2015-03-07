import os
import numpy
import unittest
import lsst.utils.tests as utilsTests
import lsst.sims.utils as utils

def controlEquationOfEquinoxes(mjd):
    """
    Taken from http://aa.usno.navy.mil/faq/docs/GAST.php

    @param [in] mjd is Terrestrial Time as a Modified Julian Date

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

def controlCalcGmstGast(mjd):
    #From http://aa.usno.navy.mil/faq/docs/GAST.php Nov. 9 2013
    mjdConv = 2400000.5
    jd2000 = 2451545.0
    mjd_o = math.floor(mjd)
    jd = mjd + mjdConv
    jd_o = mjd_o + mjdConv
    h = 24.*(jd-jd_o)
    d = jd - jd2000
    d_o = jd_o - jd2000
    t = d/36525.
    gmst = 6.697374558 + 0.06570982441908*d_o + 1.00273790935*h + 0.000026*t**2
    gast = gmst + utils.equationOfEquinoxes(mjd)
    gmst %= 24.
    gast %= 24.
    return {'GMST':gmst, 'GAST':gast}

class testCoordinateTransformations(unittest.TestCase):

    def setUp(self):
        numpy.random.seed(32)
        ntests = 100
        self.mjd = 57087.0 - 1000.0*(numpy.random.sample(ntests)-0.5)
        self.tolerance = 1.0e-5

    def testEquationOfEquinoxes(self):

        #test vectorized version
        control = controlEquationOfEquinoxes(self.mjd)
        test = utils.equationOfEquinoxes(self.mjd)
        self.assertTrue(numpy.abs(test-control).max() < self.tolerance)

        #test non-vectorized version
        for mm in self.mjd:
            control = controlEquationOfEquinoxes(mm)
            test = utils.equationOfEquinoxes(mm)
            self.assertTrue(numpy.abs(test-control) < self.tolerance)

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
