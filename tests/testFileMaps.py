import os
import unittest
import lsst.utils.tests as utilsTests

from lsst.sims.utils import defaultSpecMap

class FileMapTest(unittest.TestCase):


    def verifyFile(self, file_name, dir_name):
        """
        Verify that specMap[file_name] results in os.path.join(dir_name, file_name+'.gz')
        """
        test_name = defaultSpecMap[file_name]
        control_name = os.path.join(dir_name, file_name+'.gz')
        msg = '%s should map to %s; it actually maps to %s' % (file_name, control_name, test_name)
        self.assertEqual(test_name, control_name, msg=msg)

        add_space = file_name+' '
        self.assertNotEqual(add_space, file_name)
        test_name = defaultSpecMap[add_space]
        msg = '%s should map to %s; it actually maps to %s' % (add_space, control_name, test_name)
        self.assertEqual(test_name, control_name, msg=msg)

        add_space = ' '+file_name
        self.assertNotEqual(add_space, file_name)
        test_name = defaultSpecMap[add_space]
        msg = '%s should map to %s; it actually maps to %s' % (add_space, control_name, test_name)
        self.assertEqual(test_name, control_name, msg=msg)

        add_gz = file_name+'.gz'
        self.assertNotEqual(add_gz, file_name)
        test_name = defaultSpecMap[add_gz]
        msg = '%s should map to %s; it actually maps to %s' % (add_gz, control_name, test_name)
        self.assertEqual(test_name, control_name, msg=msg)


    def testMLT(self):
        """
        Test that defaultSpecMap correctly locates MLT dwarf spectra
        """
        self.verifyFile('lte004-3.5-0.0a+0.0.BT-Settl.spec', 'starSED/mlt')


    def test_m_spec(self):
        """
        Test that defaultSpecMap correctly finds old MLT dwarf spectra
        that begin with 'm'
        """
        self.verifyFile('m5.1Full.dat', 'starSED/old_mlt')


    def test_l4_spec(self):
        """
        Test that defaultSpecMap correctly finds l4Full.dat
        """
        self.verifyFile('l4Full.dat', 'starSED/old_mlt')


    def test_L_spec(self):
        """
        Test that defaultSpecMap correctly find the L#_# spectra
        """
        self.verifyFile('L2_0Full.dat', 'starSED/old_mlt')


    def test_burrows_spec(self):
        """
        Test that defaultSpecMap correctly find the burrows spectra
        """
        self.verifyFile('burrows+2006c91.21_T1400_g5.5_cf_0.3X', 'starSED/old_mlt')


    def testBergeron(self):
        """
        Test that defaultSpecMap correctly locates the bergeron spectra
        """
        self.verifyFile('bergeron_4750_85.dat_4900', 'starSED/wDs')


    def testKurucz(self):
        """
        Test that defaultSpecMap correctly locates the kurucz spectra
        """
        self.verifyFile('km30_5000.fits_g10_5040', 'starSED/kurucz')
        self.verifyFile('kp10_9000.fits_g40_9100', 'starSED/kurucz')


    def testGalaxy(self):
        """
        Test that defaultSpecMap correctly locates the galaxy SEDs
        """
        self.verifyFile('Const.79E06.002Z.spec', 'galaxySED')
        self.verifyFile('Inst.79E06.02Z.spec', 'galaxySED')
        self.verifyFile('Exp.40E08.02Z.spec', 'galaxySED')
        self.verifyFile('Burst.40E08.002Z.spec', 'galaxySED')

def suite():
    """Returns a suite containing all the test cases in this module."""
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(FileMapTest)

    return unittest.TestSuite(suites)

def run(shouldExit=False):
    """Run the tests"""
    utilsTests.run(suite(), shouldExit)

if __name__ == "__main__":
    run(True)
