import numpy as np
import unittest
import lsst.utils.tests as utilsTests


from lsst.sims.utils import ObservationMetaData
from lsst.sims.utils import samplePatchOnSphere
from lsst.sims.utils import sample_obsmetadata
from lsst.sims.catUtils.utils import ObservationMetaDataGenerator

class SamplingTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        """
	cls.obsMetaDataforCat = ObservationMetaData(boundType='circle',
                                          boundLength=np.degrees(0.25),
                                          pointingRA=np.degrees(0.13),
                                          pointingDec=np.degrees(-1.2),
                                          bandpassName=['r'],
					  mjd=49350.)
        ObsMetaData = cls.obsMetaDataforCat
        cls.samples = sample_obsmetadata(ObsMetaData, size=1000)

        cls.theta_c = np.radians(-60.)
        cls.phi_c = np.radians(30.)
        cls.delta = np.radians(30.)
        cls.size = 1000000

        cls.dense_samples = samplePatchOnSphere(phi=cls.phi_c, theta=cls.theta_c,
                                                delta=cls.delta, size=cls.size,
                                                seed=42)
       

    def setUp(self):
        pass


    def test_samplePatchOnSphere(self):
        

        A = lambda theta_min, theta_max: np.sin(theta_max) - np.sin(theta_min)

        theta_min = self.theta_c - self.delta
        theta_max = self.theta_c + self.delta
        tvals = np.arange(theta_min, theta_max, 0.001) 
        tvalsShifted = np.zeros(len(tvals))
        tvalsShifted[:-1] = tvals[1:]

        area = A(tvals, tvalsShifted)
        
        binsize = np.unique(np.diff(tvals))
        assert binsize.size == 1
        normval = np.sum(area) * binsize[0]

        resids = area[:-2] / normval - np.histogram(self.dense_samples[1],
                                                    bins=tvals[:-1],
                                                    normed=True)[0]

        assert all(resids < 0.3)
        

def suite():
    utilsTests.init()
    suites = []
    suites += unittest.makeSuite(SamplingTests)
    return unittest.TestSuite(suites)


def run(shouldExit=False):
    utilsTests.run(suite(), shouldExit)

if __name__ == '__main__':
    run(True)
