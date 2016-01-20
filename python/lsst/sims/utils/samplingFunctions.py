import numpy as np
from lsst.sims.utils import ObservationMetaData

__all__ = ['sample_obsmetadata', 'samplePatchOnSphere']

def sample_obsmetadata(obsmetadata, size=1, seed=1):
    """
    Sample a square patch on the sphere overlapping obsmetadata
    field of view by picking the area enclosed in
    obsmetadata.pointingRA \pm obsmetadata.boundLength
    obsmetadata.pointingDec \pm obsmetadata.boundLength

    Parameters
    ----------
    obsmetadata: instance of
        `sims.catalogs.generation.db.ObservationMetaData`
    size: integer, optional, defaults to 1
        number of samples

    seed: integer, optional, defaults to 1
        Random Seed used in generating random values 
    Returns
    -------
    tuple of ravals, decvalues
    """

    mydict = obsmetadata.summary
    phi = np.radians(mydict['pointingRA'])
    theta = np.radians(mydict['pointingDec'])
    equalrange = np.radians(mydict['boundLength'])
    ravals, thetavals = samplePatchOnSphere(phi=phi,
					    theta=theta,
					    delta=equalrange,
					    size=size,
                                            seed=seed)
    return ravals, thetavals


def samplePatchOnSphere(phi, theta, delta, size, seed=1):
    """
    Uniformly distributes samples on a patch on a sphere between phi \pm delta,
    and theta \pm delta on a sphere. Uniform distribution implies that the
    number of points in a patch of sphere is proportional to the area of the
    patch. Here, the coordinate system is the usual
    spherical coordinate system but with the azimuthal angle theta going from
    pi/2.0 at the North Pole, to - pi/2.0 at the South Pole, through 0. at the
    equator. This function does not work at the poles. The region must not go
    outside the range of theta due to delta.
 
    Parameters
    ----------
    phi: float, mandatory, radians
	center of the spherical patch in ra with range 
    theta: float, mandatory, radians
    delta: float, mandatory, radians
    size: int, mandatory
        number of samples
    seed : int, optional, defaults to 1
        random Seed used for generating values
    Returns
    -------
    tuple of (phivals, thetavals) where phivals and thetavals are arrays of 
        size size.
    """
    np.random.seed(seed)
    u = np.random.uniform(size=size)
    v = np.random.uniform(size=size)

    phivals = 2. * delta* u + (phi - delta)
    phivals = np.where ( phivals >= 0., phivals, phivals + 2. * np.pi)
    
    # use conventions in spherical coordinates
    theta = np.pi/2.0 - theta
 
    thetamax = theta + delta
    thetamin = theta - delta

    # CDF is cos(thetamin) - cos(theta) / cos(thetamin) - cos(thetamax)
    a = np.cos(thetamin) - np.cos(thetamax)
    thetavals = np.arccos(-v * a + np.cos(thetamin))

    # Get back to -pi/2 to pi/2 range of decs
    thetavals = np.pi/2.0 - thetavals 
    return phivals, thetavals
