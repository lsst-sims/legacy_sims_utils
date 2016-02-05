import numpy as np
import healpy as hp
from scipy.stats import binned_statistic

__all__ = ['hpid2RaDec', 'raDec2Hpid', 'healbin']


def hpid2RaDec(nside, hpids):
    """
    Correct for healpy being silly and running dec from 0-180.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    hpids : np.array
        Array (or single value) of healpixel IDs.

    Returns
    -------
    raRet : float (or np.array)
        RA positions of the input healpixel IDs. In radians.
    decRet : float (or np.array)
        Dec positions of the input healpixel IDs. In radians.
    """

    lat, lon = hp.pix2ang(nside, hpids)
    decRet = np.pi/2. - lat
    raRet = lon

    return raRet, decRet


def raDec2Hpid(nside, ra, dec):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels.
    dec : np.array
        Dec values to assign to healpixels.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    lat = np.pi/2. - dec
    hpids = hp.ang2pix(nside, lat, ra)
    return hpids


def healbin(ra, dec, values, nside=128, reduceFunc=np.mean):
    """
    Take arrays of ra's, dec's, and value and bin into healpixels. Like numpy.hexbin but for
    bins on a sphere.

    Parameters
    ----------
    ra : np.array
        RA positions of the data points.
    dec : np.array
        Dec positions of the data points.
    values : np.array
        The values at each ra,dec position.
    nside : int
        Healpixel nside resolution. Must be a value of 2^N.
    reduceFunc : function (numpy.mean)
        A function that will return a single value given a subset of `values`.

    Returns
    -------
    mapVals : np.array
        A numpy array that is a valid Healpixel map.
    """

    hpids = raDec2Hpid(nside, ra, dec)
    # Bins centered on the healpixel int values
    bins = np.arange(0, hp.nside2npix(nside)+1)-.5
    mapVals, bin_edges, binnumber = binned_statistic(hpids, values, statistic=reduceFunc, bins=bins)

    # Change any NaNs to healpy mask value
    mapVals[np.isnan(mapVals)] = hp.UNSEEN

    return mapVals
