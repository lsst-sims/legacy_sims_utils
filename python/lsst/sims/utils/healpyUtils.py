import numpy as np
import healpy as hp

__all__ = ['hpid2RaDec', 'raDec2Hpid', 'healbin', '_hpid2RaDec', '_raDec2Hpid', '_healbin']


def _hpid2RaDec(nside, hpids):
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
        RA positions of the input healpixel IDs. In degrees.
    decRet : float (or np.array)
        Dec positions of the input healpixel IDs. In degrees.
    """
    ra, dec = _hpid2RaDec(nside, hpids)
    return np.degrees(ra), np.degrees(dec)


def _raDec2Hpid(nside, ra, dec):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels. Radians.
    dec : np.array
        Dec values to assign to healpixels. Radians.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    lat = np.pi/2. - dec
    hpids = hp.ang2pix(nside, lat, ra)
    return hpids


def raDec2Hpid(nside, ra, dec):
    """
    Assign ra,dec points to the correct healpixel.

    Parameters
    ----------
    nside : int
        Must be a value of 2^N.
    ra : np.array
        RA values to assign to healpixels. Degrees.
    dec : np.array
        Dec values to assign to healpixels. Degrees.

    Returns
    -------
    hpids : np.array
        Healpixel IDs for the input positions.
    """
    return _raDec2Hpid(nside, np.radians(ra), np.radians(dec))


def _healbin(ra, dec, values, nside=128, reduceFunc=np.mean, dtype=float):
    """
    Take arrays of ra's, dec's, and value and bin into healpixels. Like numpy.hexbin but for
    bins on a sphere.

    Parameters
    ----------
    ra : np.array
        RA positions of the data points. Radians.
    dec : np.array
        Dec positions of the data points. Radians
    values : np.array
        The values at each ra,dec position.
    nside : int
        Healpixel nside resolution. Must be a value of 2^N.
    reduceFunc : function (numpy.mean)
        A function that will return a single value given a subset of `values`.
    dtype : dtype ('float')
        Data type of the resulting mask

    Returns
    -------
    mapVals : np.array
        A numpy array that is a valid Healpixel map.
    """

    hpids = raDec2Hpid(nside, ra, dec)

    order = np.argsort(hpids)
    hpids = hpids[order]
    values = values[order]
    pixids = np.unique(hpids)

    left = np.searchsorted(hpids, pixids)
    right = np.searchsorted(hpids, pixids, side='right')

    mapVals = np.zeros(hp.nside2npix(nside), dtype=dtype)+hp.UNSEEN

    # Wow, I thought histogram would be faster than the loop, but this has been faster!
    for i, idx in enumerate(pixids):
        mapVals[idx] = reduceFunc(values[left[i]:right[i]])

    # Change any NaNs to healpy mask value
    mapVals[np.isnan(mapVals)] = hp.UNSEEN

    return mapVals


def healbin(ra, dec, values, nside=128, reduceFunc=np.mean, dtype=float):
    """
    Take arrays of ra's, dec's, and value and bin into healpixels. Like numpy.hexbin but for
    bins on a sphere.

    Parameters
    ----------
    ra : np.array
        RA positions of the data points. Degrees.
    dec : np.array
        Dec positions of the data points. Degrees.
    values : np.array
        The values at each ra,dec position.
    nside : int
        Healpixel nside resolution. Must be a value of 2^N.
    reduceFunc : function (numpy.mean)
        A function that will return a single value given a subset of `values`.
    dtype : dtype ('float')
        Data type of the resulting mask

    Returns
    -------
    mapVals : np.array
        A numpy array that is a valid Healpixel map.
    """
    return _healbin(np.radians(ra), np.radians(dec), values, nside=nside,
                    reduceFunc=reduceFunc, dtype=dtype)
