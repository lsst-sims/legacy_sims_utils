from __future__ import division
import numpy as np
import healpy as hp

__all__ = ['hpid2RaDec', 'raDec2Hpid', 'healbin', '_hpid2RaDec', '_raDec2Hpid', '_healbin', 'moc2array']


def _hpid2RaDec(nside, hpids, **kwargs):
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

    lat, lon = hp.pix2ang(nside, hpids, **kwargs)
    decRet = np.pi / 2.0 - lat
    raRet = lon

    return raRet, decRet


def hpid2RaDec(nside, hpids, **kwargs):
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
    ra, dec = _hpid2RaDec(nside, hpids, **kwargs)
    return np.degrees(ra), np.degrees(dec)


def _raDec2Hpid(nside, ra, dec, **kwargs):
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
    lat = np.pi / 2.0 - dec
    hpids = hp.ang2pix(nside, lat, ra, **kwargs)
    return hpids


def raDec2Hpid(nside, ra, dec, **kwargs):
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
    return _raDec2Hpid(nside, np.radians(ra), np.radians(dec), **kwargs)


def _healbin(ra, dec, values, nside=128, reduceFunc=np.mean, dtype=float, fillVal=hp.UNSEEN):
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

    hpids = _raDec2Hpid(nside, ra, dec)

    order = np.argsort(hpids)
    hpids = hpids[order]
    values = values[order]
    pixids = np.unique(hpids)

    left = np.searchsorted(hpids, pixids)
    right = np.searchsorted(hpids, pixids, side='right')

    mapVals = np.zeros(hp.nside2npix(nside), dtype=dtype) + fillVal

    # Wow, I thought histogram would be faster than the loop, but this has been faster!
    for i, idx in enumerate(pixids):
        mapVals[idx] = reduceFunc(values[left[i]:right[i]])

    # Change any NaNs to fill value
    mapVals[np.isnan(mapVals)] = fillVal

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


def moc2array(data, uniq, nside=128, reduceFunc=np.sum, density=True, fillVal=0.):
    """Convert a Multi-Order Coverage Map to a single nside HEALPix array. Useful
    for converting maps output by LIGO alerts. Expect that future versions of
    healpy or astropy will be able to replace this functionality. Note that this is
    a convienence function that will probably degrade portions of the MOC that are
    sampled at high resolution.

    Details of HEALPix Mulit-Order Coverage map: http://ivoa.net/documents/MOC/20190404/PR-MOC-1.1-20190404.pdf

    Parameters
    ----------
    data : np.array
        Data values for the MOC map
    uniq : np.array
        The UNIQ values for the MOC map
    nside : int (128)
        The output map nside
    reduceFunc : function (np.sum)
        The function to use to combine data into single healpixels.
    density : bool (True)
        If True, multiplies data values by pixel area before applying reduceFunc, and divides
        the final array by the output pixel area. Should be True if working on a probability density MOC.
    fillVal : float (0.)
        Value to fill empty HEALPixels with. Good choices include 0 (default), hp.UNSEEN, and np.nan.

    Returns
    -------
    np.array : HEALpy array of nside. Units should be the same as the input map as processed by reduceFunc.
    """

    # NUNIQ packing, from page 12 of http://ivoa.net/documents/MOC/20190404/PR-MOC-1.1-20190404.pdf
    orders = np.floor(np.log2(uniq / 4) / 2).astype(int)
    npixs = (uniq - 4 * 4**orders).astype(int)

    nsides = 2**orders
    names = ['ra', 'dec', 'area']
    types = [float]*len(names)
    data_points = np.zeros(data.size, dtype=list(zip(names, types)))
    for order in np.unique(orders):
        good = np.where(orders == order)
        ra, dec = _hpid2RaDec(nsides[good][0], npixs[good], nest=True)
        data_points['ra'][good] = ra
        data_points['dec'][good] = dec
        data_points['area'][good] = hp.nside2pixarea(nsides[good][0])

    if density:
        tobin_data = data*data_points['area']
    else:
        tobin_data = data

    result = _healbin(data_points['ra'], data_points['dec'], tobin_data, nside=nside,
                      reduceFunc=reduceFunc, fillVal=fillVal)

    if density:
        good = np.where(result != fillVal)
        result[good] = result[good] / hp.nside2pixarea(nside)

    return result
