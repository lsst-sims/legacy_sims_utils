"""
This file contains coordinate transformation methods and utilities for converting an ra,dec coordinate set to
cartesian coordinates and to grid id using a spatial tree.
"""
from __future__ import division

import numpy as np
from scipy.spatial import cKDTree as kdtree

__all__ = ['_treexyz', '_rad_length', '_buildTree']


def _treexyz(ra, dec):
    """
    Utility to convert RA,dec positions in x,y,z space, useful for constructing KD-trees.

    Parameters
    ----------
    ra : float or array
        RA in radians
    dec : float or array
        Dec in radians

    Returns
    -------
    x,y,z : floats or arrays
        The position of the given points on the unit sphere.
    """
    # Note ra/dec can be arrays.
    x = np.cos(dec) * np.cos(ra)
    y = np.cos(dec) * np.sin(ra)
    z = np.sin(dec)
    return x, y, z


def _rad_length(radius=1.75):
    """
    Convert an angular radius into a physical radius for a kdtree search.

    Parameters
    ----------
    radius : float
        Radius in degrees.
    """
    x0, y0, z0 = (1, 0, 0)
    x1, y1, z1 = _treexyz(np.radians(radius), 0)
    result = np.sqrt((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)
    return result


def _buildTree(ra, dec, leafsize=100):
    """
    Build KD tree on simDataRA/Dec and set radius (via setRad) for matching.

    Parameters
    ----------
    ra, dec = RA and Dec values (in radians).
    leafsize = the number of Ra/Dec pointings in each leaf node.
    """
    if np.any(np.abs(ra) > np.pi * 2.0) or np.any(np.abs(dec) > np.pi * 2.0):
        raise ValueError('Expecting RA and Dec values to be in radians.')
    x, y, z = _treexyz(ra, dec)
    data = list(zip(x, y, z))
    if np.size(data) > 0:
        try:
            opsimtree = kdtree(data, leafsize=leafsize, balanced_tree=False, compact_nodes=False)
        except TypeError:
            opsimtree = kdtree(data, leafsize=leafsize)
    else:
        raise ValueError('SimDataRA and Dec should have length greater than 0.')

    return opsimtree
