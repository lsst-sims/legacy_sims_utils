"""
This file contains coordinate transformation methods and utilities for converting an ra,dec coordinate set to
cartesian coordinates and to grid id using a spatial tree.
"""
from __future__ import division

import numpy as np
from scipy.spatial import cKDTree as kdTree

from lsst.sims.utils.CoordinateTransformations import _xyz_from_ra_dec

__all__ = ['_buildTree']


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
    x, y, z = _xyz_from_ra_dec(ra, dec)
    data = list(zip(x, y, z))
    if np.size(data) > 0:
        try:
            tree = kdTree(data, leafsize=leafsize, balanced_tree=False, compact_nodes=False)
        except TypeError:
            tree = kdTree(data, leafsize=leafsize)
    else:
        raise ValueError('ra and dec should have length greater than 0.')

    return tree
