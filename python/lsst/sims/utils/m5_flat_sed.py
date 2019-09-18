import numpy as np

__all__ = ['m5_flat_sed', 'm5_scale']


def m5_scale(expTime, nexp, musky, FWHMeff, airmass, darkSkyMag, Cm, dCm_infinity, kAtm,
             tauCloud=0, baseExpTime=15):
    """ Return m5 (scaled) value for all filters.
    """
    # Calculate adjustment if readnoise is significant for exposure time
    # (see overview paper, equation 7)
    Tscale = expTime / baseExpTime * np.power(10.0, -0.4 * (musky - darkSkyMag))
    dCm = 0.
    dCm += dCm_infinity
    dCm -= 1.25 * np.log10(1 + (10**(0.8 * dCm_infinity) - 1)/Tscale)
    # Calculate m5 for 1 exp - constants here come from definition of Cm/dCm_infinity
    m5 = (Cm + dCm + 0.50 * (musky - 21.0) + 2.5 * np.log10(0.7 / FWHMeff) +
          1.25 * np.log10(expTime / 30.0) - kAtm * (airmass - 1.0) - 1.1 * tauCloud)
    if nexp > 1:
        m5 = 1.25 * np.log10(nexp * 10**(0.8 * m5))
    return m5


def m5_flat_sed(visitFilter, musky, FWHMeff, expTime, airmass, nexp=1, tauCloud=0):
    """Calculate the m5 value, using photometric scaling.  Note, does not include shape of the object SED.

    Parameters
    ----------
    visitFilter : str
         One of u,g,r,i,z,y
    musky : float
        Surface brightness of the sky in mag/sq arcsec
    FWHMeff : float
        The seeing effective FWHM (arcsec)
    expTime : float
        Exposure time for each exposure in the visit.
    airmass : float
        Airmass of the observation (unitless)
    nexp : int, opt
        The number of exposures. Default 1.  (total on-sky time = expTime * nexp)
    tauCloud : float (0.)
        Any extinction from clouds in magnitudes (positive values = more extinction)

    Output
    ------
    m5 : float
        The five-sigma limiting depth of a point source observed in the given conditions.
    """

    # Set up expected extinction (kAtm) and m5 normalization values (Cm) for each filter.
    # The Cm values must be changed when telescope and site parameters are updated.
    #
    # These values are calculated using $SYSENG_THROUGHPUTS/python/calcM5.py.
    # This set of values are calculated using v1.2 of the SYSENG_THROUGHPUTS repo.
    # The exposure time scaling depends on knowing the value of the exposure time used to calculate Cm/etc.

    # Only define the dicts once on initial call
    if not hasattr(m5_flat_sed, 'Cm'):
        # Using Cm / dCm_infinity values calculated for a 1x15s visit.
        # This results in an error of about 0.01 mag in u band for 2x15s visits (0.007 g, <0.005 other bands)
        # but only at most 0.004 mag errors for 1x30s visits.
        # In contrast, using the values from 2x15s visits results in negligible errors for 2x15s visits but
        # 0.01 mag errors in u band for 1x30s visits (<0.003 in other bands).
        # Similarly, using the values from 1x30s visits results in 0 errors for 1x30s visits but
        # 0.015 mag errors in u band for 2x15s visits (<0.005 mag errors in other bands).
        baseExpTime = 15
        m5_flat_sed.Cm = {'u': 23.056,
                          'g': 24.407,
                          'r': 24.433,
                          'i': 24.320,
                          'z': 24.153,
                          'y': 23.726}
        m5_flat_sed.dCm_infinity = {'u': 0.622,
                                    'g': 0.178,
                                    'r': 0.097,
                                    'i': 0.071,
                                    'z': 0.048,
                                    'y': 0.037}
        m5_flat_sed.kAtm = {'u': 0.492,
                            'g': 0.213,
                            'r': 0.126,
                            'i': 0.096,
                            'z': 0.069,
                            'y': 0.170}
        m5_flat_sed.msky = {'u': 22.989,
                            'g': 22.256,
                            'r': 21.196,
                            'i': 20.478,
                            'z': 19.600,
                            'y': 18.612}
    # Calculate adjustment if readnoise is significant for exposure time
    # (see overview paper, equation 7)
    Tscale = expTime / m5_flat_sed.baseExpTime * np.power(10.0, -0.4 * (musky - m5_flat_sed.msky[visitFilter]))
    dCm = 0.
    dCm += m5_flat_sed.dCm_infinity[visitFilter]
    dCm -= 1.25 * np.log10(1 + (10**(0.8 * m5_flat_sed.dCm_infinity[visitFilter]) - 1) / Tscale)
    # Calculate m5 for 1 exp - 30s and other constants here come from definition of Cm/dCm_infinity
    m5 = (m5_flat_sed.Cm[visitFilter] + dCm + 0.50 * (musky - 21.0) + 2.5 * np.log10(0.7 / FWHMeff) +
          1.25 * np.log10(expTime / 30.0) - m5_flat_sed.kAtm[visitFilter] * (airmass - 1.0) - 1.1 * tauCloud)
    # Then combine with coadd if >1 exposure
    if nexp > 1:
        m5 = 1.25 * np.log10(nexp * 10**(0.8 * m5))
    return m5
