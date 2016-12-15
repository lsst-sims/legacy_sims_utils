import numpy as np

__all__ = ['photo_m5']


def photo_m5(visitFilter, filtsky, FWHMeff, expTime, airmass, tauCloud=0):
    """Calculate the m5 value, using photometric scaling.  Note, does not include shape of the object SED."""

    # Set up expected extinction (kAtm) and m5 normalization values (Cm) for each filter.
    # The Cm values must be changed when telescope and site parameters are updated.
    #
    # These values are calculated using $SYSENG_THROUGHPUTS/python/calcM5.py.
    # This set of values are calculated using v1.2 of the SYSENG_THROUGHPUTS repo.
    Cm = {'u': 22.74,
          'g': 24.38,
          'r': 24.43,
          'i': 24.30,
          'z': 24.15,
          'y': 23.70}
    dCm_infinity = {'u': 0.75,
                    'g': 0.19,
                    'r': 0.10,
                    'i': 0.07,
                    'z': 0.05,
                    'y': 0.04}
    kAtm = {'u': 0.50,
            'g': 0.21,
            'r': 0.13,
            'i': 0.10,
            'z': 0.07,
            'y': 0.18}
    msky = {'u': 22.95,
            'g': 22.24,
            'r': 21.20,
            'i': 20.47,
            'z': 19.60,
            'y': 18.63}
    # Calculate adjustment if readnoise is significant for exposure time
    # (see overview paper, equation 7)
    Tscale = expTime / 30.0 * np.power(10.0, -0.4*(filtsky - msky[visitFilter]))
    dCm = dCm_infinity[visitFilter] - 1.25*np.log10(1 + (10**(0.8*dCm_infinity[visitFilter]) - 1)/Tscale)
    # Calculate fiducial m5
    m5 = (Cm[visitFilter] + dCm + 0.50*(filtsky-21.0) + 2.5*np.log10(0.7/FWHMeff) +
          1.25*np.log10(expTime/30.0) - kAtm[visitFilter]*(airmass-1.0) + 1.1*tauCloud)
    return m5
