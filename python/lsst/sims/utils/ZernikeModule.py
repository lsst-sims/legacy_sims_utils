import numpy as np
import numbers

__all__ = ["_FactorialGenerator", "ZernikePolynomialGenerator"]

class _FactorialGenerator(object):

    def __init__(self):
        self._values = {0:1, 1:1}
        self._max_i = 1

    def evaluate(self, num):
        if num<0:
            raise RuntimeError("Cannot handle negative factorial")

        i_num = int(np.round(num));
        if i_num in self._values:
            return self._values[i_num]

        val = self._values[self._max_i]
        for ii in range(self._max_i, num):
            val *= (ii+1)
            self._values[ii+1] = val

        self._max_i = num
        return self._values[num]


class ZernikePolynomialGenerator(object):

    def __init__(self):
        self._factorial = _FactorialGenerator()
        self._coeffs = {}
        self._powers = {}

    def _validate_nm(self, n, m):
        if not isinstance(n, int) and not isinstance(n, np.int64):
            raise RuntimeError('Zernike polynomial n must be int')
        if not isinstance(m,int) and not isinstance(m, np.int64):
            raise RuntimeError('Zernike polynomial m must be int')

        if n<0:
            raise RuntimeError('Radial Zernike n cannot be negative')
        if m<0:
            raise RuntimeError('Radial Zernike m cannot be negative')
        if n<m:
            raise RuntimeError('Radial Zerniki n must be >= m')

        n = int(n)
        m = int(m)

        return (n, m)

    def _make_polynomial(self, n, m):

        n, m = self._validate_nm(n, m)

        # coefficients taken from
        # https://en.wikipedia.org/wiki/Zernike_polynomials

        n_coeffs = 1+(n-m)//2
        local_coeffs = np.zeros(n_coeffs, dtype=float)
        local_powers = np.zeros(n_coeffs, dtype=float)
        for k in range(0, n_coeffs):
            if k%2 == 0:
                sgn = 1.0
            else:
                sgn = -1.0

            num_fac = self._factorial.evaluate(n-k)
            k_fac = self._factorial.evaluate(k)
            d1_fac = self._factorial.evaluate(((n+m)//2)-k)
            d2_fac = self._factorial.evaluate(((n-m)//2)-k)

            local_coeffs[k] = sgn*num_fac/(k_fac*d1_fac*d2_fac)
            local_powers[k] = n-2*k

        self._coeffs[(n,m)] = local_coeffs
        self._powers[(n,m)] = local_powers

    def _evaluate_radial(self, r, n, m):

        if not isinstance(r, numbers.Number):
            raise RuntimeError("Cannot yet handle arrays of r in Zernike")

        nm_tuple = self._validate_nm(n,m)

        if nm_tuple[0]-nm_tuple[1] % 2 == 1:
            return 0.0

        if nm_tuple not in self._coeffs:
            self._make_polynomial(nm_tuple[0], nm_tuple[1])

        r_term = np.power(r, self._powers[nm_tuple])
        return (self._coeffs[nm_tuple]*r_term).sum()

    def evaluate(self, r, phi, n, m):
        """
        phi is in radians
        """
        radial_part = self._evaluate_radial(r, n, np.abs(m))
        if m>=0:
            return radial_part*np.cos(m*phi)
        return radial_part*np.sin(m*phi)

    def norm(self, n, m):
        nm_tuple = self._validate_nm(n, np.abs(m))
        if nm_tuple[1] == 0:
            eps = 2.0
        else:
            eps = 1.0
        return eps*np.pi/(nm_tuple[0]*2+2)