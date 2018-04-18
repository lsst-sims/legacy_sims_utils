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

    def _evaluate_radial_number(self, r, nm_tuple):
        r_term = np.power(r, self._powers[nm_tuple])
        return (self._coeffs[nm_tuple]*r_term).sum()

    def _evaluate_radial_array(self, r, nm_tuple):
        log_r = np.where(r>0.0, np.log(r), -1.0e10)
        r_power = np.exp(np.outer(log_r, self._powers[nm_tuple]))
        return np.dot(r_power, self._coeffs[nm_tuple])

    def _evaluate_radial(self, r, n, m):

        is_array = False
        if not isinstance(r, numbers.Number):
            is_array = True

        nm_tuple = self._validate_nm(n,m)

        if (nm_tuple[0]-nm_tuple[1]) % 2 == 1:
            if is_array:
                return np.zeros(len(r), dtype=float)
            return 0.0

        if nm_tuple not in self._coeffs:
            self._make_polynomial(nm_tuple[0], nm_tuple[1])

        if is_array:
            return self._evaluate_radial_array(r, nm_tuple)

        return self._evaluate_radial_number(r, nm_tuple)

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

    def evaluate_xy(self, x, y, n, m):
        r = np.sqrt(x**2+y**2)
        cos_phi = np.where(r>0.0, x/r, 0.0)
        arccos_phi = np.arccos(cos_phi)
        phi = np.where(y>=0.0, arccos_phi, 0.0-arccos_phi)
        return self.evaluate(r, phi, n, m)
