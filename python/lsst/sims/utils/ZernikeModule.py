import numpy as np

__all__ = ["_FactorialGenerator"]

class _FactorialGenerator(object):

    def __init__(self):
        self._values = {0:1, 1:1}
        self._max_i = 1

    def evaluate(self, num):
        i_num = int(np.round(num));
        if i_num in self._values:
            return self._values[i_num]

        val = self._values[self._max_i]
        for ii in range(self._max_i, num):
            val *= (ii+1)
            self._values[ii+1] = val

        self._max_i = num
        return self._values[num]
