import numpy as np
from pycompss.api.api import compss_wait_on
from pycompss.api.task import task


def fft(a):
    """ Compute the one-dimensional discrete Fourier transform using a
    distributed version of the fast Fourier transform algorithm.

    Parameters
    ----------
    a : ndarray
        Input array.

    Returns
    -------

    out : ndarray
        The transformed input.

    Examples
    --------
    >>> from dislib.fft import fft
    >>> fft(np.exp(2j * np.pi * np.arange(8) / 8))
    """
    x = a.flatten()
    n = x.size

    # precompute twiddle factors
    w = np.zeros((n, n), dtype=complex)

    for i in range(n):
        for j in range(n):
            w[i, j] = np.exp(-2 * np.pi * 1j * j / (i + 1))

    lin = []

    for xk in x:
        lin.append(np.array(xk, ndmin=1, dtype=complex))

    while len(lin) > 1:
        lout = []
        ln = len(lin)
        ln2 = int(ln / 2)

        for k in range(ln2):
            lout.append(_reduce(lin[k], lin[k + ln2], w))

        lin = lout

    return compss_wait_on(lin[0])


@task(returns=1)
def _reduce(even, odd, w):
    x = np.concatenate((even, odd))
    n = len(x)
    n2 = int(n / 2)

    for k in range(n2):
        e = x[k]
        o = x[k + n2]
        wk = w[n - 1, k]

        x[k] = e + wk * o
        x[k + n2] = e - wk * o

    return x
