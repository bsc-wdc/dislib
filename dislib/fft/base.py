import numpy as np
from pycompss.api.task import task


def fft(a):
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

        for k in range(ln / 2):
            lout.append(_reduce(lin[k], lin[k + ln / 2], w))

        lin = lout

    return lin[0]


@task(returns=1)
def _reduce(even, odd, w):
    x = np.concatenate((even, odd))
    n = len(x)

    for k in range(n / 2):
        e = x[k]
        o = x[k + n / 2]
        wk = w[n - 1, k]

        x[k] = e + wk * o
        x[k + n / 2] = e - wk * o

    return x


def _base(even, odd, w):
    n = len(even) + len(odd)
    x = np.zeros(n, dtype=complex)
    e = 0
    o = 0

    for k in range(n):
        for m in range(n / 2):
            e += even[m] * w[2 * m, k]
            o += odd[m] * w[2 * m, k]

        x[k] = e + np.exp(-2 * np.pi * 1j * k / n) * o

    return x
