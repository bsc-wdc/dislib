import numpy as np
from numba import jit


@jit(nopython=True)
def gini_criteria_proxy(l_weight, l_length, r_weight, r_length, not_repeated):
    """
    Maximizing the Gini gain is equivalent to minimizing this proxy function.

    """
    return -(l_weight / l_length + r_weight / r_length) * not_repeated


@jit(nopython=True)
def test_split(sample, y_s, feature, n_classes):
    size = y_s.shape[0]
    if size == 0:
        return 1.7976931348623157e+308, np.float64(np.inf)

    f = feature[sample]
    sort_indices = np.argsort(f)
    y_sorted = y_s[sort_indices]
    f_sorted = f[sort_indices]

    not_repeated = np.empty(size, dtype=np.bool_)
    not_repeated[0: size - 1] = (f_sorted[1:] != f_sorted[:-1])
    not_repeated[size - 1] = True

    l_freq = np.zeros((n_classes, size), dtype=np.int64)
    for i in np.arange(size):
        l_freq[y_sorted[i], i] = 1

    r_freq = np.zeros((n_classes, size), dtype=np.int64)
    r_freq[:, 1:] = l_freq[:, :0:-1]

    for i in np.arange(n_classes):
        l_freq[i] = np.cumsum(l_freq[i])
    for i in np.arange(n_classes):
        r_freq[i] = np.cumsum(r_freq[i])

    l_weight = np.sum(np.square(l_freq), 0)
    r_weight = np.sum(np.square(r_freq), 0)[::-1]

    l_length = np.arange(1, size + 1, dtype=np.int32)
    r_length = np.arange(size - 1, -1, -1, dtype=np.int32)
    r_length[size - 1] = 1  # Avoid div by zero, the right score is 0 anyways

    scores = gini_criteria_proxy(l_weight, l_length, r_weight, r_length,
                                 not_repeated)

    min_index = size - np.argmin(scores[::-1]) - 1

    if min_index + 1 == size:
        b_value = np.float64(np.inf)
    else:
        b_value = (f_sorted[min_index] + f_sorted[min_index + 1]) / 2
    return scores[min_index], b_value

